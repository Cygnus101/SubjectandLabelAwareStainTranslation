#!/usr/bin/env python3
"""
Train a slide-level classifier on pooled top-K patch embeddings.

Workflow:
1. Load attention-weighted patch embeddings from attention_with_top10.csv (or similar).
2. For each slide_id, sort patches by attention_weight and keep top-K entries.
3. Normalize weights within those patches and compute a weighted average embedding.
4. Use augmented_splits.json/augmented_slides.json to respect the same train/val/test split as train_aug.
5. Train a small MLP classifier on the pooled slide embeddings.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
SRC_ROOT = SCRIPT_DIR.parent

import sys

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from utils.path import ensure_project_root_on_syspath, resolve_path, get_project_root  # noqa: E402

ensure_project_root_on_syspath()
PROJECT_ROOT = get_project_root()

DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "train_topk_classifier.json"


def load_json_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text("utf-8"))
    except Exception as exc:
        raise RuntimeError(f"Failed to parse config {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"Config file {path} must contain a JSON object.")
    return data


def _bool_arg(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_splits(slides_path: Path, splits_path: Path) -> dict[str, str]:
    slides = json.loads(slides_path.read_text("utf-8"))
    splits = json.loads(splits_path.read_text("utf-8"))
    mapping: dict[str, str] = {}
    def _assign(indices: Iterable[int], split_name: str) -> None:
        for idx in indices:
            if 0 <= idx < len(slides):
                entry = slides[idx]
                slide_id = str(entry.get("ret_stain_id") or entry.get("he_stain_id"))
                if slide_id:
                    mapping[slide_id] = split_name
    _assign(splits.get("train_indices", []), "train")
    _assign(splits.get("val_indices", []), "val")
    _assign(splits.get("test_indices", []), "test")
    return mapping


def _normalize_weights(weights: np.ndarray) -> np.ndarray:
    weights = np.clip(weights, a_min=0.0, a_max=None)
    total = float(weights.sum())
    if total <= 0:
        return np.ones_like(weights) / max(1, len(weights))
    return weights / total


@dataclass
class SlideEntry:
    embedding: np.ndarray
    label: int
    split: str
    slide_id: str


class EmbeddingPathMapper:
    def __init__(self, old_prefix: Optional[Path], new_root: Optional[Path]) -> None:
        self.old_prefix = old_prefix.resolve() if old_prefix else None
        self.new_root = new_root.resolve() if new_root else None

    def map(self, raw: str) -> Path:
        path = Path(raw).expanduser()
        if self.old_prefix and self.new_root:
            try:
                rel = path.relative_to(self.old_prefix)
                return (self.new_root / rel).resolve()
            except ValueError:
                pass
        if self.new_root and not path.is_absolute():
            return (self.new_root / path).resolve()
        if self.new_root and path.is_absolute() and not path.exists():
            return (self.new_root / path.name).resolve()
        return resolve_path(path)


def _sample_bag_indices(pool_size: int, bag_size: int, rng: random.Random, with_replacement: bool) -> list[int]:
    if pool_size <= 0:
        return []
    if with_replacement or pool_size < bag_size:
        return [rng.randint(0, pool_size - 1) for _ in range(bag_size)]
    return rng.sample(range(pool_size), min(pool_size, bag_size))


def build_slide_entries(
    attention_csv: Path,
    split_map: dict[str, str],
    bag_size: int,
    bags_per_slide: int,
    flag_column: str,
    flag_value: int,
    with_replacement: bool,
    seed: int,
    mapper: EmbeddingPathMapper,
) -> list[SlideEntry]:
    df = pd.read_csv(attention_csv)
    required = {"slide_id", "embedding_path", "attention_weight", "label", flag_column}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"attention CSV missing columns: {sorted(missing)}")
    df_filtered = df[df[flag_column] == flag_value].copy()
    if df_filtered.empty:
        raise RuntimeError(
            f"No rows matched {flag_column} == {flag_value}; cannot build candidate pools."
        )
    grouped = df_filtered.groupby("slide_id")
    entries: list[SlideEntry] = []
    for slide_id, group in grouped:
        split = split_map.get(slide_id)
        if split is None:
            continue
        pool = group.reset_index(drop=True)
        if pool.empty:
            continue
        rng = random.Random(seed + hash(slide_id))
        for _ in range(max(1, bags_per_slide)):
            idxs = _sample_bag_indices(len(pool), bag_size, rng, with_replacement)
            if not idxs:
                continue
            subset = pool.iloc[idxs]
            weights = _normalize_weights(subset["attention_weight"].to_numpy(dtype=np.float32))
            embeddings = []
            dim: Optional[int] = None
            skipped = False
            for path in subset["embedding_path"]:
                try:
                    vec = np.load(mapper.map(str(path))).astype(np.float32).reshape(-1)
                except Exception as exc:
                    logging.warning("Failed to load embedding %s (%s); skipping bag for slide %s", path, exc, slide_id)
                    skipped = True
                    break
                if dim is None:
                    dim = vec.shape[0]
                elif vec.shape[0] != dim:
                    logging.warning("Embedding dimension mismatch for slide %s; skipping bag.", slide_id)
                    skipped = True
                    break
                embeddings.append(vec)
            if skipped or not embeddings:
                continue
            emb_stack = np.stack(embeddings, axis=0)
            pooled = (weights[:, None] * emb_stack).sum(axis=0)
            label = int(subset["label"].iloc[0])
            entries.append(SlideEntry(pooled, label, split, slide_id))
    if not entries:
        raise RuntimeError("No slide entries were constructed; check inputs.")
    return entries


class SlideEmbeddingDataset(Dataset):
    def __init__(self, embeddings: list[np.ndarray], labels: list[int]) -> None:
        self.features = torch.tensor(np.stack(embeddings), dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self) -> int:
        return self.features.size(0)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.features[idx], self.labels[idx]


class SlideMLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, num_classes: int, dropout: float = 0.25) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def build_datasets(
    entries: list[SlideEntry],
    splits: tuple[str, ...] = ("train", "val", "test"),
) -> dict[str, SlideEmbeddingDataset]:
    grouped: dict[str, list[SlideEntry]] = {split: [] for split in splits}
    for entry in entries:
        if entry.split in grouped:
            grouped[entry.split].append(entry)
    datasets: dict[str, SlideEmbeddingDataset] = {}
    for split, items in grouped.items():
        if items:
            embeddings = [item.embedding for item in items]
            labels = [item.label for item in items]
            datasets[split] = SlideEmbeddingDataset(embeddings, labels)
    return datasets


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    preds: list[int] = []
    targets: list[int] = []
    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            logits = model(batch_x)
            pred = logits.argmax(dim=1).cpu().tolist()
            preds.extend(pred)
            targets.extend(batch_y.tolist())
    if not preds:
        return {"accuracy": 0.0, "macro_f1": 0.0, "qwk": 0.0}
    acc = accuracy_score(targets, preds)
    macro_f1 = f1_score(targets, preds, average="macro")
    qwk = cohen_kappa_score(targets, preds, weights="quadratic")
    return {"accuracy": acc, "macro_f1": macro_f1, "qwk": qwk}


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    device: torch.device,
    epochs: int,
    lr: float,
    weight_decay: float,
) -> tuple[nn.Module, list[dict[str, float]]]:
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    history: list[dict[str, float]] = []
    best_state: Optional[dict[str, torch.Tensor]] = None
    best_metric = -float("inf")
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        batches = 0
        for batch_x, batch_y in tqdm(train_loader, desc=f"Epoch {epoch}", leave=False):
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())
            batches += 1
        avg_loss = total_loss / max(1, batches)
        metrics = {"epoch": epoch, "train_loss": avg_loss}
        if val_loader is not None:
            val_metrics = evaluate(model, val_loader, device)
            metrics.update({f"val_{k}": v for k, v in val_metrics.items()})
            score = val_metrics["accuracy"]
            if score > best_metric:
                best_metric = score
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        history.append(metrics)
        logging.info("Epoch %d | train_loss=%.4f%s", epoch, avg_loss, f" | val_acc={metrics.get('val_accuracy'):.4f}" if "val_accuracy" in metrics else "")
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    base = argparse.ArgumentParser(add_help=False)
    base.add_argument("--config", type=str, default=str(DEFAULT_CONFIG))
    config_ns, remaining = base.parse_known_args(argv)
    config = load_json_config(Path(config_ns.config))

    parser = argparse.ArgumentParser(description=__doc__, parents=[base])
    parser.add_argument("--attention-csv", type=str, default=config.get("attention_csv", "attention_with_top10.csv"))
    parser.add_argument("--augmented-slides", type=str, default=config.get("augmented_slides", "augmented_slides.json"))
    parser.add_argument("--augmented-splits", type=str, default=config.get("augmented_splits", "augmented_splits.json"))
    parser.add_argument("--output-dir", type=str, default=config.get("output_dir", "outputs/topk_classifier"))
    parser.add_argument("--run-name", type=str, default=config.get("run_name", "topk_classifier"))
    parser.add_argument("--bag-size", type=int, default=config.get("bag_size", 16))
    parser.add_argument("--bags-per-slide", type=int, default=config.get("bags_per_slide", 4))
    parser.add_argument("--flag-column", type=str, default=config.get("flag_column", "is_top10"))
    parser.add_argument("--flag-value", type=int, default=config.get("flag_value", 1))
    parser.add_argument("--sample-with-replacement", dest="sample_with_replacement", action="store_true")
    parser.add_argument("--no-sample-with-replacement", dest="sample_with_replacement", action="store_false")
    parser.add_argument("--batch-size", type=int, default=config.get("batch_size", 32))
    parser.add_argument("--epochs", type=int, default=config.get("epochs", 50))
    parser.add_argument("--lr", type=float, default=config.get("lr", 1e-4))
    parser.add_argument("--weight-decay", type=float, default=config.get("weight_decay", 1e-4))
    parser.add_argument("--hidden-dim", type=int, default=config.get("hidden_dim", 256))
    parser.add_argument("--num-classes", type=int, default=config.get("num_classes", 4))
    parser.add_argument("--dropout", type=float, default=config.get("dropout", 0.25))
    parser.add_argument("--device", type=str, default=config.get("device"))
    parser.add_argument("--seed", type=int, default=config.get("seed", 42))
    parser.add_argument("--num-workers", type=int, default=config.get("num_workers", 0))
    parser.add_argument("--log-level", type=str, default=config.get("log_level", "INFO"))
    parser.add_argument("--save-best", action="store_true", default=config.get("save_best", True))
    parser.add_argument("--embedding-prefix", type=str, default=config.get("embedding_prefix"))
    parser.add_argument("--embedding-root", type=str, default=config.get("embedding_root"))
    parser.set_defaults(sample_with_replacement=_bool_arg(config.get("sample_with_replacement", True)))
    args = parser.parse_args(remaining)
    args.attention_csv = resolve_path(args.attention_csv)
    args.augmented_slides = resolve_path(args.augmented_slides)
    args.augmented_splits = resolve_path(args.augmented_splits)
    args.output_dir = Path(resolve_path(args.output_dir))
    args.embedding_prefix = Path(resolve_path(args.embedding_prefix)) if args.embedding_prefix else None
    args.embedding_root = Path(resolve_path(args.embedding_root)) if args.embedding_root else None
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))
    set_seed(args.seed)
    if args.device:
        device = torch.device(args.device)
    else:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    logging.info("Device: %s", device)

    split_map = load_splits(Path(args.augmented_slides), Path(args.augmented_splits))
    mapper = EmbeddingPathMapper(args.embedding_prefix, args.embedding_root)
    entries = build_slide_entries(
        Path(args.attention_csv),
        split_map,
        bag_size=args.bag_size,
        bags_per_slide=args.bags_per_slide,
        flag_column=args.flag_column,
        flag_value=args.flag_value,
        with_replacement=args.sample_with_replacement,
        seed=args.seed,
        mapper=mapper,
    )
    datasets = build_datasets(entries)
    if "train" not in datasets:
        raise RuntimeError("Training split is empty after pooling; cannot train classifier.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.output_dir / f"{args.run_name}_metrics.json"
    model_path = args.output_dir / f"{args.run_name}_best.pt"

    def _make_loader(ds: SlideEmbeddingDataset, shuffle: bool) -> DataLoader:
        return DataLoader(
            ds,
            batch_size=args.batch_size,
            shuffle=shuffle,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
        )

    train_loader = _make_loader(datasets["train"], shuffle=True)
    val_loader = _make_loader(datasets["val"], shuffle=False) if "val" in datasets else None
    test_loader = _make_loader(datasets["test"], shuffle=False) if "test" in datasets else None

    in_dim = datasets["train"].features.shape[1]
    model = SlideMLP(in_dim, args.hidden_dim, args.num_classes, dropout=args.dropout).to(device)

    model, history = train(
        model,
        train_loader,
        val_loader,
        device=device,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    metrics: dict[str, dict[str, float]] = {}
    metrics["train"] = evaluate(model, train_loader, device)
    if val_loader is not None:
        metrics["val"] = evaluate(model, val_loader, device)
    if test_loader is not None:
        metrics["test"] = evaluate(model, test_loader, device)

    payload = {"history": history, "metrics": metrics}
    metrics_path.write_text(json.dumps(payload, indent=2))
    logging.info("Saved metrics to %s", metrics_path)

    if args.save_best:
        torch.save(model.state_dict(), model_path)
        logging.info("Saved classifier weights to %s", model_path)


if __name__ == "__main__":
    main()

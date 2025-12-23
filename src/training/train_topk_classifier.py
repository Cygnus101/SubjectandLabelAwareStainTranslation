#!/usr/bin/env python3
"""
Train a slide-level classifier on pooled top-K patch embeddings.

Workflow:
1. Load attention-weighted patch embeddings from attention_with_top10.csv (or similar).
2. For each slide_id, sort patches by attention_weight and keep top-K entries (or hybrid top+random bags).
3. Normalize weights within those patches and compute a weighted average embedding.
4. Use augmented_splits.json/augmented_slides.json to respect the same train/val/test split as train_aug.
5. Train a small MLP classifier on the pooled slide embeddings.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from collections import Counter, defaultdict
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
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


@dataclass
class SlidePool:
    slide_id: str
    split: str
    label: int
    top_pool: pd.DataFrame
    rest_pool: pd.DataFrame
    full_pool: pd.DataFrame


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


def _can_draw(pool_size: int, needed: int, with_replacement: bool) -> bool:
    if needed <= 0:
        return True
    if with_replacement:
        return pool_size > 0
    return pool_size >= needed


def collect_slide_pools(
    attention_csv: Path,
    split_map: dict[str, str],
    flag_column: str,
    flag_value: int,
    pool_mode: str,
    top_fraction: float,
) -> list[SlidePool]:
    df = pd.read_csv(attention_csv)
    required = {"slide_id", "embedding_path", "attention_weight", "label"}
    if pool_mode == "flag":
        required.add(flag_column)
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"attention CSV missing columns: {sorted(missing)}")
    if pool_mode not in {"flag", "fraction"}:
        raise ValueError(f"Unknown pool_mode '{pool_mode}'. Expected 'flag' or 'fraction'.")
    if pool_mode == "fraction" and not (0 < top_fraction <= 1.0):
        raise ValueError("top_fraction must be in (0, 1] when pool_mode='fraction'.")

    pools: list[SlidePool] = []
    for slide_id, group in df.groupby("slide_id"):
        split = split_map.get(slide_id)
        if split is None:
            continue
        group = group.reset_index(drop=True)
        if pool_mode == "flag":
            mask = group[flag_column] == flag_value
            top_pool = group[mask].copy()
        else:
            sorted_group = group.sort_values("attention_weight", ascending=False)
            k = max(1, int(math.ceil(len(sorted_group) * top_fraction)))
            top_pool = sorted_group.head(k).copy()
        if top_pool.empty:
            continue
        top_indices = top_pool.index
        rest_pool = group.drop(top_indices).copy()
        label = int(group["label"].iloc[0])
        pools.append(
            SlidePool(
                slide_id=slide_id,
                split=split,
                label=label,
                top_pool=top_pool.reset_index(drop=True),
                rest_pool=rest_pool.reset_index(drop=True),
                full_pool=group.reset_index(drop=True),
            )
        )
    if not pools:
        raise RuntimeError("No slide pools were built; check inputs.")
    return pools


def _sample_subset(
    pool: pd.DataFrame,
    count: int,
    rng: random.Random,
    with_replacement: bool,
) -> pd.DataFrame:
    if count <= 0 or pool.empty:
        return pool.iloc[[]].copy()
    idxs = _sample_bag_indices(len(pool), count, rng, with_replacement)
    if not idxs:
        return pool.iloc[[]].copy()
    return pool.iloc[idxs].copy()


def generate_bag_entries(
    pools: list[SlidePool],
    bag_size: int,
    bags_per_slide: int,
    min_candidates: int,
    sample_mode: str,
    top_in_bag: int,
    random_from: str,
    with_replacement: bool,
    mapper: EmbeddingPathMapper,
    seed: int,
    log_stats: bool = False,
) -> tuple[list[SlideEntry], dict[str, Any]]:
    sample_mode = sample_mode.lower()
    random_from = random_from.lower()
    if sample_mode not in {"top_only", "hybrid"}:
        raise ValueError("sample_mode must be 'top_only' or 'hybrid'.")
    if random_from not in {"rest", "all"}:
        raise ValueError("random_from must be 'rest' or 'all'.")
    rng = random.Random(seed)
    entries: list[SlideEntry] = []
    stats: dict[str, Any] = defaultdict(int)
    per_slide_unique: dict[str, set[tuple[str, ...]]] = defaultdict(set)
    stats["slide_count"] = len(pools)

    for pool in pools:
        top_pool = pool.top_pool
        rest_pool = pool.rest_pool
        full_pool = pool.full_pool
        stats["top_pool_sizes_sum"] += len(top_pool)
        stats["rest_pool_sizes_sum"] += len(rest_pool)
        stats["slides_considered"] += 1
        if sample_mode == "top_only":
            needed = max(min_candidates, bag_size) if not with_replacement else 1
            if not _can_draw(len(top_pool), needed, with_replacement):
                stats["skipped_slides_top_only"] += 1
                continue
        else:
            top_take = min(top_in_bag, bag_size)
            random_take = bag_size - top_take
            if not _can_draw(len(top_pool), max(1, top_take), with_replacement):
                stats["skipped_slides_insufficient_top"] += 1
                continue
            random_pool = rest_pool if random_from == "rest" else full_pool
            if random_take > 0 and not _can_draw(len(random_pool), random_take, with_replacement):
                stats["skipped_slides_insufficient_rest"] += 1
                continue

        per_slide_set = per_slide_unique[pool.slide_id]
        for bag_idx in range(max(1, bags_per_slide)):
            if sample_mode == "top_only":
                subset = _sample_subset(top_pool, bag_size, rng, with_replacement)
            else:
                top_take = min(top_in_bag, bag_size)
                random_take = bag_size - top_take
                random_pool = rest_pool if random_from == "rest" else full_pool
                top_subset = _sample_subset(top_pool, top_take, rng, with_replacement)
                rand_subset = (
                    _sample_subset(random_pool, random_take, rng, with_replacement)
                    if random_take > 0
                    else None
                )
                subset = pd.concat([top_subset, rand_subset]) if rand_subset is not None else top_subset
            if subset.empty or len(subset) < bag_size:
                stats["bags_too_small"] += 1
                continue
            weights = _normalize_weights(subset["attention_weight"].to_numpy(dtype=np.float32))
            embeddings = []
            dim: Optional[int] = None
            skip_bag = False
            for path in subset["embedding_path"]:
                try:
                    vec = np.load(mapper.map(str(path))).astype(np.float32).reshape(-1)
                except Exception as exc:
                    logging.warning("Failed to load embedding %s (%s); skipping bag for slide %s", path, exc, pool.slide_id)
                    skip_bag = True
                    break
                if dim is None:
                    dim = vec.shape[0]
                elif vec.shape[0] != dim:
                    logging.warning("Embedding dimension mismatch for slide %s; skipping bag.", pool.slide_id)
                    skip_bag = True
                    break
                embeddings.append(vec)
            if skip_bag or not embeddings:
                stats["bags_skipped_load"] += 1
                continue
            emb_stack = np.stack(embeddings, axis=0)
            pooled = (weights[:, None] * emb_stack).sum(axis=0)
            entries.append(SlideEntry(pooled, pool.label, pool.split, pool.slide_id))
            stats["total_bags"] += 1
            signature = tuple(sorted(subset["patch_path"].astype(str).tolist()))
            if signature in per_slide_set:
                stats["duplicate_bags"] += 1
            per_slide_set.add(signature)

    if log_stats:
        for slide_id, sigs in per_slide_unique.items():
            logging.info("Slide %s unique bag combinations this epoch: %d", slide_id, len(sigs))
        avg_top = stats["top_pool_sizes_sum"] / max(1, stats["slides_considered"])
        avg_rest = stats["rest_pool_sizes_sum"] / max(1, stats["slides_considered"])
        logging.info(
            "Bag stats: total=%d, duplicates=%d, skipped(slides)=%d, avg_top_pool=%.1f, avg_rest_pool=%.1f",
            stats["total_bags"],
            stats["duplicate_bags"],
            stats["skipped_slides_top_only"] + stats["skipped_slides_insufficient_top"] + stats["skipped_slides_insufficient_rest"],
            avg_top,
            avg_rest,
        )
    stats["per_slide_unique_counts"] = {slide_id: len(sigs) for slide_id, sigs in per_slide_unique.items()}
    if not entries:
        raise RuntimeError("No slide entries were constructed; check inputs.")
    return entries, stats


class SlideEmbeddingDataset(Dataset):
    def __init__(self, embeddings: list[np.ndarray], labels: list[int]) -> None:
        self.features = torch.tensor(np.stack(embeddings), dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.labels_np = np.array(labels, dtype=np.int64)

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


def make_weighted_sampler(dataset: SlideEmbeddingDataset) -> WeightedRandomSampler:
    counts = Counter(dataset.labels_np.tolist())
    weights = [1.0 / counts[int(lbl)] for lbl in dataset.labels_np]
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, criterion: nn.Module) -> dict[str, float]:
    model.eval()
    preds: list[int] = []
    targets: list[int] = []
    total_loss = 0.0
    batches = 0
    with torch.no_grad():
        for batch_x, batch_y in tqdm(loader, desc="Eval", leave=False):
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            total_loss += float(loss.item())
            batches += 1
            pred = logits.argmax(dim=1).cpu().tolist()
            preds.extend(pred)
            targets.extend(batch_y.tolist())
    if not preds:
        return {"loss": 0.0, "accuracy": 0.0, "macro_f1": 0.0, "qwk": 0.0}
    acc = accuracy_score(targets, preds)
    macro_f1 = f1_score(targets, preds, average="macro")
    qwk = cohen_kappa_score(targets, preds, weights="quadratic")
    avg_loss = total_loss / max(1, batches)
    return {"loss": avg_loss, "accuracy": acc, "macro_f1": macro_f1, "qwk": qwk}


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    noise_std: float,
) -> float:
    model.train()
    total_loss = 0.0
    batches = 0
    for batch_x, batch_y in tqdm(loader, desc="Train", leave=False):
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        if noise_std > 0:
            batch_x = batch_x + torch.randn_like(batch_x) * noise_std
        logits = model(batch_x)
        loss = criterion(logits, batch_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item())
        batches += 1
    return total_loss / max(1, batches)


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
    parser.add_argument("--sample-mode", type=str, choices=["top_only", "hybrid"], default=config.get("sample_mode", "top_only"))
    parser.add_argument("--top-in-bag", type=int, default=config.get("top_in_bag", config.get("bag_size", 16)))
    parser.add_argument("--random-from", type=str, choices=["rest", "all"], default=config.get("random_from", "rest"))
    parser.add_argument("--pool-mode", type=str, choices=["flag", "fraction"], default=config.get("pool_mode", "flag"))
    parser.add_argument("--top-fraction", type=float, default=config.get("top_fraction", 1.0))
    parser.add_argument("--min-candidates", type=int, default=config.get("min_candidates", config.get("bag_size", 16)))
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
    parser.add_argument("--label-smoothing", type=float, default=config.get("label_smoothing", 0.05))
    parser.add_argument("--embedding-noise-std", type=float, default=config.get("embedding_noise_std", 0.0))
    parser.add_argument("--patience", type=int, default=config.get("patience", 6))
    parser.add_argument("--min-delta", type=float, default=config.get("min_delta", 0.0))
    parser.add_argument("--device", type=str, default=config.get("device"))
    parser.add_argument("--seed", type=int, default=config.get("seed", 42))
    parser.add_argument("--num-workers", type=int, default=config.get("num_workers", 0))
    parser.add_argument("--log-level", type=str, default=config.get("log_level", "INFO"))
    parser.add_argument("--save-best", action="store_true", default=config.get("save_best", True))
    parser.add_argument("--use-weighted-sampler", dest="use_weighted_sampler", action="store_true")
    parser.add_argument("--no-weighted-sampler", dest="use_weighted_sampler", action="store_false")
    parser.add_argument("--log-bag-stats", dest="log_bag_stats", action="store_true")
    parser.add_argument("--no-log-bag-stats", dest="log_bag_stats", action="store_false")
    parser.add_argument("--embedding-prefix", type=str, default=config.get("embedding_prefix"))
    parser.add_argument("--embedding-root", type=str, default=config.get("embedding_root"))
    parser.set_defaults(sample_with_replacement=_bool_arg(config.get("sample_with_replacement", True)))
    parser.set_defaults(use_weighted_sampler=_bool_arg(config.get("use_weighted_sampler", True)))
    parser.set_defaults(log_bag_stats=_bool_arg(config.get("log_bag_stats", False)))
    parser.add_argument("--val-draws", type=int, default=config.get("val_draws", 3))
    parser.add_argument("--test-draws", type=int, default=config.get("test_draws", 3))
    args = parser.parse_args(remaining)
    if args.pool_mode == "fraction" and not (0 < args.top_fraction <= 1.0):
        parser.error("--top-fraction must be in (0, 1] when --pool-mode=fraction")
    if args.min_candidates < 1:
        parser.error("--min-candidates must be >= 1")
    if args.bag_size < 1:
        parser.error("--bag-size must be >= 1")
    if args.top_in_bag < 0 or args.top_in_bag > args.bag_size:
        parser.error("--top-in-bag must be between 0 and bag_size")
    if args.label_smoothing < 0 or args.label_smoothing >= 1:
        parser.error("--label-smoothing must be in [0, 1)")
    if args.embedding_noise_std < 0:
        parser.error("--embedding-noise-std must be >= 0")
    if args.patience < 1:
        parser.error("--patience must be >= 1")
    args.attention_csv = resolve_path(args.attention_csv)
    args.augmented_slides = resolve_path(args.augmented_slides)
    args.augmented_splits = resolve_path(args.augmented_splits)
    args.output_dir = Path(resolve_path(args.output_dir))
    args.embedding_prefix = Path(resolve_path(args.embedding_prefix)) if args.embedding_prefix else None
    args.embedding_root = Path(resolve_path(args.embedding_root)) if args.embedding_root else None
    return args

def build_split_loader(
    pools: list[SlidePool],
    args,
    mapper: EmbeddingPathMapper,
    split_name: str,
    device: torch.device,
    seed: int,
    draws: int = 1,
) -> Optional[DataLoader]:
    if not pools:
        return None
    all_entries: list[SlideEntry] = []
    for draw_idx in range(draws):
        entries, _ = generate_bag_entries(
            pools,
            bag_size=args.bag_size,
            bags_per_slide=args.bags_per_slide,
            min_candidates=args.min_candidates,
            sample_mode=args.sample_mode,
            top_in_bag=args.top_in_bag,
            random_from=args.random_from,
            with_replacement=args.sample_with_replacement,
            mapper=mapper,
            seed=seed + draw_idx * 1231,
        )
        all_entries.extend(entries)
    datasets = build_datasets(all_entries, splits=(split_name,))
    ds = datasets.get(split_name)
    if not ds:
        return None
    return DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )


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
    pools = collect_slide_pools(
        Path(args.attention_csv),
        split_map,
        flag_column=args.flag_column,
        flag_value=args.flag_value,
        pool_mode=args.pool_mode,
        top_fraction=args.top_fraction,
    )

    split_pools: dict[str, list[SlidePool]] = defaultdict(list)
    for pool in pools:
        split_pools[pool.split].append(pool)
    if not split_pools.get("train"):
        raise RuntimeError("Training split is empty after pooling; cannot train classifier.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.output_dir / f"{args.run_name}_metrics.json"
    model_path = args.output_dir / f"{args.run_name}_best.pt"

    def _make_loader(ds: SlideEmbeddingDataset, shuffle: bool, is_train: bool = False) -> DataLoader:
        sampler = None
        if is_train and args.use_weighted_sampler:
            sampler = make_weighted_sampler(ds)
        return DataLoader(
            ds,
            batch_size=args.batch_size,
            shuffle=shuffle and sampler is None,
            sampler=sampler,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
        )

    val_loader = build_split_loader(
        split_pools.get("val", []),
        args,
        mapper,
        "val",
        device,
        seed=args.seed + 1337,
        draws=args.val_draws,
    )
    test_loader = build_split_loader(
        split_pools.get("test", []),
        args,
        mapper,
        "test",
        device,
        seed=args.seed + 2023,
        draws=args.test_draws,
    )

    # Temporary dataset to infer feature dimension
    probe_entries, _ = generate_bag_entries(
        split_pools["train"],
        bag_size=args.bag_size,
        bags_per_slide=1,
        min_candidates=args.min_candidates,
        sample_mode=args.sample_mode,
        top_in_bag=args.top_in_bag,
        random_from=args.random_from,
        with_replacement=args.sample_with_replacement,
        mapper=mapper,
        seed=args.seed + 42,
    )
    probe_dataset = build_datasets(probe_entries, splits=("train",)).get("train")
    if probe_dataset is None:
        raise RuntimeError("Unable to build probe dataset for feature inference.")
    in_dim = probe_dataset.features.shape[1]
    model = SlideMLP(in_dim, args.hidden_dim, args.num_classes, dropout=args.dropout).to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    history: list[dict[str, Any]] = []
    best_state: Optional[dict[str, torch.Tensor]] = None
    best_val_loss = float("inf")
    best_epoch = 0
    patience_counter = 0
    train_dataset: Optional[SlideEmbeddingDataset] = None

    for epoch in range(1, args.epochs + 1):
        logging.info("Epoch %d/%d", epoch, args.epochs)
        train_entries, bag_stats = generate_bag_entries(
            split_pools["train"],
            bag_size=args.bag_size,
            bags_per_slide=args.bags_per_slide,
            min_candidates=args.min_candidates,
            sample_mode=args.sample_mode,
            top_in_bag=args.top_in_bag,
            random_from=args.random_from,
            with_replacement=args.sample_with_replacement,
            mapper=mapper,
            seed=args.seed + epoch * 7919,
            log_stats=args.log_bag_stats,
        )
        bag_stats = dict(bag_stats)
        train_dataset = build_datasets(train_entries, splits=("train",)).get("train")
        if train_dataset is None:
            raise RuntimeError("Failed to build training dataset for epoch.")
        train_loader = _make_loader(train_dataset, shuffle=True, is_train=True)
        train_loss = train_one_epoch(
            model,
            train_loader,
            device=device,
            criterion=criterion,
            optimizer=optimizer,
            noise_std=args.embedding_noise_std,
        )
        epoch_record: dict[str, Any] = {"epoch": epoch, "train_loss": train_loss, "bag_stats": bag_stats}
        if val_loader is not None:
            val_loader = build_split_loader(
                split_pools.get("val", []),
                args,
                mapper,
                "val",
                device,
                seed=args.seed + 1337 + epoch,
                draws=args.val_draws,
            )
            val_metrics = evaluate(model, val_loader, device, criterion)
            epoch_record.update({f"val_{k}": v for k, v in val_metrics.items()})
            val_loss = val_metrics["loss"]
            if val_loss + args.min_delta < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                best_epoch = epoch
            else:
                patience_counter += 1
                if patience_counter >= args.patience:
                    logging.info("Early stopping at epoch %d (no val_loss improvement).", epoch)
                    history.append(epoch_record)
                    break
        history.append(epoch_record)

    if best_state is not None:
        model.load_state_dict(best_state)
        logging.info("Loaded best model from epoch %d (val_loss=%.4f)", best_epoch, best_val_loss)

    metrics: dict[str, dict[str, float]] = {}
    train_eval_entries, _ = generate_bag_entries(
        split_pools["train"],
        bag_size=args.bag_size,
        bags_per_slide=args.bags_per_slide,
        min_candidates=args.min_candidates,
        sample_mode=args.sample_mode,
        top_in_bag=args.top_in_bag,
        random_from=args.random_from,
        with_replacement=args.sample_with_replacement,
        mapper=mapper,
        seed=args.seed + 4242,
    )
    train_eval_dataset = build_datasets(train_eval_entries, splits=("train",)).get("train")
    if train_eval_dataset is None:
        raise RuntimeError("Failed to build evaluation dataset for train split.")
    train_final_loader = _make_loader(train_eval_dataset, shuffle=False)
    metrics["train"] = evaluate(model, train_final_loader, device, criterion)
    if val_loader is not None:
        metrics["val"] = evaluate(model, val_loader, device, criterion)
    if test_loader is not None:
        test_loader = build_split_loader(
            split_pools.get("test", []),
            args,
            mapper,
            "test",
            device,
            seed=args.seed + 2023 + epoch,
            draws=args.test_draws,
        )
        metrics["test"] = evaluate(model, test_loader, device, criterion)

    payload = {"history": history, "metrics": metrics}
    if val_loader is not None:
        payload["best_epoch"] = best_epoch
        payload["best_val_loss"] = best_val_loss
    metrics_path.write_text(json.dumps(payload, indent=2))
    logging.info("Saved metrics to %s", metrics_path)

    if args.save_best:
        torch.save(model.state_dict(), model_path)
        logging.info("Saved classifier weights to %s", model_path)




if __name__ == "__main__":
    main()
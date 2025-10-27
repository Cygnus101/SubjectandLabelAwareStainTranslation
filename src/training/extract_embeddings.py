#!/usr/bin/env python3
"""
Generate embeddings for patches listed in metadata.csv using a trained SimCLR encoder.
Embeddings are saved as .npy files alongside an index CSV describing the outputs.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Iterable, Iterator, Optional, Sequence

import numpy as np
import pandas as pd
import torch
from PIL import Image, UnidentifiedImageError
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
from tqdm.auto import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
SRC_ROOT = SCRIPT_DIR.parent
if str(SRC_ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(SRC_ROOT))

from utils.path import ensure_project_root_on_syspath, get_project_root, resolve_path  # noqa: E402
from models.Feature_Extractor import MODEL_REGISTRY  # noqa: E402

ensure_project_root_on_syspath()
PROJECT_ROOT = get_project_root()

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class PatchDataset(Dataset):
    """Lightweight dataset that reads patches from metadata and yields transformed tensors."""

    def __init__(
        self,
        dataframe: pd.DataFrame,
        root_candidates: Sequence[Path],
        transform: T.Compose,
        retries: int = 3,
    ) -> None:
        self.df = dataframe.reset_index(drop=True)
        self.transform = transform
        self.roots = [Path(root).resolve() for root in root_candidates]
        self.retries = max(0, int(retries))

    def __len__(self) -> int:
        return len(self.df)

    def _resolve_patch(self, rel_path: str) -> Optional[Path]:
        path = Path(rel_path).expanduser()
        if path.is_absolute() and path.exists():
            return path
        for root in self.roots:
            candidate = (root / path).resolve()
            if candidate.exists():
                return candidate
        return None

    def __getitem__(self, idx: int) -> tuple[int, torch.Tensor]:
        row = self.df.iloc[idx]
        patch_rel = row["patch_path"]

        attempts = 0
        while True:
            patch_abs = self._resolve_patch(patch_rel)
            if not patch_abs or not patch_abs.exists():
                attempts += 1
                if attempts > self.retries:
                    raise FileNotFoundError(f"Patch not found after {self.retries} retries: {patch_rel}")
                idx = torch.randint(0, len(self.df), ()).item()
                row = self.df.iloc[idx]
                patch_rel = row["patch_path"]
                continue

            try:
                with Image.open(patch_abs) as img:
                    tensor = self.transform(img.convert("RGB"))
                return idx, tensor
            except (UnidentifiedImageError, OSError):
                attempts += 1
                if attempts > self.retries:
                    raise
                idx = torch.randint(0, len(self.df), ()).item()
                row = self.df.iloc[idx]
                patch_rel = row["patch_path"]


def build_transform(image_size: int) -> T.Compose:
    return T.Compose([
        T.Resize(image_size),
        T.CenterCrop(image_size),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def load_encoder(
    checkpoint_path: Path,
    backbone: str,
    proj_hidden_dim: int,
    proj_out_dim: int,
    device: torch.device,
) -> torch.nn.Module:
    model_ctor = MODEL_REGISTRY.get(backbone.lower())
    if model_ctor is None:
        available = ", ".join(sorted(MODEL_REGISTRY))
        raise ValueError(f"Unknown backbone '{backbone}'. Available: {available}")

    model = model_ctor(
        pretrained=False,
        proj_hidden_dim=proj_hidden_dim,
        proj_out_dim=proj_out_dim,
    )

    state = torch.load(checkpoint_path, map_location="cpu")
    if "model" in state:
        model.load_state_dict(state["model"])
    elif "backbone" in state:
        model.backbone.load_state_dict(state["backbone"])
    else:
        raise KeyError(f"Checkpoint {checkpoint_path} missing 'model' or 'backbone' keys.")

    model.projector = torch.nn.Identity()
    return model.to(device).eval()


def iter_batches(loader: DataLoader) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    for indices, images in loader:
        yield indices, images


def run(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    logging.info("Using device: %s", device)

    metadata_path = resolve_path(args.metadata)
    output_root = resolve_path(args.output_dir, allow_missing=True)
    output_root.mkdir(parents=True, exist_ok=True)
    logging.info("Embeddings will be saved to %s", output_root)

    df = pd.read_csv(metadata_path)
    if args.filter_types:
        allowed = {t.lower() for t in args.filter_types}
        df = df[df["type"].str.lower().isin(allowed)]
        logging.info("Filtered to %d rows matching types %s", len(df), sorted(allowed))

    if args.subset is not None:
        subset = float(args.subset)
        if not (0.0 < subset <= 1.0):
            raise ValueError("subset must be in (0, 1].")
        df = df.sample(frac=subset, random_state=args.seed).reset_index(drop=True)
        logging.info("Random subset: %d rows (fraction=%.3f)", len(df), subset)

    root_candidates = [resolve_path(root) for root in ([args.root] if args.root else [metadata_path.parent])]
    for legacy in args.legacy_root:
        root_candidates.append(resolve_path(legacy))

    transform = build_transform(args.image_size)
    dataset = PatchDataset(df, root_candidates, transform, retries=args.max_retries)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type != "cpu"),
        drop_last=False,
    )

    encoder = load_encoder(
        resolve_path(args.checkpoint),
        args.backbone,
        args.proj_hidden_dim,
        args.proj_out_dim,
        device,
    )

    records: list[dict[str, str]] = []
    index_column = "slide_id" if "slide_id" in df.columns else None

    with torch.no_grad():
        for batch_indices, images in tqdm(iter_batches(loader), total=len(loader), desc="embed"):
            images = images.to(device)
            features, _ = encoder(images)
            features = features.cpu().numpy().astype(np.float32)

            for idx_val, embed in zip(batch_indices.tolist(), features):
                row = df.iloc[idx_val]
                patch_rel = row["patch_path"]
                embed_name = f"{Path(patch_rel).stem}.npy"
                embed_path = output_root / embed_name
                np.save(embed_path, embed)

                record = {
                    "patch_path": patch_rel,
                    "embedding_path": str(embed_path),
                }
                if index_column:
                    record[index_column] = row[index_column]
                records.append(record)

    index_path = output_root / "embeddings_index.csv"
    pd.DataFrame(records).to_csv(index_path, index=False)
    logging.info("Stored %d embeddings", len(records))
    logging.info("Index written to %s", index_path)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract patch embeddings using a trained SimCLR encoder.")
    parser.add_argument("--metadata", type=str, default=str(PROJECT_ROOT / "metadata.csv"))
    parser.add_argument("--root", type=str, default=None, help="Primary root directory for relative patch paths.")
    parser.add_argument("--legacy-root", action="append", default=[], help="Additional roots to search for patches.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to SimCLR checkpoint (.pt).")
    parser.add_argument("--backbone", type=str, default="resnet50", choices=sorted(MODEL_REGISTRY.keys()))
    parser.add_argument("--proj-hidden-dim", type=int, default=2048)
    parser.add_argument("--proj-out-dim", type=int, default=128)
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to store embeddings (.npy).")
    parser.add_argument("--subset", type=float, default=None, help="Optional random fraction of rows to process (0-1].")
    parser.add_argument("--filter-types", nargs="*", default=None, help="Optional list of stain types to keep.")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--max-retries", type=int, default=5, help="Retries for missing/unreadable patches.")
    parser.add_argument("--device", type=str, default=None, help="Device override (cuda, mps, cpu).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-level", type=str, default="INFO")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)

    if args.device is None:
        if torch.cuda.is_available():
            args.device = "cuda"
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            args.device = "mps"
        else:
            args.device = "cpu"

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    logging.info("Arguments: %s", json.dumps(vars(args), indent=2, default=str))

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    run(args)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CycleGAN dataloader driven by metadata.csv

- Supports optional --subset flag: only load the first X% of rows in metadata.csv
- Splits by stain_id (keeps patches from the same WSI together)
- Returns train, val, test DataLoaders
- Includes retry logic & logging for unreadable images
"""

import argparse, logging, random
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Sequence
import pandas as pd
from PIL import Image, UnidentifiedImageError

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T

SCRIPT_DIR = Path(__file__).resolve().parent
SRC_ROOT = SCRIPT_DIR.parent
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from utils.path import get_project_root, resolve_path as resolve_project_path

# ---------- path utilities ----------
PROJECT_ROOT = get_project_root()


# ---------- paths & knobs ----------
METADATA_CSV = "metadata.csv"                  # metadata file
LOG_FILE = resolve_project_path("outputs/logs/dataloader_skips.log", allow_missing=True)  # where to log failures

BATCH_SIZE = 64
NUM_WORKERS = 12
PREFETCH_FACTOR = 4
PIN_MEMORY = torch.cuda.is_available()
PERSISTENT_WORKERS = NUM_WORKERS > 0
DROP_LAST = True

TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
SEED = 123
PAIRING = "random"        # "random" = unpaired CycleGAN, "index" = deterministic pairing

# ---------- logging setup ----------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(str(LOG_FILE), mode="a")
    ]
)
logger = logging.getLogger("dataloader")

# ---------- transforms ----------
def ensure_512(img: Image.Image) -> Image.Image:
    """Guarantee image is exactly 512x512 (crop+pad if needed)."""
    if img.size == (512, 512):
        return img
    w, h = img.size
    cw, ch = min(w, 512), min(h, 512)
    img = T.CenterCrop((ch, cw))(img)
    pad_w, pad_h = 512 - img.size[0], 512 - img.size[1]
    if pad_w > 0 or pad_h > 0:
        img = T.Pad((0, 0, pad_w, pad_h), fill=0)(img)
    return img

def random_quadrant_rotation(img: Image.Image) -> Image.Image:
    """Rotate by 0, 90, 180, or 270 degrees randomly."""
    return img.rotate(random.choice((0, 90, 180, 270)))

train_tf = T.Compose([
    T.Lambda(ensure_512),
    T.RandomHorizontalFlip(0.5),
    T.RandomVerticalFlip(0.5),
    T.Lambda(random_quadrant_rotation),
    T.ToTensor(),
    T.Normalize((0.5,)*3, (0.5,)*3),
])

eval_tf = T.Compose([
    T.Lambda(ensure_512),
    T.ToTensor(),
    T.Normalize((0.5,)*3, (0.5,)*3),
])

# ---------- helpers ----------
def _abs_path(rel_path: str, roots: Optional[Sequence[Path]] = None) -> str:
    path = Path(rel_path)
    if path.is_absolute() and path.exists():
        return str(path)

    candidates = []
    if roots:
        candidates.extend(roots)
    candidates.append(Path.cwd())
    candidates.append(PROJECT_ROOT)

    for base in candidates:
        candidate = (Path(base) / path).resolve()
        if candidate.exists():
            return str(candidate)

    return str(resolve_project_path(rel_path, allow_missing=True))

def _split_by_group(paths: List[str], groups: List[str],
                    train_ratio: float, val_ratio: float, seed: int):
    """Split paths into train/val/test, grouped by stain_id."""
    by_group: Dict[str, List[str]] = {}
    for p, g in zip(paths, groups):
        by_group.setdefault(g, []).append(p)

    keys = list(by_group.keys())
    rng = random.Random(seed)
    rng.shuffle(keys)

    n = len(keys)
    n_train = int(n * train_ratio)
    n_val   = int(n * val_ratio)
    k_train = keys[:n_train]
    k_val   = keys[n_train:n_train+n_val]
    k_test  = keys[n_train+n_val:]

    split = {"train": [], "val": [], "test": []}
    for k in k_train: split["train"].extend(by_group[k])
    for k in k_val:   split["val"].extend(by_group[k])
    for k in k_test:  split["test"].extend(by_group[k])
    return split

# ---------- dataset ----------
class HEToReticulinFromMetadata(Dataset):
    """Unpaired CycleGAN dataset with retry & logging."""
    def __init__(self,
                 he_files: List[str],
                 ret_files: List[str],
                 pairing: str = "random",
                 he_transform: Optional[torch.nn.Module] = None,
                 ret_transform: Optional[torch.nn.Module] = None,
                 seed: int = 123,
                 max_retries: int = 5):
        assert pairing in {"random", "index"}
        if not he_files:
            raise ValueError("No H&E files in split.")
        if not ret_files:
            raise ValueError("No Reticulin files in split.")

        self.he_files = he_files
        self.ret_files = ret_files
        self.pairing = pairing
        self.he_tf = he_transform or eval_tf
        self.ret_tf = ret_transform or eval_tf
        self.rng = random.Random(seed)
        self.max_retries = max_retries

        self.failed_count = 0
        self.max_warnings = 20

    def __len__(self):
        return len(self.he_files)

    def _safe_load(self, path: str) -> Optional[Image.Image]:
        try:
            return Image.open(path).convert("RGB")
        except (FileNotFoundError, UnidentifiedImageError, OSError) as e:
            self.failed_count += 1
            if self.failed_count <= self.max_warnings:
                logger.warning(f"Skipped unreadable file: {path} ({type(e).__name__})")
            elif self.failed_count == self.max_warnings + 1:
                logger.warning("Too many unreadable files, suppressing further warnings...")
            return None

    def __getitem__(self, idx: int):
        for _ in range(self.max_retries):
            he_path = self.he_files[idx]
            if self.pairing == "random":
                ret_path = self.rng.choice(self.ret_files)
            else:
                ret_path = self.ret_files[idx % len(self.ret_files)]

            he_img_pil  = self._safe_load(he_path)
            ret_img_pil = self._safe_load(ret_path)

            if he_img_pil is not None and ret_img_pil is not None:
                he_img  = self.he_tf(he_img_pil)
                ret_img = self.ret_tf(ret_img_pil)
                return he_img, ret_img, {"he_path": he_path, "reticulin_path": ret_path}

            idx = self.rng.randint(0, len(self.he_files) - 1)

        raise RuntimeError("Too many retries, dataset may contain many bad files.")

# ---------- reporting wrapper ----------
class ReportingDataLoader:
    """Wraps DataLoader to report failures per epoch."""
    def __init__(self, dataloader: DataLoader, dataset: HEToReticulinFromMetadata):
        self.dataloader = dataloader
        self.dataset = dataset

    def __iter__(self):
        for batch in self.dataloader:
            yield batch
        if self.dataset.failed_count > 0:
            logger.info(f"[EPOCH SUMMARY] {self.dataset.failed_count} files were skipped this epoch.")
            self.dataset.failed_count = 0

    def __len__(self):
        return len(self.dataloader)

# ---------- main loader builder ----------
def make_loaders_from_metadata(
    metadata_csv: str = METADATA_CSV,
    batch_size: int = BATCH_SIZE,
    num_workers: int = NUM_WORKERS,
    pairing: str = PAIRING,
    prefetch_factor: int = PREFETCH_FACTOR,
    persistent_workers: bool = PERSISTENT_WORKERS,
    seed: int = SEED,
    train_ratio: float = TRAIN_RATIO,
    val_ratio: float = VAL_RATIO,
    subset_pct: Optional[float] = None,
    legacy_roots: Optional[Sequence[str]] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader]:

    metadata_path = resolve_project_path(metadata_csv)
    df = pd.read_csv(metadata_path)
    df["type"] = df["type"].astype(str).str.replace("&amp;", "&", regex=False).str.strip()
    root_candidates: list[Path] = []
    if legacy_roots:
        root_candidates.extend([Path(r).expanduser().resolve() for r in legacy_roots])
    df["abs_path"] = df["patch_path"].apply(lambda p: _abs_path(p, root_candidates))

    he_df  = df[df["type"] == "H&E"].copy()
    ret_df = df[df["type"].str.lower().str.contains("reticulin")].copy()

    if subset_pct is not None:
        total = len(df)
        subset_total = max(2, int(round(total * (subset_pct / 100.0))))
        per_stain = subset_total // 2
        he_df = he_df.sample(n=min(per_stain, len(he_df)), random_state=seed)
        ret_df = ret_df.sample(n=min(per_stain, len(ret_df)), random_state=seed)

    he_paths  = he_df["abs_path"].tolist()
    ret_paths = ret_df["abs_path"].tolist()

    if not he_paths: raise RuntimeError("No readable H&E tiles.")
    if not ret_paths: raise RuntimeError("No readable Reticulin tiles.")

    he_lookup  = he_df.set_index("abs_path")["stain_id"].to_dict()
    ret_lookup = ret_df.set_index("abs_path")["stain_id"].to_dict()
    he_groups  = [he_lookup[p] for p in he_paths]
    ret_groups = [ret_lookup[p] for p in ret_paths]

    he_split  = _split_by_group(he_paths,  he_groups,  train_ratio, val_ratio, seed)
    ret_split = _split_by_group(ret_paths, ret_groups, train_ratio, val_ratio, seed)

    def _mk(split: str, train: bool):
        ds = HEToReticulinFromMetadata(
            he_files=he_split[split],
            ret_files=ret_split[split],
            pairing=pairing,
            he_transform=train_tf if train else eval_tf,
            ret_transform=train_tf if train else eval_tf,
            seed=seed,
        )
        dl = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=PIN_MEMORY,
            drop_last=DROP_LAST,
            persistent_workers=persistent_workers,
            prefetch_factor=prefetch_factor if num_workers > 0 else None,
        )
        return ReportingDataLoader(dl, ds)

    return _mk("train", True), _mk("val", False), _mk("test", False)

# ---------- smoke test ----------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CycleGAN dataloaders from metadata")
    parser.add_argument("--subset", type=float, default=None,
                        help="Use only the first X percent of rows in metadata.csv")
    args = parser.parse_args()

    train_loader, val_loader, test_loader = make_loaders_from_metadata(subset_pct=args.subset)
    he, ret, meta = next(iter(train_loader))
    print("H&E batch:", he.shape, "Reticulin batch:", ret.shape)
    print("Example H&E path:", meta["he_path"][0])
    print("Example Ret path:", meta["reticulin_path"][0])

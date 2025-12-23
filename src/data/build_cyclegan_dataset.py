#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CycleGAN dataloader driven by metadata.csv

- Supports optional --subset flag: only load the first X% of rows in metadata.csv
- Splits by stain_id (keeps patches from the same WSI together)
- Returns train, val, test DataLoaders
- Includes retry logic & logging for unreadable images
"""

import argparse
import json
import logging
import random
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Sequence, Any, Set
import pandas as pd
from PIL import Image, UnidentifiedImageError

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
# import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
SRC_ROOT = SCRIPT_DIR.parent
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from utils.path import get_project_root, resolve_path as resolve_project_path

# ---------- path utilities ----------
PROJECT_ROOT = get_project_root()
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "build_cyclegan_dataset.json"

# --- Default paths for augmented split JSONs ---
DEFAULT_AUGMENTED_SLIDES = resolve_project_path("augmented_slides.json", allow_missing=True)
DEFAULT_AUGMENTED_SPLITS = resolve_project_path("augmented_splits.json", allow_missing=True)


# ---------- paths & knobs ----------
METADATA_CSV = "metadata.csv"                  # metadata file
LOG_FILE = resolve_project_path("outputs/logs/dataloader_skips.log", allow_missing=True)  # where to log failures

BATCH_SIZE = 16
NUM_WORKERS = 16
PREFETCH_FACTOR = 4
PIN_MEMORY = True
PERSISTENT_WORKERS = NUM_WORKERS > 0
DROP_LAST = True
DEFAULT_IMAGE_SIZE = 256

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


def _load_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    text = path.read_text("utf-8").strip()
    if not text:
        return {}
    try:
        if path.suffix.lower() == ".json":
            data = json.loads(text)
        else:
            data = yaml.safe_load(text)
    except Exception as exc:
        logger.warning("Failed to load config %s (%s); ignoring defaults.", path, exc)
        return {}
    if data is None:
        return {}
    if not isinstance(data, dict):
        logger.warning("Config %s must contain a mapping/object; ignoring.", path)
        return {}
    return data


def _normalize_lab_list(values: Optional[Sequence[Any]]) -> list[str]:
    labs: list[str] = []
    if values is None:
        return labs
    if isinstance(values, (str, Path)):
        iterable: Sequence[Any] = [values]
    else:
        iterable = values
    for value in iterable:
        if value is None:
            continue
        text = str(value).strip()
        if text:
            labs.append(text)
    return labs


_CONFIG_DEFAULTS = _load_config(DEFAULT_CONFIG)
_CONFIG_VAL_LABS = _normalize_lab_list(_CONFIG_DEFAULTS.get("val_lab"))
_CONFIG_TEST_LABS = _normalize_lab_list(_CONFIG_DEFAULTS.get("test_lab"))
_CONFIG_EXCLUDE_LABS = _normalize_lab_list(_CONFIG_DEFAULTS.get("exclude_lab"))
_DEFAULT_LAB_EXCLUSIONS = tuple(
    dict.fromkeys(_CONFIG_EXCLUDE_LABS + _CONFIG_VAL_LABS + _CONFIG_TEST_LABS)
)

# ---------- transforms ----------

def ensure_size(img: Image.Image, target: int) -> Image.Image:
    """Guarantee image is exactly target x target (crop+pad if needed)."""
    if img.size == (target, target):
        return img
    w, h = img.size
    cw, ch = min(w, target), min(h, target)
    img = T.CenterCrop((ch, cw))(img)
    pad_w, pad_h = target - img.size[0], target - img.size[1]
    if pad_w > 0 or pad_h > 0:
        img = T.Pad((0, 0, pad_w, pad_h), fill=0)(img)
    if img.size != (target, target):
        img = img.resize((target, target), Image.BILINEAR)
    return img

# --- Picklable wrapper for ensure_size for PyTorch DataLoader multiprocessing ---
class EnsureSizeTransform:
    """Picklable callable wrapper for ensure_size().

    Needed because DataLoader multiprocessing (spawn) on macOS requires transforms
    to be picklable; local functions inside build_transforms are not.
    """

    def __init__(self, target: int):
        self.target = int(target)

    def __call__(self, img: Image.Image) -> Image.Image:
        return ensure_size(img, self.target)

def random_quadrant_rotation(img: Image.Image) -> Image.Image:
    """Rotate by 0, 90, 180, or 270 degrees randomly."""
    return img.rotate(random.choice((0, 90, 180, 270)))

def build_transforms(image_size: int) -> Tuple[T.Compose, T.Compose]:
    train_tf = T.Compose([
        EnsureSizeTransform(image_size),
        T.RandomHorizontalFlip(0.5),
        T.RandomVerticalFlip(0.5),
        T.Lambda(random_quadrant_rotation),
        T.ToTensor(),
        T.Normalize((0.5,) * 3, (0.5,) * 3),
    ])

    eval_tf = T.Compose([
        EnsureSizeTransform(image_size),
        T.ToTensor(),
        T.Normalize((0.5,) * 3, (0.5,) * 3),
    ])
    return train_tf, eval_tf

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

def _detect_lab_column(df: pd.DataFrame) -> Optional[str]:
    for candidate in ["Lab No.", "lab_id", "Lab_No", "lab", "patient_id"]:
        if candidate in df.columns:
            return candidate
    return None

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


def _load_explicit_lab_splits(
    split_path: str,
    lab_ids: Sequence[str],
    slides_json: Optional[str] = None,
) -> Dict[str, Set[str]]:
    with Path(split_path).expanduser().resolve(strict=False).open("r", encoding="utf-8") as fp:
        payload = json.load(fp)
    if not isinstance(payload, dict):
        raise ValueError(f"Split file {split_path} must contain a mapping with train/val/test indices.")

    def _indices(key: str) -> List[int]:
        values = payload.get(key, [])
        if values is None:
            return []
        return [int(v) for v in values]

    sorted_ids = [sid for sid in sorted({str(s).strip() for s in lab_ids if str(s).strip()})]
    if not sorted_ids:
        raise RuntimeError("Cannot build explicit splits because no stain IDs were found in metadata.")

    slides_data: Optional[list] = None
    if slides_json:
        slides_path = Path(slides_json).expanduser().resolve(strict=False)
        if slides_path.exists():
            try:
                slides_data = json.loads(slides_path.read_text("utf-8"))
            except Exception as exc:
                logger.warning("Failed to read augmented slides from %s (%s); falling back to metadata order.", slides_path, exc)

    def _map_to_ids(idxs: List[int], split_name: str) -> Set[str]:
        result: Set[str] = set()
        for idx in idxs:
            if slides_data and 0 <= idx < len(slides_data):
                lab = str(slides_data[idx].get("lab_id", "")).strip()
                if lab:
                    result.add(lab)
                    continue
            if idx < 0 or idx >= len(sorted_ids):
                logger.warning(
                    "Split %s references stain index %d outside range [0, %d); skipping.",
                    split_name,
                    idx,
                    len(sorted_ids),
                )
                continue
            result.add(sorted_ids[idx])
        return result

    return {
        "train": _map_to_ids(_indices("train_indices"), "train"),
        "val": _map_to_ids(_indices("val_indices"), "val"),
        "test": _map_to_ids(_indices("test_indices"), "test"),
    }


def _split_paths_by_lab(
    paths: List[str],
    lookup: Dict[str, str],
    lab_splits: Dict[str, Set[str]],
) -> Dict[str, List[str]]:
    result = {name: [] for name in ("train", "val", "test")}
    dropped = 0
    for path in paths:
        lab = lookup.get(path, "").strip()
        assigned = False
        for split_name, allowed in lab_splits.items():
            if lab and lab in allowed:
                result.setdefault(split_name, []).append(path)
                assigned = True
                break
        if not assigned:
            dropped += 1
    if dropped:
        logger.warning("Dropped %d patch paths not covered by explicit lab splits.", dropped)
    return result

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
class EmptyCycleGANDataset(Dataset):
    """Minimal dataset placeholder for empty splits."""
    def __init__(self):
        self.failed_count = 0

    def __len__(self) -> int:
        return 0

    def __getitem__(self, index: int):
        raise IndexError("EmptyCycleGANDataset contains no items.")


class ReportingDataLoader:
    """Wraps DataLoader to report failures per epoch."""
    def __init__(self, dataloader: DataLoader, dataset: Optional[HEToReticulinFromMetadata]):
        self.dataloader = dataloader
        self.dataset = dataset

    def __iter__(self):
        for batch in self.dataloader:
            yield batch
        if self.dataset and self.dataset.failed_count > 0:
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
    subset_order: str = "random",
    image_size: int = DEFAULT_IMAGE_SIZE,
    legacy_roots: Optional[Sequence[str]] = None,
    exclude_labs: Optional[Sequence[str]] = None,
    split_json: Optional[str] = None,
    augmented_slides: Optional[str] = None,
    augmented_splits: Optional[str] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader]:

    metadata_path = resolve_project_path(metadata_csv)
    df = pd.read_csv(metadata_path)
    if "stain_id" not in df.columns:
        raise ValueError("metadata.csv must include a 'stain_id' column for explicit splits.")
    df["stain_id"] = df["stain_id"].astype(str).str.strip()
    df["type"] = df["type"].astype(str).str.replace("&amp;", "&", regex=False).str.strip()
    root_candidates: list[Path] = []
    if legacy_roots:
        root_candidates.extend([Path(r).expanduser().resolve() for r in legacy_roots])
    df["abs_path"] = df["patch_path"].apply(lambda p: _abs_path(p, root_candidates))

    lab_col = _detect_lab_column(df)
    if lab_col is None:
        raise ValueError("metadata.csv must include a lab column (e.g., 'Lab No.') for lab-based splits.")
    df["lab_norm"] = df[lab_col].astype(str).str.strip()

    # --- default to augmented split JSON if available ---
    if split_json is None:
        if augmented_splits is None and Path(DEFAULT_AUGMENTED_SPLITS).exists():
            augmented_splits = DEFAULT_AUGMENTED_SPLITS
        if augmented_slides is None and Path(DEFAULT_AUGMENTED_SLIDES).exists():
            augmented_slides = DEFAULT_AUGMENTED_SLIDES

    explicit_lab_splits: Optional[Dict[str, Set[str]]] = None
    if split_json:
        explicit_lab_splits = _load_explicit_lab_splits(split_json, df["lab_norm"].tolist(), slides_json=augmented_slides)
        logger.info(
            "Using explicit split file %s | train=%d | val=%d | test=%d lab IDs",
            split_json,
            len(explicit_lab_splits["train"]),
            len(explicit_lab_splits["val"]),
            len(explicit_lab_splits["test"]),
        )
    elif augmented_slides and augmented_splits:
        slides = json.loads(resolve_project_path(augmented_slides).read_text("utf-8"))
        splits = json.loads(resolve_project_path(augmented_splits).read_text("utf-8"))
        explicit_lab_splits = {"train": set(), "val": set(), "test": set()}
        for split_name in ("train", "val", "test"):
            key = f"{split_name}_indices"
            for idx in splits.get(key, []):
                if 0 <= idx < len(slides):
                    lab = str(slides[idx].get("lab_id", "")).strip()
                    if lab:
                        explicit_lab_splits[split_name].add(lab)
        logger.info(
            "Using augmented splits %s/%s | train=%d | val=%d | test=%d labs",
            augmented_slides,
            augmented_splits,
            len(explicit_lab_splits["train"]),
            len(explicit_lab_splits["val"]),
            len(explicit_lab_splits["test"]),
        )

    # --- Log split source ---
    if explicit_lab_splits is not None:
        logger.info("Dataset split source: explicit JSON (defaulted=%s)",
                    split_json is None)
    else:
        logger.warning("Dataset split source: random (NO explicit split found)")
    subset_via_lab = False
    if explicit_lab_splits is not None and subset_pct is not None:
        logger.warning("Explicit split file provided; ignoring subset_pct/subset_order arguments.")
        subset_pct = None
    if subset_pct is not None and lab_col is not None:
        labs = df[lab_col].astype(str).str.strip()
        labs = labs[labs.astype(bool)]
        unique_labs = labs.unique().tolist()
        if unique_labs:
            pct = max(0.0, min(100.0, float(subset_pct)))
            target = max(1, int(round(len(unique_labs) * (pct / 100.0))))
            if target < len(unique_labs):
                if subset_order == "ascending":
                    selected = sorted(unique_labs)[:target]
                elif subset_order == "descending":
                    selected = sorted(unique_labs, reverse=True)[:target]
                else:
                    rng = random.Random(seed)
                    selected = rng.sample(unique_labs, target)
            else:
                selected = unique_labs
            df = df[df[lab_col].astype(str).str.strip().isin(selected)]
            subset_via_lab = True
            logger.info(
                "Subset active (%s): %.2f%% of labs -> %d labs",
                subset_order,
                pct,
                len(selected),
            )
        else:
            logger.warning("Subset requested but lab column %s is empty.", lab_col)

    if exclude_labs is None or (isinstance(exclude_labs, Sequence) and len(exclude_labs) == 0):
        # When using explicit splits we should not also drop labs via config defaults,
        # otherwise the requested subset can disappear entirely.
        exclude_labs = () if explicit_lab_splits is not None else _DEFAULT_LAB_EXCLUSIONS
    normalized_exclude = _normalize_lab_list(exclude_labs)
    if normalized_exclude:
        if lab_col is None:
            logger.warning(
                "Lab exclusion requested for %d lab(s) but metadata is missing a lab column; skipping.",
                len(normalized_exclude),
            )
        else:
            lab_series = df[lab_col].astype(str).str.strip()
            mask = ~lab_series.isin(normalized_exclude)
            before = len(df)
            df = df[mask].copy()
            removed = before - len(df)
            if removed > 0:
                logger.info(
                    "Excluded %d rows across %d lab(s) from CycleGAN metadata.",
                    removed,
                    len(normalized_exclude),
                )
            if df.empty:
                raise RuntimeError("All rows were removed after applying lab exclusion filter.")

    he_df  = df[df["type"] == "H&E"].copy()
    ret_df = df[df["type"].str.lower().str.contains("reticulin")].copy()

    if subset_pct is not None and not subset_via_lab:
        total = len(df)
        subset_total = max(2, int(round(total * (subset_pct / 100.0))))
        per_stain = max(1, subset_total // 2)
        he_df = he_df.sample(n=min(per_stain, len(he_df)), random_state=seed)
        ret_df = ret_df.sample(n=min(per_stain, len(ret_df)), random_state=seed)
        logger.info(
            "Subset active (random rows): %.2f%% of rows -> %d H&E / %d Reticulin samples",
            subset_pct,
            len(he_df),
            len(ret_df),
        )

    he_paths  = he_df["abs_path"].tolist()
    ret_paths = ret_df["abs_path"].tolist()

    if not he_paths: raise RuntimeError("No readable H&E tiles.")
    if not ret_paths: raise RuntimeError("No readable Reticulin tiles.")

    he_lab_lookup = he_df.set_index("abs_path")["lab_norm"].to_dict()
    ret_lab_lookup = ret_df.set_index("abs_path")["lab_norm"].to_dict()
    he_groups = [he_lab_lookup.get(p, "") for p in he_paths]
    ret_groups = [ret_lab_lookup.get(p, "") for p in ret_paths]

    if explicit_lab_splits is not None:
        he_split = _split_paths_by_lab(he_paths, he_lab_lookup, explicit_lab_splits)
        ret_split = _split_paths_by_lab(ret_paths, ret_lab_lookup, explicit_lab_splits)
        logger.info(
            "Explicit split patch counts | H&E train/val/test = %s | Reticulin train/val/test = %s",
            {k: len(v) for k, v in he_split.items()},
            {k: len(v) for k, v in ret_split.items()},
        )
    else:
        he_split  = _split_by_group(he_paths,  he_groups,  train_ratio, val_ratio, seed)
        ret_split = _split_by_group(ret_paths, ret_groups, train_ratio, val_ratio, seed)

    train_tf, eval_tf = build_transforms(image_size)

    def _mk(split: str, train: bool):
        he_files = he_split.get(split, [])
        ret_files = ret_split.get(split, [])
        if not he_files or not ret_files:
            logger.warning(
                "Split %s has insufficient data (H&E=%d | Reticulin=%d); returning empty loader.",
                split,
                len(he_files),
                len(ret_files),
            )
            empty_dataset = EmptyCycleGANDataset()
            empty_loader = DataLoader(
                empty_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=0,
            )
            return ReportingDataLoader(empty_loader, empty_dataset)
        logger.info(
            "Creating %s split | H&E patches: %d | Reticulin patches: %d",
            split,
            len(he_files),
            len(ret_files),
        )
        ds = HEToReticulinFromMetadata(
            he_files=he_files,
            ret_files=ret_files,
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
    parser.add_argument(
        "--subset-order",
        type=str,
        choices=["random", "ascending", "descending"],
        default="random",
        help="Subset labs deterministically (ascending/descending) or randomly.",
    )
    parser.add_argument("--image-size", type=int, default=DEFAULT_IMAGE_SIZE)
    parser.add_argument(
        "--exclude-lab",
        action="append",
        default=None,
        help=(
            "Lab ID to exclude from all splits (repeatable). "
            "Defaults to labs listed for val/test in configs/build_cyclegan_dataset.json if not provided."
        ),
    )
    args = parser.parse_args()

    exclude_labs = args.exclude_lab if args.exclude_lab is not None else _DEFAULT_LAB_EXCLUSIONS

    train_loader, val_loader, test_loader = make_loaders_from_metadata(
        subset_pct=args.subset,
        subset_order=args.subset_order,
        image_size=args.image_size,
        exclude_labs=exclude_labs,
    )
    he, ret, meta = next(iter(train_loader))
    print("H&E batch:", he.shape, "Reticulin batch:", ret.shape)
    print("Example H&E path:", meta["he_path"][0])
    print("Example Ret path:", meta["reticulin_path"][0])

#!/usr/bin/env python3
"""Train Augmented CycleGAN with slide-level classification + contrastive losses."""

from __future__ import annotations
import os
import argparse
import json
import logging
import random
import re
import sys
import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd
from PIL import Image, UnidentifiedImageError
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms as T
from torchvision.utils import save_image
from tqdm.auto import tqdm
from distutils.util import strtobool

SCRIPT_DIR = Path(__file__).resolve().parent
SRC_ROOT = SCRIPT_DIR.parent
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from utils.path import ensure_project_root_on_syspath, get_project_root, resolve_path  # noqa: E402

ensure_project_root_on_syspath()
PROJECT_ROOT = get_project_root()
DEFAULT_TRAIN_AUGMENT_CONFIG = PROJECT_ROOT / "configs" / "train_augment_default.json"

LOGGER = logging.getLogger("train_augment")
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
GRADE_PATTERN = re.compile(r"(\d+)")


try:  # pragma: no cover - prefer the v3 architecture used in main CycleGAN training
    from models.Backbone_model.CycleGANv3 import UNetGenerator as CycleGenerator  # type: ignore
    from models.Backbone_model.CycleGANv3 import Discriminator as CycleDiscriminator  # type: ignore
except Exception:
    # Fallback to v2 if v3 is not available on this machine
    from models.Backbone_model.CycleGANv2 import UNetGenerator as CycleGenerator  # type: ignore
    from models.Backbone_model.CycleGANv2 import Discriminator as CycleDiscriminator  # type: ignore

try:
    from models.Feature_Extractor.Resnet50 import SimCLRModel as FeatureExtractor  # type: ignore
except Exception:  # pragma: no cover
    FeatureExtractor = None  # type: ignore

from models.Feature_Extractor import MODEL_REGISTRY as ENCODER_REGISTRY  # type: ignore
from models.Classification.abmil import ABMIL, SlideClassifier  # noqa: E402


# @dataclass
# class SlideRecord:
#     lab_id: str
#     he_stain_id: str
#     ret_stain_id: str
#     grade: int
#     he_paths: list[Path]
#     ret_paths: list[Path]

#     @property
#     def slide_id(self) -> str:
#         return f"{self.lab_id}|{self.he_stain_id}|{self.ret_stain_id}"

#     @property
#     def contrastive_id(self) -> str:
#         return _contrastive_group_id(self.lab_id)


@dataclass
class Slide:
    lab_id: str
    he_stain_id: str
    ret_stain_id: str
    grade: int
    he_paths: list[Path]
    he_top_flags: list[int]
    ret_paths: list[Path]
    ret_top_flags: list[int]

    @property
    def contrastive_id(self) -> str:
        return _contrastive_group_id(self.lab_id)


def _numeric_tokens(value: str) -> list[int]:
    return [int(match.group()) for match in re.finditer(r"\d+", value)]


def _bool_arg(value: str | bool | int) -> bool:
    if isinstance(value, bool):
        return value
    return bool(strtobool(str(value)))


def _stain_prefix(value: str) -> str:
    cleaned = re.sub(r"\d+", "", value)
    return cleaned.replace("_", "").replace("-", "").strip().lower()


def _contrastive_group_id(value: str) -> str:
    value = str(value).strip()
    return value.replace("-", "_") or value


def _closest_stain_id(source: str, candidates: Sequence[str]) -> Optional[str]:
    if not candidates:
        return None
    if source in candidates:
        return source

    src_tokens = _numeric_tokens(source)
    src_prefix = _stain_prefix(source)

    def score(candidate: str) -> tuple[int, int, str]:
        cand_tokens = _numeric_tokens(candidate)
        cand_prefix = _stain_prefix(candidate)
        prefix_penalty = 0 if cand_prefix == src_prefix and src_prefix else 1
        if src_tokens and cand_tokens:
            diff = abs(src_tokens[-1] - cand_tokens[-1])
        else:
            diff = abs(len(source) - len(candidate))
        return (prefix_penalty, diff, candidate)

    return min(candidates, key=score)


def _resolve_patch_path(raw: str, data_root: Optional[Path]) -> Path:
    path = Path(str(raw))
    if path.is_absolute():
        return path
    base = Path(data_root) if data_root else PROJECT_ROOT
    return (base / path).resolve()

def remap_load_slides(slides_path, data_root, required_indices):
    payload = json.loads(Path(slides_path).read_text("utf-8"))
    if not isinstance(payload, list):
        raise ValueError("Slides JSON must contain a list.")

    slides = []
    index_map = {}   # old_index → new_index

    for new_idx, old_idx in enumerate(required_indices):
        entry = payload[old_idx]

        lab_id = str(entry["lab_id"]) 
        he_id  = str(entry["he_stain_id"])
        ret_id = str(entry["ret_stain_id"])
        grade  = int(entry["grade"])

        def resolve_paths(meta):
            paths = []
            flags = []
            for p in meta:
                paths.append(_resolve_patch_path(p["patch_path"], data_root))
                flags.append(int(p.get("is_top10", 0)))
            return paths, flags

        he_paths, he_flags = resolve_paths(entry.get("he_patches", []))
        ret_paths, ret_flags = resolve_paths(entry.get("ret_patches", []))

        if not he_paths or not ret_paths:
            continue

        slides.append(
            Slide(
                lab_id=lab_id,
                he_stain_id=he_id,
                ret_stain_id=ret_id,
                grade=grade,
                he_paths=he_paths,
                he_top_flags=he_flags,
                ret_paths=ret_paths,
                ret_top_flags=ret_flags,
            )
        )

        index_map[old_idx] = len(slides) - 1

    LOGGER.info("Loaded %d filtered slides (from %d total)",
                len(slides), len(payload))

    return slides, index_map

def load_splits(split_path: Path) -> dict[str, list[int]]:
    payload = json.loads(Path(split_path).read_text("utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Split file {split_path} must contain a mapping.")

    def _extract(key: str) -> list[int]:
        values = payload.get(key, [])
        if values is None:
            return []
        return [int(v) for v in values]

    return {
        "train": _extract("train_indices"),
        "val": _extract("val_indices"),
        "test": _extract("test_indices"),
    }


def _parse_grade(value: object) -> Optional[int]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    text = str(value).strip()
    if not text:
        return None
    match = GRADE_PATTERN.search(text)
    if not match:
        return None
    grade = int(match.group(1))
    return max(0, min(3, grade))


def _build_patch_transform(image_size: int, augment: bool) -> T.Compose:
    ops: list[nn.Module] = [T.Resize(image_size), T.CenterCrop(image_size)]
    if augment:
        ops.extend([T.RandomHorizontalFlip(), T.RandomVerticalFlip(), T.RandomRotation(10)])
    ops.extend([T.ToTensor(), T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
    return T.Compose(ops)


# class SlidePatchDataset(Dataset):
    """Dataset returning N patch stacks per slide for both stains."""

    # def __init__(
    #     self,
    #     metadata_csv: Path,
    #     data_root: Optional[Path],
    #     patches_per_slide: int,
    #     image_size: int,
    #     augment: bool,
    #     seed: int,
    #     max_slides: Optional[int] = None,
    #     patch_retries: int = 8,
    #     subset_pct: Optional[float] = None,
    #     subset_order: str = "random",
    #     legacy_roots: Optional[Sequence[Path]] = None,
    #     entries: Optional[list[SlideRecord]] = None,
    # ) -> None:
    #     self.metadata_path = resolve_path(metadata_csv)
    #     self.patch_root = resolve_path(data_root or self.metadata_path.parent)
    #     self.patches_per_slide = patches_per_slide
    #     self.train_transform = _build_patch_transform(image_size, augment)
    #     self.eval_transform = _build_patch_transform(image_size, augment=False)
    #     self.patch_retries = max(1, patch_retries)
    #     self.rng = random.Random(seed)
    #     self.subset_pct = subset_pct
    #     self.subset_order = subset_order
    #     self.legacy_roots: list[Path] = []
    #     for root in legacy_roots or []:
    #         try:
    #             resolved = Path(root).expanduser().resolve()
    #         except FileNotFoundError:
    #             resolved = Path(root).expanduser()
    #         self.legacy_roots.append(resolved)
    #     self.missing_patch_paths = 0
    #     self._missing_log_limit = 20
    #     if entries is not None:
    #         self.entries = entries
    #     else:
    #         self.entries = self._build_entries(max_slides)
    #     if not self.entries:
    #         raise RuntimeError("No slides with both H&E and Reticulin patches were found.")
    #     if self.missing_patch_paths:
    #         LOGGER.warning("Skipped %d patch files that were missing on disk.", self.missing_patch_paths)

    # def _normalize_path(self, raw: str) -> Path:
    #     normalized = raw.replace("\\", "/")
    #     path = Path(normalized)

    #     # If it's an absolute path, try to remap from any legacy root to the current project layout.
    #     if path.is_absolute():
    #         if path.exists():
    #             return path
    #         for legacy_root in self.legacy_roots:
    #             try:
    #                 relative = path.relative_to(legacy_root)
    #             except ValueError:
    #                 continue
    #             remapped = (self.patch_root / relative).resolve()
    #             if remapped.exists():
    #                 return remapped
    #         # No remap possible or remapped target missing; return original path.
    #         return path

    #     candidate = (self.patch_root / path).resolve()
    #     if candidate.exists():
    #         return candidate
    #     for legacy_root in self.legacy_roots:
    #         alt = (legacy_root / path).resolve()
    #         if alt.exists():
    #             return alt
    #     return candidate

    # def _collect_valid_paths(self, raw_paths: Sequence[str]) -> list[Path]:
    #     paths: list[Path] = []
    #     for raw in raw_paths:
    #         path = self._normalize_path(str(raw))
    #         if not path.is_file():
    #             self.missing_patch_paths += 1
    #             if self.missing_patch_paths <= self._missing_log_limit:
    #                 LOGGER.warning("Patch file not found: %s", path)
    #             elif self.missing_patch_paths == self._missing_log_limit + 1:
    #                 LOGGER.warning("Too many missing patch files; suppressing additional warnings.")
    #             continue
    #         paths.append(path)
    #     return paths

    # def _build_entries(self, max_slides: Optional[int]) -> list[SlideRecord]:
    #     df = pd.read_csv(self.metadata_path)
    #     required = {"Lab No.", "stain_id", "type", "patch_path", "Reticulin Grade"}
    #     if required - set(df.columns):
    #         missing = required - set(df.columns)
    #         raise ValueError(f"metadata missing required columns: {sorted(missing)}")

    #     df["type_norm"] = df["type"].astype(str).str.lower()
    #     df["patch_path_norm"] = df["patch_path"].astype(str)
    #     df["lab_id"] = df["Lab No."].astype(str).str.strip()
    #     df = df[df["lab_id"].astype(bool)]
    #     if self.subset_pct is not None:
    #         pct = max(0.0, min(100.0, float(self.subset_pct)))
    #         unique_labs = df["lab_id"].unique().tolist()
    #         target = max(1, int(round(len(unique_labs) * (pct / 100.0))))
    #         if target < len(unique_labs):
    #             if self.subset_order == "ascending":
    #                 labs_sorted = sorted(unique_labs)
    #                 selected = labs_sorted[:target]
    #             elif self.subset_order == "descending":
    #                 labs_sorted = sorted(unique_labs, reverse=True)
    #                 selected = labs_sorted[:target]
    #             else:
    #                 selected = self.rng.sample(unique_labs, target)
    #             df = df[df["lab_id"].isin(selected)]
    #         LOGGER.info(
    #             "Subset active (%s): %.2f%% of labs -> %d labs",
    #             self.subset_order,
    #             pct,
    #             df["lab_id"].nunique(),
    #         )
    #     entries: list[SlideRecord] = []
    #     for lab_id, lab_df in df.groupby("lab_id"):
    #         he_df = lab_df[lab_df["type_norm"].str.contains("h&e", na=False)]
    #         ret_df = lab_df[lab_df["type_norm"].str.contains("reticulin", na=False)]
    #         if he_df.empty or ret_df.empty:
    #             continue

    #         he_groups = {str(sid): group for sid, group in he_df.groupby("stain_id")}
    #         ret_groups = {str(sid): group for sid, group in ret_df.groupby("stain_id")}
    #         ret_ids = list(ret_groups.keys())

    #         for he_id, he_group in he_groups.items():
    #             ret_id = _closest_stain_id(he_id, ret_ids)
    #             if ret_id is None:
    #                 continue
    #             ret_group = ret_groups[ret_id]
    #             grade_value = _parse_grade(ret_group["Reticulin Grade"].iloc[0])
    #             if grade_value is None:
    #                 continue

    #             he_paths = self._collect_valid_paths(he_group["patch_path_norm"].tolist())
    #             ret_paths = self._collect_valid_paths(ret_group["patch_path_norm"].tolist())
    #             if not he_paths or not ret_paths:
    #                 continue

    #             entries.append(
    #                 SlideRecord(
    #                     lab_id=str(lab_id),
    #                     he_stain_id=he_id,
    #                     ret_stain_id=ret_id,
    #                     grade=grade_value,
    #                     he_paths=he_paths,
    #                     ret_paths=ret_paths,
    #                 )
    #             )
    #             if max_slides is not None and len(entries) >= max_slides:
    #                 return entries
    #     return entries

    # def __len__(self) -> int:
    #     return len(self.entries)

    # def _load_stack(self, paths: list[Path], transform: Optional[T.Compose] = None) -> torch.Tensor:
    #     images: list[torch.Tensor] = []
    #     pool = list(paths)
    #     if not pool:
    #         raise RuntimeError("Slide has no available patches.")
    #     indices: list[int]
    #     if len(pool) >= self.patches_per_slide:
    #         indices = self.rng.sample(range(len(pool)), self.patches_per_slide)
    #     else:
    #         indices = [self.rng.randrange(len(pool)) for _ in range(self.patches_per_slide)]
    #     attempts = 0
    #     max_attempts = self.patch_retries * self.patches_per_slide
    #     ptr = 0
    #     while len(images) < self.patches_per_slide and attempts < max_attempts:
    #         path = pool[indices[ptr % len(indices)]]
    #         ptr += 1
    #         attempts += 1
    #         try:
    #             with Image.open(path) as img:
    #                 tfm = transform or self.train_transform
    #                 tensor = tfm(img.convert("RGB"))
    #         except (FileNotFoundError, UnidentifiedImageError, OSError) as exc:
    #             LOGGER.warning("Failed to load patch %s (%s)", path, exc)
    #             continue
    #         images.append(tensor)
    #     if len(images) < self.patches_per_slide:
    #         raise RuntimeError(f"Unable to collect {self.patches_per_slide} patches from {paths[0].parent}")
    #     return torch.stack(images, dim=0)

    # def get_slide(
    #     self,
    #     idx: int,
    #     *,
    #     deterministic: bool = False,
    # ):
    #     record = self.entries[idx]
    #     tfm = self.eval_transform if deterministic else self.train_transform
    #     he_stack = self._load_stack(record.he_paths, transform=tfm)
    #     ret_stack = self._load_stack(record.ret_paths, transform=tfm)
    #     grade = torch.tensor(record.grade, dtype=torch.long)
    #     return he_stack, ret_stack, grade, record.contrastive_id

    # def __getitem__(self, idx: int):
    #     return self.get_slide(idx)

    # def filter_slides(self, allowed_ids: set[str]) -> int:
    #     before = len(self.entries)
    #     if before == 0:
    #         return 0
    #     self.entries = [entry for entry in self.entries if entry.contrastive_id in allowed_ids]
    #     removed = before - len(self.entries)
    #     if not self.entries:
    #         raise RuntimeError("Filtering removed all slides; check embedding bank coverage.")
    #     return removed

    # def serialize_entries(self, indices: Sequence[int]) -> list[dict[str, Any]]:
    #     return [self._serialize_record(self.entries[i]) for i in indices]

    # @staticmethod
    # def _serialize_record(record: SlideRecord) -> dict[str, Any]:
    #     return {
    #         "lab_id": record.lab_id,
    #         "he_stain_id": record.he_stain_id,
    #         "ret_stain_id": record.ret_stain_id,
    #         "grade": record.grade,
    #         "he_paths": [str(p) for p in record.he_paths],
    #         "ret_paths": [str(p) for p in record.ret_paths],
    #     }

    # @staticmethod
    # def _deserialize_record(payload: dict[str, Any]) -> SlideRecord:
    #     return SlideRecord(
    #         lab_id=payload["lab_id"],
    #         he_stain_id=payload["he_stain_id"],
    #         ret_stain_id=payload["ret_stain_id"],
    #         grade=int(payload["grade"]),
    #         he_paths=[Path(p) for p in payload["he_paths"]],
    #         ret_paths=[Path(p) for p in payload["ret_paths"]],
    #     )

    # @classmethod
    # def from_serialized(
    #     cls,
    #     entries: Sequence[dict[str, Any]],
    #     args: argparse.Namespace,
    # ) -> SlidePatchDataset:
    #     records = [cls._deserialize_record(entry) for entry in entries]
    #     legacy_roots = (
    #         [Path(p) for p in getattr(args, "legacy_root", [])]
    #         if getattr(args, "legacy_root", None)
    #         else None
    #     )
    #     return cls(
    #         metadata_csv=Path(args.metadata),
    #         data_root=Path(args.data_root) if args.data_root else None,
    #         patches_per_slide=args.patches_per_slide,
    #         image_size=args.image_size,
    #         augment=not args.no_augment,
    #         seed=args.seed,
    #         patch_retries=args.patch_retries,
    #         subset_pct=None,
    #         subset_order=args.subset_order,
    #         legacy_roots=legacy_roots,
    #         entries=records,
    #     )


class SlidewisePatchDataset(Dataset):
    def __init__(
        self,
        slides: list[Slide],
        patches_per_slide: int,
        image_size: int,
        augment: bool,
        seed: int,
        patch_retries: int = 8,
        top_fraction: float = 0.25,
    ) -> None:
        if not slides:
            raise ValueError("SlidewisePatchDataset received an empty slide list.")
        self.entries = slides
        self.seed = seed
        self.patches_per_slide = patches_per_slide
        self.patch_retries = max(1, patch_retries)
        self.top_fraction = max(0.0, min(1.0, float(top_fraction)))
        self.train_transform = _build_patch_transform(image_size, augment)
        self.eval_transform = _build_patch_transform(image_size, augment=False)

    def __len__(self) -> int:
        return len(self.entries)

    def _sample_indices(
        self, total: int, flags: Sequence[int], deterministic: bool
    ) -> list[int]:
        if total <= 0:
            raise RuntimeError("Slide contains no patches.")
        if deterministic:
            indices = list(range(min(total, self.patches_per_slide)))
            while len(indices) < self.patches_per_slide:
                indices.append(indices[-1] if indices else 0)
            return indices[: self.patches_per_slide]

        top_indices = [i for i, flag in enumerate(flags) if flag]
        all_indices = list(range(total))
        k_top = int(round(self.patches_per_slide * self.top_fraction))
        if self.top_fraction > 0.0:
            k_top = max(1, k_top)
        else:
            k_top = 0
        k_top = min(k_top, self.patches_per_slide)
        k_rest = self.patches_per_slide - k_top

        def _draw(pool: list[int], count: int) -> list[int]:
            if count <= 0:
                return []
            if not pool:
                pool = all_indices
            idx = torch.randint(
                0, len(pool), (count,), dtype=torch.long
            ).tolist()
            return [pool[i] for i in idx]

        selected = _draw(top_indices or all_indices, k_top)
        selected += _draw(all_indices, k_rest)
        return selected

    def _load_stack(
        self,
        paths: Sequence[Path],
        indices: list[int],
        transform: T.Compose,
        allow_repeat: bool,
    ) -> torch.Tensor:
        images: list[torch.Tensor] = []
        attempts = 0
        max_attempts = self.patch_retries * max(1, len(indices))
        total = len(paths)
        if total == 0:
            raise RuntimeError("Slide has no patch paths.")
        pointer = 0
        while len(images) < len(indices) and attempts < max_attempts:
            idx = indices[pointer]
            pointer = (pointer + 1) % len(indices)
            path = paths[idx % total]
            try:
                with Image.open(path) as img:
                    tensor = transform(img.convert("RGB"))
            except (FileNotFoundError, UnidentifiedImageError, OSError) as exc:
                LOGGER.warning("Failed to load patch %s (%s)", path, exc)
                if allow_repeat:
                    indices[pointer - 1] = torch.randint(0, total, (1,), dtype=torch.long).item()
                attempts += 1
                continue
            images.append(tensor)
        if len(images) < len(indices):
            raise RuntimeError("Unable to collect enough patches for a slide.")
        return torch.stack(images, dim=0)

    def get_slide(
        self,
        idx: int,
        *,
        deterministic: bool = False,
    ):
        record = self.entries[idx]
        he_indices = self._sample_indices(
            len(record.he_paths), record.he_top_flags, deterministic
        )
        ret_indices = self._sample_indices(
            len(record.ret_paths), record.ret_top_flags, deterministic
        )
        transform = self.eval_transform if deterministic else self.train_transform
        he_stack = self._load_stack(
            record.he_paths, he_indices, transform, allow_repeat=not deterministic
        )
        ret_stack = self._load_stack(
            record.ret_paths, ret_indices, transform, allow_repeat=not deterministic
        )
        grade = torch.tensor(record.grade, dtype=torch.long)
        return he_stack, ret_stack, grade, record.contrastive_id

    def __getitem__(self, idx: int):
        return self.get_slide(idx)


@dataclass
class SlideChunkSchedule:
    he_chunks: list[list[int]]
    ret_chunks: list[list[int]]

    @property
    def num_chunks(self) -> int:
        return len(self.he_chunks)


def _compute_top_quota(patches_per_slide: int, top_fraction: float) -> int:
    if top_fraction <= 0.0:
        return 0
    quota = int(round(patches_per_slide * top_fraction))
    return max(1, min(patches_per_slide, quota))


def _build_chunks_for_stain(
    flags: Sequence[int],
    total_patches: int,
    patches_per_slide: int,
    top_fraction: float,
    rng: random.Random,
) -> list[list[int]]:
    top_pool = [i for i, flag in enumerate(flags[:total_patches]) if flag]
    other_pool = [i for i, flag in enumerate(flags[:total_patches]) if not flag]
    rng.shuffle(top_pool)
    rng.shuffle(other_pool)
    if not top_pool and not other_pool and total_patches > 0:
        other_pool = list(range(total_patches))
    chunks: list[list[int]] = []
    top_quota = _compute_top_quota(patches_per_slide, top_fraction) if top_pool else 0
    while top_pool or other_pool:
        chunk: list[int] = []
        remaining = patches_per_slide
        take_top = min(len(top_pool), top_quota)
        if take_top == 0 and top_pool and not other_pool:
            take_top = min(len(top_pool), remaining)
        take_top = min(take_top, remaining)
        if take_top > 0:
            chunk.extend(top_pool[:take_top])
            del top_pool[:take_top]
            remaining -= take_top
        if remaining > 0:
            take_other = min(len(other_pool), remaining)
            if take_other > 0:
                chunk.extend(other_pool[:take_other])
                del other_pool[:take_other]
                remaining -= take_other
        if remaining > 0:
            fallback = top_pool if top_pool else other_pool
            if fallback:
                take_more = min(len(fallback), remaining)
                chunk.extend(fallback[:take_more])
                del fallback[:take_more]
                remaining -= take_more
        if not chunk:
            break
        chunks.append(chunk)
    if not chunks and total_patches > 0:
        chunks.append(list(range(min(total_patches, patches_per_slide))))
    return chunks


def _pad_chunks(chunks: list[list[int]], target: int) -> list[list[int]]:
    if not chunks:
        return []
    padded = [chunk[:] for chunk in chunks]
    while len(padded) < target:
        padded.append(padded[-1][:])
    return padded[:target]


def _build_slide_schedule(
    slide: Slide,
    patches_per_slide: int,
    top_fraction: float,
    rng: random.Random,
) -> SlideChunkSchedule:
    he_chunks = _build_chunks_for_stain(
        slide.he_top_flags,
        len(slide.he_paths),
        patches_per_slide,
        top_fraction,
        rng,
    )
    ret_chunks = _build_chunks_for_stain(
        slide.ret_top_flags,
        len(slide.ret_paths),
        patches_per_slide,
        top_fraction,
        rng,
    )
    target = max(len(he_chunks), len(ret_chunks))
    he_chunks = _pad_chunks(he_chunks, target)
    ret_chunks = _pad_chunks(ret_chunks, target)
    return SlideChunkSchedule(he_chunks=he_chunks, ret_chunks=ret_chunks)


def _build_epoch_schedule(
    slides: Sequence[Slide],
    slide_indices: Sequence[int],
    patches_per_slide: int,
    top_fraction: float,
    rng: random.Random,
) -> dict[int, SlideChunkSchedule]:
    schedule: dict[int, SlideChunkSchedule] = {}
    for idx in slide_indices:
        if idx < 0 or idx >= len(slides):
            continue
        slide = slides[idx]
        schedule[idx] = _build_slide_schedule(slide, patches_per_slide, top_fraction, rng)
    return schedule


def _chunk_indices_for_stack(
    chunk: list[int],
    total_patches: int,
    target_count: int,
) -> list[int]:
    indices = list(chunk[:target_count])
    if not indices:
        indices = list(range(min(total_patches, target_count)))
    if not indices:
        raise RuntimeError("Slide contains no available patches.")
    base = indices[:]
    ptr = 0
    while len(indices) < target_count:
        indices.append(base[ptr % len(base)])
        ptr += 1
    return indices[:target_count]


def _load_chunk_stack(
    dataset: SlidewisePatchDataset,
    paths: Sequence[Path],
    chunk_indices: list[int],
    deterministic: bool,
) -> torch.Tensor:
    target = dataset.patches_per_slide
    indices = _chunk_indices_for_stack(chunk_indices, len(paths), target)
    transform = dataset.eval_transform if deterministic else dataset.train_transform
    return dataset._load_stack(paths, indices, transform, allow_repeat=not deterministic)


def _batch_slide_indices(indices: Sequence[int], batch_size: int) -> list[list[int]]:
    if batch_size <= 0:
        raise ValueError("batch_slides must be positive.")
    return [list(indices[i : i + batch_size]) for i in range(0, len(indices), batch_size)]


class SlideEmbeddingBank:
    def __init__(self, slide_embeddings: dict[str, list[torch.Tensor]]) -> None:
        if not slide_embeddings:
            raise ValueError("Slide embedding bank is empty.")
        self._group_embeddings: dict[str, torch.Tensor] = {}
        self._flat_embeddings: list[tuple[str, torch.Tensor]] = []
        for group_id, embeddings in slide_embeddings.items():
            if not embeddings:
                continue
            stacked = torch.stack([emb.float().detach().cpu() for emb in embeddings], dim=0)
            self._group_embeddings[group_id] = stacked
            for emb in stacked:
                self._flat_embeddings.append((group_id, emb))
        if not self._group_embeddings:
            raise ValueError("Slide embedding bank has no valid group embeddings.")
        self._ids = list(self._group_embeddings.keys())

    def get(self, slide_id: str) -> torch.Tensor:
        emb = self._group_embeddings.get(slide_id)
        if emb is None:
            raise KeyError(f"Slide '{slide_id}' missing from embedding bank.")
        return emb

    def maybe_get(self, slide_id: str) -> torch.Tensor | None:
        """Return stacked positive embeddings for slide_id, or None if missing."""
        return self._group_embeddings.get(slide_id)

    @property
    def slide_ids(self) -> list[str]:
        return list(self._ids)

    @property
    def num_embeddings(self) -> int:
        return len(self._flat_embeddings)

    def sample(self, exclude: set[str], k: int) -> torch.Tensor:
        candidates = [emb for sid, emb in self._flat_embeddings if sid not in exclude]
        if not candidates:
            raise RuntimeError("No available negatives for contrastive bank sampling.")
        if k <= 0:
            raise ValueError("Number of negatives K must be positive.")
        if len(candidates) < k:
            selected = candidates
        else:
            selected = random.sample(candidates, k)
        return torch.stack(selected, dim=0)


# def build_dataset(args: argparse.Namespace) -> SlidePatchDataset:
#     return SlidePatchDataset(
#         metadata_csv=Path(args.metadata),
#         data_root=Path(args.data_root) if args.data_root else None,
#         patches_per_slide=args.patches_per_slide,
#         image_size=args.image_size,
#         augment=not args.no_augment,
#         seed=args.seed,
#         max_slides=args.max_slide_count,
#         patch_retries=args.patch_retries,
#         subset_pct=args.subset,
#         subset_order=args.subset_order,
#         legacy_roots=[Path(p) for p in getattr(args, "legacy_root", [])] if getattr(args, "legacy_root", None) else None,
#     )


def build_dataloader(
    dataset: Dataset,
    args: argparse.Namespace,
    *,
    shuffle: bool = True,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=args.batch_slides,
        shuffle=shuffle,
        num_workers=args.num_workers,
        pin_memory=args.device.startswith("cuda") if args.device else False,
        drop_last=True,
    )


# def build_loader(args: argparse.Namespace) -> tuple[SlidePatchDataset, DataLoader]:
#     dataset = build_dataset(args)
#     loader = build_dataloader(dataset, args)
#     return dataset, loader


# def build_split_loaders(
#     dataset: SlidePatchDataset,
#     args: argparse.Namespace,
#     *,
#     return_indices: bool = False,
# ) -> tuple[dict[str, DataLoader | None], dict[str, list[int]]] | dict[str, DataLoader | None]:
#     total = len(dataset)
#     if total < 3:
#         loader = build_dataloader(dataset, args)
#         result = {"train": loader, "val": None, "test": None}
#         if return_indices:
#             return result, {"train": list(range(total)), "val": [], "test": []}
#         return result
#     val_count = int(total * args.val_ratio)
#     test_count = int(total * args.test_ratio)
#     if val_count + test_count > total - 2:
#         # Ensure at least 2 slides remain for training.
#         shrink = (val_count + test_count) - (total - 2)
#         if shrink > 0:
#             if test_count >= shrink:
#                 test_count -= shrink
#             else:
#                 shrink -= test_count
#                 test_count = 0
#                 val_count = max(0, val_count - shrink)
#     train_count = total - val_count - test_count
#     generator = torch.Generator().manual_seed(args.seed)
#     perm = torch.randperm(total, generator=generator).tolist()
#     train_idx = perm[:train_count]
#     val_idx = perm[train_count : train_count + val_count]
#     test_idx = perm[train_count + val_count :]

#     subsets: dict[str, Subset | None] = {
#         "train": Subset(dataset, train_idx),
#         "val": Subset(dataset, val_idx) if val_idx else None,
#         "test": Subset(dataset, test_idx) if test_idx else None,
#     }

#     loaders: dict[str, DataLoader | None] = {}
#     for name, subset in subsets.items():
#         if subset is None or len(subset) == 0:  # type: ignore[arg-type]
#             loaders[name] = None
#             continue
#         loaders[name] = build_dataloader(
#             subset,
#             args,
#             shuffle=name == "train",
#         )
#     if return_indices:
#         return loaders, {"train": train_idx, "val": val_idx, "test": test_idx}
#     return loaders


class ProjectionHead(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def freeze_module(module: nn.Module) -> nn.Module:
    module.eval()
    for param in module.parameters():
        param.requires_grad_(False)
    return module


def grad_norm(module: nn.Module) -> float:
    norms = [p.grad.norm().item() for p in module.parameters() if p.grad is not None]
    return float(sum(norms) / len(norms)) if norms else 0.0


def component_grad_norm(
    loss: torch.Tensor | None, modules: Sequence[nn.Module], retain_graph: bool = True
) -> float:
    if loss is None or not isinstance(loss, torch.Tensor) or not loss.requires_grad:
        return 0.0
    params = [p for module in modules for p in module.parameters() if p.requires_grad]
    if not params:
        return 0.0
    grads = torch.autograd.grad(loss, params, retain_graph=retain_graph, allow_unused=True)
    norms = [g.norm().item() for g in grads if g is not None]
    return float(sum(norms) / len(norms)) if norms else 0.0


def _load_cache(path_str: Optional[str], label: str, no_cache: bool):
    if not path_str or no_cache:
        return None
    cache_path = Path(path_str)
    if not cache_path.exists():
        return None
    LOGGER.info("Loading cached %s from %s", label, cache_path)
    try:
        # For PyTorch 2.6+, default weights_only=True breaks loading custom classes
        return torch.load(cache_path, weights_only=False)
    except TypeError:
        # Older PyTorch versions do not support weights_only
        return torch.load(cache_path)
    except Exception as exc:
        LOGGER.warning(
            "Failed to load cached %s from %s (%s); ignoring cache and recomputing.",
            label,
            cache_path,
            exc,
        )
        return None


def _save_cache(obj: Any, path_str: Optional[str], label: str, no_cache: bool) -> None:
    if not path_str or no_cache:
        return
    cache_path = Path(path_str)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(obj, cache_path)
    LOGGER.info("Saved %s cache to %s", label, cache_path)


def evaluate_split(
    split_name: str,
    loader: DataLoader | None,
    *,
    device: torch.device,
    mean: torch.Tensor,
    std: torch.Tensor,
    feature_extractor: nn.Module,
    abmil: ABMIL,
    slide_classifier: SlideClassifier,
    projector: ProjectionHead,
    G_H2R: nn.Module,
    embedding_bank: SlideEmbeddingBank,
    args: argparse.Namespace,
    criterion_cls: nn.Module,
) -> dict[str, float]:
    if loader is None or len(loader) == 0:
        return {}
    G_H2R.eval()
    projector.eval()
    total_loss_cls = 0.0
    total_loss_con = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for he_batch, ret_batch, grades, slide_ids in loader:
            he_batch = he_batch.to(device, non_blocking=True)
            ret_batch = ret_batch.to(device, non_blocking=True)
            grades = grades.to(device, non_blocking=True)

            he_flat = he_batch.view(-1, *he_batch.shape[2:])
            fake_ret = G_H2R(he_flat).view_as(ret_batch)

            z_fake = extract_patch_features(
                feature_extractor_model, fake_ret, mean, std, require_grad=False
            )
            cls_loss, logits = compute_classification_loss(
                z_fake, grades, abmil, slide_classifier, criterion_cls
            )
            contrastive_loss = compute_contrastive_loss_with_bank(
                z_fake,
                slide_ids,
                embedding_bank,
                abmil,
                projector,
                args.contrastive_negatives,
                args.temperature,
            )
            preds = logits.argmax(dim=1)
            total_correct += (preds == grades).sum().item()
            total_samples += grades.size(0)
            total_loss_cls += cls_loss.item()
            total_loss_con += contrastive_loss.item()

    avg_cls = total_loss_cls / max(1, len(loader))
    avg_con = total_loss_con / max(1, len(loader))
    acc = total_correct / max(1, total_samples)
    LOGGER.info(
        "[%s] cls_loss=%.4f | con_loss=%.4f | acc=%.3f",
        split_name,
        avg_cls,
        avg_con,
        acc,
    )
    return {"cls": avg_cls, "con": avg_con, "acc": acc}


def plot_metrics(history: list[dict[str, float]], out_path: Path) -> None:
    if not history:
        return
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        LOGGER.warning("matplotlib not available; skipping metric plot.")
        return

    epochs = [row["epoch"] for row in history]
    plt.figure(figsize=(8, 5))
    train_cls = [row.get("train_cls", row.get("cls")) for row in history]
    plt.plot(epochs, train_cls, label="train cls")
    if any(row.get("val_cls") is not None for row in history):
        plt.plot(epochs, [row.get("val_cls") for row in history], label="val cls")
    if any(row.get("test_cls") is not None for row in history):
        plt.plot(epochs, [row.get("test_cls") for row in history], label="test cls")
    plt.xlabel("Epoch")
    plt.ylabel("Classification Loss")
    plt.title("Train/Val/Test Classification Loss")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.4)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    LOGGER.info("Saved metric plot to %s", out_path)


def _load_state_dict(module: nn.Module, checkpoint_path: Path, key: Optional[str] = None) -> None:
    state = torch.load(checkpoint_path, map_location="cpu")
    if key and key in state:
        payload = state[key]
    elif isinstance(state, dict) and "state_dict" in state:
        payload = state["state_dict"]
    else:
        payload = state
    module.load_state_dict(payload, strict=False)


def build_feature_extractor(args: argparse.Namespace, device: torch.device) -> nn.Module:
    if FeatureExtractor is not None:
        model = FeatureExtractor(  # type: ignore[call-arg]
            pretrained=False,
            proj_hidden_dim=args.encoder_proj_hidden,
            proj_out_dim=args.encoder_proj_out,
        )
        _load_state_dict(model, resolve_path(args.pretrained_encoder))
    else:
        ctor = ENCODER_REGISTRY.get(args.encoder_backbone.lower())
        if ctor is None:
            raise ValueError(f"Unknown encoder backbone '{args.encoder_backbone}'.")
        model = ctor(
            pretrained=False,
            proj_hidden_dim=args.encoder_proj_hidden,
            proj_out_dim=args.encoder_proj_out,
        )
        state = torch.load(resolve_path(args.pretrained_encoder), map_location="cpu")
        if "model" in state:
            model.load_state_dict(state["model"])
        elif "backbone" in state:
            model.backbone.load_state_dict(state["backbone"])
        else:
            model.load_state_dict(state)
        model.projector = nn.Identity()
    return freeze_module(model).to(device)


def forward_encoder_features(encoder: nn.Module, images: torch.Tensor) -> torch.Tensor:
    outputs = encoder(images)
    if isinstance(outputs, tuple):
        return outputs[0]
    return outputs


def normalize_for_encoder(patches: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    flat = patches.view(-1, *patches.shape[2:]).to(dtype=mean.dtype)
    flat = (flat + 1.0) * 0.5
    return (flat - mean) / std


def extract_patch_features(
    encoder: nn.Module,
    patches: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
    require_grad: bool,
) -> torch.Tensor:
    normalized = normalize_for_encoder(patches, mean, std)
    if require_grad:
        feats = forward_encoder_features(encoder, normalized)
    else:
        with torch.no_grad():
            feats = forward_encoder_features(encoder, normalized)
    feat_dim = feats.shape[1]
    return feats.view(patches.size(0), patches.size(1), feat_dim)


def compute_classification_loss(
    z_fake: torch.Tensor,
    grades: torch.Tensor,
    abmil: ABMIL,
    classifier: SlideClassifier,
    criterion: nn.Module,
) -> tuple[torch.Tensor, torch.Tensor]:
    logits: list[torch.Tensor] = []
    for idx in range(z_fake.size(0)):
        bag = z_fake[idx]
        slide_vec, _ = abmil.pool(bag) #
        logit = classifier(slide_vec.unsqueeze(0))
        logits.append(logit)
    stacked = torch.cat(logits, dim=0)
    loss = criterion(stacked, grades)
    return loss, stacked


def compute_contrastive_loss(
    z_real: torch.Tensor,
    z_fake: torch.Tensor,
    projector: ProjectionHead,
    temperature: float,
) -> torch.Tensor:
    bsz = z_real.size(0)
    e_real = z_real.mean(dim=1)
    e_fake = z_fake.mean(dim=1)
    q_real = F.normalize(projector(e_real), dim=1)
    q_fake = F.normalize(projector(e_fake), dim=1)
    embeddings = torch.cat([q_real, q_fake], dim=0)
    sim = embeddings @ embeddings.t() / temperature
    mask = torch.eye(sim.size(0), device=sim.device, dtype=torch.bool)
    sim = sim.masked_fill(mask, float("-inf"))

    losses: list[torch.Tensor] = []
    for idx in range(bsz):
        pos = idx + bsz
        losses.append(F.logsumexp(sim[idx], dim=0) - sim[idx, pos])
    for idx in range(bsz, 2 * bsz):
        pos = idx - bsz
        losses.append(F.logsumexp(sim[idx], dim=0) - sim[idx, pos])
    return torch.stack(losses).mean()


def compute_contrastive_loss_with_bank(
    z_fake: torch.Tensor,
    slide_ids: Sequence[str],
    bank: SlideEmbeddingBank,
    abmil: ABMIL,
    projector: ProjectionHead,
    num_negatives: int,
    temperature: float,
) -> torch.Tensor:
    batch = z_fake.size(0)
    if batch != len(slide_ids):
        raise ValueError("slide_ids length must match batch size.")
    device = z_fake.device

    # For contrastive loss, we get embeddings for the real reticulin patches and the fake/generated reticulin patches
    # For the real reticulin patches, we already get pooled embeddings
    # For the fake/generated patches, we get individual patch embeddings and we pool them here. Here pooling is done from the trained ABMIL model
    # We pass the real and fake pooled embeddings through a MLP that gets their features.
    # Anchors are the fake patch embeddings that are generated from the pooling. 
    # These are used to calculate the distances from +ve and -ve samples.
    # Positive pairs: Anchor vs embeddings of real reticulin samples from the "same" subject that is in the training batch
    # Negative pairs: Anchor vs embeddings of real reticulin samples from "different" subject that is in the training batch

    # Collect only those slides that have an embedding in the bank.
    valid_indices: list[int] = []
    anchor_embeddings: list[torch.Tensor] = []
    positive_embedding_sets: list[torch.Tensor] = []
    valid_ids: list[str] = []
    for idx, sid in enumerate(slide_ids):
        positive_set = bank.maybe_get(sid)
        if positive_set is None:
            continue
        slide_vec, _ = abmil.pool(z_fake[idx])
        valid_indices.append(idx)
        anchor_embeddings.append(slide_vec)
        positive_embedding_sets.append(positive_set)
        valid_ids.append(sid)

    # If no slides in this batch have embeddings, return zero loss (no contrastive signal).
    if not valid_indices:
        return torch.zeros((), device=device)

    e_fake = torch.stack(anchor_embeddings, dim=0)
    q_fake = F.normalize(projector(e_fake), dim=1)

    losses: list[torch.Tensor] = []
    # Exclude all valid ids from being used as negatives for any anchor.
    exclude_batch = set(valid_ids)
    for local_idx, sid in enumerate(valid_ids):
        exclude = set(exclude_batch) #remove the samples from any ids that are in this training set
        exclude.add(sid)
        negatives = bank.sample(exclude, num_negatives).to(device)
        q_neg = F.normalize(projector(negatives), dim=1)
        pos_tensor = positive_embedding_sets[local_idx].to(device)
        q_pos = F.normalize(projector(pos_tensor), dim=1)

        anchor = q_fake[local_idx]
        pos_logits = torch.matmul(q_pos, anchor)/ temperature #maximize this, s+
        neg_logits = torch.matmul(q_neg, anchor)/ temperature #minimize this, s-
        all_logits = torch.cat([pos_logits, neg_logits], dim=0) # [s+, s-]
        loss = -(torch.logsumexp(pos_logits, dim=0) - torch.logsumexp(all_logits, dim=0))
        losses.append(loss)

    return torch.stack(losses).mean()


def _pool_real_slide_embeddings(vectors: Sequence[torch.Tensor], top_flags: Sequence[int] | None) -> torch.Tensor:
    stacked = torch.stack([vec.float() for vec in vectors], dim=0)
    if stacked.size(0) == 1:
        return stacked[0]

    selected = stacked
    if top_flags:
        flags = torch.tensor(list(top_flags[: stacked.size(0)]), dtype=torch.bool)
        if flags.numel() < stacked.size(0):
            pad = torch.zeros(stacked.size(0) - flags.numel(), dtype=torch.bool)
            flags = torch.cat([flags, pad], dim=0)
        if flags.any():
            selected = stacked[flags]

    if selected.size(0) == 1:
        return selected[0]

    #mean-shift, radial basis (RBF) kernel attention, or soft-medoid pooling
    center = selected.mean(dim=0, keepdim=True)
    scale = max(selected.size(1), 1) ** 0.5
    attn_logits = (selected * center).sum(dim=1) / scale
    attn = torch.softmax(attn_logits, dim=0)
    return torch.sum(attn.unsqueeze(1) * selected, dim=0)


def load_embedding_bank(root: Path, dataset: SlidewisePatchDataset) -> SlideEmbeddingBank:
    #we get one embedding for the entire slide instead.
    root = resolve_path(root)
    if not root.exists():
        raise FileNotFoundError(f"Reticulin embedding directory not found: {root}")
    file_lookup: dict[str, Path] = {}
    for npy_path in root.rglob("*.npy"):
        key = npy_path.stem
        if key in file_lookup:
            LOGGER.warning("Duplicate embedding for %s; keeping the first instance (%s)", key, file_lookup[key])
            continue
        file_lookup[key] = npy_path

    grouped_vectors: dict[str, list[torch.Tensor]] = {}
    missing_vectors = 0
    for record in dataset.entries:
        vectors: list[torch.Tensor] = []
        for patch_path in record.ret_paths:
            key = patch_path.stem
            emb_path = file_lookup.get(key)
            if emb_path is None:
                missing_vectors += 1
                continue
            vec = torch.from_numpy(np.load(emb_path)).float()
            vectors.append(vec)
        if not vectors:
            continue
        top_flags = getattr(record, "ret_top_flags", None)
        record_embedding = _pool_real_slide_embeddings(vectors, top_flags)
        grouped_vectors.setdefault(record.contrastive_id, []).append(record_embedding)
    slide_embeddings = grouped_vectors
    if not slide_embeddings:
        raise RuntimeError(
            "Unable to build embedding bank: no matched slides between metadata and "
            f"embeddings located under {root}."
        )
    LOGGER.info(
        "Loaded real reticulin embedding bank for %d contrastive groups / %d slide embeddings (missing patch embeddings: %d)",
        len(slide_embeddings),
        sum(len(embeddings) for embeddings in slide_embeddings.values()),
        missing_vectors,
    )
    return SlideEmbeddingBank(slide_embeddings)


def load_cyclegan_weights(root: Path, generators: tuple[nn.Module, nn.Module], discriminators: tuple[nn.Module, nn.Module]) -> None:
    root = resolve_path(root)
    G_H2R, G_R2H = generators
    D_R, D_H = discriminators
    if root.is_file():
        checkpoint = torch.load(root, map_location="cpu")
        for key, module in {
            "G_H2R": G_H2R,
            "G_R2H": G_R2H,
            "D_R": D_R,
            "D_H": D_H,
        }.items():
            state = checkpoint.get(key)
            if state is None:
                continue
            module.load_state_dict(state)
        return

    def latest(prefix: str) -> Path:
        pattern = re.compile(rf"{re.escape(prefix)}_epoch(\d+)\.")
        best_epoch = -1
        best_path: Optional[Path] = None
        for path in root.glob(f"{prefix}_epoch*.pth*"):
            match = pattern.match(path.name)
            if not match:
                continue
            epoch = int(match.group(1))
            if epoch > best_epoch:
                best_epoch = epoch
                best_path = path
        if best_path is None:
            raise FileNotFoundError(f"No checkpoint found for {prefix} under {root}")
        return best_path

    for name, module in [("G_H2R", G_H2R), ("G_R2H", G_R2H), ("D_R", D_R), ("D_H", D_H)]:
        ckpt_path = latest(name)
        state = torch.load(ckpt_path, map_location="cpu")
        payload = state.get("state_dict", state)
        module.load_state_dict(payload)


def adversarial_loss(discriminator: nn.Module, real: torch.Tensor, fake: torch.Tensor, criterion: nn.Module) -> tuple[torch.Tensor, torch.Tensor]:
    pred_real = discriminator(real)
    pred_fake = discriminator(fake.detach())
    loss_real = criterion(pred_real, torch.ones_like(pred_real))
    loss_fake = criterion(pred_fake, torch.zeros_like(pred_fake))
    return (loss_real + loss_fake) * 0.5, pred_real


def infer_feature_dim(encoder: nn.Module, sample: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> int:
    with torch.no_grad():
        feats = forward_encoder_features(encoder, normalize_for_encoder(sample, mean, std))
    return feats.shape[1]


def save_checkpoints(
    run_dir: Path,
    epoch: int,
    modules: dict[str, nn.Module],
    optimizers: dict[str, torch.optim.Optimizer | None],
    scalers: dict[str, GradScaler | None],
) -> None:
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {"epoch": epoch}
    for name, module in modules.items():
        payload[name] = module.state_dict()
    for name, opt in optimizers.items():
        if opt is not None:
            payload[name] = opt.state_dict()
    for name, scaler in scalers.items():
        if scaler is not None:
            payload[name] = scaler.state_dict()
    path = ckpt_dir / f"train_state_epoch{epoch}.pt"
    torch.save(payload, path)


def load_checkpoints(
    run_dir: Path,
    epoch: int,
    modules: dict[str, nn.Module],
    optimizers: dict[str, torch.optim.Optimizer | None],
    scalers: dict[str, GradScaler | None],
) -> None:
    ckpt_dir = Path(run_dir) / "checkpoints"
    path = ckpt_dir / f"train_state_epoch{epoch}.pt"
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint {path} not found.")
    state = torch.load(path, map_location="cpu")
    for name, module in modules.items():
        payload = state.get(name)
        if payload is None:
            raise KeyError(f"Checkpoint missing state for module {name}.")
        module.load_state_dict(payload)
    for name, opt in optimizers.items():
        if opt is not None and name in state:
            opt.load_state_dict(state[name])
    for name, scaler in scalers.items():
        if scaler is not None and name in state:
            scaler.load_state_dict(state[name])


def _load_cli_defaults(config_path: Optional[str]) -> dict[str, Any]:
    if not config_path:
        return {}
    resolved = resolve_path(config_path, allow_missing=True)
    if not resolved.exists():
        return {}
    try:
        with resolved.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Failed to parse training config {resolved}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Training config {resolved} must contain a JSON object.")
    return payload


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    base_parser = argparse.ArgumentParser(add_help=False)
    base_parser.add_argument(
        "--config",
        type=str,
        default=str(DEFAULT_TRAIN_AUGMENT_CONFIG),
        help="Path to JSON config specifying default CLI arguments.",
    )
    config_ns, _ = base_parser.parse_known_args(argv)
    try:
        config_defaults = _load_cli_defaults(config_ns.config)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    parser = argparse.ArgumentParser(description="Augmented CycleGAN training", parents=[base_parser])
    parser.add_argument(
        "--legacy-root",
        type=str,
        action="append",
        default=None,
        help="Legacy absolute root directory to remap from (can be used multiple times).",
    )
    parser.add_argument("--metadata", type=str, default=str(PROJECT_ROOT / "metadata.csv"))
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument(
        "--augmented-slides",
        type=str,
        required=False, #shubham
        help="Path to slides.json produced by build_augmented_dataset.py",
    )
    parser.add_argument(
        "--augmented-splits",
        type=str,
        required=False, #shubham
        help="Path to splits.json produced by build_augmented_dataset.py",
    )
    parser.add_argument("--batch-slides", type=int, default=2)
    parser.add_argument("--patches-per-slide", type=int, default=32)
    parser.add_argument("--lambda-cycle", type=float, default=50.0)
    parser.add_argument("--lambda-identity", type=float, default=25.0)
    parser.add_argument("--lambda-cls", type=float, default=0.0)
    parser.add_argument("--lambda-con", type=float, default=0.0)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr-gen", type=float, default=2e-4)
    parser.add_argument("--lr-disc", type=float, default=2e-4)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--pretrained-cyclegan", type=str, default=None)
    parser.add_argument("--pretrained-encoder", type=str, default=None)
    parser.add_argument("--pretrained-abmil", type=str, default=None)
    parser.add_argument("--pretrained-classifier", type=str, default=None)
    parser.add_argument("--log-dir", type=str, default=str(PROJECT_ROOT / "outputs" / "logs" / "cyclegan"))
    parser.add_argument(
        "--samples-dir",
        type=str,
        default=str(PROJECT_ROOT / "outputs" / "samples" / "cyclegan"),
        help="Directory where epoch sample grids are stored.",
    )
    parser.add_argument("--run-id", type=str, required=True)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--encoder-backbone", type=str, default="resnet50", choices=sorted(ENCODER_REGISTRY.keys()))
    parser.add_argument("--encoder-proj-hidden", type=int, default=2048)
    parser.add_argument("--encoder-proj-out", type=int, default=128)
    parser.add_argument("--abmil-attn-dim", type=int, default=256)
    parser.add_argument("--classifier-hidden-dim", type=int, default=512)
    parser.add_argument("--abmil-dropout", type=float, default=0.25)
    parser.add_argument("--num-classes", type=int, default=4)
    parser.add_argument("--proj-hidden-dim", type=int, default=512)
    parser.add_argument("--proj-out-dim", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--save-every", type=int, default=5)
    parser.add_argument(
        "--save-samples-every",
        type=int,
        default=1,
        help="Save deterministic sample grids every N epochs.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-augment", action="store_true")
    parser.add_argument(
        "--subset",
        type=float,
        default=None,
        help="Use only X%% of labs from metadata for quick experiments.",
    )
    parser.add_argument(
        "--subset-order",
        type=str,
        choices=["random", "ascending", "descending"],
        default="random",
        help="When --subset is set, keep labs from the top/bottom (alphabetical) or sample randomly.",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.0,
        help="Fraction of slides reserved for validation (0-1 range).",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.0,
        help="Fraction of slides reserved for testing (0-1 range).",
    )
    parser.add_argument("--max-slide-count", type=int, default=None)
    parser.add_argument("--patch-retries", type=int, default=8)
    parser.add_argument(
        "--profile-steps",
        action="store_true",
        help="Log per-step CUDA memory usage and iteration time (may slow training).",
    )
    parser.add_argument(
        "--amp",
        action="store_true",
        help="Enable torch.cuda.amp mixed precision when running on CUDA devices.",
    )
    parser.add_argument(
        "--disable-identity-loss",
        action="store_true",
        help="Disable identity loss term (sets its weight to zero).",
    )
    parser.add_argument(
        "--reticulin-embedding-dir",
        type=str,
        default=None,
        help="Directory containing precomputed real reticulin embeddings (.npy).",
    )
    parser.add_argument(
        "--contrastive-negatives",
        type=int,
        default=32,
        help="Number of negative samples drawn from the real embedding bank per slide.",
    )
    parser.add_argument(
        "--cache-embedding-bank",
        type=str,
        default=None,
        help="Path to cache the computed embedding bank (torch.save format).",
    )
    parser.add_argument(
        "--cache-index",
        type=str,
        default=None,
        help="Path to cache serialized slide indices/splits.",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable reading/writing caches even if cache paths are provided.",
    )
    parser.add_argument(
        "--grad-req",
        type=_bool_arg,
        default=True,
        help="Whether to enable gradient computation/backprop (set to false for inference-only runs).",
    )
    parser.add_argument(
        "--top-fraction",
        type=float,
        default=0.25,
        help="Fraction of patches drawn from the top attention pool (default 0.25).",
    )
    parser.add_argument(
        "--resume-dir",
        type=str,
        default=None,
        help="Optional run directory to resume from (expects checkpoints there).",
    )
    parser.add_argument(
        "--resume-epoch",
        type=int,
        default=None,
        help="Epoch number to resume from (requires --resume-dir).",
    )
    parser.set_defaults(**config_defaults)
    args = parser.parse_args(argv)

    # Fill missing pretrained paths from the config defaults (supports legacy keys)
    for arg_name, keys in [
        ("pretrained_cyclegan", ["pretrained_cyclegan", "cyclegan_checkpoint"]),
        ("pretrained_encoder", ["pretrained_encoder", "encoder_checkpoint"]),
        ("pretrained_abmil", ["pretrained_abmil", "abmil_checkpoint"]),
        ("pretrained_classifier", ["pretrained_classifier", "classifier_checkpoint"]),
    ]:
        current = getattr(args, arg_name)
        if current is None or (isinstance(current, str) and not current.strip()):
            for key in keys:
                if key in config_defaults and config_defaults[key]:
                    setattr(args, arg_name, config_defaults[key])
                    break
    if args.reticulin_embedding_dir:
        args.reticulin_embedding_dir = str(resolve_path(args.reticulin_embedding_dir, allow_missing=True))
    if args.contrastive_negatives <= 0:
        raise ValueError("--contrastive-negatives must be positive.")
    if args.subset is not None:
        if args.subset <= 0 or args.subset > 100:
            raise ValueError("--subset must be between 0 and 100 (exclusive of 0).")
    if args.val_ratio < 0 or args.test_ratio < 0 or args.val_ratio >= 1 or args.test_ratio >= 1:
        raise ValueError("--val-ratio and --test-ratio must be within [0, 1).")
    if args.val_ratio + args.test_ratio >= 0.95:
        raise ValueError("val_ratio + test_ratio must be less than 0.95 to leave room for training.")
    if not (0.0 <= args.top_fraction <= 1.0):
        raise ValueError("--top-fraction must be within [0, 1].")
    if (args.resume_dir and args.resume_epoch is None) or (args.resume_epoch and not args.resume_dir):
        raise ValueError("--resume-dir and --resume-epoch must be provided together.")
    
    #shubham
    def debug_print_args(args):
        LOGGER.info("\n========== Parsed Arguments ==========")
        for k, v in vars(args).items():
            LOGGER.info(f"{k:25s} : {v}")
        LOGGER.info("======================================\n")

    debug_print_args(args)


    required = [
        "pretrained_cyclegan",
        "pretrained_encoder",
        "pretrained_abmil",
        "pretrained_classifier",
        "run_id",
    ]
    missing = [
        name
        for name in required
        if not getattr(args, name)
        or (isinstance(getattr(args, name), str) and not getattr(args, name).strip())
    ]
    if missing:
        raise ValueError(
            "Missing required arguments (provide via config or CLI): " + ", ".join(sorted(missing))
        )
    args.metadata = str(resolve_path(args.metadata))
    if args.data_root:
        args.data_root = str(resolve_path(args.data_root, allow_missing=True))
    args.augmented_slides = str(resolve_path(args.augmented_slides))
    args.augmented_splits = str(resolve_path(args.augmented_splits))
    args.log_dir = str(resolve_path(args.log_dir, allow_missing=True))
    args.samples_dir = str(resolve_path(args.samples_dir, allow_missing=True))
    if args.cache_embedding_bank:
        args.cache_embedding_bank = str(resolve_path(args.cache_embedding_bank, allow_missing=True))
    if args.cache_index:
        args.cache_index = str(resolve_path(args.cache_index, allow_missing=True))
    args.grad_req = _bool_arg(args.grad_req)
    if args.resume_dir:
        args.resume_dir = str(resolve_path(args.resume_dir, allow_missing=True))
    if args.legacy_root:
        # Resolve legacy roots but allow them to be missing on the current machine;
        # we only use them for prefix-stripping and remapping.
        args.legacy_root = [
            str(resolve_path(path, allow_missing=True)) for path in args.legacy_root
        ]

    
    config_dict = vars(args)
    with open(os.path.join("PROJECT_ROOT/outputs/logs/cyclegan/" + run_id + '_config.json'), "w") as :
        json.dump(config_dict, json_file, indent=4)

    return args


def setup_logging(run_dir: Path) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    log_file = run_dir / "train.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file, mode="w"),
        ],
    )
    LOGGER.info("Logging to %s", log_file)


def determine_device(requested: Optional[str]) -> torch.device:
    if requested:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train(args: argparse.Namespace) -> None:
    device = determine_device(args.device)
    args.device = device.type if device.index is None else f"{device.type}:{device.index}"

    log_root = resolve_path(args.log_dir, allow_missing=True)
    samples_root = resolve_path(args.samples_dir, allow_missing=True)
    run_dir = log_root / args.run_id
    samples_dir = samples_root / args.run_id
    setup_logging(run_dir)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    if not args.reticulin_embedding_dir:
        raise ValueError(
            "Slide-level contrastive training now requires --reticulin-embedding-dir "
            "with precomputed reticulin embeddings."
        )
    
    splits = load_splits(Path(args.augmented_splits))

    required_indices = sorted(set(splits["train"] + splits["val"] + splits["test"]))
    
    slides, index_map = remap_load_slides(
    Path(args.slides),
    Path(args.data_root) if args.data_root else None,
    required_indices,
)


    #shubham    
    def show_split(name, indices):
        LOGGER.info(f"\n========== {name.upper()} ==========")
        LOGGER.info(f"Total slides: {len(indices)}")

        # for idx in indices:
        #     if idx < 0 or idx >= len(slides):
        #         continue

        #     slide = slides[idx]

        #     print("\n--- Slide ---")
        #     print("Slide ID :", f"{slide.lab_id}|{slide.he_stain_id}|{slide.ret_stain_id}")
        #     print("Grade    :", slide.grade)

        #     print("\nH&E patches:")
        #     for p in slide.he_paths:
        #         print("   ", p)

        #     print("\nReticulin patches:")
        #     for p in slide.ret_paths:
        #         print("   ", p)

    def remap(indices):
        return [index_map[i] for i in indices if i in index_map]

    splits["train"] = remap(splits["train"])
    splits["val"]   = remap(splits["val"])
    splits["test"]  = remap(splits["test"])

    dataset = SlidewisePatchDataset(
        slides=slides,
        patches_per_slide=args.patches_per_slide,
        image_size=args.image_size,
        augment=not args.no_augment,
        seed=args.seed,
        patch_retries=args.patch_retries,
        top_fraction=args.top_fraction,
    )

    def _build_subset(indices: list[int]) -> Subset | None:
        valid = [idx for idx in indices if 0 <= idx < len(dataset)]
        if not valid:
            return None
        return Subset(dataset, valid)

    train_slide_indices = [idx for idx in splits["train"] if 0 <= idx < len(slides)]
    if not train_slide_indices:
        raise RuntimeError("Training indices list is empty; cannot start training.")
    val_subset = _build_subset(splits["val"])
    test_subset = _build_subset(splits["test"])

    val_loader = build_dataloader(val_subset, args, shuffle=False) if val_subset else None
    test_loader = build_dataloader(test_subset, args, shuffle=False) if test_subset else None
    dataset_for_preview = dataset #he_stack, ret_stack, grade, record.contrastive_id


    embedding_bank = _load_cache(args.cache_embedding_bank, "embedding bank", args.no_cache) #
    if embedding_bank is None:
        embedding_bank = load_embedding_bank(Path(args.reticulin_embedding_dir), dataset)
        LOGGER.info(
            "Embedding bank covers %d slide(s); slides without embeddings will be skipped in contrastive loss.",
            len(embedding_bank.slide_ids),
        )
        _save_cache(embedding_bank, args.cache_embedding_bank, "embedding bank", args.no_cache)
    else:
        LOGGER.info(
            "Loaded embedding bank cache covering %d slide(s).",
            len(embedding_bank.slide_ids),
        )

    LOGGER.info(
        "Slide counts | train=%d | val=%d | test=%d",
        len(train_slide_indices),
        len(splits["val"]),
        len(splits["test"]),
    )
    LOGGER.info(
        "Evaluation loader sizes | val: %s | test: %s",
        len(val_loader) if val_loader is not None else 0,
        len(test_loader) if test_loader is not None else 0,
    )

    LOGGER.info(
        "Using embedding bank for contrastive loss with %d negatives per slide.",
        args.contrastive_negatives,
    )

    feature_extractor_model = build_feature_extractor(args, device) #SimCLR being loaded
    mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, -1, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=device).view(1, -1, 1, 1)

    sample_he, sample_ret, _, _ = dataset_for_preview[0] #he_stack, ret_stack, grade, record.contrastive_id
    feat_dim = infer_feature_dim(feature_extractor_model, sample_ret.unsqueeze(0).to(device), mean, std)
    LOGGER.info("Feature dimension inferred as %d", feat_dim)

    # abmil = ABMIL(
    #     in_dim=feat_dim,
    #     attn_dim=args.abmil_attn_dim,
    #     classifier_hidden=args.classifier_hidden_dim,
    #     num_classes=args.num_classes,
    #     dropout=args.abmil_dropout,
    # ).to(device)
    # _load_state_dict(abmil, resolve_path(args.pretrained_abmil))
    # freeze_module(abmil)

    # slide_classifier = SlideClassifier(
    #     feat_dim,
    #     args.classifier_hidden_dim,
    #     args.num_classes,
    #     dropout=args.abmil_dropout,
    # ).to(device)
    # _load_state_dict(slide_classifier, resolve_path(args.pretrained_classifier))
    # freeze_module(slide_classifier)

    abmil = ABMIL(
        in_dim=feat_dim,
        attn_dim=args.abmil_attn_dim,
        classifier_hidden=args.classifier_hidden_dim,
        num_classes=args.num_classes,
        dropout=args.abmil_dropout,
    ).to(device)

    _load_state_dict(abmil, resolve_path(args.pretrained_abmil))
    abmil.train() ###

    slide_classifier = SlideClassifier(
        feat_dim,
        args.classifier_hidden_dim,
        args.num_classes,
        dropout=args.abmil_dropout,
    ).to(device)

    _load_state_dict(slide_classifier, resolve_path(args.pretrained_classifier))
    slide_classifier.train() ###

    projector = ProjectionHead(feat_dim, args.proj_hidden_dim, args.proj_out_dim).to(device)

    G_H2R = CycleGenerator(img_channels=3).to(device)
    G_R2H = CycleGenerator(img_channels=3).to(device)
    D_R = CycleDiscriminator(in_channels=3).to(device)
    D_H = CycleDiscriminator(in_channels=3).to(device)
    load_cyclegan_weights(Path(args.pretrained_cyclegan), (G_H2R, G_R2H), (D_R, D_H))

    optimizer_G = torch.optim.Adam(
        list(G_H2R.parameters()) +
        list(G_R2H.parameters()) +
        list(projector.parameters()), ### the ProjectorHead
        lr=args.lr_gen,
        betas=(0.5, 0.999),
    )

    optimizer_cls = torch.optim.Adam(
        list(abmil.parameters()) +
        list(slide_classifier.parameters()),
        lr=1e-4,
        betas=(0.9, 0.999),
    )

    optimizer_D = torch.optim.Adam(
        list(D_R.parameters()) +
        list(D_H.parameters()),
        lr=args.lr_disc,
        betas=(0.5, 0.999),
    )

    criterion_gan = nn.MSELoss()
    l1_loss = nn.L1Loss()
    criterion_cls = nn.CrossEntropyLoss()
    
    grad_enabled = bool(args.grad_req)
    if not grad_enabled:
        LOGGER.warning("grad_req set to False – running in inference-only mode (no optimizer steps).")
    
    use_amp = bool(args.amp and device.type == "cuda" and grad_enabled)
    if args.amp and not use_amp and device.type != "cuda":
        LOGGER.warning("AMP requested but device %s does not support CUDA autocast.", device.type)
    LOGGER.info("AMP enabled: %s", use_amp)
    
    scaler_G = torch.amp.GradScaler("cuda") if use_amp else None
    scaler_D = torch.amp.GradScaler("cuda") if use_amp else None
    amp_context = (lambda: torch.amp.autocast(device_type="cuda")) if use_amp else nullcontext
    grad_modules = (G_H2R, G_R2H, projector, abmil, slide_classifier)
    module_states = {
        "G_H2R": G_H2R,
        "G_R2H": G_R2H,
        "Proj": projector,
        "D_R": D_R,
        "D_H": D_H,
        "ABMIL": abmil,
        "SlideClassifier": slide_classifier,
    }
    optimizer_states = {
        "opt_G": optimizer_G if grad_enabled else None,
        "opt_C": optimizer_cls if grad_enabled else None,
        "opt_D": optimizer_D if grad_enabled else None,
    }
    scaler_states = {
        "scaler_G": scaler_G if use_amp else None,
        "scaler_D": scaler_D if use_amp else None,
    }

    start_epoch = 1
    if args.resume_dir:
        load_checkpoints(
            Path(args.resume_dir),
            int(args.resume_epoch),
            module_states,
            optimizer_states,
            scaler_states,
        )
        start_epoch = int(args.resume_epoch) + 1
        LOGGER.info("Resuming training from epoch %d (next epoch %d).", args.resume_epoch, start_epoch)
    if start_epoch > args.epochs:
        LOGGER.info(
            "Start epoch (%d) exceeds configured epochs (%d); no training to run.",
            start_epoch,
            args.epochs,
        )
        return

    samples_dir.mkdir(parents=True, exist_ok=True)

    history_path = run_dir / "history.json"
    history: list[dict[str, float]] = []
    if args.resume_dir and Path(args.resume_dir) == run_dir and history_path.exists():
        try:
            history = json.loads(history_path.read_text("utf-8"))
        except Exception as exc:
            LOGGER.warning("Failed to load existing history from %s (%s).", history_path, exc)
            history = []
    preview_he: torch.Tensor | None = None
    preview_ret: torch.Tensor | None = None
    preview_index = train_slide_indices[0] if train_slide_indices else 0
    try:
        he_stack, ret_stack, _, _ = dataset_for_preview.get_slide(preview_index, deterministic=True)
        # he_stack: [P, C, H, W] -> add a slide dimension -> [1, P, C, H, W]
        preview_he = he_stack.unsqueeze(0).to(device, non_blocking=True)
        preview_ret = ret_stack.unsqueeze(0).to(device, non_blocking=True)
    except Exception as exc:
        LOGGER.warning("Unable to collect preview samples (%s); skipping sample grids.", exc)

    base_train_indices = list(train_slide_indices)
    step_idx = 0
    for epoch in range(start_epoch, args.epochs + 1):
        G_H2R.train()
        G_R2H.train()
        D_R.train()
        D_H.train()
        abmil.train() ###
        slide_classifier.train() ###
        projector.train() ###

        epoch_metrics = {
            "G": 0.0,
            "D": 0.0,
            "cls": 0.0,
            "con": 0.0,
            "cycle": 0.0,
            "identity": 0.0,
            "adv": 0.0,
        }
        epoch_seed = args.seed + epoch
        order_rng = random.Random(epoch_seed)
        chunk_rng = random.Random(epoch_seed * 9973 + 17)
        epoch_indices = base_train_indices[:]
        order_rng.shuffle(epoch_indices)
        epoch_schedule = _build_epoch_schedule(
            slides,
            epoch_indices,
            args.patches_per_slide,
            args.top_fraction,
            chunk_rng,
        )
        slide_batches = _batch_slide_indices(epoch_indices, args.batch_slides)
        total_steps = 0
        for batch_indices in slide_batches:
            valid_batch = [
                idx for idx in batch_indices if idx in epoch_schedule and epoch_schedule[idx].num_chunks > 0
            ]
            if not valid_batch:
                continue
            total_steps += max(epoch_schedule[idx].num_chunks for idx in valid_batch)
        progress = tqdm(total=total_steps, desc=f"Epoch {epoch}") if total_steps else None
        batches = 0

        for batch_indices in slide_batches:
            valid_batch = [
                idx for idx in batch_indices if idx in epoch_schedule and epoch_schedule[idx].num_chunks > 0
            ]
            if not valid_batch:
                continue
            T_batch = max(epoch_schedule[idx].num_chunks for idx in valid_batch)
            for chunk_idx in range(T_batch):
                step_idx += 1
                if args.profile_steps and device.type == "cuda":
                    torch.cuda.reset_peak_memory_stats(device)
                    mem_start = torch.cuda.memory_allocated(device)
                    t_start = time.perf_counter()
                he_slide_stacks = []
                ret_slide_stacks = []
                grades = []
                slide_ids = []
                for slide_idx in valid_batch:
                    schedule_entry = epoch_schedule[slide_idx]
                    chunk_pos = min(chunk_idx, schedule_entry.num_chunks - 1)
                    slide = slides[slide_idx]
                    he_chunk = schedule_entry.he_chunks[chunk_pos]
                    ret_chunk = schedule_entry.ret_chunks[chunk_pos]
                    he_stack = _load_chunk_stack(
                        dataset,
                        slide.he_paths,
                        he_chunk,
                        deterministic=False,
                    )
                    ret_stack = _load_chunk_stack(
                        dataset,
                        slide.ret_paths,
                        ret_chunk,
                        deterministic=False,
                    )
                    he_slide_stacks.append(he_stack)
                    ret_slide_stacks.append(ret_stack)
                    grades.append(slide.grade)
                    slide_ids.append(slide.contrastive_id)
                if not he_slide_stacks or not ret_slide_stacks:
                    continue
                batches += 1
                he_batch = torch.stack(he_slide_stacks, dim=0).to(device, non_blocking=True)
                ret_batch = torch.stack(ret_slide_stacks, dim=0).to(device, non_blocking=True)
                grades_tensor = torch.tensor(grades, dtype=torch.long, device=device)

                he_flat = he_batch.view(-1, *he_batch.shape[2:])
                ret_flat = ret_batch.view(-1, *ret_batch.shape[2:])

                with amp_context():
                    
                    #Generator
                    fake_ret_flat = G_H2R(he_flat)
                    fake_he_flat = G_R2H(ret_flat)
                    fake_ret = fake_ret_flat.view_as(ret_batch)
                    fake_he = fake_he_flat.view_as(he_batch)

                    rec_he = G_R2H(fake_ret_flat).view_as(he_batch)
                    rec_ret = G_H2R(fake_he_flat).view_as(ret_batch)

                    cycle_loss = l1_loss(rec_he, he_batch) + l1_loss(rec_ret, ret_batch)

                    #Discriminator
                    pred_fake_ret = D_R(fake_ret_flat)
                    pred_fake_he = D_H(fake_he_flat)
                    adv_loss = criterion_gan(pred_fake_ret, torch.ones_like(pred_fake_ret)) + criterion_gan(
                        pred_fake_he, torch.ones_like(pred_fake_he)
                    )

                    if args.lambda_identity > 0:
                        id_ret = G_H2R(ret_flat).view_as(ret_batch) # G_H2R should still return Reticulin if it is given an input of Reticulin 
                        id_he = G_R2H(he_flat).view_as(he_batch) # G_R2H should still return HE if it is given an input of HE
                        id_loss = l1_loss(id_ret, ret_batch) + l1_loss(id_he, he_batch)
                    else:
                        id_loss = torch.zeros((), device=device)

                z_fake = extract_patch_features(feature_extractor_model, fake_ret, mean, std, require_grad=True)
                cls_loss, _ = compute_classification_loss(z_fake, grades_tensor, abmil, slide_classifier, criterion_cls)
                contrastive_loss = compute_contrastive_loss_with_bank(
                    z_fake,
                    slide_ids,
                    embedding_bank,
                    abmil,
                    projector,
                    args.contrastive_negatives,
                    args.temperature,
                )

                id_weight = 0.0 if args.disable_identity_loss else args.lambda_identity

                weighted_adv = adv_loss
                weighted_cycle = args.lambda_cycle * cycle_loss
                weighted_identity = id_weight * id_loss
                weighted_cls = args.lambda_cls * cls_loss ###
                weighted_con = args.lambda_con * contrastive_loss ###

                if grad_enabled:
                    grad_adv = component_grad_norm(weighted_adv, grad_modules)
                    grad_cycle = component_grad_norm(weighted_cycle, grad_modules)
                    grad_identity = component_grad_norm(weighted_identity, grad_modules)
                    grad_cls = component_grad_norm(weighted_cls, grad_modules)
                    grad_con = component_grad_norm(weighted_con, grad_modules)
                else:
                    grad_adv = grad_cycle = grad_identity = grad_cls = grad_con = 0.0

                loss_G = weighted_adv + weighted_cycle + weighted_identity + weighted_cls + weighted_con ###

                if grad_enabled:
                    optimizer_G.zero_grad(set_to_none=True)
                    if use_amp:
                        assert scaler_G is not None
                        scaler_G.scale(loss_G).backward()
                        scaler_G.step(optimizer_G)
                        scaler_G.update()

                        scaler_G.scale(loss_G).backward() ###
                        scaler_G.step(optimizer_cls) ###
                        scaler_G.update()
                    else:
                        loss_G.backward() ###
                        optimizer_G.step() ###

                        loss_G.backward() ###
                        optimizer_cls.step() ###
                else:
                    loss_G = loss_G.detach()

                with amp_context():
                    D_R_loss, _ = adversarial_loss(D_R, ret_flat, fake_ret_flat, criterion_gan)
                    D_H_loss, _ = adversarial_loss(D_H, he_flat, fake_he_flat, criterion_gan)
                    loss_D = D_R_loss + D_H_loss

                if grad_enabled:
                    optimizer_D.zero_grad(set_to_none=True)
                    if use_amp:
                        assert scaler_D is not None
                        scaler_D.scale(loss_D).backward()
                        scaler_D.step(optimizer_D)
                        scaler_D.update()
                    else:
                        loss_D.backward()
                        optimizer_D.step()
                else:
                    loss_D = loss_D.detach()

                # -------------------------
                # Classifier update
                # -------------------------
                # if grad_enabled and args.lambda_cls > 0:
                #     optimizer_cls.zero_grad(set_to_none=True)
                #     with amp_context():
                #         cls_loss_c, _ = compute_classification_loss(
                #             z_fake.detach(),
                #             grades_tensor,
                #             abmil,
                #             slide_classifier,
                #             criterion_cls,
                #         )
                #         weighted_cls_c = args.lambda_cls * cls_loss_c

                #     if use_amp:
                #         assert scaler_G is not None
                #         scaler_G.scale(weighted_cls_c).backward()
                #         scaler_G.step(optimizer_cls)
                #         scaler_G.update()
                #     else:
                #         weighted_cls_c.backward()
                #         optimizer_cls.step()
                        
                # if not grad_enabled:
                #     loss_G = loss_G.detach()
                #     loss_D = loss_D.detach()
                #     cls_loss = cls_loss.detach()
                #     contrastive_loss = contrastive_loss.detach()

                LOGGER.info(
                    "[GRADS] adv=%.6e | cycle=%.6e | identity=%.6e | cls=%.6e | con=%.6e || "
                    "G_H2R=%.6e | G_R2H=%.6e | Proj=%.6e | ABMIL=%.6e | CLS=%.6e",
                    grad_adv,
                    grad_cycle,
                    grad_identity,
                    grad_cls,
                    grad_con,
                    grad_norm(G_H2R) if grad_enabled else 0.0,
                    grad_norm(G_R2H) if grad_enabled else 0.0,
                    grad_norm(projector) if grad_enabled else 0.0,
                    grad_norm(abmil) if grad_enabled else 0.0,
                    grad_norm(slide_classifier) if grad_enabled else 0.0,
                )

                if args.profile_steps and device.type == "cuda":
                    torch.cuda.synchronize(device)
                    t_end = time.perf_counter()
                    mem_end = torch.cuda.memory_allocated(device)
                    mem_peak = torch.cuda.max_memory_allocated(device)
                    LOGGER.info(
                        "Profile | epoch=%d step=%d | time=%.3fs | mem_start=%.1f MB | mem_end=%.1f MB | mem_peak=%.1f MB",
                        epoch,
                        step_idx,
                        t_end - t_start,
                        mem_start / (1024**2),
                        mem_end / (1024**2),
                        mem_peak / (1024**2),
                    )

                epoch_metrics["G"] += loss_G.item()
                epoch_metrics["D"] += loss_D.item()
                epoch_metrics["cls"] += cls_loss.item()
                epoch_metrics["con"] += contrastive_loss.item()
                epoch_metrics["cycle"] += cycle_loss.item()
                epoch_metrics["identity"] += id_loss.item()
                epoch_metrics["adv"] += adv_loss.item()
                if progress:
                    progress.update(1)

        if progress:
            progress.close()

        for key in epoch_metrics:
            epoch_metrics[key] /= max(1, batches)

        LOGGER.info(
            "Epoch %d | G: %.4f | D: %.4f | Adv: %.4f | Cycle: %.4f | Id: %.4f | Cls: %.4f | Con: %.4f",
            epoch,
            epoch_metrics["G"],
            epoch_metrics["D"],
            epoch_metrics["adv"],
            epoch_metrics["cycle"],
            epoch_metrics["identity"],
            epoch_metrics["cls"],
            epoch_metrics["con"],
        )

        val_metrics = evaluate_split(
            "val",
            val_loader,
            device=device,
            mean=mean,
            std=std,
            feature_extractor=feature_extractor_model,
            abmil=abmil,
            slide_classifier=slide_classifier,
            projector=projector,
            G_H2R=G_H2R,
            embedding_bank=embedding_bank,
            args=args,
            criterion_cls=criterion_cls,
        )
        test_metrics = evaluate_split(
            "test",
            test_loader,
            device=device,
            mean=mean,
            std=std,
            feature_extractor=feature_extractor_model,
            abmil=abmil,
            slide_classifier=slide_classifier,
            projector=projector,
            G_H2R=G_H2R,
            embedding_bank=embedding_bank,
            args=args,
            criterion_cls=criterion_cls,
        )
        G_H2R.train()
        G_R2H.train()
        projector.train()

        history.append(
            {
                "epoch": epoch,
                **epoch_metrics,
                "val_cls": val_metrics.get("cls") if val_metrics else None,
                "val_con": val_metrics.get("con") if val_metrics else None,
                "val_acc": val_metrics.get("acc") if val_metrics else None,
                "test_cls": test_metrics.get("cls") if test_metrics else None,
                "test_con": test_metrics.get("con") if test_metrics else None,
                "test_acc": test_metrics.get("acc") if test_metrics else None,
            }
        )
        with history_path.open("w", encoding="utf-8") as fp:
            json.dump(history, fp, indent=2)

        if (
            preview_he is not None
            and preview_ret is not None
            and args.save_samples_every > 0
            and epoch % args.save_samples_every == 0
        ):
            G_H2R.eval()
            G_R2H.eval()
            with torch.no_grad():
                he_sample = preview_he
                ret_sample = preview_ret
                he_flat = he_sample.view(-1, *he_sample.shape[2:])
                ret_flat = ret_sample.view(-1, *ret_sample.shape[2:])
                fake_ret_sample = G_H2R(he_flat).view_as(he_sample)
                fake_he_sample = G_R2H(ret_flat).view_as(ret_sample)
            he_pair = torch.cat(
                [
                    he_sample.view(-1, *he_sample.shape[2:]),
                    fake_ret_sample.view(-1, *fake_ret_sample.shape[2:]),
                ],
                dim=0,
            )
            ret_pair = torch.cat(
                [
                    ret_sample.view(-1, *ret_sample.shape[2:]),
                    fake_he_sample.view(-1, *fake_he_sample.shape[2:]),
                ],
                dim=0,
            )
            save_image(
                he_pair * 0.5 + 0.5,
                str(samples_dir / f"he_to_ret_epoch{epoch}.jpg"),
                nrow=he_sample.shape[1],
            )
            save_image(
                ret_pair * 0.5 + 0.5,
                str(samples_dir / f"ret_to_he_epoch{epoch}.jpg"),
                nrow=ret_sample.shape[1],
            )
            G_H2R.train()
            G_R2H.train()
            projector.train()

        if epoch % args.save_every == 0 or epoch == args.epochs:
            save_checkpoints(
                run_dir,
                epoch,
                module_states,
                optimizer_states,
                scaler_states,
            )

    plot_metrics(history, run_dir / "metrics.png")
    LOGGER.info("Training complete. History saved to %s", history_path)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    train(args)


if __name__ == "__main__":
    main()
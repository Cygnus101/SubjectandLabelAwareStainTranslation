#!/usr/bin/env python3
"""Build augmented slide dataset with attention flags and constrained splits."""

from __future__ import annotations

import argparse
import json
import logging
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence

import pandas as pd
import yaml

LOGGER = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "build_augmented_dataset.yaml"


def _closest_stain_id(source: str, candidates: Sequence[str]) -> str | None:
    if not candidates:
        return None
    if source in candidates:
        return source

    def _tokens(val: str) -> List[int]:
        return [int(match.group()) for match in re.finditer(r"\d+", val)]

    def _prefix(val: str) -> str:
        return re.sub(r"\d+", "", val).replace("_", "").replace("-", "").lower()

    src_tokens = _tokens(source)
    src_prefix = _prefix(source)

    def score(candidate: str) -> tuple[int, int, str]:
        cand_tokens = _tokens(candidate)
        cand_prefix = _prefix(candidate)
        prefix_penalty = 0 if cand_prefix == src_prefix and src_prefix else 1
        if src_tokens and cand_tokens:
            diff = abs(src_tokens[-1] - cand_tokens[-1])
        else:
            diff = abs(len(source) - len(candidate))
        return (prefix_penalty, diff, candidate)

    return min(candidates, key=score)


def _parse_grade(value: Any) -> int | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    text = str(value).strip()
    if not text:
        return None
    match = re.search(r"(\d+)", text)
    if not match:
        return None
    grade = int(match.group(1))
    return max(0, min(3, grade))


@dataclass
class SlideRecord:
    lab_id: str
    he_stain_id: str
    ret_stain_id: str
    grade: int
    he_patches: list[dict[str, Any]]
    ret_patches: list[dict[str, Any]]


def _load_attention_lookup(csv_path: Path) -> Dict[tuple[str, str], int]:
    df = pd.read_csv(csv_path)
    required = {"slide_id", "patch_path", "is_top10"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"attention CSV missing columns: {sorted(missing)}")
    lookup: Dict[tuple[str, str], int] = {}
    for _, row in df.iterrows():
        slide = str(row["slide_id"]).strip()
        patch = str(row["patch_path"]).strip()
        flag = int(row.get("is_top10", 0))
        lookup[(slide, patch)] = 1 if flag else 0
    return lookup


def _apply_subset(df: pd.DataFrame, subset_pct: float | None, order: str, rng: random.Random) -> pd.DataFrame:
    if subset_pct is None:
        return df
    pct = max(0.0, min(100.0, float(subset_pct)))
    labs = df["lab_id"].unique().tolist()
    if not labs:
        return df
    target = max(1, int(round(len(labs) * (pct / 100.0))))
    if target >= len(labs):
        return df
    if order == "ascending":
        selected = sorted(labs)[:target]
    elif order == "descending":
        selected = sorted(labs, reverse=True)[:target]
    else:
        selected = rng.sample(labs, target)
    LOGGER.info("Subset active (%s): %.2f%% of labs -> %d labs", order, pct, len(selected))
    return df[df["lab_id"].isin(selected)].copy()


def build_slide_records(metadata_path: Path, attention_path: Path, subset_pct: float | None, subset_order: str, seed: int) -> list[SlideRecord]:
    rng = random.Random(seed)
    df = pd.read_csv(metadata_path)
    required_cols = {"Lab No.", "stain_id", "type", "patch_path", "Reticulin Grade"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"metadata missing columns: {sorted(missing)}")
    df["lab_id"] = df["Lab No."].astype(str).str.strip()
    df["type_norm"] = df["type"].astype(str).str.lower()
    df["patch_path_norm"] = df["patch_path"].astype(str)
    df = df[df["lab_id"].astype(bool)]
    df = _apply_subset(df, subset_pct, subset_order, rng)

    attention_lookup = _load_attention_lookup(attention_path)
    slides: list[SlideRecord] = []

    for lab_id, lab_df in df.groupby("lab_id"):
        he_df = lab_df[lab_df["type_norm"].str.contains("h&e", na=False)]
        ret_df = lab_df[lab_df["type_norm"].str.contains("reticulin", na=False)]
        if he_df.empty or ret_df.empty:
            continue
        he_groups = {str(sid): group for sid, group in he_df.groupby("stain_id")}
        ret_groups = {str(sid): group for sid, group in ret_df.groupby("stain_id")}
        for he_id, he_group in he_groups.items():
            ret_id = _closest_stain_id(he_id, list(ret_groups.keys()))
            if ret_id is None:
                continue
            ret_group = ret_groups[ret_id]
            grade = _parse_grade(ret_group["Reticulin Grade"].iloc[0])
            if grade is None:
                continue
            he_paths = he_group["patch_path_norm"].astype(str).tolist()
            ret_paths = ret_group["patch_path_norm"].astype(str).tolist()
            if not he_paths or not ret_paths:
                continue

            he_patches = [
                {"patch_path": path, "is_top10": attention_lookup.get((he_id, path), 0)}
                for path in he_paths
            ]
            ret_patches = [
                {"patch_path": path, "is_top10": attention_lookup.get((ret_id, path), 0)}
                for path in ret_paths
            ]
            slides.append(
                SlideRecord(
                    lab_id=str(lab_id),
                    he_stain_id=he_id,
                    ret_stain_id=ret_id,
                    grade=grade,
                    he_patches=he_patches,
                    ret_patches=ret_patches,
                )
            )
    if not slides:
        raise RuntimeError("No eligible slides were constructed from metadata + attention inputs.")
    LOGGER.info("Constructed %d slide records.", len(slides))
    return slides


def build_splits(
    slides: list[SlideRecord],
    train_labs: Sequence[str],
    val_labs: Sequence[str],
    test_labs: Sequence[str],
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> tuple[list[int], list[int], list[int]]:
    train_set = {lab.strip() for lab in train_labs}
    val_set = {lab.strip() for lab in val_labs}
    test_set = {lab.strip() for lab in test_labs}
    if train_set & val_set or train_set & test_set or val_set & test_set:
        LOGGER.warning("Overlap detected between train/val/test lab constraints; precedence train > val > test.")
    val_fixed: list[int] = []
    test_fixed: list[int] = []
    train_fixed: list[int] = []
    remaining: list[int] = []
    for idx, slide in enumerate(slides):
        lab = slide.lab_id
        if lab in train_set:
            train_fixed.append(idx)
        elif lab in val_set:
            val_fixed.append(idx)
        elif lab in test_set:
            test_fixed.append(idx)
        else:
            remaining.append(idx)

    rng = random.Random(seed)
    N = len(remaining)

    if N == 0:
        if not train_fixed:
            raise RuntimeError("No slides available for training after applying lab constraints.")
        LOGGER.info(
            "No remaining slides for random split; using fixed assignments (train=%d, val=%d, test=%d).",
            len(train_fixed),
            len(val_fixed),
            len(test_fixed),
        )
        return train_fixed, val_fixed, test_fixed

    n_val_extra = int(round(val_ratio * N))
    n_test_extra = int(round(test_ratio * N))
    max_allowed = max(0, N - 2)
    if n_val_extra + n_test_extra > max_allowed:
        overflow = n_val_extra + n_test_extra - max_allowed
        reduce_test = min(n_test_extra, overflow)
        n_test_extra -= reduce_test
        overflow -= reduce_test
        if overflow > 0:
            n_val_extra = max(0, n_val_extra - overflow)
    perm = remaining[:]
    rng.shuffle(perm)
    train_cut = len(perm) - n_val_extra - n_test_extra
    train_indices = perm[:train_cut]
    val_extra = perm[train_cut : train_cut + n_val_extra]
    test_extra = perm[train_cut + n_val_extra :]

    train_all = train_fixed + train_indices
    val_all = val_fixed + val_extra
    test_all = test_fixed + test_extra

    if not train_all:
        raise RuntimeError("Training split ended up empty. Adjust ratios or constraints.")
    LOGGER.info(
        "Split sizes | train=%d | val=%d | test=%d",
        len(train_all),
        len(val_all),
        len(test_all),
    )
    return train_all, val_all, test_all


def _load_config(path: str | None) -> dict[str, Any]:
    if not path:
        return {}
    cfg_path = Path(path)
    if not cfg_path.exists():
        return {}
    data = yaml.safe_load(cfg_path.read_text("utf-8"))
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"Config file {cfg_path} must contain a mapping/object.")
    return data


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    base_parser = argparse.ArgumentParser(add_help=False)
    base_parser.add_argument(
        "--config",
        type=str,
        default=str(DEFAULT_CONFIG),
        help="Path to YAML config with default arguments.",
    )
    config_ns, remaining = base_parser.parse_known_args(argv)
    config_defaults = _load_config(config_ns.config)

    parser = argparse.ArgumentParser(description=__doc__, parents=[base_parser])
    parser.add_argument("--metadata", type=Path, default=Path("metadata.csv"))
    parser.add_argument("--attention-csv", type=Path, default=Path("attention_with_top10.csv"))
    parser.add_argument("--output-slides", type=Path, default=Path("augmented_slides.json"))
    parser.add_argument("--output-splits", type=Path, default=Path("augmented_splits.json"))
    parser.add_argument("--subset-pct", type=float, default=None)
    parser.add_argument(
        "--subset-order",
        type=str,
        choices=["random", "ascending", "descending"],
        default="random",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--train-lab", action="append", default=[], help="Lab IDs forced into training")
    parser.add_argument("--val-lab", action="append", default=[], help="Lab IDs forced into validation")
    parser.add_argument("--test-lab", action="append", default=[], help="Lab IDs forced into test")
    if config_defaults:
        parser.set_defaults(**config_defaults)
    args = parser.parse_args(argv)

    # Normalize val/test lab lists even when provided via config file.
    def _normalize_lab_list(value):
        if value is None:
            return []
        if isinstance(value, (list, tuple)):
            return list(value)
        return [value]

    args.train_lab = [str(lab).strip() for lab in _normalize_lab_list(args.train_lab)]
    args.val_lab = [str(lab).strip() for lab in _normalize_lab_list(args.val_lab)]
    args.test_lab = [str(lab).strip() for lab in _normalize_lab_list(args.test_lab)]
    return args


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    args = parse_args()

    slides = build_slide_records(
        metadata_path=args.metadata,
        attention_path=args.attention_csv,
        subset_pct=args.subset_pct,
        subset_order=args.subset_order,
        seed=args.seed,
    )
    train_idx, val_idx, test_idx = build_splits(
        slides,
        train_labs=args.train_lab,
        val_labs=args.val_lab,
        test_labs=args.test_lab,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )

    slides_payload = [slide.__dict__ for slide in slides]
    args.output_slides.parent.mkdir(parents=True, exist_ok=True)
    with args.output_slides.open("w", encoding="utf-8") as fp:
        json.dump(slides_payload, fp, indent=2)
    LOGGER.info("Wrote %d slide records to %s", len(slides), args.output_slides)

    splits_payload = {
        "train_indices": train_idx,
        "val_indices": val_idx,
        "test_indices": test_idx,
    }
    args.output_splits.parent.mkdir(parents=True, exist_ok=True)
    with args.output_splits.open("w", encoding="utf-8") as fp:
        json.dump(splits_payload, fp, indent=2)
    LOGGER.info("Wrote split indices to %s", args.output_splits)


if __name__ == "__main__":
    main()

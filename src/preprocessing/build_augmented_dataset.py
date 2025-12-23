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


def build_slide_records(
    metadata_path: Path,
    attention_path: Path | None,
    subset_pct: float | None,
    subset_order: str,
    seed: int,
) -> list[SlideRecord]:
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

    attention_lookup = _load_attention_lookup(attention_path) if attention_path else {}
    slides: list[SlideRecord] = []

    for lab_id, lab_df in df.groupby("lab_id"):
        he_df = lab_df[lab_df["type_norm"].str.contains("h&e", na=False)]
        ret_df = lab_df[lab_df["type_norm"].str.contains("reticulin", na=False)]
        if he_df.empty or ret_df.empty:
            continue
        he_groups = {str(sid): group for sid, group in he_df.groupby("stain_id")}
        ret_groups = {str(sid): group for sid, group in ret_df.groupby("stain_id")}

        # Track which stain IDs have already been used in a pair so we can add "ret-only" pairs
        paired_he: set[str] = set()
        paired_ret: set[str] = set()

        # 1) Primary pairing: for each H&E stain, pick the closest Reticulin stain (existing behavior)
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

            paired_he.add(he_id)
            paired_ret.add(ret_id)

        # 2) Secondary pairing: for each Reticulin stain that didn't get used above,
        #    map it to the closest available H&E stain in the same lab.
        #    This retains Reticulin labels and prevents dropping Ret-only stains.
        for ret_id, ret_group in ret_groups.items():
            if ret_id in paired_ret:
                continue

            he_id = _closest_stain_id(ret_id, list(he_groups.keys()))
            if he_id is None:
                # No H&E stains exist in this lab (should be rare since we checked earlier)
                continue

            he_group = he_groups[he_id]
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
    """Build lab-safe train/val/test splits.

    Key change vs previous behavior:
    - Ratios are enforced over the TOTAL number of labs (not just the remaining/unfixed labs).
    - Any labs passed via train_labs/val_labs/test_labs are guaranteed to be included.
    """

    # Normalize + strip
    train_set = {str(lab).strip() for lab in train_labs if str(lab).strip()}
    val_set = {str(lab).strip() for lab in val_labs if str(lab).strip()}
    test_set = {str(lab).strip() for lab in test_labs if str(lab).strip()}

    if train_set & val_set or train_set & test_set or val_set & test_set:
        LOGGER.warning("Overlap detected between train/val/test lab constraints; precedence train > val > test.")

    # Slide indices by lab
    by_lab: dict[str, list[int]] = {}
    for idx, slide in enumerate(slides):
        by_lab.setdefault(slide.lab_id, []).append(idx)

    all_labs = sorted(by_lab.keys())
    N = len(all_labs)
    if N == 0:
        raise RuntimeError("No labs found in constructed slides; cannot build splits.")

    # Fixed assignments with precedence train > val > test
    train_fixed: set[str] = set()
    val_fixed: set[str] = set()
    test_fixed: set[str] = set()

    for lab in all_labs:
        if lab in train_set:
            train_fixed.add(lab)
        elif lab in val_set:
            val_fixed.add(lab)
        elif lab in test_set:
            test_fixed.add(lab)

    remaining = [lab for lab in all_labs if lab not in (train_fixed | val_fixed | test_fixed)]

    # Ratio targets over TOTAL labs
    vr = max(0.0, min(0.9, float(val_ratio)))
    tr = max(0.0, min(0.9, float(test_ratio)))

    n_val_target = int(round(vr * N))
    n_test_target = int(round(tr * N))

    # Ensure fixed labs fit
    n_val_target = max(n_val_target, len(val_fixed))
    n_test_target = max(n_test_target, len(test_fixed))

    # Ensure at least 1 train lab
    n_train_target = N - n_val_target - n_test_target
    if n_train_target < 1:
        deficit = 1 - n_train_target
        reducible_test = max(0, n_test_target - len(test_fixed))
        take = min(reducible_test, deficit)
        n_test_target -= take
        deficit -= take

        reducible_val = max(0, n_val_target - len(val_fixed))
        take = min(reducible_val, deficit)
        n_val_target -= take
        deficit -= take

        n_train_target = N - n_val_target - n_test_target
        if n_train_target < 1:
            raise RuntimeError(
                "Unable to allocate at least 1 training lab given fixed constraints. "
                "Reduce fixed val/test labs or ratios."
            )

    # Sample extra labs
    n_val_extra = max(0, n_val_target - len(val_fixed))
    n_test_extra = max(0, n_test_target - len(test_fixed))

    rng = random.Random(seed)
    rng.shuffle(remaining)

    val_extra = remaining[:n_val_extra]
    test_extra = remaining[n_val_extra : n_val_extra + n_test_extra]
    train_extra = remaining[n_val_extra + n_test_extra :]

    train_labs_final = sorted(train_fixed | set(train_extra))
    val_labs_final = sorted(val_fixed | set(val_extra))
    test_labs_final = sorted(test_fixed | set(test_extra))

    def _indices_for(labs: list[str]) -> list[int]:
        out: list[int] = []
        for lab in labs:
            out.extend(by_lab.get(lab, []))
        return out

    train_idx = _indices_for(train_labs_final)
    val_idx = _indices_for(val_labs_final)
    test_idx = _indices_for(test_labs_final)

    if not train_idx:
        raise RuntimeError("Training split ended up empty. Adjust ratios or constraints.")

    LOGGER.info(
        "Lab targets (total=%d) | train=%d | val=%d | test=%d",
        N,
        len(train_labs_final),
        len(val_labs_final),
        len(test_labs_final),
    )
    LOGGER.info(
        "Slide indices | train=%d | val=%d | test=%d",
        len(train_idx),
        len(val_idx),
        len(test_idx),
    )

    return train_idx, val_idx, test_idx


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
    parser.add_argument(
        "--attention-csv",
        type=Path,
        default=None,
        help="Optional attention CSV to mark top-k patches; omit to skip.",
    )
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

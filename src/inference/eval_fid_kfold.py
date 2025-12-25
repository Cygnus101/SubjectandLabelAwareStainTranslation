#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Evaluate CycleGAN generators by computing FID on explicit train/val/test splits sourced from augmented_splits.json.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Sequence

import pandas as pd
import torch
from tqdm.auto import tqdm

from src.inference import eval_fid as base
from utils.path import resolve_path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def _load_split_lists(splits_path: Path, slides_path: Path) -> Dict[str, list[str]]:
    splits_payload = json.loads(splits_path.read_text("utf-8"))
    if not isinstance(splits_payload, dict):
        raise ValueError(f"{splits_path} must contain a JSON object (augmented_splits format).")
    slides_payload = json.loads(slides_path.read_text("utf-8"))
    if not isinstance(slides_payload, list):
        raise ValueError(f"{slides_path} must contain a JSON list (augmented_slides format).")

    def _stain_id(idx: int) -> str:
        entry = slides_payload[idx]
        return str(entry.get("ret_stain_id") or entry.get("he_stain_id") or f"idx_{idx}")

    result: Dict[str, list[str]] = {"train": [], "val": [], "test": []}
    for split in result.keys():
        key = f"{split}_indices"
        for idx in splits_payload.get(key, []):
            if isinstance(idx, int) and 0 <= idx < len(slides_payload):
                result[split].append(_stain_id(idx))

    if not any(result.values()):
        raise RuntimeError(f"No split indices found in {splits_path}.")
    return result


def _get_rows_for_stain(df: pd.DataFrame, stain_id: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    stain_mask = df["stain_id"].astype(str) == stain_id
    subset = df[stain_mask]
    if subset.empty:
        raise RuntimeError(f"Stain '{stain_id}' missing from metadata.")
    type_series = subset["type"].astype(str).str.lower()
    clean = type_series.str.replace(r"[^a-z]", "", regex=True)
    he_rows = subset[clean == "he"].reset_index(drop=True)
    ret_rows = subset[clean == "reticulin"].reset_index(drop=True)
    if he_rows.empty or ret_rows.empty:
        raise RuntimeError(f"Stain '{stain_id}' lacks H&E or Reticulin rows.")
    return he_rows, ret_rows


def _compute_fid_for_stain(
    generator: torch.nn.Module,
    df: pd.DataFrame,
    stain_id: str,
    patch_size: int,
    device: torch.device,
    work_dir: Path,
) -> float:
    he_rows, ret_rows = _get_rows_for_stain(df, stain_id)
    real_entries = base._rows_to_entries(ret_rows)
    patches_dir = work_dir / f"stain_{stain_id}_generated"
    fake_entries = base._generate_reticulin_patches(
        generator,
        he_rows,
        patch_size,
        device,
        patches_dir,
    )
    fid_value = base._compute_fid(
        real_paths=[entry.path for entry in real_entries],
        fake_paths=[entry.path for entry in fake_entries],
        device=device,
    )
    return fid_value


def _fmt(score: float | None) -> str:
    return "N/A" if score is None else f"{score:.4f}"


def _compute_split_fid(
    generator: torch.nn.Module,
    df: pd.DataFrame,
    stain_ids: Sequence[str],
    split_name: str,
    patch_size: int,
    device: torch.device,
    ckpt_dir: Path,
) -> float | None:
    scores: list[float] = []
    for stain_id in stain_ids:
        try:
            fid = _compute_fid_for_stain(
                generator,
                df,
                stain_id,
                patch_size,
                device,
                ckpt_dir,
            )
            scores.append(fid)
            logging.info("Split=%s | stain=%s | FID=%.4f", split_name, stain_id, fid)
        except Exception as exc:
            logging.warning("Skipping stain %s (split=%s) due to error: %s", stain_id, split_name, exc)
    if not scores:
        return None
    return float(sum(scores) / len(scores))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate FID per splits defined in augmented_splits.json.")
    parser.add_argument("--checkpoint", type=Path, default=None, help="Path to a single CycleGAN H2R checkpoint.")
    parser.add_argument(
        "--checkpoint-root",
        type=Path,
        default=None,
        help="Directory to recursively search for G_H2R checkpoints.",
    )
    parser.add_argument("--metadata", type=Path, default=base.DEFAULT_METADATA)
    parser.add_argument("--augmented-splits", type=Path, default=Path("augmented_splits.json"))
    parser.add_argument("--augmented-slides", type=Path, default=Path("augmented_slides.json"))
    parser.add_argument("--output-dir", type=Path, default=base.DEFAULT_OUTPUT_DIR / "splits")
    parser.add_argument("--patch-size", type=int, default=512)
    parser.add_argument("--device", type=str, default=base._default_device())
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    checkpoints = base._discover_checkpoints(args.checkpoint, args.checkpoint_root)

    df = pd.read_csv(resolve_path(args.metadata))
    required_cols = {"patch_path", "type", "x", "y", "stain_id"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"metadata missing required columns: {sorted(missing)}")

    split_map = _load_split_lists(resolve_path(args.augmented_splits), resolve_path(args.augmented_slides))
    if not any(split_map.values()):
        raise RuntimeError("No stain IDs found in the provided split CSV.")

    run_dir = resolve_path(args.output_dir, allow_missing=True)
    run_dir.mkdir(parents=True, exist_ok=True)

    for ckpt_path in tqdm(checkpoints, desc="Checkpoints"):
        generator = base.load_generator(ckpt_path, device)
        ckpt_dir = run_dir / ckpt_path.stem
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        split_scores: dict[str, float | None] = {}
        for split_name, stains in split_map.items():
            if not stains:
                continue
            score = _compute_split_fid(
                generator,
                df,
                stains,
                split_name,
                args.patch_size,
                device,
                ckpt_dir,
            )
            split_scores[split_name] = score

        logging.info(
            "[%s] FID train=%s | val=%s | test=%s",
            ckpt_path.name,
            _fmt(split_scores.get("train")),
            _fmt(split_scores.get("val")),
            _fmt(split_scores.get("test")),
        )


if __name__ == "__main__":
    main()

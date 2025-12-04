#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
K-fold variant of eval_fid: evaluate CycleGAN generators across multiple labs by
partitioning labs into K folds and averaging per-fold FID.
"""

from __future__ import annotations

import argparse
import logging
import random
from pathlib import Path
from typing import Sequence

import pandas as pd
import torch
from tqdm.auto import tqdm

from inference import eval_fid as base
from utils.path import resolve_path


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def _labs_with_both(df: pd.DataFrame, lab_col: str) -> list[str]:
    type_series = df["type"].astype(str).str.lower()
    clean = type_series.str.replace(r"[^a-z]", "", regex=True)
    he_mask = clean == "he"
    ret_mask = clean == "reticulin"
    return sorted(set(df.loc[he_mask, lab_col]) & set(df.loc[ret_mask, lab_col]))


def _split_kfold(items: Sequence[str], k: int, rng: random.Random) -> list[list[str]]:
    items = list(items)
    rng.shuffle(items)
    folds = [[] for _ in range(k)]
    for idx, item in enumerate(items):
        folds[idx % k].append(item)
    return folds


def _pick_rows_for_lab(df: pd.DataFrame, lab_col: str, lab_id: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    subset = df[df[lab_col] == lab_id]
    type_series = subset["type"].astype(str).str.lower()
    clean = type_series.str.replace(r"[^a-z]", "", regex=True)
    he_df = subset[clean == "he"].copy()
    ret_df = subset[clean == "reticulin"].copy()
    if he_df.empty or ret_df.empty:
        raise RuntimeError(f"Lab {lab_id} missing H&E or Reticulin rows.")
    he_stain = he_df["stain_id"].iloc[0]
    ret_stain = ret_df["stain_id"].iloc[0]
    he_rows = he_df[he_df["stain_id"] == he_stain].reset_index(drop=True)
    ret_rows = ret_df[ret_df["stain_id"] == ret_stain].reset_index(drop=True)
    if he_rows.empty or ret_rows.empty:
        raise RuntimeError(f"Lab {lab_id} missing stain match.")
    return he_rows, ret_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="K-fold FID evaluation for CycleGAN generators.")
    parser.add_argument("--checkpoint", type=Path, default=None, help="Path to a single CycleGAN H2R checkpoint.")
    parser.add_argument("--checkpoint-root", type=Path, default=None, help="Root directory to search for G_H2R checkpoints.")
    parser.add_argument("--metadata", type=Path, default=base.DEFAULT_METADATA)
    parser.add_argument("--output-dir", type=Path, default=base.DEFAULT_OUTPUT_DIR)
    parser.add_argument("--patch-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=base._default_device())
    parser.add_argument("--lab-col", type=str, default=None, help="Column used to group slides/patients.")
    parser.add_argument("--kfolds", type=int, default=5)
    parser.add_argument("--downsample", type=float, default=4.0, help="Downsample factor for saved canvases.")
    parser.add_argument("--max-labs", type=int, default=None, help="Optional limit of labs to evaluate.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    device = torch.device(args.device)
    checkpoints = base._discover_checkpoints(args.checkpoint, args.checkpoint_root)

    metadata_path = resolve_path(args.metadata)
    df = pd.read_csv(metadata_path)
    required_cols = {"patch_path", "type", "x", "y"}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(f"metadata missing required columns: {sorted(missing_cols)}")

    lab_col = base._detect_lab_column(df, args.lab_col)
    labs = _labs_with_both(df, lab_col)
    if args.max_labs:
        labs = labs[: args.max_labs]
    if not labs:
        raise RuntimeError("No labs with both H&E and Reticulin patches.")
    folds = _split_kfold(labs, max(1, args.kfolds), rng)

    run_dir = resolve_path(args.output_dir, allow_missing=True) / "kfold"
    run_dir.mkdir(parents=True, exist_ok=True)

    logging.info("Evaluating %d checkpoints across %d folds (%d labs).", len(checkpoints), len(folds), len(labs))

    for ckpt_path in tqdm(checkpoints, desc="Checkpoints"):
        generator = base.load_generator(ckpt_path, device)
        ckpt_dir = run_dir / ckpt_path.stem
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        fold_scores: list[float] = []

        for fold_idx, fold_labs in enumerate(folds, start=1):
            fold_dir = ckpt_dir / f"fold_{fold_idx}"
            fold_dir.mkdir(parents=True, exist_ok=True)
            lab_scores: list[float] = []

            for lab_id in tqdm(fold_labs, desc=f"Fold {fold_idx}", leave=False):
                he_rows, ret_rows = _pick_rows_for_lab(df, lab_col, lab_id)
                he_entries = base._rows_to_entries(he_rows)
                ret_entries = base._rows_to_entries(ret_rows)

                he_canvas_path = fold_dir / f"{lab_id}_he_original.png"
                if not he_canvas_path.exists():
                    he_canvas = base._assemble_slide(he_entries, args.patch_size)
                    base._save_canvas(he_canvas, he_canvas_path, downsample=args.downsample)

                ret_canvas_path = fold_dir / f"{lab_id}_real_reticulin.png"
                if not ret_canvas_path.exists():
                    ret_canvas = base._assemble_slide(ret_entries, args.patch_size)
                    base._save_canvas(ret_canvas, ret_canvas_path, downsample=args.downsample)

                generated_entries = base._generate_reticulin_patches(
                    generator,
                    he_rows,
                    args.patch_size,
                    device,
                    fold_dir / f"{lab_id}_generated_patches",
                )
                gen_canvas = base._assemble_slide(generated_entries, args.patch_size)
                gen_canvas_path = fold_dir / f"{lab_id}_generated_reticulin.png"
                base._save_canvas(gen_canvas, gen_canvas_path, downsample=args.downsample)

                fid_value = base._compute_fid(
                    real_paths=[e.path for e in ret_entries],
                    fake_paths=[e.path for e in generated_entries],
                    device=device,
                )
                lab_scores.append(fid_value)
                logging.info("[%s | fold %d | lab %s] FID=%.4f", ckpt_path.stem, fold_idx, lab_id, fid_value)

            if lab_scores:
                fold_mean = float(sum(lab_scores) / len(lab_scores))
                fold_scores.append(fold_mean)
                logging.info("[%s | fold %d] Mean FID=%.4f over %d labs", ckpt_path.stem, fold_idx, fold_mean, len(lab_scores))

        if fold_scores:
            overall = float(sum(fold_scores) / len(fold_scores))
            logging.info("[%s] Overall mean FID=%.4f across %d folds", ckpt_path.stem, overall, len(fold_scores))


if __name__ == "__main__":
    main()

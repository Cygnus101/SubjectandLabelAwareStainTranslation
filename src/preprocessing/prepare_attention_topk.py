#!/usr/bin/env python3
"""Mark top-10% attention patches per slide and save to CSV."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("attention_weights.csv"),
        help="Path to attention_weights.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("attention_with_top10.csv"),
        help="Path to write attention_with_top10.csv",
    )
    parser.add_argument(
        "--top-fraction",
        type=float,
        default=0.10,
        help="Fraction of patches per slide to mark as top (default 0.10 => top 10%).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.input)
    required = {"slide_id", "patch_path", "attention_weight"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Input CSV missing required columns: {sorted(missing)}")

    frac = max(0.0, min(1.0, args.top_fraction))
    result_frames = []
    for slide_id, group in df.groupby("slide_id", sort=False):
        group = group.copy()
        n = len(group)
        if n == 0:
            continue
        k = max(1, int(round(frac * n))) if frac > 0 else 0
        group = group.sort_values("attention_weight", ascending=False)
        group["is_top10"] = 0
        if k > 0:
            group.loc[group.index[:k], "is_top10"] = 1
        result_frames.append(group)

    if not result_frames:
        raise RuntimeError("No slides found in the input CSV.")

    out_df = pd.concat(result_frames, axis=0)
    out_df.to_csv(args.output, index=False)
    print(f"Wrote top-k annotations to {args.output}")


if __name__ == "__main__":
    main()

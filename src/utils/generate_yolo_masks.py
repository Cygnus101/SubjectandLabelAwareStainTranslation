"""
Utility to build YOLO-friendly masks/labels from patch metadata.

The metadata CSV is expected to have the following columns:
    Lab No., stain_id, Reticulin Grade, type, source_ome, patch_path, x, y

Each row represents one extracted patch. The patch size is parsed from
`patch_path`, which is assumed to contain `_w{width}_h{height}`.

Outputs
-------
- YOLO segmentation label files (.txt), one per `source_ome`, where each line
  is a rectangle polygon in normalized coordinates:
      <class_id> x1 y1 x2 y2 x3 y3 x4 y4
  The default class mapping is by stain type (H&E -> 0, Reticulin -> 1).
- Optional binary preview masks (.png) showing the covered regions. These can
  be downscaled to avoid enormous canvas sizes.

Example
-------
python -m src.utils.generate_yolo_masks \\
    --csv metadata.csv \\
    --output-root outputs/yolo_masks \\
    --write-png --scale 0.1 \\
    --limit 2
"""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path
from typing import Iterable, Tuple

import pandas as pd
from PIL import Image, ImageDraw
from tqdm import tqdm


PATCH_SIZE_RE = re.compile(r"_w(\d+)_h(\d+)")
TYPE_TO_CLASS = {"h&e": 0, "he": 0, "h&e.": 0, "reticulin": 1, "retic": 1}
TYPE_TO_FOLDER = {"h&e": "he", "he": "he", "h&e.": "he", "reticulin": "reticulin", "retic": "reticulin"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("metadata.csv"),
        help="Path to the metadata CSV.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/yolo_masks"),
        help="Directory where labels (and optional PNG masks) are written.",
    )
    parser.add_argument(
        "--class-mode",
        choices=["type", "single"],
        default="type",
        help="How to assign class ids. `type` uses H&E=0, Reticulin=1; "
        "`single` uses class id 0 for everything.",
    )
    parser.add_argument(
        "--write-png",
        action="store_true",
        help="Also write binary preview masks (useful for debugging).",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=1.0,
        help="Scale factor applied to the PNG mask canvas and coordinates.",
    )
    parser.add_argument(
        "--max-dim",
        type=int,
        default=None,
        help="If set, caps the largest side of the PNG mask canvas by "
        "adjusting the effective scale. Ignored when --write-png is not set.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only this many source_ome entries (for quick tests).",
    )
    return parser.parse_args()


def _clean_stem(source_ome: str) -> str:
    """
    Turn a source_ome path into a filesystem-friendly stem.

    Examples:
        H&E/A-2653/A_2653_1.ome.tiff -> A_2653_1
        Reticulin/A-443/A_443_2.ome.tiff -> A_443_2
    """
    stem = Path(source_ome).stem  # drops .tiff
    return Path(stem).stem  # drops trailing .ome if present


def _parse_patch_sizes(df: pd.DataFrame) -> pd.DataFrame:
    size_df = df["patch_path"].str.extract(PATCH_SIZE_RE)
    if size_df.isna().any().any():
        raise ValueError("Failed to parse patch sizes from patch_path.")
    size_df = size_df.astype(int)
    df = df.copy()
    df["w"] = size_df[0]
    df["h"] = size_df[1]
    df["x2"] = df["x"] + df["w"]
    df["y2"] = df["y"] + df["h"]
    return df


def _class_from_type(stain_type: str) -> int:
    key = (stain_type or "").strip().lower()
    if key in TYPE_TO_CLASS:
        return TYPE_TO_CLASS[key]
    raise ValueError(f"Unknown stain type '{stain_type}'")


def _type_folder(stain_type: str) -> str:
    key = (stain_type or "").strip().lower()
    if key in TYPE_TO_FOLDER:
        return TYPE_TO_FOLDER[key]
    return key.replace("&", "and").replace(" ", "_") or "unknown"


def _make_label_lines(
    group: pd.DataFrame, class_mode: str, slide_w: int, slide_h: int
) -> Iterable[str]:
    for _, row in group.iterrows():
        class_id = 0 if class_mode == "single" else _class_from_type(row["type"])
        x1, y1 = float(row["x"]), float(row["y"])
        x2, y2 = x1 + float(row["w"]), y1 + float(row["h"])
        polygon = [
            x1 / slide_w,
            y1 / slide_h,
            x2 / slide_w,
            y1 / slide_h,
            x2 / slide_w,
            y2 / slide_h,
            x1 / slide_w,
            y2 / slide_h,
        ]
        yield " ".join([str(class_id)] + [f"{p:.6f}" for p in polygon])


def _resolve_scale(slide_w: int, slide_h: int, scale: float, max_dim: int | None):
    effective_scale = scale
    if max_dim:
        largest_side = max(slide_w, slide_h)
        if largest_side * scale > max_dim:
            effective_scale = max_dim / float(largest_side)
    return effective_scale


def _draw_mask(
    group: pd.DataFrame, slide_w: int, slide_h: int, scale: float, dest: Path
) -> None:
    scaled_w = max(1, int(math.ceil(slide_w * scale)))
    scaled_h = max(1, int(math.ceil(slide_h * scale)))
    mask = Image.new("1", (scaled_w, scaled_h), 0)
    draw = ImageDraw.Draw(mask)
    for _, row in group.iterrows():
        x1 = row["x"] * scale
        y1 = row["y"] * scale
        x2 = (row["x"] + row["w"]) * scale
        y2 = (row["y"] + row["h"]) * scale
        draw.rectangle([x1, y1, x2, y2], fill=1)
    dest.parent.mkdir(parents=True, exist_ok=True)
    mask.save(dest)


def build_masks(
    csv_path: Path,
    output_root: Path,
    class_mode: str = "type",
    write_png: bool = False,
    scale: float = 1.0,
    max_dim: int | None = None,
    limit: int | None = None,
) -> Tuple[int, Path]:
    df = pd.read_csv(csv_path)
    df = _parse_patch_sizes(df)

    labels_root = output_root / "labels"
    masks_root = output_root / "masks"

    grouped = df.groupby("source_ome")
    processed = 0
    for source_ome, group in tqdm(
        grouped, total=len(grouped), desc="Building masks", unit="slide"
    ):
        if limit is not None and processed >= limit:
            break

        slide_w = int(group["x2"].max())
        slide_h = int(group["y2"].max())
        if slide_w <= 0 or slide_h <= 0:
            raise ValueError(f"Invalid slide dimensions for {source_ome}")

        slide_stem = _clean_stem(source_ome)
        stain_type = (group["type"].iloc[0] or "").lower()
        type_folder = _type_folder(stain_type)

        label_path = labels_root / type_folder / f"{slide_stem}.txt"
        label_path.parent.mkdir(parents=True, exist_ok=True)
        label_lines = list(_make_label_lines(group, class_mode, slide_w, slide_h))
        label_path.write_text("\n".join(label_lines))

        if write_png:
            effective_scale = _resolve_scale(slide_w, slide_h, scale, max_dim)
            mask_path = masks_root / type_folder / f"{slide_stem}.png"
            _draw_mask(group, slide_w, slide_h, effective_scale, mask_path)

        processed += 1

    return processed, output_root


def main() -> None:
    args = parse_args()
    count, out_dir = build_masks(
        csv_path=args.csv,
        output_root=args.output_root,
        class_mode=args.class_mode,
        write_png=args.write_png,
        scale=args.scale,
        max_dim=args.max_dim,
        limit=args.limit,
    )
    print(f"Wrote labels for {count} slide(s) to {out_dir}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Patch augmented_slides.json / augmented_splits.json with missing Reticulin stains.

- Finds Reticulin stain_ids present in metadata.csv that are missing from augmented_slides.json
- For each missing stain, pairs it with the closest available H&E stain (same lab) so we can keep
  Reticulin labels intact.
- Appends the new SlideRecord entry and inserts its index into the appropriate split based on the lab.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Sequence

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
SRC_ROOT = SCRIPT_DIR.parent
import sys

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from preprocessing.build_augmented_dataset import SlideRecord, _closest_stain_id, _parse_grade  # noqa: E402


def load_json(path: Path) -> dict:
    return json.loads(path.read_text("utf-8"))


def save_json(path: Path, data: dict | list) -> None:
    path.write_text(json.dumps(data, indent=2))


def slide_split_map(slides: list[dict], splits: dict[str, list[int]]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for split_name, indices in splits.items():
        for idx in indices:
            if 0 <= idx < len(slides):
                slide = slides[idx]
                slide_id = str(slide.get("ret_stain_id") or slide.get("he_stain_id"))
                if slide_id:
                    mapping[slide_id] = split_name
    return mapping


def build_patch_list(group: pd.DataFrame) -> list[dict[str, object]]:
    return [{"patch_path": str(p), "is_top10": int(row.get("is_top10", 0))} for _, row in group.iterrows() for p in [row["patch_path"]]]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, default=Path("metadata.csv"))
    parser.add_argument("--augmented-slides", type=Path, default=Path("augmented_slides.json"))
    parser.add_argument("--augmented-splits", type=Path, default=Path("augmented_splits.json"))
    parser.add_argument("--backup", action="store_true", help="Create .bak copies before modifying JSON files.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    slides_path = args.augmented_slides
    splits_path = args.augmented_splits

    slides = load_json(slides_path)
    splits = load_json(splits_path)

    slide_ids = {slide["ret_stain_id"] for slide in slides}
    meta = pd.read_csv(args.metadata)
    meta["type_norm"] = meta["type"].astype(str).str.lower()
    meta["stain_id"] = meta["stain_id"].astype(str)
    meta["lab_id"] = meta["Lab No."].astype(str).str.strip()
    ret_df = meta[meta["type_norm"].str.contains("reticulin")]
    he_df = meta[meta["type_norm"].str.contains("h&e")]

    missing_stains = sorted(set(ret_df["stain_id"]) - slide_ids)
    if not missing_stains:
        logging.info("No missing Reticulin stains detected; nothing to patch.")
        return

    if args.backup:
        slides_path.write_text(slides_path.read_text())
        splits_path.write_text(splits_path.read_text())

    split_map = slide_split_map(slides, splits)
    added = 0
    for stain in missing_stains:
        ret_group = ret_df[ret_df["stain_id"] == stain]
        if ret_group.empty:
            continue
        lab_id = ret_group["lab_id"].iloc[0]
        he_candidates = he_df[he_df["lab_id"] == lab_id]
        if he_candidates.empty:
            logging.warning("Skipping %s: no H&E candidates in lab %s", stain, lab_id)
            continue
        he_choice = _closest_stain_id(stain, he_candidates["stain_id"].tolist())
        if he_choice is None:
            logging.warning("Skipping %s: unable to match H&E stain.", stain)
            continue
        he_group = he_candidates[he_candidates["stain_id"] == he_choice]
        grade = _parse_grade(ret_group["Reticulin Grade"].iloc[0])
        if grade is None:
            logging.warning("Skipping %s: missing grade.", stain)
            continue
        he_paths = he_group["patch_path"].astype(str).tolist()
        ret_paths = ret_group["patch_path"].astype(str).tolist()
        if not he_paths or not ret_paths:
            logging.warning("Skipping %s: missing patch paths.", stain)
            continue
        record = SlideRecord(
            lab_id=lab_id,
            he_stain_id=he_choice,
            ret_stain_id=stain,
            grade=grade,
            he_patches=[{"patch_path": path, "is_top10": 0} for path in he_paths],
            ret_patches=[{"patch_path": path, "is_top10": 0} for path in ret_paths],
        )
        slides.append(record.__dict__)
        target_split = split_map.get(he_choice, "train")
        splits.setdefault(f"{target_split}", []).append(len(slides) - 1)
        added += 1
        logging.info("Added %s paired with %s to %s split.", stain, he_choice, target_split)

    save_json(slides_path, slides)
    save_json(splits_path, splits)
    logging.info("Patched augmented slides/splits with %d new entries.", added)


if __name__ == "__main__":
    main()

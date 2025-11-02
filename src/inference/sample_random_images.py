#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Sample a fixed number of images from a source tree into a target folder."""

from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path

SUPPORTED_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def collect_images(root: Path) -> list[Path]:
    return [p for p in root.rglob("*") if p.suffix.lower() in SUPPORTED_SUFFIXES]


def copy_subset(files: list[Path], dest: Path, count: int, seed: int) -> list[Path]:
    rng = random.Random(seed)
    chosen = files if len(files) <= count else rng.sample(files, count)
    dest.mkdir(parents=True, exist_ok=True)
    copied: list[Path] = []
    for src in chosen:
        target_path = dest / src.name
        suffix = 1
        while target_path.exists():
            target_path = dest / f"{src.stem}_{suffix}{src.suffix}"
            suffix += 1
        shutil.copy2(src, target_path)
        copied.append(target_path)
    return copied


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Copy a random subset of images to a destination folder.")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--dest", type=Path, required=True)
    parser.add_argument("--count", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    files = collect_images(args.source)
    if not files:
        raise SystemExit(f"No supported images found under {args.source}")
    copied = copy_subset(files, args.dest, args.count, args.seed)
    print(f"Copied {len(copied)} images to {args.dest}")


if __name__ == "__main__":
    main()

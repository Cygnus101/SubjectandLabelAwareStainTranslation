#!/usr/bin/env python3
"""Concatenate metadata CSVs from a chosen directory tree."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Optional

SCRIPT_DIR = Path(__file__).resolve().parent
SRC_ROOT = SCRIPT_DIR.parent
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from utils.path import get_project_root, resolve_path as resolve_project_path

PROJECT_ROOT = get_project_root()


def concat_metadata(root: str | Path, out_path: str | Path = "metadata_master.csv") -> Path:
    scan_root = resolve_project_path(root)
    output_path = resolve_project_path(out_path, allow_missing=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    csv_files = sorted(
        p for p in scan_root.rglob("metadata*.csv")
        if p.is_file() and not p.name.startswith("._")
    )

    if not csv_files:
        raise SystemExit(f"No metadata*.csv files found under {scan_root}")

    common_cols: set[str] | None = None
    ordered_cols: list[str] = []
    all_rows: list[list[dict[str, str]]] = []

    for path in csv_files:
        with path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            cols = reader.fieldnames or []
            if common_cols is None:
                common_cols = set(cols)
                ordered_cols = cols[:]
            else:
                common_cols &= set(cols)

            rows = [row for row in reader]
            all_rows.append(rows)

    if not common_cols:
        raise SystemExit("No shared columns across metadata files.")

    ordered_cols = [col for col in ordered_cols if col in common_cols]

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=ordered_cols)
        writer.writeheader()
        for rows in all_rows:
            for row in rows:
                writer.writerow({col: row[col] for col in ordered_cols})

    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Concatenate metadata CSV files into a master CSV.")
    parser.add_argument(
        "folder",
        nargs="?",
        default=str(PROJECT_ROOT),
        help="Directory tree to scan for metadata*.csv files (default: project root).",
    )
    parser.add_argument(
        "--out",
        default="metadata_master.csv",
        help="Output CSV filename or path (default: metadata_master.csv in the project root).",
    )
    return parser.parse_args()


def main(args: Optional[argparse.Namespace] = None):
    args = args or parse_args()
    result = concat_metadata(root=args.folder, out_path=args.out)
    print(f"Wrote {result} using metadata files under {resolve_project_path(args.folder)}")


if __name__ == "__main__":
    main()

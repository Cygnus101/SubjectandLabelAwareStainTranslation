#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Optional

import sys

SCRIPT_DIR = Path(__file__).resolve().parent
SRC_ROOT = SCRIPT_DIR.parent
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from utils.path import get_project_root, resolve_path as resolve_project_path

PROJECT_ROOT = get_project_root()
DEFAULT_METADATA_NAME = "metadata.csv"
DEFAULT_LOG_NAME = "missing_patch_files.csv"
DEFAULT_SUMMARY_LIMIT = 20  # how many missing entries to print inline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check metadata.csv for missing patch files.")
    parser.add_argument(
        "--root",
        type=str,
        default=None,
        help="Base directory that relative paths in metadata.csv refer to (default: project root).",
    )
    parser.add_argument(
        "--metadata",
        type=str,
        default=DEFAULT_METADATA_NAME,
        help="Metadata CSV to inspect (default: metadata.csv at the project root).",
    )
    parser.add_argument(
        "--log",
        type=str,
        default=DEFAULT_LOG_NAME,
        help="Where to write the detailed missing-file report (default: missing_patch_files.csv in the root).",
    )
    parser.add_argument(
        "--summary-limit",
        type=int,
        default=DEFAULT_SUMMARY_LIMIT,
        help="How many missing entries to print inline (default: 20).",
    )
    return parser.parse_args()


def main(args: Optional[argparse.Namespace] = None) -> None:
    args = args or parse_args()

    root_candidate = args.root or PROJECT_ROOT
    root = resolve_project_path(root_candidate, allow_missing=True)

    metadata_path = Path(args.metadata).expanduser()
    if not metadata_path.is_absolute():
        metadata_path = (root / metadata_path).resolve()

    if not metadata_path.is_file():
        raise SystemExit(f"metadata CSV not found: {metadata_path}")

    log_path = Path(args.log).expanduser()
    if not log_path.is_absolute():
        log_path = (root / log_path).resolve()

    summary_limit = max(0, int(args.summary_limit))

    missing: list[tuple[str, dict[str, str]]] = []
    total = 0
    fieldnames: list[str] | None = None

    with metadata_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        if not fieldnames:
            raise SystemExit(f"{metadata_path.name} has no header row")

        for row in reader:
            total += 1
            rel_path = (row.get("patch_path") or "").strip()
            if not rel_path:
                missing.append(("missing patch_path value", row))
                continue

            patch_file = Path(rel_path).expanduser()
            if not patch_file.is_absolute():
                patch_file = (root / patch_file).resolve()
            if not patch_file.is_file():
                missing.append((str(patch_file), row))

    print(f"Checked {total} rows.")
    print(f"Missing files: {len(missing)}")

    if missing:
        for entry, row in missing[:summary_limit]:
            print(f"✖ {entry} (Lab: {row.get('Lab No.')}, stain: {row.get('stain_id')})")
        if len(missing) > summary_limit:
            print(f"... {len(missing) - summary_limit} more omitted")

        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("w", newline="", encoding="utf-8") as out:
            writer = csv.DictWriter(out, fieldnames=["missing_path", *fieldnames])
            writer.writeheader()
            for entry, row in missing:
                writer.writerow({"missing_path": entry, **row})

        print(f"\nDetailed list written to {log_path}")


if __name__ == "__main__":
    main()

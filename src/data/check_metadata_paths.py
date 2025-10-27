#!/usr/bin/env python3

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd
from tqdm.auto import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT_CANDIDATE = SCRIPT_DIR.parent.parent
if str(PROJECT_ROOT_CANDIDATE) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_CANDIDATE))

try:
    from src.utils.path import ensure_project_root_on_syspath, resolve_path, get_project_root
except ModuleNotFoundError as exc:  # pragma: no cover - defensive fallback
    raise ModuleNotFoundError(
        "Could not import 'src.utils.path'. Run this script from the project root "
        "or install the package so 'src' is on PYTHONPATH."
    ) from exc

ensure_project_root_on_syspath()
PROJECT_ROOT = get_project_root()


def _iter_resolution_bases(base: Path | None) -> Iterable[Path]:
    if base is not None:
        yield Path(base).expanduser().resolve()
    yield Path.cwd().resolve()
    yield PROJECT_ROOT


def _abs_path(rel_path: str, roots: Sequence[Path] | None = None) -> Path:
    """
    Mirror the resolution logic used by the CycleGAN dataloader.

    This follows src/data/build_cyclegan_dataset._abs_path verbatim so the
    results match the training script.
    """
    path = Path(rel_path)
    if path.is_absolute() and path.exists():
        return path

    candidates: list[Path] = []
    if roots:
        candidates.extend(roots)
    candidates.extend(_iter_resolution_bases(None))

    for base in candidates:
        candidate = (Path(base) / path).resolve()
        if candidate.exists():
            return candidate

    fallback = resolve_path(rel_path, allow_missing=True)
    return fallback if isinstance(fallback, Path) else Path(fallback)


def _prepare_roots(legacy_roots: Sequence[str]) -> list[Path]:
    prepared: list[Path] = []
    for root in legacy_roots:
        resolved = resolve_path(root, allow_missing=True)
        resolved_path = resolved if isinstance(resolved, Path) else Path(resolved)
        prepared.append(resolved_path.resolve())
    return prepared


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check metadata paths using the same logic as train_cyclegan.py"
    )
    parser.add_argument(
        "--metadata",
        default="metadata.csv",
        help="Path to metadata CSV (defaults to metadata.csv)",
    )
    parser.add_argument(
        "--legacy-root",
        action="append",
        default=[],
        help="Additional root directories to search for patch_path entries.",
    )
    args = parser.parse_args()

    metadata_path = resolve_path(args.metadata)
    metadata_path = metadata_path if isinstance(metadata_path, Path) else Path(metadata_path)
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    df = pd.read_csv(metadata_path)
    if "patch_path" not in df.columns:
        raise KeyError("metadata is missing 'patch_path' column")

    roots = _prepare_roots(args.legacy_root)
    missing_records: list[dict[str, object]] = []
    metadata_paths: dict[Path, dict[str, object]] = {}
    kept_dir_names: set[str] = set()

    for idx, rel_path in tqdm(
        df["patch_path"].items(),
        total=len(df),
        desc="Checking metadata paths",
    ):
        abs_path = _abs_path(rel_path, roots)
        abs_path = abs_path.resolve()
        metadata_paths[abs_path] = {
            "index": idx,
            "patch_path": rel_path,
        }

        first_part = Path(rel_path).parts[0] if Path(rel_path).parts else None
        if first_part and first_part.lower().startswith("kept"):
            kept_dir_names.add(first_part)

        if not abs_path.exists():
            entry = {
                "index": idx,
                "patch_path": rel_path,
                "resolved_path": abs_path,
            }
            if "type" in df.columns:
                entry["type"] = df.at[idx, "type"]
            if "stain_id" in df.columns:
                entry["stain_id"] = df.at[idx, "stain_id"]
            missing_records.append(entry)

    print(f"Scanned {len(df)} rows from {metadata_path}")
    print(f"Missing files: {len(missing_records)}")

    if missing_records:
        print("\nMissing entries:")
        for record in missing_records:
            parts = [f"{k}={record[k]}" for k in ("index", "type", "stain_id") if k in record]
            parts.append(f"patch_path={record['patch_path']}")
            parts.append(f"resolved={record['resolved_path']}")
            print(" - " + ", ".join(parts))

    if kept_dir_names:
        print("\nChecking kept patch directories for untracked files...")
    else:
        print("\nNo kept patch directories referenced in metadata; skipping directory diff.")

    for dir_name in sorted(kept_dir_names):
        dir_path = _abs_path(dir_name, roots).resolve()
        if not dir_path.exists() or not dir_path.is_dir():
            print(f" - {dir_name}: directory not found at {dir_path}")
            continue

        directory_files: set[Path] = set()
        for file_path in tqdm(
            dir_path.rglob("*"),
            desc=f"Scanning {dir_name}",
        ):
            if file_path.is_file():
                directory_files.add(file_path.resolve())

        extras = sorted(path for path in directory_files if path not in metadata_paths)
        print(f" - {dir_name}: {len(extras)} files missing from metadata")
        if extras:
            for path in extras:
                try:
                    rel = path.relative_to(dir_path)
                except ValueError:
                    rel = path
                print(f"   - {rel}")


if __name__ == "__main__":
    main()

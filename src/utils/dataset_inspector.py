#!/usr/bin/env python3
"""
Utility script that mirrors the DatasetChecker notebook to inspect
PatchDataset and SlideBagDataset outputs and persist the findings.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

SCRIPT_DIR = Path(__file__).resolve().parent
SRC_ROOT = SCRIPT_DIR.parent

import sys

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from training.extract_embeddings import PatchDataset, build_transform, collate_skip_missing  # noqa: E402
from training.train_classifier import SlideBagDataset, collate_fn, _extract_label  # noqa: E402
from utils.path import ensure_project_root_on_syspath, resolve_path  # noqa: E402

ensure_project_root_on_syspath()


def inspect_patch_dataset(args: argparse.Namespace) -> Dict[str, Any]:
    metadata_path = resolve_path(args.metadata)
    metadata_df = pd.read_csv(metadata_path)
    transform = build_transform(args.image_size)

    root_candidates: List[Path] = [metadata_path.parent]
    for legacy in args.legacy_root:
        root_candidates.append(resolve_path(legacy))

    dataset = PatchDataset(
        dataframe=metadata_df.head(args.patch_limit),
        root_candidates=root_candidates,
        transform=transform,
        retries=args.patch_retries,
    )

    results: Dict[str, Any] = {
        "length": len(dataset),
        "sample_indices": [],
        "samples": [],
        "missing_indices": [],
    }

    for idx in range(min(args.sample_count, len(dataset))):
        sample = dataset[idx]
        results["sample_indices"].append(idx)
        if sample is None:
            results["samples"].append(None)
            results["missing_indices"].append(idx)
            continue
        item_idx, tensor = sample
        results["samples"].append(
            {
                "returned_index": int(item_idx),
                "tensor_shape": list(tensor.shape),
                "tensor_dtype": str(tensor.dtype),
                "tensor_mean": float(tensor.mean().item()),
                "tensor_std": float(tensor.std().item()),
            }
        )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_skip_missing,
    )

    first_batch: Optional[Dict[str, Any]] = None
    for batch_num, batch in enumerate(loader, start=1):
        if batch is None:
            continue
        indices, images = batch
        first_batch = {
            "batch_number": batch_num,
            "indices": [int(i) for i in indices],
            "tensor_shape": list(images.shape),
            "tensor_dtype": str(images.dtype),
            "tensor_mean": float(images.mean().item()),
            "tensor_std": float(images.std().item()),
        }
        break

    results["first_batch"] = first_batch

    missing_samples = [
        idx for idx in range(len(dataset)) if dataset[idx] is None
    ]
    results["missing_indices_extended"] = missing_samples

    return results


def inspect_slide_dataset(
    args: argparse.Namespace,
) -> tuple[Dict[str, Any], tuple[tuple[str, int, List[str], List[str]], ...]]:
    metadata_path = resolve_path(args.metadata)
    embeddings_path = resolve_path(args.embeddings_csv)

    metadata_df = pd.read_csv(metadata_path)
    metadata_df = metadata_df[metadata_df["type"].astype(str).str.lower().str.contains("reticulin")].copy()
    metadata_df["patch_path_norm"] = metadata_df["patch_path"].astype(str).str.replace("\\", "/", regex=False)

    embeddings_df = pd.read_csv(embeddings_path)
    embeddings_df = embeddings_df.copy()
    embeddings_df["patch_path_norm"] = embeddings_df["patch_path"].astype(str).str.replace("\\", "/", regex=False)

    joined = embeddings_df.merge(
        metadata_df[["patch_path_norm", "stain_id", "Reticulin Grade"]],
        on="patch_path_norm",
        how="inner",
    ).rename(columns={"stain_id": "slide_id", "Reticulin Grade": "label"})

    dataset = SlideBagDataset(joined)

    summary: Dict[str, Any] = {
        "length": len(dataset),
        "sample": None,
        "loader_sample": None,
    }

    if len(dataset) == 0:
        return summary, tuple()

    slide_id, label, bag_tensor, patch_paths, embed_paths = dataset[0]
    summary["sample"] = {
        "slide_id": slide_id,
        "label": int(label),
        "bag_shape": list(bag_tensor.shape),
        "bag_dtype": str(bag_tensor.dtype),
        "patch_path_count": len(patch_paths),
        "embed_path_count": len(embed_paths),
        "first_patch_paths": patch_paths[:3],
        "first_embedding_paths": embed_paths[:3],
    }

    loader = DataLoader(
        dataset,
        batch_size=args.slide_batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )

    for slide_ids, labels, bags, patch_lists, embed_lists in loader:
        summary["loader_sample"] = {
            "slide_ids": list(slide_ids),
            "labels": [int(lb) for lb in labels],
            "num_bags": len(bags),
            "first_bag_shape": list(bags[0].shape),
            "first_patch_count": len(patch_lists[0]),
            "first_embedding_count": len(embed_lists[0]),
        }
        break

    return summary, tuple(dataset.entries)


def collect_missing_label_stain_ids(metadata_path: Path) -> List[str]:
    metadata_df = pd.read_csv(metadata_path)
    if "stain_id" not in metadata_df.columns or "Reticulin Grade" not in metadata_df.columns or "type" not in metadata_df.columns:
        return []
    metadata_df = metadata_df[metadata_df["type"].astype(str).str.lower().str.contains("reticulin")].copy()
    missing = metadata_df[
        metadata_df["Reticulin Grade"].apply(lambda val: _extract_label(val) is None)
    ]
    stain_ids = sorted({str(sid) for sid in missing["stain_id"].dropna().astype(str)})
    return stain_ids


def build_slide_label_records(
    slide_entries: Sequence[Tuple[str, int, List[str], List[str]]],
) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for slide_id, label, patch_paths, embed_paths in slide_entries:
        records.append(
            {
                "slide_id": slide_id,
                "label": int(label),
                "num_patches": len(patch_paths),
                "num_embeddings": len(embed_paths),
            }
        )
    return records


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Inspect datasets used in embedding extraction and classifier training.")
    parser.add_argument("--metadata", type=str, default="metadata.csv", help="Path to metadata CSV.")
    parser.add_argument("--embeddings-csv", type=str, default="outputs/embeddings_test/embeddings_index.csv")
    parser.add_argument("--output", type=str, default="outputs/dataset_inspection.json", help="Where to write JSON results.")
    parser.add_argument("--label-csv", type=str, default="outputs/dataset_inspector_slide_labels.csv", help="CSV path storing slide_id to label mapping.")
    parser.add_argument("--image-size", type=int, default=512, help="Image transform size for PatchDataset.")
    parser.add_argument("--sample-count", type=int, default=3, help="Number of PatchDataset samples to inspect.")
    parser.add_argument("--patch-limit", type=int, default=10, help="Limit rows when constructing PatchDataset.")
    parser.add_argument("--patch-retries", type=int, default=1, help="Retries for missing/unreadable patches.")
    parser.add_argument("--batch-size", type=int, default=3, help="PatchDataset DataLoader batch size.")
    parser.add_argument("--slide-batch-size", type=int, default=1, help="SlideBag DataLoader batch size.")
    parser.add_argument("--legacy-root", action="append", default=[], help="Additional roots to locate patches.")
    parser.add_argument("--log-level", type=str, default="INFO")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    logging.info("Inspecting patch dataset...")
    patch_info = inspect_patch_dataset(args)

    logging.info("Inspecting slide dataset...")
    slide_info, slide_entries = inspect_slide_dataset(args)

    missing_label_stain_ids = collect_missing_label_stain_ids(resolve_path(args.metadata))
    label_records = build_slide_label_records(slide_entries)

    label_csv_path = resolve_path(args.label_csv, allow_missing=True)
    label_csv_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(label_records).to_csv(label_csv_path, index=False)

    output_path = resolve_path(args.output, allow_missing=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "patch_dataset": patch_info,
        "slide_dataset": slide_info,
        "missing_label_stain_ids": missing_label_stain_ids,
        "label_csv_path": str(label_csv_path),
    }

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    logging.info("Inspection written to %s", output_path)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Assemble a random H&E slide from metadata patches, translate it to Reticulin using
a trained CycleGAN generator, reconstruct the translated slide, and compute FID
against the real Reticulin slide from the same patient/lab.
"""

from __future__ import annotations

import argparse
import logging
import shutil
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd
from PIL import Image, UnidentifiedImageError

import torch
from torchvision import transforms as T
from torchmetrics.image.fid import FrechetInceptionDistance
from tqdm.auto import tqdm

from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
SRC_ROOT = SCRIPT_DIR.parent
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from utils.path import ensure_project_root_on_syspath, get_project_root, resolve_path

ensure_project_root_on_syspath()

from src.models.Backbone_model.CycleGANv3 import UNetGenerator

PROJECT_ROOT = get_project_root()
DEFAULT_METADATA = PROJECT_ROOT / "metadata.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "eval_fid"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

TensorTransform = T.Compose([
    T.ToTensor(),
    T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])


def _default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _detect_lab_column(df: pd.DataFrame, override: str | None) -> str:
    if override and override in df.columns:
        return override
    candidates = ["Lab No.", "lab_no", "Lab_No", "lab", "patient_id", "stain_id"]
    for name in candidates:
        if name in df.columns:
            return name
    raise ValueError("Could not infer a lab/patient column. Use --lab-col to specify one explicitly.")


@dataclass
class PatchEntry:
    x: int
    y: int
    path: Path


def _resolve_patch(path_value: str | Path) -> Path:
    resolved = resolve_path(path_value)
    if not resolved.exists():
        raise FileNotFoundError(f"Patch not found: {resolved}")
    return resolved


def _assemble_slide(entries: Sequence[PatchEntry], patch_size: int) -> Image.Image:
    if not entries:
        raise ValueError("Cannot assemble slide with zero patches.")
    max_x = max(entry.x for entry in entries)
    max_y = max(entry.y for entry in entries)
    width = int(max_x) + patch_size
    height = int(max_y) + patch_size
    canvas = Image.new("RGB", (width, height), color=(0, 0, 0))
    for entry in entries:
        try:
            with Image.open(entry.path) as patch_img:
                canvas.paste(patch_img.convert("RGB"), (int(entry.x), int(entry.y)))
        except (UnidentifiedImageError, OSError) as exc:  # pragma: no cover - depends on data
            logging.warning("Skipping unreadable patch %s (%s)", entry.path, type(exc).__name__)
    return canvas


def _tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    tensor = tensor.detach().cpu().clamp(-1, 1)
    tensor = (tensor * 0.5 + 0.5).clamp(0, 1)
    return T.ToPILImage()(tensor)


def _generate_reticulin_patches(
    generator: torch.nn.Module,
    he_rows: pd.DataFrame,
    patch_size: int,
    device: torch.device,
    patch_dir: Path,
) -> list[PatchEntry]:

    generator.eval()
    if patch_dir.exists():
        shutil.rmtree(patch_dir)
    patch_dir.mkdir(parents=True, exist_ok=True)
    entries: list[PatchEntry] = []

    for idx, row in he_rows.iterrows():
        try:
            src_path = _resolve_patch(row["patch_path"])
            with Image.open(src_path) as img:
                img = img.convert("RGB")
        except (FileNotFoundError, UnidentifiedImageError, OSError) as exc:
            logging.warning("Skipping H&E patch %s (%s)", row["patch_path"], type(exc).__name__)
            continue

        tensor = TensorTransform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            fake = generator(tensor)
        fake_img = _tensor_to_pil(fake.squeeze(0))
        out_path = patch_dir / f"patch_{idx:05d}.png"
        fake_img.save(out_path)
        entries.append(PatchEntry(int(row["x"]), int(row["y"]), out_path))

    if not entries:
        raise RuntimeError("No H&E patches were successfully translated.")
    return entries


def _rows_to_entries(rows: pd.DataFrame) -> list[PatchEntry]:
    entries: list[PatchEntry] = []
    for _, row in rows.iterrows():
        try:
            patch_path = _resolve_patch(row["patch_path"])
        except FileNotFoundError as exc:
            logging.warning("%s", exc)
            continue
        entries.append(PatchEntry(int(row["x"]), int(row["y"]), patch_path))
    if not entries:
        raise RuntimeError("No valid patch paths found for the selected lab.")
    return entries


def _compute_fid(real_paths: Sequence[Path], fake_paths: Sequence[Path], device: torch.device) -> float:
    fid_device = torch.device("cuda" if device.type == "cuda" else "cpu")
    fid = FrechetInceptionDistance(normalize=True).to(fid_device)
    if device.type == "mps":
        logging.info("FID computed on CPU due to MPS limitations.")
    tfm = T.ToTensor()

    def update(paths: Sequence[Path], real: bool) -> None:
        for path in paths:
            with Image.open(path) as img:
                tensor = tfm(img.convert("RGB")).unsqueeze(0).to(fid_device)
            fid.update(tensor, real=real)

    update(real_paths, True)
    update(fake_paths, False)
    return float(fid.compute().cpu())


def _select_lab(
    df: pd.DataFrame,
    lab_col: str,
    rng: random.Random,
) -> tuple[str, pd.DataFrame, pd.DataFrame]:
    type_series = df["type"].astype(str).str.lower()
    type_clean = type_series.str.replace(r"[^a-z]", "", regex=True)
    he_mask = type_clean == "he"
    ret_mask = type_clean == "reticulin"
    labs = sorted(set(df.loc[he_mask, lab_col]) & set(df.loc[ret_mask, lab_col]))
    if not labs:
        raise RuntimeError("No lab/patient has both H&E and Reticulin entries.")
    chosen = rng.choice(labs)
    he_subset = df[(df[lab_col] == chosen) & he_mask].copy()
    ret_subset = df[(df[lab_col] == chosen) & ret_mask].copy()
    he_stain = he_subset["stain_id"].iloc[0]
    ret_stain = ret_subset["stain_id"].iloc[0]
    he_rows = he_subset[he_subset["stain_id"] == he_stain].reset_index(drop=True)
    ret_rows = ret_subset[ret_subset["stain_id"] == ret_stain].reset_index(drop=True)
    if he_rows.empty or ret_rows.empty:
        raise RuntimeError(f"Selected lab {chosen} does not contain both stains.")
    return str(chosen), he_rows, ret_rows


def _save_canvas(canvas: Image.Image, path: Path, downsample: float = 1.0) -> None:
    if downsample > 1.0:
        new_size = (
            max(1, int(canvas.width / downsample)),
            max(1, int(canvas.height / downsample)),
        )
        canvas = canvas.resize(new_size, Image.Resampling.LANCZOS)
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)
    logging.info("Saved %s", path)


def load_generator(checkpoint_path: Path, device: torch.device) -> UNetGenerator:
    generator = UNetGenerator().to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    state = ckpt.get("state_dict") or ckpt
    generator.load_state_dict(state)
    logging.info("Loaded generator checkpoint %s", checkpoint_path)
    return generator


def _discover_checkpoints(single: Path | None, root: Path | None) -> list[Path]:
    if single is None and root is None:
        raise ValueError("Provide --checkpoint or --checkpoint-root.")
    if single is not None and root is not None:
        raise ValueError("Provide only one of --checkpoint or --checkpoint-root.")
    checkpoints: list[Path] = []
    if single is not None:
        checkpoints.append(resolve_path(single))
    else:
        base = resolve_path(root)
        patterns = ["G_H2R*.pth", "G_H2R*.pth.tar", "G_H2R*.pt", "G_H2R*.ckpt"]
        for pattern in patterns:
            checkpoints.extend(base.rglob(pattern))
        checkpoints = sorted({p.resolve() for p in checkpoints})
    if not checkpoints:
        raise FileNotFoundError("No G_H2R checkpoints found with the provided arguments.")
    return checkpoints


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate CycleGAN generator by assembling slides and computing FID.")
    parser.add_argument("--checkpoint", type=Path, default=None, help="Path to a single CycleGAN H2R checkpoint.")
    parser.add_argument("--checkpoint-root", type=Path, default=None, help="Root directory to search for G_H2R checkpoints.")
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--patch-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=_default_device())
    parser.add_argument("--lab-col", type=str, default=None, help="Column used to group slides/patients.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    device = torch.device(args.device)
    checkpoints = _discover_checkpoints(args.checkpoint, args.checkpoint_root)

    metadata_path = resolve_path(args.metadata)
    df = pd.read_csv(metadata_path)
    required_cols = {"patch_path", "type", "x", "y"}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(f"metadata missing required columns: {sorted(missing_cols)}")

    lab_col = _detect_lab_column(df, args.lab_col)
    run_dir = resolve_path(args.output_dir, allow_missing=True)
    run_dir.mkdir(parents=True, exist_ok=True)

    lab_id, he_rows, ret_rows = _select_lab(df, lab_col, rng)
    logging.info("Selected lab %s with %d H&E patches and %d Reticulin patches.", lab_id, len(he_rows), len(ret_rows))

    he_entries = _rows_to_entries(he_rows)
    ret_entries = _rows_to_entries(ret_rows)

    downsample_factor = 4.0

    he_canvas_path = run_dir / f"{lab_id}_he_original.png"
    if not he_canvas_path.exists():
        he_canvas = _assemble_slide(he_entries, args.patch_size)
        _save_canvas(he_canvas, he_canvas_path, downsample_factor)

    ret_canvas_path = run_dir / f"{lab_id}_real_reticulin.png"
    if not ret_canvas_path.exists():
        ret_canvas = _assemble_slide(ret_entries, args.patch_size)
        _save_canvas(ret_canvas, ret_canvas_path, downsample_factor)

    for ckpt_path in tqdm(checkpoints, desc="Evaluating checkpoints"):
        ckpt_out = run_dir / ckpt_path.stem
        ckpt_out.mkdir(parents=True, exist_ok=True)

        generator = load_generator(ckpt_path, device)

        generated_entries = _generate_reticulin_patches(
            generator,
            he_rows,
            args.patch_size,
            device,
            ckpt_out / "generated_patches"
        )

        generated_canvas = _assemble_slide(generated_entries, args.patch_size)
        gen_canvas_path = ckpt_out / f"{lab_id}_generated_reticulin.png"
        _save_canvas(generated_canvas, gen_canvas_path, downsample_factor)

        fid_value = _compute_fid(
            real_paths=[entry.path for entry in ret_entries],
            fake_paths=[entry.path for entry in generated_entries],
            device=device,
        )

        logging.info("[%s] FID (generated vs real Reticulin): %.4f", ckpt_path.name, fid_value)


if __name__ == "__main__":
    main()

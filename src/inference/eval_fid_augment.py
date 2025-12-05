#!/usr/bin/env python3
"""Evaluate train_augment generators by assembling slides and computing FID."""

from __future__ import annotations

import argparse
import logging
import random
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import pandas as pd
from PIL import Image, UnidentifiedImageError

import torch
from torchvision import transforms as T
from torchmetrics.image.fid import FrechetInceptionDistance
from tqdm.auto import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
SRC_ROOT = SCRIPT_DIR.parent
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from utils.path import ensure_project_root_on_syspath, get_project_root, resolve_path

ensure_project_root_on_syspath()
PROJECT_ROOT = get_project_root()

try:  # pragma: no cover
    from models.Backbone_model.CycleGAN import UNetGenerator as CycleGenerator  # type: ignore
except Exception:  # pragma: no cover
    from models.Backbone_model.CycleGANv2 import UNetGenerator as CycleGenerator  # type: ignore

DEFAULT_METADATA = PROJECT_ROOT / "metadata.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "eval_fid_augment"
DEFAULT_CHECKPOINT_ROOT = PROJECT_ROOT / "outputs" / "logs" / "augmented_cyclegan"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

TensorTransform = T.Compose(
    [
        T.ToTensor(),
        T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)


def _default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _detect_lab_column(df: pd.DataFrame, override: str | None) -> str:
    if override and override in df.columns:
        return override
    for cand in ["Lab No.", "lab_no", "Lab_No", "lab", "patient_id", "stain_id"]:
        if cand in df.columns:
            return cand
    raise ValueError("Unable to infer lab column. Please supply --lab-col explicitly.")


@dataclass
class PatchEntry:
    x: int
    y: int
    path: Path


LEGACY_ROOT = Path("/Volumes/Expansion/BiopsyExtractedSet")

def _resolve_patch(path_value: str | Path) -> Path:
    p = resolve_path(path_value)
    if p.exists():
        return p

    # Try legacy root
    legacy_path = LEGACY_ROOT / Path(path_value)
    if legacy_path.exists():
        return legacy_path

    raise FileNotFoundError(f"Patch not found in standard or legacy roots: {path_value}")


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
        raise RuntimeError("No valid patches found for selected slide.")
    return entries


def _assemble_slide(entries: Sequence[PatchEntry], patch_size: int) -> Image.Image:
    max_x = max(entry.x for entry in entries)
    max_y = max(entry.y for entry in entries)
    width = max_x + patch_size
    height = max_y + patch_size
    canvas = Image.new("RGB", (width, height), color=(0, 0, 0))
    for entry in entries:
        with Image.open(entry.path) as img:
            canvas.paste(img.convert("RGB"), (entry.x, entry.y))
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
            with Image.open(_resolve_patch(row["patch_path"])) as img:
                img = img.convert("RGB")
        except (FileNotFoundError, UnidentifiedImageError, OSError) as exc:
            logging.warning("Skipping H&E patch %s (%s)", row["patch_path"], exc)
            continue
        tensor = TensorTransform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            fake = generator(tensor)
        fake_img = _tensor_to_pil(fake.squeeze(0))
        out_path = patch_dir / f"patch_{idx:05d}.png"
        fake_img.save(out_path)
        entries.append(PatchEntry(int(row["x"]), int(row["y"]), out_path))

    if not entries:
        raise RuntimeError("No translated patches were produced.")
    return entries


def _compute_fid(real_paths: Sequence[Path], fake_paths: Sequence[Path], device: torch.device) -> float:
    fid_device = torch.device("cuda" if device.type == "cuda" else "cpu")
    fid = FrechetInceptionDistance(normalize=True).to(fid_device)
    if device.type == "mps":
        logging.info("FID computed on CPU since MPS lacks CUDA support.")
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
    lab_override: str | None = None,
    he_stain_override: str | None = None,
    ret_stain_override: str | None = None,
) -> tuple[str, pd.DataFrame, pd.DataFrame]:
    type_series = df["type"].astype(str).str.lower()
    type_clean = type_series.str.replace(r"[^a-z]", "", regex=True)
    he_mask = type_clean == "he"
    ret_mask = type_clean == "reticulin"

    labs_with_both = sorted(set(df.loc[he_mask, lab_col]) & set(df.loc[ret_mask, lab_col]))
    if not labs_with_both:
        raise RuntimeError("No lab has both H&E and Reticulin entries.")

    if lab_override is not None:
        if lab_override not in labs_with_both:
            raise RuntimeError(
                f"Requested lab {lab_override!r} does not have both H&E and Reticulin entries."
            )
        chosen = lab_override
    else:
        chosen = rng.choice(labs_with_both)

    he_subset = df[(df[lab_col] == chosen) & he_mask].copy()
    ret_subset = df[(df[lab_col] == chosen) & ret_mask].copy()

    # Choose stain IDs (either from overrides or inferred from the data)
    he_stains = sorted(he_subset["stain_id"].astype(str).unique())
    ret_stains = sorted(ret_subset["stain_id"].astype(str).unique())

    if he_stain_override is not None:
        if he_stain_override not in he_stains:
            raise RuntimeError(
                f"Requested H&E stain_id {he_stain_override!r} not found for lab {chosen}."
            )
        he_stain = he_stain_override
    else:
        he_stain = he_stains[0]

    if ret_stain_override is not None:
        if ret_stain_override not in ret_stains:
            raise RuntimeError(
                f"Requested Reticulin stain_id {ret_stain_override!r} not found for lab {chosen}."
            )
        ret_stain = ret_stain_override
    else:
        ret_stain = ret_stains[0]

    he_rows = he_subset[he_subset["stain_id"].astype(str) == str(he_stain)].reset_index(drop=True)
    ret_rows = ret_subset[ret_subset["stain_id"].astype(str) == str(ret_stain)].reset_index(drop=True)

    if he_rows.empty or ret_rows.empty:
        raise RuntimeError(
            f"Selected lab {chosen} does not contain both stains (he_stain={he_stain}, ret_stain={ret_stain})."
        )

    logging.info(
        "Selected lab %s with H&E stain_id=%s (%d patches) and Reticulin stain_id=%s (%d patches).",
        chosen,
        he_stain,
        len(he_rows),
        ret_stain,
        len(ret_rows),
    )

    return str(chosen), he_rows, ret_rows


def _save_canvas(canvas: Image.Image, path: Path, downsample: float) -> None:
    if downsample > 1.0:
        new_size = (
            max(1, int(canvas.width / downsample)),
            max(1, int(canvas.height / downsample)),
        )
        canvas = canvas.resize(new_size, Image.Resampling.LANCZOS)
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)
    logging.info("Saved %s", path)


def load_generator(checkpoint_path: Path, device: torch.device) -> CycleGenerator:
    generator = CycleGenerator(img_channels=3).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    state = ckpt.get("state_dict") or ckpt.get("G_H2R") or ckpt
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
        patterns = ["G_H2R_epoch*.pt", "G_H2R_epoch*.pth", "G_H2R_epoch*.pth.tar"]
        for pattern in patterns:
            checkpoints.extend(base.rglob(pattern))
        checkpoints = sorted({p.resolve() for p in checkpoints})
    if not checkpoints:
        raise FileNotFoundError("No G_H2R checkpoints found with the provided arguments.")
    return checkpoints


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate train_augment generators via FID.")
    parser.add_argument("--checkpoint", type=Path, default=None, help="Path to a single G_H2R checkpoint.")
    parser.add_argument(
        "--checkpoint-root",
        type=Path,
        default=DEFAULT_CHECKPOINT_ROOT,
        help="Directory containing train_augment run checkpoints (search recursively).",
    )
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--patch-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=_default_device())
    parser.add_argument("--lab-col", type=str, default=None)
    parser.add_argument(
        "--lab-id",
        type=str,
        default=None,
        help="Explicit lab ID to evaluate (overrides random selection).",
    )
    parser.add_argument(
        "--he-stain-id",
        type=str,
        default=None,
        help="Explicit H&E stain_id to use within the selected lab.",
    )
    parser.add_argument(
        "--ret-stain-id",
        type=str,
        default=None,
        help="Explicit Reticulin stain_id to use within the selected lab.",
    )
    parser.add_argument(
        "--ret-only",
        action="store_true",
        help="Use only Reticulin patches (skip H&E and skip generation).",
    )
    parser.add_argument("--downsample", type=float, default=4.0, help="Downsample factor when saving mosaics.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    device = torch.device(args.device)

    # If a specific checkpoint is given, ignore checkpoint_root
    checkpoint_root = None if args.checkpoint is not None else args.checkpoint_root
    checkpoints = _discover_checkpoints(args.checkpoint, checkpoint_root)

    metadata_path = resolve_path(args.metadata)
    df = pd.read_csv(metadata_path)
    required_cols = {"patch_path", "type", "x", "y"}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(f"metadata missing required columns: {sorted(missing_cols)}")

    run_dir = resolve_path(args.output_dir, allow_missing=True)
    run_dir.mkdir(parents=True, exist_ok=True)

    if args.ret_only:
        lab_col = _detect_lab_column(df, args.lab_col)
        type_series = df["type"].astype(str).str.lower()
        type_clean = type_series.str.replace(r"[^a-z]", "", regex=True)
        ret_mask = type_clean == "reticulin"

        if args.lab_id is not None:
            chosen = args.lab_id
        else:
            chosen = sorted(df.loc[ret_mask, lab_col].unique())[0]

        ret_rows = df[(df[lab_col] == chosen) & ret_mask].copy()
        ret_rows = ret_rows.reset_index(drop=True)

        if ret_rows.empty:
            raise RuntimeError("No Reticulin patches found for selected lab.")

        logging.info(
            "Selected lab %s with ONLY Reticulin (%d patches).",
            chosen,
            len(ret_rows),
        )

        # convert rows to entries
        ret_entries = _rows_to_entries(ret_rows)

        # assemble real canvas
        ret_canvas_path = run_dir / f"{chosen}_real_reticulin.png"
        if not ret_canvas_path.exists():
            ret_canvas = _assemble_slide(ret_entries, args.patch_size)
            _save_canvas(ret_canvas, ret_canvas_path, args.downsample)

        # skip H&E and generator; compute FID of real vs itself = 0
        logging.info("RET-ONLY: FID(real vs real) = 0.0000")
        return

    lab_col = _detect_lab_column(df, args.lab_col)
    lab_id, he_rows, ret_rows = _select_lab(
        df,
        lab_col,
        rng,
        lab_override=args.lab_id,
        he_stain_override=args.he_stain_id,
        ret_stain_override=args.ret_stain_id,
    )

    he_entries = _rows_to_entries(he_rows)
    ret_entries = _rows_to_entries(ret_rows)

    he_canvas_path = run_dir / f"{lab_id}_he_original.png"
    if not he_canvas_path.exists():
        he_canvas = _assemble_slide(he_entries, args.patch_size)
        _save_canvas(he_canvas, he_canvas_path, args.downsample)

    ret_canvas_path = run_dir / f"{lab_id}_real_reticulin.png"
    if not ret_canvas_path.exists():
        ret_canvas = _assemble_slide(ret_entries, args.patch_size)
        _save_canvas(ret_canvas, ret_canvas_path, args.downsample)

    for ckpt_path in tqdm(checkpoints, desc="Evaluating checkpoints"):
        ckpt_out = run_dir / ckpt_path.stem
        ckpt_out.mkdir(parents=True, exist_ok=True)

        generator = load_generator(ckpt_path, device)

        generated_entries = _generate_reticulin_patches(
            generator,
            he_rows,
            args.patch_size,
            device,
            ckpt_out / "generated_patches",
        )

        generated_canvas = _assemble_slide(generated_entries, args.patch_size)
        gen_canvas_path = ckpt_out / f"{lab_id}_generated_reticulin.png"
        _save_canvas(generated_canvas, gen_canvas_path, args.downsample)

        fid_value = _compute_fid(
            real_paths=[entry.path for entry in ret_entries],
            fake_paths=[entry.path for entry in generated_entries],
            device=device,
        )

        logging.info("[%s] FID (generated vs real Reticulin): %.4f", ckpt_path.name, fid_value)


if __name__ == "__main__":
    main()
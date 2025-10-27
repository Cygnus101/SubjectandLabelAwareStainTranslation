#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
WSI tiling + OD-based keep + CSV manifest + preview stitching.

Notes
-----
- Requires: openslide (library) and openslide-python (bindings), Pillow, NumPy.
- OD calculations and thresholds are identical to the user-provided versions:
  * Global I0 is estimated from a downscaled thumbnail (per-channel high percentile).
  * Tile OD uses that global I0 and reduces (mean/median) across the WHOLE tile.
  * Keep if od_stat > od_thresh.
- After tiles are saved + CSV written, you can reconstruct a low-res preview PNG
  that stitches the kept patches back into their positions.

This module provides:
  1) tile_keep_save_ome_tiff_global_OD
  2) batch_tile_keep_save_ome_folder
  3) reconstruct_canvas_from_csv
  4) save_preview_png_from_csv
  5) save_previews_for_batch_results
"""

from __future__ import annotations

import csv
import sys
import traceback
from pathlib import Path
import argparse

import numpy as np
from PIL import Image, ImageDraw
import openslide
# Optional, but handy when run in notebooks:
try:
    from IPython.display import display  # noqa: F401
except Exception:  # pragma: no cover
    display = None


# -------------------------
# Path helpers
# -------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
SRC_ROOT = SCRIPT_DIR.parent
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from utils.path import get_project_root, resolve_path as resolve_project_path

PROJECT_ROOT = get_project_root()


# -------------------------
# OD helpers (UNCHANGED MATH)
# -------------------------

def _estimate_global_I0(
    slide: openslide.OpenSlide,
    *,
    thumb_long_edge: int = 4096,
    hi_pct: float = 99.9
):
    """
    Estimate per-channel I0 (white reference) from the WHOLE slide by analyzing a downscaled thumbnail.
    Returns a float32 np.array([I0_R, I0_G, I0_B]) in the nominal 0..255 range.
    """
    w0, h0 = slide.level_dimensions[0]
    scale = min(1.0, float(thumb_long_edge) / max(w0, h0))
    tw, th = max(1, int(w0 * scale)), max(1, int(h0 * scale))
    thumb = slide.get_thumbnail((tw, th)).convert("RGB")
    arr = np.asarray(thumb, dtype=np.float32)  # (th, tw, 3) in 0..255 sRGB

    # Use a high percentile as "white" estimate (robust to a bit of ink)
    I0_R = np.percentile(arr[..., 0], hi_pct)
    I0_G = np.percentile(arr[..., 1], hi_pct)
    I0_B = np.percentile(arr[..., 2], hi_pct)

    # Guardrails: clamp to [1, 255] to avoid log blowups and nonsense
    I0 = np.clip(np.array([I0_R, I0_G, I0_B], dtype=np.float32), 1.0, 255.0)
    return I0


def _rgb_to_od_with_I0(
    img_rgb: Image.Image | np.ndarray,
    I0_vec: np.ndarray,
    eps: float = 1.0
):
    """
    Compute per-pixel OD using a per-slide, per-channel I0 vector: OD = -ln((I + eps) / I0).
    img_rgb: PIL.Image RGB or np.ndarray uint8(H,W,3)
    I0_vec: float32 array (3,) with [I0_R, I0_G, I0_B]
    Returns float32 OD array (H,W,3)
    """
    if isinstance(img_rgb, Image.Image):
        arr = np.asarray(img_rgb.convert("RGB"), dtype=np.float32)
    else:
        arr = img_rgb.astype(np.float32)
    # Broadcast I0 over HxW, add eps to avoid log(0)
    return -np.log((arr + float(eps)) / I0_vec.reshape(1, 1, 3))


def _od_magnitude(od_arr: np.ndarray):
    """L2 magnitude across channels → float32 (H,W)."""
    return np.sqrt(np.sum(od_arr * od_arr, axis=2, dtype=np.float32))


# -------------------------
# Tiling pass on a single OME-TIFF (UNCHANGED CALC)
# -------------------------

def tile_keep_save_ome_tiff_global_OD(
    ome_tiff_path: str | Path,
    out_dir: str | Path,
    *,
    patch_size: int = 512,
    stride: int | None = None,
    od_thresh: float = 0.2,
    reduce: str = "mean",         # "mean" or "median"
    eps: float = 1.0,
    save_format: str = "tif",     # "tif"|"png"|"jpg"
    tiff_compression: str = "tiff_deflate",
    thumb_long_edge: int = 4096,  # for global I0 estimation
    hi_pct: float = 99.9          # percentile for I0 estimation
):
    """
    Tiling pass on a .ome.tiff at level-0 with OD computed relative to the ENTIRE slide:
      1) Estimate per-slide I0 (R,G,B) from a downscaled thumbnail (99.9th percentile).
      2) Slide a grid over level-0 (patch_size, stride).
      3) For each tile, compute OD using that global I0, reduce (mean/median) over WHOLE tile.
      4) Keep if OD_stat > od_thresh, save patch, and log to CSV.

    CSV columns: source_ome, patch_path, x, y, od_stat
    """
    ome_tiff_path = resolve_project_path(ome_tiff_path)
    out_dir = resolve_project_path(out_dir, allow_missing=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    if stride is None:
        stride = patch_size

    slide = openslide.OpenSlide(str(ome_tiff_path))
    W0, H0 = slide.level_dimensions[0]

    # --- Global I0 from the whole slide ---
    I0_vec = _estimate_global_I0(slide, thumb_long_edge=thumb_long_edge, hi_pct=hi_pct)  # float32 [R,G,B]

    # --- CSV manifest ---
    csv_path = out_dir / f"metadata_{ome_tiff_path.stem}.csv"
    kept = 0
    total = 0

    with open(csv_path, "w", newline="") as fcsv:
        writer = csv.writer(fcsv)
        writer.writerow(["source_ome", "patch_path", "x", "y", "od_stat"])

        for y in range(0, H0, stride):
            rh = min(patch_size, H0 - y)
            if rh <= 0:
                break
            for x in range(0, W0, stride):
                rw = min(patch_size, W0 - x)
                if rw <= 0:
                    break

                total += 1
                patch = slide.read_region((x, y), 0, (rw, rh)).convert("RGB")

                # --- OD using GLOBAL per-slide I0 (UNCHANGED) ---
                od = _rgb_to_od_with_I0(patch, I0_vec=I0_vec, eps=eps)
                mag = _od_magnitude(od)
                od_stat = float(np.median(mag) if reduce == "median" else np.mean(mag))

                if od_stat > float(od_thresh):
                    kept += 1
                    fname = f"{ome_tiff_path.stem}_x{x}_y{y}_w{rw}_h{rh}.{save_format}"
                    fpath = out_dir / fname
                    if save_format.lower() == "tif":
                        patch.save(str(fpath), compression=tiff_compression)
                    else:
                        patch.save(str(fpath))
                    writer.writerow([str(ome_tiff_path), str(fpath), x, y, od_stat])

    slide.close()
    return {
        "slide": str(ome_tiff_path),
        "total_tiles": total,
        "kept_tiles": kept,
        "csv": str(csv_path),
        "out_dir": str(out_dir),
        "I0_vec": list(map(float, I0_vec)),
    }


# -------------------------
# Batch tiling over a folder (UNCHANGED CALC)
# -------------------------
def batch_tile_keep_save_ome_folder(
    input_dir: str | Path,
    out_root: str | Path,
    *,
    recursive: bool = True,
    patch_size: int = 512,
    stride: int | None = None,
    od_thresh: float = 0.2,
    reduce: str = "mean",
    eps: float = 1.0,
    save_format: str = "tif",
    tiff_compression: str = "tiff_deflate",
    thumb_long_edge: int = 4096,
    hi_pct: float = 99.9,
    exts: tuple[str, ...] = (".ome.tiff", ".ome.tif"),
    include_only: list[str] | set[str] | tuple[str, ...] | None = None,  # NEW
):
    """
    Process all OME-TIFFs in input_dir. If include_only is provided, only
    process slides inside those immediate subfolders (e.g., "A-3291", ...).
    Skips macOS AppleDouble files (names starting with '._').
    """
    input_dir = resolve_project_path(input_dir)
    out_root = resolve_project_path(out_root, allow_missing=True)
    out_root.mkdir(parents=True, exist_ok=True)

    allow = set(include_only) if include_only else None

    # Gather candidates
    def good_file(p: Path) -> bool:
        if not p.is_file():
            return False
        name_lower = p.name.lower()
        if name_lower.startswith("._"):
            return False
        suff = p.suffix.lower()
        suffs = "".join(p.suffixes).lower()
        if (suff in exts) or (suffs in exts):
            # if include_only is set, only allow files in those subdirs
            if allow is None:
                return True
            # assume structure H&E/<CASE>/file.ome.tif*
            return p.parent.name in allow
        return False

    if recursive:
        candidates = [p for p in input_dir.rglob("*") if good_file(p)]
    else:
        candidates = [p for p in input_dir.iterdir() if good_file(p)]

    candidates = sorted(candidates)
    if not candidates:
        print(f"[batch] No matching OME-TIFF files under: {input_dir}")
        return {"slides": [], "master_csv": str(out_root / "metadata_master.csv"),
                "errors": [], "out_root": str(out_root), "num_processed": 0, "num_errors": 0}

    master_csv = out_root / "metadata_master.csv"
    with open(master_csv, "w", newline="") as fmaster:
        mw = csv.writer(fmaster)
        mw.writerow(["source_ome", "patch_path", "x", "y", "od_stat"])

    results, errors = [], []
    for i, slide_path in enumerate(candidates, 1):
        slide_stem = slide_path.stem
        per_out = out_root / slide_stem
        print(f"[{i}/{len(candidates)}] Processing: {slide_path}")

        try:
            res = tile_keep_save_ome_tiff_global_OD(
                ome_tiff_path=slide_path,
                out_dir=per_out,
                patch_size=patch_size,
                stride=stride,
                od_thresh=od_thresh,
                reduce=reduce,
                eps=eps,
                save_format=save_format,
                tiff_compression=tiff_compression,
                thumb_long_edge=thumb_long_edge,
                hi_pct=hi_pct,
            )
            results.append(res)

            with open(res["csv"], "r", newline="") as fcsv, open(master_csv, "a", newline="") as fmaster:
                reader = csv.reader(fcsv)
                mw = csv.writer(fmaster)
                next(reader, None)  # skip header
                for row in reader:
                    mw.writerow(row)

        except Exception as e:
            errors.append((str(slide_path), repr(e)))
            print(f"[error] {slide_path}")
            traceback.print_exc(file=sys.stdout)

    return {
        "slides": results,
        "master_csv": str(master_csv),
        "errors": errors,
        "out_root": str(out_root),
        "num_processed": len(results),
        "num_errors": len(errors),
    }


# -------------------------
# Reconstruction (stitch kept patches back)
# -------------------------

def _safe_paste(canvas: Image.Image, patch: Image.Image, x: int, y: int):
    """
    Paste patch onto canvas at (x,y), safely cropping if the patch extends beyond canvas bounds.
    """
    cw, ch = canvas.size
    pw, ph = patch.size
    # Destination bbox on canvas
    dx0, dy0 = x, y
    dx1, dy1 = x + pw, y + ph

    # Intersection with canvas
    ix0 = max(0, dx0); iy0 = max(0, dy0)
    ix1 = min(cw, dx1); iy1 = min(ch, dy1)
    if ix0 >= ix1 or iy0 >= iy1:
        return  # fully outside

    # Corresponding crop on the patch
    sx0 = ix0 - dx0; sy0 = iy0 - dy0
    sx1 = sx0 + (ix1 - ix0); sy1 = sy0 + (iy1 - iy0)

    patch_cropped = patch.crop((sx0, sy0, sx1, sy1))
    canvas.paste(patch_cropped, (ix0, iy0))


def reconstruct_canvas_from_csv(
    csv_path: str | Path,
    *,
    preview_max: int = 4096,     # max long edge for preview
    background=(0, 0, 0),        # preview background color
    build_fullres: bool = False, # guarded full-res (can be huge)
    fullres_max_px: int = 16000  # safety cap on long edge for full-res
):
    """
    Reads CSV with columns: source_ome, patch_path, x, y, [od_stat]
    Reconstructs a downscaled preview canvas (and optional full-res) by pasting patches at their (x,y).

    Returns: dict with 'preview' (PIL.Image), optional 'fullres', 'scale', and summary info.
    """
    csv_path = resolve_project_path(csv_path)
    assert csv_path.exists(), f"CSV not found: {csv_path}"

    # Gather rows
    rows = []
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        required = {"source_ome", "patch_path", "x", "y"}
        missing = required - set(reader.fieldnames or [])
        assert not missing, f"CSV missing columns: {missing}"
        for r in reader:
            rows.append(r)
    assert rows, "CSV has no data rows."

    # Use the first row's source_ome (assume all rows are from the same slide)
    source_ome = resolve_project_path(rows[0]["source_ome"], allow_missing=True)
    assert source_ome.exists(), f"Source OME-TIFF not found: {source_ome}"

    # Open slide to get true level-0 dimensions
    slide = openslide.OpenSlide(str(source_ome))
    W0, H0 = slide.level_dimensions[0]
    slide.close()

    # Decide preview scale (keep aspect ratio)
    long_edge = max(W0, H0)
    scale = min(1.0, float(preview_max) / float(long_edge))
    Pw, Ph = max(1, int(W0 * scale)), max(1, int(H0 * scale))

    # Prepare canvases
    preview = Image.new("RGB", (Pw, Ph), background)

    do_fullres = build_fullres and (max(W0, H0) <= fullres_max_px)
    fullres = Image.new("RGB", (W0, H0), background) if do_fullres else None

    # Stitch patches
    missing_files = 0
    for r in rows:
        ppath = resolve_project_path(r["patch_path"], allow_missing=True)
        if not ppath.exists():
            missing_files += 1
            continue
        try:
            x = int(float(r["x"]))  # tolerate CSVs that wrote x,y as floats
            y = int(float(r["y"]))
        except Exception:
            raise ValueError(f"Non-integer coords in CSV row: {r}")

        patch = Image.open(ppath).convert("RGB")
        pw, ph = patch.size

        # Preview paste
        if scale != 1.0:
            spw = max(1, int(pw * scale))
            sph = max(1, int(ph * scale))
            patch_small = patch.resize((spw, sph), Image.BILINEAR)
            _safe_paste(preview, patch_small, int(x * scale), int(y * scale))
        else:
            _safe_paste(preview, patch, x, y)

        # Optional full-res
        if fullres is not None:
            _safe_paste(fullres, patch, x, y)

    info = {
        "preview": preview,
        "fullres": fullres,          # may be None
        "scale": scale,
        "slide_size_level0": (W0, H0),
        "preview_size": (Pw, Ph),
        "csv": str(csv_path),
        "source_ome": str(source_ome),
        "missing_patches": missing_files,
        "total_rows": len(rows),
    }

    # Display inline if available (e.g., Jupyter)
    if display is not None:
        display(preview)
        if fullres is not None:
            print("[info] Built full-res canvas (not auto-displayed to avoid UI lag).")

    return info


# -------------------------
# Save preview PNG helpers (no OD math)
# -------------------------

def save_preview_png_from_csv(
    csv_path: str | Path,
    out_png_dir: str | Path = "Kept_patches_png",
    *,
    preview_max: int = 4096,
    background=(0, 0, 0),
    build_fullres: bool = False,
    fullres_max_px: int = 16000,
    overwrite: bool = True
):
    """
    Build preview from a single per-slide CSV and save as PNG in out_png_dir.
    Does NOT change any OD calculations; only stitches the already-saved patches.
    """
    csv_path = resolve_project_path(csv_path)
    out_png_dir = resolve_project_path(out_png_dir, allow_missing=True)
    out_png_dir.mkdir(parents=True, exist_ok=True)

    info = reconstruct_canvas_from_csv(
        csv_path=csv_path,
        preview_max=preview_max,
        background=background,
        build_fullres=build_fullres,
        fullres_max_px=fullres_max_px
    )
    # Name file by the source slide stem
    src = Path(info["source_ome"])
    stem = src.stem  # e.g., A_429_1.ome
    out_png = out_png_dir / f"{stem}_kept_preview.png"
    if out_png.exists() and not overwrite:
        print(f"[preview] Exists, skipping: {out_png}")
        return str(out_png)

    info["preview"].save(out_png, format="PNG", compress_level=6, optimize=True)
    print(f"[preview] Wrote: {out_png}")
    return str(out_png)


def save_previews_for_batch_results(
    batch_summary: dict,
    out_png_dir: str | Path = "Kept_patches_png",
    *,
    preview_max: int = 4096,
    background=(0, 0, 0),
    build_fullres: bool = False,
    fullres_max_px: int = 16000,
    overwrite: bool = True
):
    """
    Given the dict returned by batch_tile_keep_save_ome_folder, render & save a preview PNG for each slide.
    No OD math is recomputed; this only reads CSV + patches and stitches.
    """
    out_png_dir = resolve_project_path(out_png_dir, allow_missing=True)
    out_png_dir.mkdir(parents=True, exist_ok=True)

    written = []
    errors = []
    for res in batch_summary.get("slides", []):
        csv_path = resolve_project_path(res["csv"], allow_missing=True)
        try:
            png_path = save_preview_png_from_csv(
                csv_path=csv_path,
                out_png_dir=out_png_dir,
                preview_max=preview_max,
                background=background,
                build_fullres=build_fullres,
                fullres_max_px=fullres_max_px,
                overwrite=overwrite
            )
            written.append(png_path)
        except Exception as e:
            errors.append((str(csv_path), repr(e)))
            print(f"[preview-error] {csv_path}")
            traceback.print_exc(file=sys.stdout)
    return {"written_pngs": written, "errors": errors, "out_png_dir": str(out_png_dir)}


# -------------------------
# Optional: quick examples (commented)
# -------------------------
def _parse_cli_args(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(
        description="Batch tiling helpers for OME-TIFF slides with optional per-folder filtering."
    )
    parser.add_argument("input_dir", nargs="?", default="Reticulin",
                        help="Folder containing case subdirectories with OME-TIFF files (default: Reticulin)")
    parser.add_argument("out_root", nargs="?", default="KeptPatches_Reticulin",
                        help="Output folder root for kept patches (default: KeptPatches_Reticulin)")
    parser.add_argument("--include", nargs="*", metavar="CASE",
                        help="Optional list of immediate subfolders/case IDs to process (e.g. A-3291 A-3510).")
    parser.add_argument("--include-file", type=Path,
                        help="Path to text file with one case ID per line; combined with --include when both are given.")
    parser.add_argument("--no-recursive", action="store_true",
                        help="Disable recursive search; only look at files directly under the case directories.")
    parser.add_argument("--preview-dir", default="Kept_patches_R_png",
                        help="Directory to store preview PNGs (default: Kept_patches_R_png)")
    parser.add_argument("--skip-previews", action="store_true",
                        help="Skip preview generation after tiling.")
    parser.add_argument("--previews-only", action="store_true",
                        help="Only render previews from existing metadata_* CSVs under --csv-root or out_root.")
    parser.add_argument("--csv-root", type=Path,
                        help="Root folder to search for metadata CSVs when using --previews-only (default: out_root).")
    return parser.parse_args(argv)


def _resolve_include_list(cli_args) -> list[str] | None:
    include: list[str] = []
    if cli_args.include:
        include.extend(cli_args.include)
    if cli_args.include_file:
        include_file = resolve_project_path(cli_args.include_file)
        try:
            with include_file.open("r", encoding="utf-8") as f:
                include.extend(line.strip() for line in f if line.strip())
        except FileNotFoundError as exc:
            raise SystemExit(f"[cli] Include file not found: {include_file}\n{exc}")
    return include or None


if __name__ == "__main__":
    args = _parse_cli_args()
    include_only = _resolve_include_list(args)

    if args.previews_only:
        csv_root = resolve_project_path(args.csv_root or args.out_root)
        if not csv_root.exists():
            raise SystemExit(f"[previews-only] CSV root does not exist: {csv_root}")

        csv_files = sorted(csv_root.rglob("metadata_*.csv"))
        if not csv_files:
            raise SystemExit(f"[previews-only] No metadata_*.csv files found under {csv_root}")

        batch_summary = {"slides": [{"csv": str(csv_path)} for csv_path in csv_files]}
        previews = save_previews_for_batch_results(
            batch_summary=batch_summary,
            out_png_dir=args.preview_dir,
            preview_max=4096,
            background=(0, 0, 0),
            build_fullres=False,
            fullres_max_px=16000
        )
        print("Saved previews:", len(previews["written_pngs"]))
        if previews["errors"]:
            print("Preview errors:", previews["errors"])
        raise SystemExit(0)

    summary = batch_tile_keep_save_ome_folder(
        input_dir=args.input_dir,
        out_root=args.out_root,
        patch_size=512,
        stride=512,
        od_thresh=0.2,
        reduce="mean",
        save_format="tif",
        recursive=not args.no_recursive,
        include_only=include_only,
    )

    if not args.skip_previews:
        previews = save_previews_for_batch_results(
            batch_summary=summary,
            out_png_dir=args.preview_dir,
            preview_max=4096,
            background=(0, 0, 0),
            build_fullres=False,
            fullres_max_px=16000
        )
        print("Saved previews:", len(previews["written_pngs"]))
        if previews["errors"]:
            print("Preview errors:", previews["errors"])

    # _ = save_preview_png_from_csv(
    #     csv_path="KeptPatches_HE/A_429_1/metadata_A_429_1.ome.csv",
    #     out_png_dir="Kept_patches_png",
    #     preview_max=4096,
    #     background=(0, 0, 0),
    #     build_fullres=False
    # )

#!/usr/bin/env python3
"""Evaluate round-trip SSIM for CycleGAN checkpoints on selected Reticulin patches."""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
from PIL import Image, UnidentifiedImageError
from skimage.metrics import structural_similarity
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
from tqdm.auto import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
SRC_ROOT = SCRIPT_DIR.parent
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from utils.path import ensure_project_root_on_syspath, get_project_root, resolve_path

ensure_project_root_on_syspath()
PROJECT_ROOT = get_project_root()
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "eval_ssim.json"

try:
    from src.models.Backbone_model.CycleGANv3 import UNetGenerator as UNetGeneratorV3  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    UNetGeneratorV3 = None  # type: ignore[assignment]

try:
    from src.models.Backbone_model.CycleGAN import UNetGenerator as UNetGeneratorLegacy  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    UNetGeneratorLegacy = None  # type: ignore[assignment]


LOGGER = logging.getLogger("eval_ssim")


def _load_config(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    text = path.read_text("utf-8").strip()
    if not text:
        return {}
    if path.suffix.lower() == ".json":
        data = json.loads(text)
    else:
        import yaml

        data = yaml.safe_load(text)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"Config file {path} must contain a mapping/object.")
    return data


def _normalize_list(values: Any | None) -> list[str]:
    if values is None:
        return []
    if isinstance(values, (str, Path)):
        values = [values]
    result: list[str] = []
    for value in values:  # type: ignore[assignment]
        if value is None:
            continue
        text = str(value).strip()
        if text:
            result.append(text)
    return result


def _detect_lab_column(df: pd.DataFrame) -> str | None:
    for candidate in ("Lab No.", "lab_id", "Lab_No", "lab", "patient_id"):
        if candidate in df.columns:
            return candidate
    return None


def _resolve_patch_path(raw: str, data_root: Path | None, metadata_path: Path) -> Path:
    path = Path(raw)
    if path.is_absolute():
        return path
    base = data_root if data_root else metadata_path.parent
    return (base / path).resolve()


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _determine_device(spec: str | None) -> torch.device:
    if spec:
        return torch.device(spec)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@dataclass
class PatchRecord:
    path: Path
    stain_id: str
    lab_id: str


def _prepare_reticulin_records(
    metadata_csv: Path,
    chosen_stains: Sequence[str],
    num_samples: int,
    seed: int,
    data_root: Path | None,
) -> list[PatchRecord]:
    return _prepare_patch_records(
        metadata_csv=metadata_csv,
        chosen_stains=chosen_stains,
        num_samples=num_samples,
        seed=seed,
        data_root=data_root,
        type_keywords=("reticulin",),
        description="Reticulin",
        require_matches=True,
    )


def _prepare_patch_records(
    metadata_csv: Path,
    chosen_stains: Sequence[str],
    num_samples: int,
    seed: int,
    data_root: Path | None,
    type_keywords: Sequence[str],
    description: str,
    require_matches: bool = True,
) -> list[PatchRecord]:
    if not chosen_stains:
        raise ValueError("chosen_stains list is empty; specify at least one stain_id or lab.")
    target = {str(stain).strip() for stain in chosen_stains if str(stain).strip()}
    if not target:
        raise ValueError("After normalization, chosen_stains is empty.")

    df = pd.read_csv(metadata_csv)
    required = {"type", "stain_id", "patch_path"}
    if missing := required - set(df.columns):
        raise ValueError(f"metadata missing required columns: {sorted(missing)}")
    df["type_norm"] = df["type"].astype(str).str.lower()
    lab_col = _detect_lab_column(df)
    df["stain_norm"] = df["stain_id"].astype(str).str.strip()
    if lab_col:
        df["lab_norm"] = df[lab_col].astype(str).str.strip()
    else:
        df["lab_norm"] = ""

    normalized_keywords = [str(keyword).strip().lower() for keyword in type_keywords if str(keyword).strip()]
    if not normalized_keywords:
        type_mask = pd.Series(True, index=df.index)
    else:
        type_mask = pd.Series(False, index=df.index)
        for keyword in normalized_keywords:
            type_mask |= df["type_norm"].str.contains(keyword, na=False)
    stain_mask = df["stain_norm"].isin(target)
    lab_mask = df["lab_norm"].isin(target)
    df = df[type_mask & (stain_mask | lab_mask)].copy()
    if df.empty:
        msg = f"No {description} entries found for the provided stains/labs."
        if require_matches:
            raise RuntimeError(msg)
        LOGGER.warning(msg)
        return []

    df["abs_path"] = df["patch_path"].astype(str).apply(
        lambda p: _resolve_patch_path(p, data_root, metadata_csv)
    )
    df["exists"] = df["abs_path"].apply(lambda p: Path(p).is_file())
    missing = df[~df["exists"]]
    if not missing.empty:
        LOGGER.warning("%d entries point to missing patch files; skipping.", len(missing))
    df = df[df["exists"]].copy()
    if df.empty:
        msg = f"All candidate {description} patches are missing on disk."
        if require_matches:
            raise RuntimeError(msg)
        LOGGER.warning(msg)
        return []

    if len(df) > num_samples:
        df = df.sample(n=num_samples, random_state=seed)
    records: list[PatchRecord] = []
    for _, row in df.iterrows():
        records.append(
            PatchRecord(
                path=Path(row["abs_path"]),
                stain_id=str(row["stain_norm"]),
                lab_id=str(row["lab_norm"] or row["stain_norm"]),
            )
        )
    LOGGER.info("Using %d %s patches for processing.", len(records), description)
    return records


def _build_eval_transform(image_size: int) -> T.Compose:
    return T.Compose(
        [
            T.Resize(image_size),
            T.CenterCrop(image_size),
            T.ToTensor(),
            T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )


class PatchDataset(Dataset):
    def __init__(self, records: Sequence[PatchRecord], image_size: int) -> None:
        if not records:
            raise ValueError("PatchDataset requires at least one record.")
        self.records = list(records)
        self.transform = _build_eval_transform(image_size)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        record = self.records[idx]
        try:
            with Image.open(record.path) as img:
                tensor = self.transform(img.convert("RGB"))
        except (FileNotFoundError, UnidentifiedImageError, OSError) as exc:
            raise RuntimeError(f"Failed to load patch {record.path}: {exc}") from exc
        return tensor, record.stain_id, record.lab_id, str(record.path)


def _tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    array = tensor.detach().cpu().clamp(-1.0, 1.0)
    array = (array + 1.0) * 0.5
    array = (array.clamp(0.0, 1.0) * 255.0).permute(1, 2, 0).byte().numpy()
    return Image.fromarray(array)


def _select_random_records(records: Sequence[PatchRecord], count: int, seed: int) -> list[PatchRecord]:
    if count <= 0:
        return []
    if len(records) <= count:
        return list(records)
    rng = random.Random(seed)
    return rng.sample(list(records), count)


def compute_ssim_batch(real_batch: torch.Tensor, recon_batch: torch.Tensor) -> list[float]:
    real = real_batch.detach().cpu()
    recon = recon_batch.detach().cpu()
    real = (real + 1.0) * 0.5
    recon = (recon + 1.0) * 0.5
    real_np = real.permute(0, 2, 3, 1).numpy()
    recon_np = recon.permute(0, 2, 3, 1).numpy()
    scores: list[float] = []
    for a, b in zip(real_np, recon_np):
        try:
            score = structural_similarity(a, b, channel_axis=-1, data_range=1.0)
        except TypeError:
            score = structural_similarity(a, b, multichannel=True, data_range=1.0)  # type: ignore[arg-type]
        scores.append(float(score))
    return scores


def _extract_state_from_payload(payload: Any, role: str | None = None) -> dict[str, Any]:
    """
    Extract a state_dict from a checkpoint payload.

    Handles:
    - Plain state_dicts
    - Dicts with 'state_dict' or 'model'
    - Train-state style checkpoints with keys like 'G_R2H' / 'G_H2R',
      where the nested dict may itself contain 'state_dict' or 'model'.
    """
    if isinstance(payload, dict):
        # Prefer explicit role (G_R2H / G_H2R) when present, e.g. train_state checkpoints.
        if role and role in payload and isinstance(payload[role], dict):
            sub = payload[role]
            if "state_dict" in sub and isinstance(sub["state_dict"], dict):
                return sub["state_dict"]
            if "model" in sub and isinstance(sub["model"], dict):
                return sub["model"]
            if all(isinstance(k, str) for k in sub.keys()):
                # Looks like a state_dict already
                return sub  # type: ignore[return-value]

        # Standard patterns at top level
        if "state_dict" in payload and isinstance(payload["state_dict"], dict):
            return payload["state_dict"]
        if "model" in payload and isinstance(payload["model"], dict):
            return payload["model"]

        # Fallback: assume the whole dict is already a state_dict
        return payload  # type: ignore[return-value]

    raise TypeError(f"Unsupported checkpoint payload type for state dict extraction: {type(payload)}")


def _load_generator(path: Path, device: torch.device, img_channels: int = 3, role: str | None = None) -> torch.nn.Module:
    """
    Load a generator from the given checkpoint path.

    - If the checkpoint is a train_state.pt with both G_R2H and G_H2R,
      the `role` argument ('G_R2H' or 'G_H2R') selects the submodule.
    - Tries CycleGANv3 UNetGenerator first; if state_dict keys mismatch, falls
      back to the legacy CycleGAN UNetGenerator.
    """
    payload = torch.load(path, map_location="cpu")
    state = _extract_state_from_payload(payload, role=role)

    last_error: Exception | None = None

    # First, strict load: try v3 then legacy
    for GenCls in (UNetGeneratorV3, UNetGeneratorLegacy):
        if GenCls is None:
            continue
        model = GenCls(img_channels=img_channels).to(device)  # type: ignore[operator]
        try:
            model.load_state_dict(state, strict=True)
            return model
        except RuntimeError as exc:
            last_error = exc

    # If strict failed for all, try non-strict loading as a best-effort fallback.
    for GenCls in (UNetGeneratorV3, UNetGeneratorLegacy):
        if GenCls is None:
            continue
        model = GenCls(img_channels=img_channels).to(device)  # type: ignore[operator]
        try:
            model.load_state_dict(state, strict=False)
            msg = str(last_error) if last_error is not None else "unknown mismatch"
            LOGGER.warning(
                "Loaded checkpoint %s with non-strict matching using %s due to mismatch: %s",
                path,
                GenCls.__module__,
                msg,
            )
            return model
        except RuntimeError as exc:
            last_error = exc

    raise RuntimeError(f"Failed to load checkpoint {path} into any UNetGenerator variant: {last_error}")


def evaluate_checkpoints(
    dataset: PatchDataset,
    batch_size: int,
    device: torch.device,
    checkpoint_pairs: Sequence[dict[str, str]],
    num_workers: int,
) -> list[dict[str, Any]]:
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    results: list[dict[str, Any]] = []
    for pair in checkpoint_pairs:
        name = pair["name"]
        g_r2h_path = resolve_path(pair["g_r2h"])
        g_h2r_path = resolve_path(pair["g_h2r"])
        LOGGER.info("Evaluating checkpoint '%s'", name)
        G_R2H = _load_generator(g_r2h_path, device=device, img_channels=3, role="G_R2H")
        G_H2R = _load_generator(g_h2r_path, device=device, img_channels=3, role="G_H2R")
        G_R2H.eval()
        G_H2R.eval()

        all_scores: list[float] = []
        per_meta: list[tuple[str, str]] = []
        with torch.no_grad():
            for real_batch, stain_ids, lab_ids, _ in tqdm(loader, desc=f"SSIM {name}"):
                real_batch = real_batch.to(device)
                fake_he = G_R2H(real_batch)
                recon_ret = G_H2R(fake_he)
                scores = compute_ssim_batch(real_batch, recon_ret)
                all_scores.extend(scores)
                per_meta.extend(zip(stain_ids, lab_ids))

        if not all_scores:
            LOGGER.warning("No SSIM scores computed for checkpoint %s; skipping.", name)
            continue
        scores_np = np.array(all_scores, dtype=np.float32)
        summary = {
            "name": name,
            "g_r2h": str(g_r2h_path),
            "g_h2r": str(g_h2r_path),
            "mean_ssim": float(scores_np.mean()),
            "std_ssim": float(scores_np.std(ddof=0)),
            "median_ssim": float(np.median(scores_np)),
            "min_ssim": float(scores_np.min()),
            "max_ssim": float(scores_np.max()),
        }
        per_stain: dict[str, list[float]] = {}
        for (stain_id, _), score in zip(per_meta, all_scores):
            per_stain.setdefault(stain_id, []).append(score)
        summary["per_stain"] = {
            stain: {"mean": float(np.mean(vals)), "count": len(vals)}
            for stain, vals in per_stain.items()
        }
        results.append(summary)
    return results


def _export_samples_for_domain(
    dataset: PatchDataset,
    generator: torch.nn.Module,
    destination: Path,
    batch_size: int,
    device: torch.device,
    prefix: str,
) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    loader = DataLoader(
        dataset,
        batch_size=max(1, min(batch_size, len(dataset))),
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )
    counter = 0
    with torch.no_grad():
        for batch, stain_ids, lab_ids, paths in loader:
            batch = batch.to(device)
            outputs = generator(batch)
            for idx in range(outputs.size(0)):
                image = _tensor_to_pil(outputs[idx])
                source_name = Path(paths[idx]).stem
                stain = str(stain_ids[idx]).replace(" ", "_")
                lab = str(lab_ids[idx]).replace(" ", "_")
                filename = f"{prefix}_{counter:03d}_{stain}_{lab}_{source_name}.jpg"
                image.save(destination / filename, quality=95)
                counter += 1


# New helper function: _export_side_by_side_samples
def _export_side_by_side_samples(
    dataset: PatchDataset,
    generator: torch.nn.Module,
    destination: Path,
    batch_size: int,
    device: torch.device,
    prefix: str,
    left_label: str,
    right_label: str,
) -> None:
    """
    Save a 2 x N panel of ORIGINAL (top) and TRANSLATED (bottom) patches.

    This mimics the panel-style visualizations in train_augment.py:
    - top row: input domain (e.g., H&E)
    - bottom row: translated domain (e.g., Reticulin)

    Only the first batch is used; `batch_size` controls N (capped by dataset size).
    """
    destination.mkdir(parents=True, exist_ok=True)
    loader = DataLoader(
        dataset,
        batch_size=max(1, min(batch_size, len(dataset))),
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )

    with torch.no_grad():
        for batch, stain_ids, lab_ids, paths in loader:
            batch = batch.to(device)
            outputs = generator(batch)

            k = outputs.size(0)
            # Convert to PIL and collect
            real_pils: list[Image.Image] = []
            fake_pils: list[Image.Image] = []
            for idx in range(k):
                real_pils.append(_tensor_to_pil(batch[idx]))
                fake_pils.append(_tensor_to_pil(outputs[idx]))

            if not real_pils:
                return

            # Assume all images are same size; enforce via resize
            w, h = real_pils[0].size
            for i in range(k):
                real_pils[i] = real_pils[i].resize((w, h), Image.BILINEAR)
                fake_pils[i] = fake_pils[i].resize((w, h), Image.BILINEAR)

            # Create 2 x N canvas: top = real, bottom = fake
            panel_width = w * k
            panel_height = h * 2
            canvas = Image.new("RGB", (panel_width, panel_height))

            # Paste real (top row) and fake (bottom row)
            for i in range(k):
                canvas.paste(real_pils[i], (i * w, 0))
                canvas.paste(fake_pils[i], (i * w, h))

            # Build a concise filename; we don't try to encode all patch metadata here
            base_stain = str(stain_ids[0]).replace(" ", "_") if len(stain_ids) > 0 else "mixed"
            base_lab = str(lab_ids[0]).replace(" ", "_") if len(lab_ids) > 0 else "mixed"
            filename = f"{prefix}_panel_{base_stain}_{base_lab}_{left_label}_to_{right_label}.jpg"

            canvas.save(destination / filename, quality=95)
            # Only need one panel per call (first batch only)
            break


def generate_sample_images(
    checkpoint_pairs: Sequence[dict[str, str]],
    he_records: Sequence[PatchRecord],
    ret_records: Sequence[PatchRecord],
    sample_count: int,
    output_dir: Path,
    device: torch.device,
    image_size: int,
    seed: int,
) -> None:
    he_selection = _select_random_records(he_records, sample_count, seed)
    ret_selection = _select_random_records(ret_records, sample_count, seed)
    if not he_selection and not ret_selection:
        LOGGER.warning("Sample option requested but no suitable patches were found; skipping sample export.")
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    he_dataset = PatchDataset(he_selection, image_size) if he_selection else None
    ret_dataset = PatchDataset(ret_selection, image_size) if ret_selection else None
    for pair in checkpoint_pairs:
        name = pair["name"]
        LOGGER.info("Generating sample images for checkpoint '%s'", name)
        g_r2h_path = resolve_path(pair["g_r2h"])
        g_h2r_path = resolve_path(pair["g_h2r"])
        G_R2H = _load_generator(g_r2h_path, device=device, img_channels=3, role="G_R2H")
        G_H2R = _load_generator(g_h2r_path, device=device, img_channels=3, role="G_H2R")
        G_R2H.eval()
        G_H2R.eval()

        pair_dir = output_dir / name

        # H&E → Reticulin panels (left = original H&E, right = translated Reticulin)
        if he_dataset:
            _export_side_by_side_samples(
                dataset=he_dataset,
                generator=G_H2R,
                destination=pair_dir / "he_to_ret_panels",
                batch_size=sample_count,
                device=device,
                prefix="he2ret",
                left_label="he",
                right_label="ret",
            )

        # Reticulin → H&E panels (left = original Reticulin, right = translated H&E)
        if ret_dataset:
            _export_side_by_side_samples(
                dataset=ret_dataset,
                generator=G_R2H,
                destination=pair_dir / "ret_to_he_panels",
                batch_size=sample_count,
                device=device,
                prefix="ret2he",
                left_label="ret",
                right_label="he",
            )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    base = argparse.ArgumentParser(add_help=False)
    base.add_argument("--config", type=str, default=str(DEFAULT_CONFIG))
    config_ns, remaining = base.parse_known_args(argv)
    config_defaults = _load_config(Path(config_ns.config)) if config_ns.config else {}

    parser = argparse.ArgumentParser(description=__doc__, parents=[base])
    parser.add_argument("--metadata-csv", type=str, default=config_defaults.get("metadata_csv", "metadata.csv"))
    parser.add_argument("--data-root", type=str, default=config_defaults.get("data_root"))
    parser.add_argument("--num-samples", type=int, default=config_defaults.get("num_samples", 1000))
    parser.add_argument("--seed", type=int, default=config_defaults.get("seed", 42))
    parser.add_argument("--batch-size", type=int, default=config_defaults.get("batch_size", 16))
    parser.add_argument("--num-workers", type=int, default=config_defaults.get("num_workers", 4))
    parser.add_argument("--image-size", type=int, default=config_defaults.get("image_size", 256))
    parser.add_argument("--device", type=str, default=config_defaults.get("device"))
    parser.add_argument("--output-json", type=str, default=config_defaults.get("output_json", "outputs/ssim_results.json"))
    parser.add_argument("--output-csv", type=str, default=config_defaults.get("output_csv", "outputs/ssim_results.csv"))
    parser.add_argument("--sample", dest="sample", action="store_true", help="Enable sample image export.")
    parser.add_argument("--no-sample", dest="sample", action="store_false", help="Disable sample image export.")
    parser.set_defaults(sample=bool(config_defaults.get("sample", False)))
    parser.add_argument("--sample-count", type=int, default=config_defaults.get("sample_count", 15))
    parser.add_argument(
        "--sample-output-dir",
        type=str,
        default=config_defaults.get("sample_output_dir", "outputs/metrics/samples"),
    )
    parser.add_argument("--chosen-stain", action="append", dest="chosen_stains", default=None)
    parser.add_argument(
        "--checkpoint",
        action="append",
        help="Format: name:path_to_G_R2H:path_to_G_H2R (overrides config checkpoint list)",
    )
    args = parser.parse_args(remaining)

    args.metadata_csv = resolve_path(args.metadata_csv)
    args.data_root = resolve_path(args.data_root) if args.data_root else None
    args.output_json = resolve_path(args.output_json)
    args.output_csv = resolve_path(args.output_csv)
    if args.sample_output_dir:
        args.sample_output_dir = resolve_path(args.sample_output_dir)
    else:
        args.sample_output_dir = args.output_json.parent / "samples"
    args.chosen_stains = (
        _normalize_list(args.chosen_stains)
        if args.chosen_stains
        else _normalize_list(config_defaults.get("chosen_stains"))
    )
    if args.checkpoint:
        ckpt_pairs: list[dict[str, str]] = []
        for spec in args.checkpoint:
            parts = spec.split(":")
            if len(parts) != 3:
                raise ValueError(f"--checkpoint entry '{spec}' must be name:path_r2h:path_h2r")
            ckpt_pairs.append({"name": parts[0], "g_r2h": parts[1], "g_h2r": parts[2]})
        args.checkpoint_pairs = ckpt_pairs
    else:
        config_pairs = config_defaults.get("checkpoint_pairs", [])
        if not isinstance(config_pairs, list):
            raise ValueError("checkpoint_pairs in config must be a list")
        args.checkpoint_pairs = config_pairs
    return args


def save_results(results: Sequence[dict[str, Any]], json_path: Path, csv_path: Path) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with json_path.open("w", encoding="utf-8") as fp:
        json.dump(results, fp, indent=2)
    summary_rows = [
        {
            "name": row["name"],
            "g_r2h": row["g_r2h"],
            "g_h2r": row["g_h2r"],
            "mean_ssim": row["mean_ssim"],
            "std_ssim": row["std_ssim"],
            "median_ssim": row["median_ssim"],
            "min_ssim": row["min_ssim"],
            "max_ssim": row["max_ssim"],
        }
        for row in results
    ]
    pd.DataFrame(summary_rows).to_csv(csv_path, index=False)
    LOGGER.info("Saved SSIM summary to %s and %s", json_path, csv_path)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    device = _determine_device(args.device)
    _set_seed(args.seed)
    he_sample_records: list[PatchRecord] = []
    if args.sample and args.sample_count > 0:
        he_sample_records = _prepare_patch_records(
            metadata_csv=args.metadata_csv,
            chosen_stains=args.chosen_stains,
            num_samples=max(args.sample_count, 1),
            seed=args.seed,
            data_root=args.data_root,
            type_keywords=("h&e", "he", "hematoxylin"),
            description="H&E",
            require_matches=False,
        )
    elif args.sample:
        LOGGER.warning("Sample flag enabled but sample_count <= 0; skipping sample generation.")
    records = _prepare_reticulin_records(
        metadata_csv=args.metadata_csv,
        chosen_stains=args.chosen_stains,
        num_samples=args.num_samples,
        seed=args.seed,
        data_root=args.data_root,
    )
    dataset = PatchDataset(records, image_size=args.image_size)
    results = evaluate_checkpoints(
        dataset,
        batch_size=args.batch_size,
        device=device,
        checkpoint_pairs=args.checkpoint_pairs,
        num_workers=args.num_workers,
    )
    if args.sample and args.sample_count > 0:
        generate_sample_images(
            checkpoint_pairs=args.checkpoint_pairs,
            he_records=he_sample_records,
            ret_records=records,
            sample_count=args.sample_count,
            output_dir=args.sample_output_dir,
            device=device,
            image_size=args.image_size,
            seed=args.seed,
        )
    if not results:
        LOGGER.warning("No results to save; exiting.")
        return
    save_results(results, args.output_json, args.output_csv)


if __name__ == "__main__":
    main()

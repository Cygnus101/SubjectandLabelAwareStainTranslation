#!/usr/bin/env python3
"""
Compare how two CycleGAN generators (checkpoint A vs checkpoint B) affect a frozen
ABMIL + slide classifier pipeline on the validation/test split from augmented_splits.json.

High-level procedure:
1. Load frozen ABMIL attention module + slide classifier weights.
2. Load SimCLR/feature-extractor weights (used to produce patch embeddings).
3. Load both CycleGAN generators (G_H2R) from checkpoints A and B.
4. For each slide in the chosen split, gather H&E patches (exactly as train_aug uses).
5. Run patches through generator A -> SimCLR encoder -> ABMIL -> classifier.
6. Repeat with generator B.
7. Compare predicted classes, logits, and slide vectors (cosine similarity).
8. Report summary metrics and optionally dump per-slide diagnostics to JSON.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms as T
from tqdm.auto import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
SRC_ROOT = SCRIPT_DIR.parent

import sys

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from utils.path import ensure_project_root_on_syspath, resolve_path, get_project_root  # noqa: E402

ensure_project_root_on_syspath()
PROJECT_ROOT = get_project_root()

from models.Feature_Extractor import MODEL_REGISTRY as ENCODER_REGISTRY  # noqa: E402
from models.Classification.abmil import ABMIL, SlideClassifier  # noqa: E402

try:  # CycleGAN variants (v3 first, fallback to legacy)
    from models.Backbone_model.CycleGANv3 import UNetGenerator as UNetGeneratorV3  # type: ignore
except Exception:  # pragma: no cover
    UNetGeneratorV3 = None  # type: ignore[assignment]

try:
    from models.Backbone_model.CycleGAN import UNetGenerator as UNetGeneratorLegacy  # type: ignore
except Exception:  # pragma: no cover
    UNetGeneratorLegacy = None  # type: ignore[assignment]

from training.train_aug import (  # noqa: E402
    IMAGENET_MEAN,
    IMAGENET_STD,
    load_augmented_slides,
    load_augmented_splits,
)

LOGGER = logging.getLogger("compare_generators")


def _extract_state_from_payload(payload: Any, role: str | None = None) -> dict[str, Any]:
    if isinstance(payload, dict):
        if role and role in payload and isinstance(payload[role], dict):
            sub = payload[role]
            if "state_dict" in sub and isinstance(sub["state_dict"], dict):
                return sub["state_dict"]
            if "model" in sub and isinstance(sub["model"], dict):
                return sub["model"]
            if all(isinstance(k, str) for k in sub.keys()):
                return sub  # type: ignore[return-value]
        if "state_dict" in payload and isinstance(payload["state_dict"], dict):
            return payload["state_dict"]
        if "model" in payload and isinstance(payload["model"], dict):
            return payload["model"]
        return payload  # type: ignore[return-value]
    raise TypeError(f"Unsupported checkpoint payload type: {type(payload)}")


def load_generator(path: Path, device: torch.device, role: str | None = "G_H2R") -> nn.Module:
    payload = torch.load(path, map_location="cpu")
    state = _extract_state_from_payload(payload, role=role)
    last_error: Exception | None = None

    for gen_cls in (UNetGeneratorV3, UNetGeneratorLegacy):
        if gen_cls is None:
            continue
        model = gen_cls(img_channels=3).to(device)  # type: ignore[operator]
        try:
            model.load_state_dict(state, strict=True)
            model.eval()
            return model
        except RuntimeError as exc:
            last_error = exc

    for gen_cls in (UNetGeneratorV3, UNetGeneratorLegacy):
        if gen_cls is None:
            continue
        model = gen_cls(img_channels=3).to(device)  # type: ignore[operator]
        try:
            model.load_state_dict(state, strict=False)
            LOGGER.warning(
                "Loaded generator %s with non-strict matching due to mismatch: %s",
                path,
                last_error or "unknown mismatch",
            )
            model.eval()
            return model
        except RuntimeError as exc:
            last_error = exc
    raise RuntimeError(f"Failed to load generator {path}: {last_error}")


def build_feature_extractor(
    backbone: str,
    checkpoint: Path,
    proj_hidden_dim: int,
    proj_out_dim: int,
    device: torch.device,
) -> nn.Module:
    ctor = ENCODER_REGISTRY.get(backbone.lower())
    if ctor is None:
        raise ValueError(f"Unknown encoder backbone '{backbone}'. Available: {sorted(ENCODER_REGISTRY.keys())}")
    model = ctor(
        pretrained=False,
        proj_hidden_dim=proj_hidden_dim,
        proj_out_dim=proj_out_dim,
    )
    state = torch.load(checkpoint, map_location="cpu")
    if isinstance(state, dict):
        if "model" in state:
            model.load_state_dict(state["model"])
        elif "backbone" in state:
            model.backbone.load_state_dict(state["backbone"])
        else:
            model.load_state_dict(state)
    else:
        model.load_state_dict(state)
    if hasattr(model, "projector"):
        model.projector = nn.Identity()
    for param in model.parameters():
        param.requires_grad_(False)
    return model.to(device).eval()


def normalize_for_encoder(patches: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    flat = patches.view(-1, *patches.shape[2:])
    flat = (flat + 1.0) * 0.5
    return (flat - mean) / std


def extract_patch_features(
    encoder: nn.Module,
    patches: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
) -> torch.Tensor:
    normalized = normalize_for_encoder(patches, mean, std)
    with torch.no_grad():
        feats = encoder(normalized)
        if isinstance(feats, tuple):
            feats = feats[0]
    feat_dim = feats.shape[1]
    return feats.view(patches.size(0), patches.size(1), feat_dim)


def infer_feature_dim(
    encoder: nn.Module,
    generator: nn.Module,
    sample_patch: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
    device: torch.device,
) -> int:
    with torch.no_grad():
        fake = generator(sample_patch.to(device))
        fake = fake.unsqueeze(0)  # [1, 1, C, H, W]
        feats = extract_patch_features(encoder, fake, mean, std)
        return feats.shape[-1]


def load_patches(paths: Sequence[Path], transform: T.Compose, max_patches: int | None = None) -> torch.Tensor:
    selected = list(paths if max_patches is None else paths[:max_patches])
    tensors: list[torch.Tensor] = []
    for path in selected:
        try:
            with Image.open(path) as img:
                tensors.append(transform(img.convert("RGB")))
        except (FileNotFoundError, OSError) as exc:
            LOGGER.warning("Skipping patch %s (%s)", path, exc)
    if not tensors:
        raise RuntimeError("No H&E patches could be loaded for slide.")
    return torch.stack(tensors, dim=0)


def generator_inference(
    he_patches: torch.Tensor,
    generator: nn.Module,
    encoder: nn.Module,
    abmil: ABMIL,
    classifier: SlideClassifier,
    mean: torch.Tensor,
    std: torch.Tensor,
    device: torch.device,
    batch_size: int = 16,
) -> dict[str, Any]:
    generator.eval()
    chunks: list[torch.Tensor] = []
    with torch.no_grad():
        for idx in range(0, he_patches.size(0), batch_size):
            chunk = he_patches[idx : idx + batch_size].to(device)
            fake = generator(chunk)
            chunks.append(fake.cpu())
    fake_ret = torch.cat(chunks, dim=0).unsqueeze(0)  # [1, N, C, H, W]
    z = extract_patch_features(encoder, fake_ret.to(device), mean, std)[0]
    with torch.no_grad():
        slide_vec, attn = abmil.pool(z)
        logits = classifier(slide_vec.unsqueeze(0))
        probs = torch.softmax(logits, dim=1)
        pred = int(probs.argmax(dim=1).item())
    return {
        "prediction": pred,
        "logits": logits.detach().cpu().numpy().tolist(),
        "slide_vec": slide_vec.detach().cpu().numpy().tolist(),
        "attention": attn.detach().cpu().numpy().tolist(),
    }


def compute_summary(per_slide: list[dict[str, Any]]) -> dict[str, float]:
    total = len(per_slide)
    if total == 0:
        return {}
    changed = sum(1 for row in per_slide if row["pred_a"] != row["pred_b"])
    logit_diffs = []
    cosines = []
    for row in per_slide:
        logits_a = torch.tensor(row["logits_a"])
        logits_b = torch.tensor(row["logits_b"])
        logit_diffs.append(torch.mean(torch.abs(logits_a - logits_b)).item())
        vec_a = torch.tensor(row["slide_vec_a"])
        vec_b = torch.tensor(row["slide_vec_b"])
        if torch.norm(vec_a) > 0 and torch.norm(vec_b) > 0:
            cos = F.cosine_similarity(vec_a.unsqueeze(0), vec_b.unsqueeze(0)).item()
            cosines.append(cos)
    return {
        "slides_compared": total,
        "prediction_change_pct": (changed / total) * 100.0,
        "mean_abs_logit_diff": float(np.mean(logit_diffs)) if logit_diffs else 0.0,
        "mean_cosine_similarity": float(np.mean(cosines)) if cosines else 0.0,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare classifier decisions between two CycleGAN generators.")
    parser.add_argument("--augmented-slides", type=Path, default=PROJECT_ROOT / "augmented_slides.json")
    parser.add_argument("--augmented-splits", type=Path, default=PROJECT_ROOT / "augmented_splits.json")
    parser.add_argument("--split", type=str, choices=("val", "test"), default="val")
    parser.add_argument("--data-root", type=Path, default=None, help="Root directory for patch paths (optional).")
    parser.add_argument("--abmil-checkpoint", type=Path, required=True, help="Pretrained ABMIL weights (.pt).")
    parser.add_argument("--classifier-checkpoint", type=Path, required=True, help="Pretrained slide classifier weights.")
    parser.add_argument("--encoder-backbone", type=str, default="resnet50", help="Key from models.Feature_Extractor registry.")
    parser.add_argument("--encoder-checkpoint", type=Path, required=True, help="SimCLR/feature extractor weights.")
    parser.add_argument("--encoder-proj-hidden", type=int, default=2048)
    parser.add_argument("--encoder-proj-out", type=int, default=256)
    parser.add_argument("--generator-a", type=Path, required=True, help="Checkpoint path for generator A (G_H2R).")
    parser.add_argument("--generator-b", type=Path, required=True, help="Checkpoint path for generator B (G_H2R).")
    parser.add_argument("--attn-dim", type=int, default=None, help="ABMIL attention dimension (defaults to feat_dim // 2).")
    parser.add_argument("--classifier-hidden", type=int, default=None, help="Slide classifier hidden dimension (defaults to feat_dim).")
    parser.add_argument("--num-classes", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--max-patches", type=int, default=None, help="Optional cap on H&E patches per slide.")
    parser.add_argument("--batch-size", type=int, default=16, help="Generator batch size for patch conversion.")
    parser.add_argument("--device", type=str, default=None, help="Device spec (cuda, mps, cpu).")
    parser.add_argument("--output-json", type=Path, default=None, help="Optional path to dump per-slide comparisons.")
    parser.add_argument("--log-level", type=str, default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    LOGGER.info("Using device: %s", device)

    slides = load_augmented_slides(Path(args.augmented_slides), args.data_root)
    splits = load_augmented_splits(Path(args.augmented_splits))
    split_indices = splits[args.split]
    if not split_indices:
        raise RuntimeError(f"No indices found for split '{args.split}'.")
    LOGGER.info("Loaded %d slides; split '%s' has %d indices.", len(slides), args.split, len(split_indices))
    target_slides = [slides[idx] for idx in split_indices if 0 <= idx < len(slides)]

    transform = T.Compose(
        [
            T.Resize(args.image_size),
            T.CenterCrop(args.image_size),
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )

    generator_a = load_generator(resolve_path(args.generator_a), device=device)
    generator_b = load_generator(resolve_path(args.generator_b), device=device)
    encoder = build_feature_extractor(
        args.encoder_backbone,
        resolve_path(args.encoder_checkpoint),
        args.encoder_proj_hidden,
        args.encoder_proj_out,
        device,
    )

    mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, -1, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=device).view(1, -1, 1, 1)

    # Infer feature dimension using first slide + generator A.
    sample_slide = next(slide for slide in target_slides if slide.he_paths)
    with Image.open(sample_slide.he_paths[0]) as img:
        sample_patch = transform(img.convert("RGB")).unsqueeze(0)
    feat_dim = infer_feature_dim(
        encoder=encoder,
        generator=generator_a,
        sample_patch=sample_patch,
        mean=mean,
        std=std,
        device=device,
    )
    LOGGER.info("Inferred feature dimension: %d", feat_dim)

    attn_dim = args.attn_dim if args.attn_dim is not None else max(64, feat_dim // 2)
    hidden_dim = args.classifier_hidden if args.classifier_hidden is not None else feat_dim
    abmil = ABMIL(
        in_dim=feat_dim,
        attn_dim=attn_dim,
        classifier_hidden=hidden_dim,
        num_classes=args.num_classes,
        dropout=args.dropout,
    ).to(device)
    abmil.load_state_dict(torch.load(resolve_path(args.abmil_checkpoint), map_location="cpu"))
    abmil.eval()

    slide_classifier = SlideClassifier(
        in_dim=feat_dim,
        hidden_dim=hidden_dim,
        num_classes=args.num_classes,
        dropout=args.dropout,
    ).to(device)
    slide_classifier.load_state_dict(torch.load(resolve_path(args.classifier_checkpoint), map_location="cpu"))
    slide_classifier.eval()

    per_slide: list[dict[str, Any]] = []
    for slide in tqdm(target_slides, desc=f"Comparing generators on {args.split}"):
        try:
            he_tensor = load_patches(slide.he_paths, transform, args.max_patches)
        except RuntimeError as exc:
            LOGGER.warning("Skipping slide %s due to patch loading failure: %s", slide.lab_id, exc)
            continue
        slide_batch = he_tensor.to(device)

        result_a = generator_inference(
            he_patches=slide_batch,
            generator=generator_a,
            encoder=encoder,
            abmil=abmil,
            classifier=slide_classifier,
            mean=mean,
            std=std,
            device=device,
            batch_size=args.batch_size,
        )
        result_b = generator_inference(
            he_patches=slide_batch,
            generator=generator_b,
            encoder=encoder,
            abmil=abmil,
            classifier=slide_classifier,
            mean=mean,
            std=std,
            device=device,
            batch_size=args.batch_size,
        )
        per_slide.append(
            {
                "slide_id": slide.he_stain_id,
                "lab_id": slide.lab_id,
                "grade": slide.grade,
                "pred_a": result_a["prediction"],
                "pred_b": result_b["prediction"],
                "logits_a": result_a["logits"],
                "logits_b": result_b["logits"],
                "slide_vec_a": result_a["slide_vec"],
                "slide_vec_b": result_b["slide_vec"],
            }
        )

    summary = compute_summary(per_slide)
    LOGGER.info("Summary: %s", json.dumps(summary, indent=2))
    if args.output_json:
        payload = {"summary": summary, "per_slide": per_slide}
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(payload, indent=2))
        LOGGER.info("Wrote comparison details to %s", args.output_json)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Deliberately broken CycleGAN training to demonstrate chain-rule breaking."""

from __future__ import annotations

import argparse
import json
import logging
import random
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Optional, Sequence

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
SRC_ROOT = SCRIPT_DIR.parent
if str(SRC_ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(SRC_ROOT))

from utils.path import ensure_project_root_on_syspath, get_project_root, resolve_path  # noqa: E402

ensure_project_root_on_syspath()
PROJECT_ROOT = get_project_root()

from training.train_augment import (  # noqa: E402
    IMAGENET_MEAN,
    IMAGENET_STD,
    DEFAULT_TRAIN_AUGMENT_CONFIG,
    SlidePatchDataset,
    ProjectionHead,
    freeze_module,
    _load_state_dict,
    build_feature_extractor,
    compute_classification_loss,
    compute_contrastive_loss,
    compute_contrastive_loss_with_bank,
    load_embedding_bank,
    load_cyclegan_weights,
    adversarial_loss,
    determine_device,
    setup_logging,
    _load_cli_defaults,
    extract_patch_features,
)

try:  # pragma: no cover
    from models.Backbone_model.CycleGAN import UNetGenerator as CycleGenerator, Discriminator as CycleDiscriminator  # type: ignore
except Exception:  # pragma: no cover
    try:
        from models.Backbone_model.CycleGANv3 import UNetGenerator as CycleGenerator, Discriminator as CycleDiscriminator  # type: ignore
    except Exception:
        from models.Backbone_model.CycleGANv2 import UNetGenerator as CycleGenerator, Discriminator as CycleDiscriminator  # type: ignore

from models.Classification.abmil import ABMIL, SlideClassifier  # noqa: E402

LOGGER = logging.getLogger("train_broken")


def grad_norm(module: nn.Module) -> float:
    norms = [p.grad.norm().item() for p in module.parameters() if p.grad is not None]
    return float(sum(norms) / len(norms)) if norms else 0.0


def component_grad_norm(loss: torch.Tensor, modules: Sequence[nn.Module], retain_graph: bool = True) -> float:
    if not loss.requires_grad:
        return 0.0
    params = [p for m in modules for p in m.parameters() if p.requires_grad]
    if not params:
        return 0.0
    grads = torch.autograd.grad(loss, params, retain_graph=retain_graph, allow_unused=True)
    norms = [g.norm().item() for g in grads if g is not None]
    return float(sum(norms) / len(norms)) if norms else 0.0


def _resolve_data_root(primary: Optional[str], legacy_roots: Sequence[str], metadata_path: Path) -> Optional[Path]:
    """
    Try primary root, then any legacy roots. Fall back to metadata parent.
    """
    candidates: list[str] = []
    if primary:
        candidates.append(primary)
    candidates.extend(legacy_roots or [])
    for raw in candidates:
        if not raw:
            continue
        candidate = resolve_path(raw, allow_missing=True)
        if candidate.exists():
            LOGGER.info("Using data root: %s", candidate)
            return candidate
    fallback = metadata_path.parent.resolve()
    LOGGER.info("Using metadata parent as data root: %s", fallback)
    return fallback


def build_loader(args: argparse.Namespace) -> tuple[SlidePatchDataset, DataLoader]:
    dataset = SlidePatchDataset(
        metadata_csv=Path(args.metadata),
        data_root=Path(args.data_root) if args.data_root else None,
        patches_per_slide=args.patches_per_slide,
        image_size=args.image_size,
        augment=not args.no_augment,
        seed=args.seed,
        max_slides=args.max_slide_count,
        patch_retries=args.patch_retries,
        subset_pct=args.subset,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_slides,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=args.device.startswith("cuda") if args.device else False,
        drop_last=True,
    )
    return dataset, loader


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    base_parser = argparse.ArgumentParser(add_help=False)
    base_parser.add_argument(
        "--config",
        type=str,
        default=str(DEFAULT_TRAIN_AUGMENT_CONFIG),
        help="Path to JSON config specifying default CLI arguments.",
    )
    config_ns, _ = base_parser.parse_known_args(argv)
    try:
        config_defaults = _load_cli_defaults(config_ns.config)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    parser = argparse.ArgumentParser(description="Broken CycleGAN training (detached fakes)", parents=[base_parser])
    parser.add_argument("--metadata", type=str, default=str(PROJECT_ROOT / "metadata.csv"))
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument(
        "--legacy-root",
        action="append",
        default=[],
        help="Additional roots to try when resolving patch paths (legacy locations).",
    )
    parser.add_argument("--batch-slides", type=int, default=2)
    parser.add_argument("--patches-per-slide", type=int, default=32)
    parser.add_argument("--lambda-adv", type=float, default=1.0)
    parser.add_argument("--lambda-cls", type=float, default=1.0)
    parser.add_argument("--lambda-con", type=float, default=1.0)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr-gen", type=float, default=2e-4)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--pretrained-cyclegan", type=str, default=None)
    parser.add_argument("--pretrained-encoder", type=str, default=None)
    parser.add_argument("--pretrained-abmil", type=str, default=None)
    parser.add_argument("--pretrained-classifier", type=str, default=None)
    parser.add_argument(
        "--log-dir",
        type=str,
        default=str(PROJECT_ROOT / "outputs" / "logs" / "broken_cyclegan"),
    )
    parser.add_argument("--run-id", type=str, required=True)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--encoder-backbone", type=str, default="resnet50")
    parser.add_argument("--encoder-proj-hidden", type=int, default=2048)
    parser.add_argument("--encoder-proj-out", type=int, default=128)
    parser.add_argument("--abmil-attn-dim", type=int, default=256)
    parser.add_argument("--classifier-hidden-dim", type=int, default=512)
    parser.add_argument("--abmil-dropout", type=float, default=0.25)
    parser.add_argument("--num-classes", type=int, default=4)
    parser.add_argument("--proj-hidden-dim", type=int, default=512)
    parser.add_argument("--proj-out-dim", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--save-every", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-augment", action="store_true")
    parser.add_argument(
        "--subset",
        type=float,
        default=None,
        help="Use only X%% of labs from metadata for quick experiments.",
    )
    parser.add_argument("--max-slide-count", type=int, default=None)
    parser.add_argument("--patch-retries", type=int, default=8)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument(
        "--reticulin-embedding-dir",
        type=str,
        default=None,
        help="Directory containing precomputed real reticulin embeddings (.npy).",
    )
    parser.add_argument(
        "--contrastive-negatives",
        type=int,
        default=32,
        help="Number of negative samples drawn from the real embedding bank per slide.",
    )

    parser.set_defaults(**config_defaults)
    args = parser.parse_args(argv)

    # Fill missing pretrained paths from config defaults (supports legacy keys)
    for arg_name, keys in [
        ("pretrained_cyclegan", ["pretrained_cyclegan", "cyclegan_checkpoint"]),
        ("pretrained_encoder", ["pretrained_encoder", "encoder_checkpoint"]),
        ("pretrained_abmil", ["pretrained_abmil", "abmil_checkpoint"]),
        ("pretrained_classifier", ["pretrained_classifier", "classifier_checkpoint"]),
    ]:
        current = getattr(args, arg_name)
        if current is None or (isinstance(current, str) and not current.strip()):
            for key in keys:
                if key in config_defaults and config_defaults[key]:
                    setattr(args, arg_name, config_defaults[key])
                    break

    if args.reticulin_embedding_dir:
        args.reticulin_embedding_dir = str(resolve_path(args.reticulin_embedding_dir, allow_missing=True))
    if args.contrastive_negatives <= 0:
        raise ValueError("--contrastive-negatives must be positive.")
    if args.subset is not None:
        if args.subset <= 0 or args.subset > 100:
            raise ValueError("--subset must be between 0 and 100 (exclusive of 0).")

    required = [
        "pretrained_cyclegan",
        "pretrained_encoder",
        "pretrained_abmil",
        "pretrained_classifier",
        "run_id",
    ]
    missing = [
        name
        for name in required
        if not getattr(args, name)
        or (isinstance(getattr(args, name), str) and not getattr(args, name).strip())
    ]
    if missing:
        raise ValueError(
            "Missing required arguments (provide via config or CLI): " + ", ".join(sorted(missing))
        )

    args.metadata = str(resolve_path(args.metadata))
    args.data_root = _resolve_data_root(args.data_root, args.legacy_root, Path(args.metadata))
    return args


def train(args: argparse.Namespace) -> None:
    device = determine_device(args.device)
    args.device = device.type if device.index is None else f"{device.type}:{device.index}"

    run_dir = resolve_path(args.log_dir, allow_missing=True) / args.run_id
    setup_logging(run_dir)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    dataset, loader = build_loader(args)
    LOGGER.info("Slides available: %d", len(dataset))
    embedding_bank = None
    if args.reticulin_embedding_dir:
        embedding_bank = load_embedding_bank(Path(args.reticulin_embedding_dir), dataset)
        LOGGER.info(
            "Using embedding bank for contrastive loss with %d negatives per slide.",
            args.contrastive_negatives,
        )

    feature_extractor = build_feature_extractor(args, device)
    mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, -1, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=device).view(1, -1, 1, 1)

    sample_he, sample_ret, _, _ = dataset[0]
    feat_dim = extract_patch_features(
        feature_extractor,
        sample_ret.unsqueeze(0).to(device),
        mean,
        std,
        require_grad=False,
    ).shape[-1]
    LOGGER.info("Feature dimension inferred as %d", feat_dim)

    abmil = ABMIL(
        in_dim=feat_dim,
        attn_dim=args.abmil_attn_dim,
        classifier_hidden=args.classifier_hidden_dim,
        num_classes=args.num_classes,
        dropout=args.abmil_dropout,
    ).to(device)
    _load_state_dict(abmil, resolve_path(args.pretrained_abmil))
    freeze_module(abmil)

    slide_classifier = SlideClassifier(
        feat_dim,
        args.classifier_hidden_dim,
        args.num_classes,
        dropout=args.abmil_dropout,
    ).to(device)
    _load_state_dict(slide_classifier, resolve_path(args.pretrained_classifier))
    freeze_module(slide_classifier)

    projector = ProjectionHead(feat_dim, args.proj_hidden_dim, args.proj_out_dim).to(device)

    G_H2R = CycleGenerator(img_channels=3).to(device)
    G_R2H = CycleGenerator(img_channels=3).to(device)
    D_R = CycleDiscriminator(in_channels=3).to(device)
    D_H = CycleDiscriminator(in_channels=3).to(device)

    load_cyclegan_weights(
        Path(args.pretrained_cyclegan),
        (G_H2R, G_R2H),
        (D_R, D_H),
    )
    freeze_module(D_R)
    freeze_module(D_H)

    opt_G = torch.optim.Adam(
        list(G_H2R.parameters()) + list(G_R2H.parameters()) + list(projector.parameters()),
        lr=args.lr_gen,
        betas=(0.5, 0.999),
    )

    criterion_cls = nn.CrossEntropyLoss()
    criterion_gan = nn.MSELoss()
    use_amp = bool(args.amp and device.type == "cuda")
    if args.amp and not use_amp:
        LOGGER.warning("AMP requested but device %s does not support CUDA autocast.", device.type)
    LOGGER.info("AMP enabled: %s", use_amp)
    scaler_G = GradScaler(enabled=use_amp)
    amp_context = autocast if use_amp else nullcontext

    for epoch in range(1, args.epochs + 1):
        G_H2R.train()
        G_R2H.train()
        projector.train()
        epoch_metrics = {"G": 0.0, "adv": 0.0, "cls": 0.0, "con": 0.0}
        batches = 0

        for he_batch, ret_batch, grades, slide_ids in tqdm(loader, desc=f"Epoch {epoch}"):
            batches += 1
            he_batch = he_batch.to(device, non_blocking=True)
            ret_batch = ret_batch.to(device, non_blocking=True)
            grades = grades.to(device, non_blocking=True)

            he_flat = he_batch.view(-1, *he_batch.shape[2:])
            ret_flat = ret_batch.view(-1, *ret_batch.shape[2:])

            with amp_context():
                fake_ret_flat = G_H2R(he_flat)
                fake_ret = fake_ret_flat.view_as(ret_batch)
                fake_ret_for_losses = fake_ret.detach()
                pred_fake_ret = D_R(fake_ret_flat)
                adv_loss = criterion_gan(pred_fake_ret, torch.ones_like(pred_fake_ret))

            z_fake = extract_patch_features(
                feature_extractor, fake_ret_for_losses, mean, std, require_grad=True
            )
            if embedding_bank is not None:
                contrastive_loss = compute_contrastive_loss_with_bank(
                    z_fake,
                    slide_ids,
                    embedding_bank,
                    projector,
                    args.contrastive_negatives,
                    args.temperature,
                )
            else:
                z_real = extract_patch_features(
                    feature_extractor, ret_batch, mean, std, require_grad=False
                )
                contrastive_loss = compute_contrastive_loss(z_real, z_fake, projector, args.temperature)

            cls_loss, _ = compute_classification_loss(z_fake, grades, abmil, slide_classifier, criterion_cls)

            weighted_adv = args.lambda_adv * adv_loss
            weighted_cls = args.lambda_cls * cls_loss
            weighted_con = args.lambda_con * contrastive_loss

            # Component-wise gradients: separately for generators vs projector
            adv_grad_G = component_grad_norm(weighted_adv, [G_H2R, G_R2H], retain_graph=True)
            adv_grad_P = component_grad_norm(weighted_adv, [projector], retain_graph=True)

            cls_grad_G = component_grad_norm(weighted_cls, [G_H2R, G_R2H], retain_graph=True)
            cls_grad_P = component_grad_norm(weighted_cls, [projector], retain_graph=True)

            con_grad_G = component_grad_norm(weighted_con, [G_H2R, G_R2H], retain_graph=True)
            con_grad_P = component_grad_norm(weighted_con, [projector], retain_graph=True)

            loss_G = weighted_adv + weighted_cls + weighted_con

            opt_G.zero_grad(set_to_none=True)
            if use_amp:
                scaler_G.scale(loss_G).backward()
                scaler_G.unscale_(opt_G)
            else:
                loss_G.backward()

            LOGGER.info(
                "[BROKEN] grads | adv(G)=%.6e adv(P)=%.6e | cls(G)=%.6e cls(P)=%.6e | con(G)=%.6e con(P)=%.6e || G_H2R=%.6e | G_R2H=%.6e | Proj=%.6e",
                adv_grad_G,
                adv_grad_P,
                cls_grad_G,
                cls_grad_P,
                con_grad_G,
                con_grad_P,
                grad_norm(G_H2R),
                grad_norm(G_R2H),
                grad_norm(projector),
            )

            if use_amp:
                scaler_G.step(opt_G)
                scaler_G.update()
            else:
                opt_G.step()

            epoch_metrics["G"] += loss_G.item()
            epoch_metrics["adv"] += adv_loss.item()
            epoch_metrics["cls"] += cls_loss.item()
            epoch_metrics["con"] += contrastive_loss.item()

        for key in epoch_metrics:
            epoch_metrics[key] = epoch_metrics[key] / max(1, batches)

        LOGGER.info(
            "Epoch %d | loss_G=%.4f | adv=%.4f | cls=%.4f | con=%.4f",
            epoch,
            epoch_metrics["G"],
            epoch_metrics["adv"],
            epoch_metrics["cls"],
            epoch_metrics["con"],
        )

    LOGGER.info("Training complete.")


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    train(args)


if __name__ == "__main__":
    main()

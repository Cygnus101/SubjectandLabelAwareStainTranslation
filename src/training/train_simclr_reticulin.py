#!/usr/bin/env python3
"""
SimCLR training script for Reticulin patches.

Loads 512x512 Reticulin patches from ``metadata.csv`` (or a compatible file),
trains a ResNet-50 backbone with a projection MLP using the NT-Xent loss, and
persists checkpoints alongside an encoder state dict that emits 2048-D feature
vectors suitable for downstream tasks.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Optional, Sequence, Tuple

import pandas as pd
from PIL import Image, UnidentifiedImageError

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
from tqdm.auto import tqdm

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:  # matplotlib optional; plotting disabled if unavailable
    plt = None

SCRIPT_DIR = Path(__file__).resolve().parent
SRC_ROOT = SCRIPT_DIR.parent
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from utils.path import ensure_project_root_on_syspath, get_project_root
from models.Feature_Extractor import MODEL_REGISTRY

ensure_project_root_on_syspath()
PROJECT_ROOT = get_project_root()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_METADATA = PROJECT_ROOT / "metadata.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "checkpoints" / "simclr_reticulin"
DEFAULT_LOG_DIR = PROJECT_ROOT / "outputs" / "logs"
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# ---------------------------------------------------------------------------
# Model Factory
# ---------------------------------------------------------------------------


def build_backbone(
    name: str,
    *,
    pretrained: bool,
    proj_hidden_dim: int,
    proj_out_dim: int,
) -> nn.Module:
    lookup = MODEL_REGISTRY.get(name.lower())
    if lookup is None:
        available = ", ".join(sorted(MODEL_REGISTRY))
        raise ValueError(f"Unknown backbone '{name}'. Available options: {available}")
    return lookup(
        pretrained=pretrained,
        proj_hidden_dim=proj_hidden_dim,
        proj_out_dim=proj_out_dim,
    )


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def setup_logging(log_dir: Path, run_name: str) -> Path:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{run_name}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(log_path), logging.StreamHandler()]
    )
    logging.info("Logging to %s", log_path)
    return log_path


# ---------------------------------------------------------------------------
# Dataset & Augmentations
# ---------------------------------------------------------------------------


def _resolve_path(patch_path: str, candidate_roots: Sequence[Path]) -> Path:
    path = Path(patch_path).expanduser()
    if path.is_absolute() and path.exists():
        return path

    for root in candidate_roots:
        candidate = (root / path).resolve()
        if candidate.exists():
            return candidate

    if path.is_absolute():
        raise FileNotFoundError(f"Patch not found: {path}")
    raise FileNotFoundError(f"Patch not found: {patch_path}")


class SimCLRPatchDataset(Dataset):
    """Dataset that returns two augmented views for SimCLR training."""

    def __init__(
        self,
        metadata_csv: Path,
        transform: Callable[[Image.Image], torch.Tensor],
        root_dir: Optional[Path] = None,
        subset: Optional[float] = None,
        seed: int = 123,
        legacy_roots: Optional[Sequence[Path]] = None,
        missing_retry_limit: int = 5,
    ) -> None:
        metadata_csv = metadata_csv.resolve()
        self.primary_root = root_dir.resolve() if root_dir else metadata_csv.parent
        df = pd.read_csv(metadata_csv)
        df = df[df["type"].str.lower() == "reticulin"].reset_index(drop=True)
        if df.empty:
            raise ValueError("No Reticulin rows found in metadata.")

        if subset is not None:
            if not (0.0 < subset <= 1.0):
                raise ValueError("subset must be in (0, 1].")
            rng = random.Random(seed)
            indices = list(range(len(df)))
            rng.shuffle(indices)
            keep = indices[: max(1, int(len(indices) * subset))]
            df = df.iloc[sorted(keep)].reset_index(drop=True)

        self._df = df
        self.transform = transform
        self.seed = seed
        self._rng = random.Random(seed)
        self.max_missing_retries = max(0, int(missing_retry_limit))

        roots: list[Path] = []
        def _add_root(path: Path) -> None:
            resolved = path.resolve()
            if resolved not in roots:
                roots.append(resolved)

        _add_root(self.primary_root)
        if legacy_roots:
            for root in legacy_roots:
                _add_root(root)
        self._root_candidates = roots

    def __len__(self) -> int:
        return len(self._df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        attempts = 0
        while True:
            row = self._df.iloc[idx]
            patch_rel = row["patch_path"]
            try:
                path = _resolve_path(patch_rel, self._root_candidates)
            except FileNotFoundError:
                attempts += 1
                if attempts > self.max_missing_retries:
                    raise
                logging.warning(
                    "Missing patch %s (attempt %d/%d); sampling replacement.",
                    patch_rel,
                    attempts,
                    self.max_missing_retries,
                )
                idx = self._rng.randint(0, len(self._df) - 1)
                continue

            try:
                with Image.open(path) as img:
                    img = img.convert("RGB")
                    view_one = self.transform(img)
                    view_two = self.transform(img)
                    return view_one, view_two
            except (UnidentifiedImageError, OSError) as exc:
                attempts += 1
                if attempts > self.max_missing_retries:
                    raise
                logging.warning(
                    "Unreadable patch %s (%s) (attempt %d/%d); sampling replacement.",
                    path,
                    type(exc).__name__,
                    attempts,
                    self.max_missing_retries,
                )
                idx = self._rng.randint(0, len(self._df) - 1)
                continue


class SimCLRTransform:
    """Construct the standard SimCLR augmentation pipeline."""

    def __init__(self, image_size: int, blur_kernel: int, color_jitter: float) -> None:
        if blur_kernel % 2 == 0:
            blur_kernel += 1
        self.transform = T.Compose([
            T.RandomResizedCrop(image_size, scale=(0.5, 1.0)),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.2),
            T.RandomApply(
                [T.ColorJitter(brightness=color_jitter, contrast=color_jitter,
                                saturation=color_jitter, hue=0.02)],
                p=0.8,
            ),
            T.GaussianBlur(kernel_size=blur_kernel, sigma=(0.1, 2.0)),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

    def __call__(self, img: Image.Image) -> torch.Tensor:
        return self.transform(img)




# ---------------------------------------------------------------------------
# Loss & Metrics
# ---------------------------------------------------------------------------


def nt_xent_loss(z_i: torch.Tensor, z_j: torch.Tensor, temperature: float) -> torch.Tensor:
    """Normalized temperature-scaled cross entropy loss."""
    z = torch.cat([z_i, z_j], dim=0)
    sim = torch.mm(z, z.t()) / temperature
    batch_size = z_i.size(0)
    labels = torch.arange(batch_size, device=z.device)
    labels = torch.cat([labels + batch_size, labels])

    diag_mask = torch.eye(2 * batch_size, dtype=torch.bool, device=z.device)
    sim = sim.masked_fill(diag_mask, float('-inf'))

    loss = F.cross_entropy(sim, labels)
    return loss


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------


def save_checkpoint(
    output_dir: Path,
    run_name: str,
    epoch: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: Optional[GradScaler],
    scheduler_state: Optional[dict],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = output_dir / f"{run_name}_epoch{epoch:03d}.pt"
    torch.save(
        {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict() if scaler else None,
            "scheduler": scheduler_state,
        },
        ckpt_path,
    )
    encoder_path = output_dir / f"{run_name}_encoder_epoch{epoch:03d}.pt"
    torch.save({"backbone": model.backbone.state_dict()}, encoder_path)
    logging.info("Saved checkpoints to %s", ckpt_path)


def load_checkpoint(
    checkpoint_path: Path,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scaler: Optional[GradScaler] = None,
) -> int:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["model"])
    if optimizer and checkpoint.get("optimizer"):
        optimizer.load_state_dict(checkpoint["optimizer"])
    if scaler and checkpoint.get("scaler"):
        scaler.load_state_dict(checkpoint["scaler"])
    return int(checkpoint.get("epoch", 0))


# ---------------------------------------------------------------------------
# Metrics Persistence & Plotting
# ---------------------------------------------------------------------------


def persist_training_artifacts(
    metrics: Sequence[dict[str, float]],
    output_dir: Path,
    run_name: str,
    enable_plot: bool,
) -> None:
    if not metrics:
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / f"{run_name}_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as fp:
        json.dump(list(metrics), fp, indent=2)
    logging.info("Saved training metrics to %s", metrics_path)

    if not enable_plot:
        return
    if plt is None:
        logging.warning("matplotlib not available; skipping loss curve plot.")
        return

    epochs = [entry["epoch"] for entry in metrics]
    losses = [entry["loss"] for entry in metrics]
    lrs = [entry["lr"] for entry in metrics]

    fig, ax1 = plt.subplots(figsize=(8, 5), constrained_layout=True)
    ax1.plot(epochs, losses, marker="o", color="#1f77b4", label="Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("NT-Xent Loss", color="#1f77b4")
    ax1.tick_params(axis="y", labelcolor="#1f77b4")

    if any(lrs):
        ax2 = ax1.twinx()
        ax2.plot(epochs, lrs, linestyle="--", color="#ff7f0e", label="LR")
        ax2.set_ylabel("Learning Rate", color="#ff7f0e")
        ax2.tick_params(axis="y", labelcolor="#ff7f0e")

    ax1.set_title(f"SimCLR Training ({run_name})")
    ax1.grid(True, linestyle="--", alpha=0.3)

    plot_path = output_dir / f"{run_name}_metrics.png"
    fig.savefig(plot_path, dpi=120)
    plt.close(fig)
    logging.info("Saved loss curve plot to %s", plot_path)


def load_metrics_history(metrics_path: Path, upto_epoch: Optional[int] = None) -> list[dict[str, float]]:
    if not metrics_path.exists():
        return []
    try:
        with metrics_path.open("r", encoding="utf-8") as fp:
            data = json.load(fp)
    except Exception as exc:
        logging.warning("Failed to load metrics from %s: %s", metrics_path, exc)
        return []

    if not isinstance(data, list):
        logging.warning("Metrics file %s is not a list; ignoring.", metrics_path)
        return []

    history: list[dict[str, float]] = []
    for entry in data:
        if not isinstance(entry, dict) or "epoch" not in entry:
            continue
        try:
            epoch_value = float(entry["epoch"])
        except (TypeError, ValueError):
            continue
        if upto_epoch is not None and epoch_value > upto_epoch:
            continue
        history.append(
            {
                "epoch": epoch_value,
                "loss": float(entry.get("loss", 0.0)),
                "lr": float(entry.get("lr", 0.0)),
            }
        )

    history.sort(key=lambda item: item["epoch"])
    return history


def find_latest_checkpoint(output_dir: Path, run_name: str) -> Optional[tuple[Path, int]]:
    pattern = re.compile(rf"{re.escape(run_name)}_epoch(\\d+)\\.pt$")
    latest_epoch = -1
    latest_path: Optional[Path] = None
    for path in output_dir.glob(f"{run_name}_epoch*.pt"):
        if "encoder" in path.stem:
            continue
        match = pattern.match(path.name)
        if not match:
            continue
        epoch_num = int(match.group(1))
        if epoch_num > latest_epoch:
            latest_epoch = epoch_num
            latest_path = path
    if latest_path is None:
        return None
    return latest_path, latest_epoch


# ---------------------------------------------------------------------------
# Training Loop
# ---------------------------------------------------------------------------


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: Optional[GradScaler],
    device: torch.device,
    temperature: float,
    use_amp: bool,
) -> float:
    model.train()
    running_loss = 0.0
    for images_i, images_j in tqdm(dataloader, desc="train", leave=False):
        images_i = images_i.to(device, non_blocking=True)
        images_j = images_j.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=use_amp):
            _, z_i = model(images_i)
            _, z_j = model(images_j)
            loss = nt_xent_loss(z_i, z_j, temperature)

        if scaler:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        running_loss += loss.item() * images_i.size(0)

    return running_loss / (len(dataloader.dataset))


# ---------------------------------------------------------------------------
# Argument Parsing
# ---------------------------------------------------------------------------


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train SimCLR on Reticulin patches")
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA, help="Path to metadata CSV")
    parser.add_argument("--root", type=Path, default=None, help="Optional root directory for relative patch paths")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory to store checkpoints")
    parser.add_argument("--run-name", type=str, default="simclr_reticulin", help="Run identifier used in artifact names")
    parser.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR, help="Directory for training logs")
    parser.add_argument(
        "--disable-plot",
        dest="plot",
        action="store_false",
        help="Disable saving a loss curve plot alongside metrics (enabled by default).",
    )

    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--subset", type=float, default=None, help="Optional fraction of data to use (0-1]")
    parser.add_argument(
        "--legacy-root",
        action="append",
        default=[],
        type=Path,
        help="Additional base directory to search when resolving patch paths (can be used multiple times).",
    )

    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--blur-kernel", type=int, default=23)
    parser.add_argument("--color-jitter", type=float, default=0.2)
    parser.add_argument(
        "--backbone",
        type=str,
        choices=sorted(MODEL_REGISTRY.keys()),
        default="resnet50",
        help="Feature extractor backbone to use (default: resnet50).",
    )

    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-6)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--proj-hidden-dim", type=int, default=2048)
    parser.add_argument("--proj-out-dim", type=int, default=128)
    
    parser.add_argument(
        "--no-pretrained",
        action="store_false",
        dest="pretrained",
        help="Disable ImageNet initialization (enabled by default)",
    )
    parser.set_defaults(pretrained=True, plot=True)

    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device specifier (e.g. cuda, cuda:1, mps, cpu). Defaults to CUDA, then MPS, then CPU.",
    )
    parser.add_argument("--resume", type=Path, default=None, help="Path to a checkpoint to resume from")
    parser.add_argument(
        "--resume-latest",
        action="store_true",
        help="Resume from the most recent checkpoint in --output-dir for this run.",
    )
    parser.add_argument("--amp", action="store_true", help="Use automatic mixed precision")

    args = parser.parse_args(argv)
    args.metadata = args.metadata.expanduser().resolve()
    if args.root is not None:
        args.root = args.root.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    args.log_dir = args.log_dir.expanduser().resolve()
    if args.resume is not None:
        args.resume = args.resume.expanduser().resolve()
    args.backbone = args.backbone.lower()
    args.legacy_root = [path.expanduser().resolve() for path in args.legacy_root]
    return args


# ---------------------------------------------------------------------------
# Main Entrypoint
# ---------------------------------------------------------------------------


def main(args: Optional[argparse.Namespace] = None) -> None:
    if args is None:
        args = parse_args()

    if args.device:
        device = torch.device(args.device)
    else:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    use_amp = args.amp and device.type == "cuda"

    set_seed(args.seed)
    log_path = setup_logging(args.log_dir, args.run_name)
    logging.info("Configuration:\n%s", json.dumps(vars(args), default=str, indent=2))
    logging.info("Using device: %s | AMP: %s", device, use_amp)

    transform = SimCLRTransform(args.image_size, args.blur_kernel, args.color_jitter)
    dataset = SimCLRPatchDataset(
        metadata_csv=args.metadata,
        transform=transform,
        root_dir=args.root,
        subset=args.subset,
        seed=args.seed,
        legacy_roots=args.legacy_root,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=args.num_workers > 0,
    )

    logging.info("Backbone: %s | pretrained=%s", args.backbone, args.pretrained)
    model = build_backbone(
        args.backbone,
        pretrained=args.pretrained,
        proj_hidden_dim=args.proj_hidden_dim,
        proj_out_dim=args.proj_out_dim,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))
    scaler = GradScaler(enabled=use_amp)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.output_dir / f"{args.run_name}_metrics.json"

    if args.resume_latest and args.resume is not None:
        logging.warning("Both --resume and --resume-latest specified; proceeding with explicit --resume path %s", args.resume)
    elif args.resume_latest and args.resume is None:
        latest = find_latest_checkpoint(args.output_dir, args.run_name)
        if latest:
            args.resume, latest_epoch = latest
            logging.info("Auto-resuming from latest checkpoint %s (epoch %d)", args.resume, latest_epoch)
        else:
            logging.info("No checkpoints found in %s to auto-resume from.", args.output_dir)

    start_epoch = 0
    if args.resume:
        logging.info("Resuming from %s", args.resume)
        start_epoch = load_checkpoint(args.resume, model, optimizer=optimizer, scaler=scaler)
        scheduler.last_epoch = start_epoch - 1 if start_epoch > 0 else -1

    if start_epoch > 0:
        metrics = load_metrics_history(metrics_path, upto_epoch=start_epoch)
        if metrics:
            last_recorded = metrics[-1]["epoch"]
            if int(last_recorded) != int(start_epoch):
                logging.warning(
                    "Metrics file %s latest epoch %.0f does not match checkpoint epoch %d; truncating to resume epoch.",
                    metrics_path,
                    last_recorded,
                    start_epoch,
                )
            else:
                logging.info("Loaded %d historical metrics entries from %s", len(metrics), metrics_path)
        else:
            logging.info(
                "No existing metrics entries found in %s up to epoch %d; starting metrics fresh.",
                metrics_path,
                start_epoch,
            )
    else:
        metrics = []

    for epoch in range(start_epoch + 1, args.epochs + 1):
        logging.info("Epoch %d/%d", epoch, args.epochs)
        avg_loss = train_epoch(
            model,
            dataloader,
            optimizer,
            scaler if use_amp else None,
            device,
            args.temperature,
            use_amp,
        )
        scheduler.step()
        lr = scheduler.get_last_lr()[0]
        logging.info("Epoch %d complete | loss=%.4f | lr=%.6f", epoch, avg_loss, lr)
        save_checkpoint(args.output_dir, args.run_name, epoch, model, optimizer, scaler if use_amp else None, scheduler.state_dict())
        metrics.append({"epoch": float(epoch), "loss": float(avg_loss), "lr": float(lr)})
        persist_training_artifacts(metrics, args.output_dir, args.run_name, enable_plot=args.plot)

    persist_training_artifacts(metrics, args.output_dir, args.run_name, enable_plot=args.plot)
    logging.info("Training complete. Latest encoder weights stored in %s", args.output_dir)
    logging.info("Log file: %s", log_path)


if __name__ == "__main__":
    main()

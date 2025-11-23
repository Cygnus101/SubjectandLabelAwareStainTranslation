#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Train CycleGAN for H&E ↔ Reticulin stain translation using metadata-driven dataloaders.
"""

import argparse
import itertools
import json
import logging
import re
import random
import sys
from contextlib import ExitStack, nullcontext
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import autocast, GradScaler
from torch.profiler import (
    ProfilerActivity,
    profile as torch_profile,
    schedule as profiler_schedule,
    tensorboard_trace_handler,
)
from torchvision.utils import save_image
from tqdm.auto import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
SRC_ROOT = SCRIPT_DIR.parent
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from utils.path import ensure_project_root_on_syspath, get_project_root, resolve_path

ensure_project_root_on_syspath()
PROJECT_ROOT = get_project_root()

# --- Import your custom modules ---
from src.models.Backbone_model.CycleGANv3 import UNetGenerator, Discriminator
from data.build_cyclegan_dataset import make_loaders_from_metadata, METADATA_CSV, BATCH_SIZE

DEFAULT_OUTPUTS_DIR = PROJECT_ROOT / "outputs"
DEFAULT_CHECKPOINTS_DIR = DEFAULT_OUTPUTS_DIR / "checkpoints" / "cyclegan"
DEFAULT_LOGS_DIR = DEFAULT_OUTPUTS_DIR / "logs"
DEFAULT_PROFILER_DIR = DEFAULT_LOGS_DIR / "profiler"
DEFAULT_SAMPLES_DIR = DEFAULT_OUTPUTS_DIR / "samples" / "cyclegan"
CHECKPOINT_PREFIXES = ("G_H2R", "G_R2H", "D_H", "D_R")
HISTORY_FILENAME = "history.json"
PREVIEW_SAMPLE_COUNT = 9

# ==============================================================================
# Replay Buffer (stabilizes discriminator training)
# ==============================================================================
class ReplayBuffer:
    def __init__(self, max_size=50):
        self.max_size = max_size
        self.data = []

    def push_and_pop(self, data_batch):
        images_to_return = []
        for element in data_batch.detach():
            element = torch.unsqueeze(element, 0)
            if len(self.data) < self.max_size:
                self.data.append(element.cpu())
                images_to_return.append(element)
            else:
                if random.random() > 0.5:
                    i = random.randint(0, self.max_size - 1)
                    cached = self.data[i].to(element.device, non_blocking=True)
                    images_to_return.append(cached)
                    self.data[i] = element.cpu()
                else:
                    images_to_return.append(element)
        return torch.cat(images_to_return)


def add_instance_noise(x: torch.Tensor, step: int, total_steps: int, sigma0: float) -> torch.Tensor:
    """Optionally add annealed Gaussian noise to discriminator inputs.

    sigma0 decays linearly to zero once `step` reaches `total_steps`.
    """
    if sigma0 <= 0 or total_steps <= 0:
        return x
    progress = min(max(step, 0) / float(total_steps), 1.0)
    sigma = sigma0 * max(0.0, 1.0 - progress)
    if sigma == 0:
        return x
    noise = sigma * torch.randn_like(x)
    return (x + noise).clamp(-1.0, 1.0)

# ==============================================================================
# Checkpoint Utilities
# ==============================================================================
def save_checkpoint(model, optimizer, scaler, filename):
    logging.info(f"Saving checkpoint → {filename}")
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "scaler": scaler.state_dict() if scaler is not None else None,
    }
    torch.save(checkpoint, filename)

def load_checkpoint(model, optimizer, scaler, lr, checkpoint_file):
    logging.info(f"Loading checkpoint ← {checkpoint_file}")
    try:
        checkpoint = torch.load(checkpoint_file, map_location="cpu")
        model.load_state_dict(checkpoint["state_dict"])
        if optimizer is not None and checkpoint.get("optimizer"):
            optimizer.load_state_dict(checkpoint["optimizer"])
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr
        if scaler is not None and checkpoint.get("scaler"):
            scaler.load_state_dict(checkpoint["scaler"])
        logging.info("Checkpoint loaded successfully.")
        return True
    except Exception as e:
        logging.error(f"Failed to load checkpoint {checkpoint_file}: {e}")
        return False


def _checkpoint_filename(prefix: str, epoch: int) -> str:
    return f"{prefix}_epoch{epoch}.pth.tar"


def _available_checkpoint_epochs(run_dir: Path) -> list[int]:
    """Return epochs that have a full set of checkpoints in run_dir."""
    if not run_dir.exists():
        return []
    epoch_sets: list[set[int]] = []
    for prefix in CHECKPOINT_PREFIXES:
        pattern = re.compile(rf"{re.escape(prefix)}_epoch(\d+)\.pth\.tar$")
        epochs = {
            int(match.group(1))
            for path in run_dir.glob(f"{prefix}_epoch*.pth.tar")
            if (match := pattern.match(path.name))
        }
        epoch_sets.append(epochs)
    if not epoch_sets:
        return []
    if not all(epoch_sets):
        return []
    epochs = set.intersection(*epoch_sets)
    return sorted(epochs)


def _load_history_file(history_path: Path, upto_epoch: int | None = None) -> list[dict[str, float]]:
    """Load persisted history JSON (if present) and optionally trim to epoch."""
    if not history_path.exists():
        return []
    try:
        with history_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as exc:
        logging.warning(f"Failed to load history from {history_path}: {exc}")
        return []
    if not isinstance(data, list):
        logging.warning(f"History file {history_path} is not a list; ignoring.")
        return []

    history: list[dict[str, float]] = []
    for row in data:
        if not isinstance(row, dict) or "epoch" not in row:
            continue
        try:
            epoch_value = int(row["epoch"])
        except (TypeError, ValueError):
            continue
        if upto_epoch is not None and epoch_value > upto_epoch:
            continue
        history.append(
            {
                "epoch": epoch_value,
                "gen_loss": float(row.get("gen_loss", 0.0)),
                "disc_loss": float(row.get("disc_loss", 0.0)),
            }
        )
    history.sort(key=lambda item: item["epoch"])
    return history


def _save_history_file(history_path: Path, history: list[dict[str, float]]) -> None:
    """Persist history JSON alongside checkpoints."""
    try:
        with history_path.open("w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)
    except Exception as exc:
        logging.warning(f"Failed to save history to {history_path}: {exc}")


def _select_resume_epoch(run_dir: Path, requested_epoch: int | None) -> int:
    """Determine which epoch to resume from."""
    available = _available_checkpoint_epochs(run_dir)
    if not available:
        raise FileNotFoundError(f"No complete checkpoints found in {run_dir}")
    if requested_epoch is None:
        return available[-1]
    if requested_epoch not in available:
        raise ValueError(
            f"Requested resume epoch {requested_epoch} not available in {run_dir}. "
            f"Available epochs: {available}"
        )
    return requested_epoch


def _save_loss_plot(history, output_path: str | Path) -> None:
    """Persist generator/discriminator loss curves for the run."""
    if not history:
        return
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        logging.warning(f"Skipping loss plot (matplotlib unavailable): {exc}")
        return

    epochs = [row["epoch"] for row in history]
    gen_losses = [row["gen_loss"] for row in history]
    disc_losses = [row["disc_loss"] for row in history]

    plt.figure()
    plt.plot(epochs, gen_losses, label="Generator Loss")
    plt.plot(epochs, disc_losses, label="Discriminator Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("CycleGAN Training Losses")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    plt.close()
    logging.info(f"Saved loss curves to {out_path}")


def _build_profiler_context(args, device: torch.device):
    """Return context manager for optional PyTorch profiler."""
    if not args.profile:
        return nullcontext()

    activities = [ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(ProfilerActivity.CUDA)

    trace_dir = Path(args.profile_dir) / args.run_id
    trace_dir.mkdir(parents=True, exist_ok=True)
    handler = tensorboard_trace_handler(str(trace_dir))
    schedule = profiler_schedule(
        wait=args.profile_wait,
        warmup=args.profile_warmup,
        active=args.profile_active,
        repeat=args.profile_repeat,
        skip_first=args.profile_skip_first,
    )
    logging.info(
        "Profiler enabled (wait=%d, warmup=%d, active=%d, repeat=%d, skip_first=%d). Traces → %s",
        args.profile_wait,
        args.profile_warmup,
        args.profile_active,
        args.profile_repeat,
        args.profile_skip_first,
        trace_dir,
    )
    return torch_profile(
        activities=activities,
        schedule=schedule,
        on_trace_ready=handler,
        record_shapes=args.profile_record_shapes,
        profile_memory=args.profile_memory,
        with_stack=args.profile_with_stack,
    )

# ==============================================================================
# Training Function
# ==============================================================================
def main(args):
    # --- Setup device & directories ---
    if torch.cuda.is_available():
        device = torch.device("cuda")
        use_amp = True
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        device = torch.device("mps")
        use_amp = False
    else:
        device = torch.device("cpu")
        use_amp = False  # gate AMP on CUDA only
    autocast_ctx = (lambda: autocast("cuda")) if use_amp else nullcontext

    checkpoints_dir = Path(args.checkpoints_dir)
    logs_dir = Path(args.logs_dir)
    samples_dir = Path(args.samples_dir)
    resume_dir = Path(args.resume_dir).expanduser().resolve(strict=False) if args.resume_dir else None
    run_ckpt_dir = resume_dir if resume_dir is not None else checkpoints_dir / args.run_id
    run_samples_dir = samples_dir / args.run_id

    if args.save_model_every <= 0:
        raise ValueError("--save-model-every must be a positive integer")
    if args.save_samples_every <= 0:
        raise ValueError("--save-samples-every must be a positive integer")
    if args.d_steps <= 0 or args.g_steps <= 0:
        raise ValueError("--d-steps and --g-steps must be positive integers")

    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    run_ckpt_dir.mkdir(parents=True, exist_ok=True)
    run_samples_dir.mkdir(parents=True, exist_ok=True)

    # --- Setup logging ---
    log_file = logs_dir / f"train_{args.run_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()]
    )

    logging.info(f"Run ID: {args.run_id}")
    if resume_dir is not None:
        if resume_dir.name != args.run_id:
            logging.info(
                f"Resuming from directory {resume_dir} (folder name differs from run_id)."
            )
        else:
            logging.info(f"Resuming from directory {resume_dir}")
    logging.info(f"Device: {device} | AMP: {use_amp}")
    logging.info(f"Hyperparameters: {vars(args)}")

    # --- Seed for reproducibility ---
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    # --- Compute persistent_workers from CLI (fix for num_workers=0) ---
    persistent_workers = args.num_workers > 0

    # --- Build dataloaders ---
    train_loader, val_loader, test_loader = make_loaders_from_metadata(
        metadata_csv=args.metadata,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
        persistent_workers=persistent_workers,
        subset_pct=args.subset,
        legacy_roots=args.legacy_root,
    )
    logging.info(f"Train batches: {len(train_loader)}, Val: {len(val_loader)}, Test: {len(test_loader)}")
    if len(train_loader) == 0:
        raise RuntimeError("Training dataloader is empty; cannot proceed with training.")

    if args.gen_warmup_epochs > 0:
        logging.info(
            f"Generator warmup enabled for {args.gen_warmup_epochs} epoch(s) using cycle/identity losses only."
        )

    # --- Cache a deterministic validation pair for previews ---
    # Fallback to test loader if val is empty.
    try_loader = val_loader if len(val_loader) > 0 else test_loader
    collected_he = []
    collected_ret = []
    total_collected = 0
    for he_batch, ret_batch, _ in try_loader:
        collected_he.append(he_batch)
        collected_ret.append(ret_batch)
        total_collected += he_batch.size(0)
        if total_collected >= PREVIEW_SAMPLE_COUNT:
            break

    if not collected_he or not collected_ret:
        raise RuntimeError("Failed to collect preview samples for deterministic sampling.")

    fixed_he = torch.cat(collected_he, dim=0)
    fixed_ret = torch.cat(collected_ret, dim=0)
    preview_count = min(PREVIEW_SAMPLE_COUNT, fixed_he.size(0), fixed_ret.size(0))
    fixed_he = fixed_he[:preview_count].cpu()
    fixed_ret = fixed_ret[:preview_count].cpu()

    # --- Initialize models ---
    G_H2R = UNetGenerator().to(device)   # H&E → Reticulin
    G_R2H = UNetGenerator().to(device)   # Reticulin → H&E
    D_H   = Discriminator().to(device)   # Discriminator for H&E
    D_R   = Discriminator().to(device)   # Discriminator for Reticulin

    opt_G = optim.Adam(itertools.chain(G_H2R.parameters(), G_R2H.parameters()), lr=args.lr_gen, betas=(0.5, 0.999))
    opt_D = optim.Adam(itertools.chain(D_H.parameters(), D_R.parameters()), lr=args.lr_disc, betas=(0.5, 0.999))

    adv_loss = nn.MSELoss()
    cycle_loss = nn.L1Loss()
    identity_loss = nn.L1Loss()

    scaler_G = GradScaler(device="cuda") if use_amp else None
    scaler_D = GradScaler(device="cuda") if use_amp else None

    # --- Replay buffers ---
    buffer_fake_H = ReplayBuffer()
    buffer_fake_R = ReplayBuffer()

    # --- Training Loop ---
    history_path = run_ckpt_dir / HISTORY_FILENAME
    history: list[dict[str, float]] = []
    resume_epoch: int | None = None

    if resume_dir is not None:
        selected_epoch = _select_resume_epoch(run_ckpt_dir, args.resume_epoch)
        checkpoints = {
            prefix: run_ckpt_dir / _checkpoint_filename(prefix, selected_epoch)
            for prefix in CHECKPOINT_PREFIXES
        }
        for prefix, ckpt_path in checkpoints.items():
            if not ckpt_path.exists():
                raise FileNotFoundError(f"Missing checkpoint for {prefix} at epoch {selected_epoch}: {ckpt_path}")

        if not load_checkpoint(G_H2R, opt_G, scaler_G, args.lr_gen, checkpoints["G_H2R"]):
            raise RuntimeError(f"Failed to load generator checkpoint {checkpoints['G_H2R']}")
        if not load_checkpoint(G_R2H, None, None, args.lr_gen, checkpoints["G_R2H"]):
            raise RuntimeError(f"Failed to load generator checkpoint {checkpoints['G_R2H']}")
        if not load_checkpoint(D_H, opt_D, scaler_D, args.lr_disc, checkpoints["D_H"]):
            raise RuntimeError(f"Failed to load discriminator checkpoint {checkpoints['D_H']}")
        if not load_checkpoint(D_R, None, None, args.lr_disc, checkpoints["D_R"]):
            raise RuntimeError(f"Failed to load discriminator checkpoint {checkpoints['D_R']}")

        history = _load_history_file(history_path, selected_epoch)
        if history and history[-1]["epoch"] != selected_epoch:
            logging.warning(
                f"History file {history_path} does not contain epoch {selected_epoch}; "
                f"latest recorded epoch is {history[-1]['epoch']}."
            )
        resume_epoch = selected_epoch
        _save_history_file(history_path, history)
    else:
        resume_epoch = None

    start_epoch = 1 if resume_epoch is None else resume_epoch + 1

    if resume_epoch is not None:
        logging.info(
            f"Resuming training from epoch {resume_epoch} located at {run_ckpt_dir}. "
            f"Starting at epoch {start_epoch} with target {args.num_epochs}."
        )

    if start_epoch > args.num_epochs:
        logging.info(
            f"No epochs left to train (start_epoch={start_epoch} > num_epochs={args.num_epochs})."
        )

    warmup_epochs = args.gen_warmup_epochs
    steps_per_epoch = max(1, len(train_loader) * max(1, args.d_steps))
    # Convert epoch-based schedule into total discriminator steps for linear annealing.
    noise_total_steps = args.instance_noise_epochs * steps_per_epoch if args.instance_noise else 0

    if warmup_epochs > 0 and start_epoch > warmup_epochs + 1:
        logging.info("Resuming after generator warmup; adversarial training active.")

    if args.instance_noise and resume_epoch is not None:
        completed_adv_epochs = max(0, (start_epoch - 1) - warmup_epochs)
        global_d_step = completed_adv_epochs * steps_per_epoch
    else:
        global_d_step = 0

    with ExitStack() as stack:
        profiler = stack.enter_context(_build_profiler_context(args, device))
        for epoch in range(start_epoch, args.num_epochs + 1):
            loop = tqdm(train_loader, desc=f"Epoch [{epoch}/{args.num_epochs}]")

            running_G, running_D = 0.0, 0.0
            batch_count = 0
            warmup_phase = epoch <= warmup_epochs

            if warmup_epochs > 0 and epoch == warmup_epochs + 1:
                logging.info("Generator warmup complete; enabling adversarial training.")

            for _, (real_H, real_R, _) in enumerate(loop):
                real_H = real_H.to(device, non_blocking=True)
                real_R = real_R.to(device, non_blocking=True)

                disc_loss_acc = 0.0
                gen_loss_acc = 0.0

                # --- Train Discriminators ---
                if not warmup_phase:
                    for _ in range(args.d_steps):
                        with torch.no_grad():
                            fake_R = G_H2R(real_H)
                            fake_H = G_R2H(real_R)

                        fake_H_buffer = buffer_fake_H.push_and_pop(fake_H)
                        fake_R_buffer = buffer_fake_R.push_and_pop(fake_R)

                        with autocast_ctx():

                            if args.instance_noise:
                                real_H_in = add_instance_noise(
                                    real_H, global_d_step, noise_total_steps, args.instance_noise_sigma
                                )
                                real_R_in = add_instance_noise(
                                    real_R, global_d_step, noise_total_steps, args.instance_noise_sigma
                                )
                                fake_H_in = add_instance_noise(
                                    fake_H_buffer, global_d_step, noise_total_steps, args.instance_noise_sigma
                                )
                                fake_R_in = add_instance_noise(
                                    fake_R_buffer, global_d_step, noise_total_steps, args.instance_noise_sigma
                                )
                            else:
                                real_H_in = real_H
                                real_R_in = real_R
                                fake_H_in = fake_H_buffer
                                fake_R_in = fake_R_buffer

                            D_H_real = D_H(real_H_in)
                            D_H_fake = D_H(fake_H_in)
                            D_H_loss = 0.5 * (
                                adv_loss(D_H_real, torch.ones_like(D_H_real))
                                + adv_loss(D_H_fake, torch.zeros_like(D_H_fake))
                            )

                            D_R_real = D_R(real_R_in)
                            D_R_fake = D_R(fake_R_in)
                            D_R_loss = 0.5 * (
                                adv_loss(D_R_real, torch.ones_like(D_R_real))
                                + adv_loss(D_R_fake, torch.zeros_like(D_R_fake))
                            )

                            loss_D = 0.5 * (D_H_loss + D_R_loss)

                        opt_D.zero_grad()
                        if use_amp:
                            scaler_D.scale(loss_D).backward()
                            scaler_D.step(opt_D)
                            scaler_D.update()
                        else:
                            loss_D.backward()
                            opt_D.step()

                        disc_loss_acc += float(loss_D.detach().cpu())
                        global_d_step += 1

                # --- Train Generators ---
                for _ in range(args.g_steps):
                    for p in D_H.parameters():
                        p.requires_grad_(False)
                    for p in D_R.parameters():
                        p.requires_grad_(False)
                    with autocast_ctx():
                        fake_R = G_H2R(real_H)
                        fake_H = G_R2H(real_R)

                        cycled_H = G_R2H(fake_R)
                        cycled_R = G_H2R(fake_H)
                        loss_cycle = cycle_loss(real_H, cycled_H) + cycle_loss(real_R, cycled_R)

                        loss_id = identity_loss(real_H, G_R2H(real_H)) + identity_loss(real_R, G_H2R(real_R))

                        if warmup_phase:
                            loss_G = (
                                args.lambda_cycle * loss_cycle
                                + args.lambda_identity * loss_id
                            )
                        else:
                            pred_fake_H = D_H(fake_H)
                            pred_fake_R = D_R(fake_R)
                            loss_G_H_adv = adv_loss(pred_fake_H, torch.ones_like(pred_fake_H))
                            loss_G_R_adv = adv_loss(pred_fake_R, torch.ones_like(pred_fake_R))

                            loss_G = (
                                loss_G_H_adv
                                + loss_G_R_adv
                                + args.lambda_cycle * loss_cycle
                                + args.lambda_identity * loss_id
                            )

                    opt_G.zero_grad()
                    if use_amp:
                        scaler_G.scale(loss_G).backward()
                        scaler_G.step(opt_G)
                        scaler_G.update()
                    else:
                        loss_G.backward()
                        opt_G.step()
                    for p in D_H.parameters():
                        p.requires_grad_(True)
                    for p in D_R.parameters():
                        p.requires_grad_(True)

                    gen_loss_acc += float(loss_G.detach().cpu())

                avg_d_loss = disc_loss_acc / max(1, args.d_steps)
                avg_g_loss = gen_loss_acc / max(1, args.g_steps)

                loop.set_postfix(G_loss=avg_g_loss, D_loss=avg_d_loss)
                running_G += avg_g_loss
                running_D += avg_d_loss
                batch_count += 1
                if profiler is not None:
                    profiler.step()

            history.append(
                {
                    "epoch": epoch,
                    "gen_loss": running_G / max(1, batch_count),
                    "disc_loss": running_D / max(1, batch_count),
                }
            )
            _save_history_file(history_path, history)

            # --- Save samples (deterministic cached pair) ---
            if epoch % args.save_samples_every == 0:
                G_H2R.eval()
                G_R2H.eval()
                with torch.no_grad():
                    he_cpu = fixed_he.to(device, non_blocking=True)
                    ret_cpu = fixed_ret.to(device, non_blocking=True)
                    fake_R_sample = G_H2R(he_cpu)
                    fake_H_sample = G_R2H(ret_cpu)
                he_pair = torch.cat([he_cpu.cpu(), fake_R_sample.cpu()], dim=0)
                ret_pair = torch.cat([ret_cpu.cpu(), fake_H_sample.cpu()], dim=0)
                save_image(
                    he_pair * 0.5 + 0.5,
                    str(run_samples_dir / f"he_to_ret_epoch{epoch}.jpg"),
                    nrow=preview_count,
                )
                save_image(
                    ret_pair * 0.5 + 0.5,
                    str(run_samples_dir / f"ret_to_he_epoch{epoch}.jpg"),
                    nrow=preview_count,
                )
                G_H2R.train()
                G_R2H.train()

            # --- Persist loss curve every epoch ---
            loss_plot_path = logs_dir / f"{args.run_id}_loss_curve.png"
            _save_loss_plot(history, loss_plot_path)

            # --- Save checkpoints ---
            if epoch % args.save_model_every == 0:
                save_checkpoint(G_H2R, opt_G, scaler_G, run_ckpt_dir / f"G_H2R_epoch{epoch}.pth.tar")
                save_checkpoint(G_R2H, opt_G, scaler_G, run_ckpt_dir / f"G_R2H_epoch{epoch}.pth.tar")
                save_checkpoint(D_H,   opt_D, scaler_D, run_ckpt_dir / f"D_H_epoch{epoch}.pth.tar")
                save_checkpoint(D_R,   opt_D, scaler_D, run_ckpt_dir / f"D_R_epoch{epoch}.pth.tar")

    logging.info("Training complete")
    loss_plot_path = logs_dir / f"{args.run_id}_loss_curve.png"
    _save_history_file(history_path, history)
    _save_loss_plot(history, loss_plot_path)
    if history:
        logging.info(f"Final loss curves available at {loss_plot_path}")
    else:
        logging.info("No training history recorded; skipping final loss plot.")


# ==============================================================================
# CLI helpers
# ==============================================================================
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train CycleGAN for H&E ↔ Reticulin translation")

    # Data/dataloader
    parser.add_argument(
        "--metadata",
        type=str,
        default=str(PROJECT_ROOT / METADATA_CSV),
        help="Path to metadata.csv",
    )
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--prefetch-factor", type=int, default=4, help="Ignored if --num-workers 0")
    parser.add_argument("--subset", type=float, default=None, help="Use only X%% of metadata (for quick runs)")

    # Training
    parser.add_argument("--run-id", type=str, required=True, help="Unique run name")
    parser.add_argument("--num-epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=None, help="Legacy base learning rate (overrides lr_disc and lr_gen when provided).")
    parser.add_argument("--lr-gen", type=float, default=4e-4, help="Generator learning rate.")
    parser.add_argument("--lr-disc", type=float, default=2e-4, help="Discriminator learning rate.")
    parser.add_argument("--lambda-cycle", type=float, default=10.0)
    parser.add_argument("--lambda-identity", type=float, default=5.0)
    parser.add_argument("--g-steps", type=int, default=1, help="Generator updates per iteration.")
    parser.add_argument("--d-steps", type=int, default=1, help="Discriminator updates per iteration.")
    parser.add_argument(
        "--gen-warmup-epochs",
        type=int,
        default=0,
        help="Train generators with only cycle/identity losses for this many initial epochs.",
    )
    parser.add_argument(
        "--instance-noise",
        action="store_true",
        help="Enable annealed instance noise for discriminator inputs.",
    )
    parser.add_argument(
        "--instance-noise-sigma",
        type=float,
        default=0.04,
        help="Initial noise standard deviation when instance noise is enabled.",
    )
    parser.add_argument(
        "--instance-noise-epochs",
        type=int,
        default=5,
        help="Number of epochs over which to anneal the noise to zero.",
    )

    # Logging/checkpoints
    parser.add_argument(
        "--checkpoints-dir",
        type=str,
        default=str(DEFAULT_CHECKPOINTS_DIR),
        help="Directory to store model checkpoints",
    )
    parser.add_argument(
        "--logs-dir",
        type=str,
        default=str(DEFAULT_LOGS_DIR),
        help="Directory for training logs",
    )
    parser.add_argument(
        "--samples-dir",
        type=str,
        default=str(DEFAULT_SAMPLES_DIR),
        help="Directory for sample outputs",
    )
    parser.add_argument("--save-model-every", type=int, default=1)
    parser.add_argument("--save-samples-every", type=int, default=1)
    parser.add_argument(
        "--resume-dir",
        type=str,
        default=None,
        help="Directory containing previous checkpoints to resume from.",
    )
    parser.add_argument(
        "--resume-epoch",
        type=int,
        default=None,
        help="Epoch to resume from (defaults to latest available in --resume-dir).",
    )
    # Profiler
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Enable PyTorch profiler around the training loop.",
    )
    parser.add_argument(
        "--profile-dir",
        type=str,
        default=str(DEFAULT_PROFILER_DIR),
        help="Directory where profiler traces (TensorBoard) should be stored.",
    )
    parser.add_argument("--profile-wait", type=int, default=1, help="Profiler schedule wait steps.")
    parser.add_argument("--profile-warmup", type=int, default=1, help="Profiler warmup steps.")
    parser.add_argument("--profile-active", type=int, default=3, help="Profiler active steps.")
    parser.add_argument("--profile-repeat", type=int, default=1, help="Profiler schedule repeats.")
    parser.add_argument("--profile-skip-first", type=int, default=0, help="Profiler steps to skip before scheduling.")
    parser.add_argument(
        "--profile-record-shapes",
        action="store_true",
        help="Record tensor shapes in profiler traces.",
    )
    parser.add_argument(
        "--profile-memory",
        action="store_true",
        help="Track memory usage in profiler traces.",
    )
    parser.add_argument(
        "--profile-with-stack",
        action="store_true",
        help="Capture Python stack traces in profiler events.",
    )

    # Misc
    parser.add_argument("--legacy-root", action="append", default=[], help="Additional directories to resolve patch paths from.")
    parser.add_argument("--seed", type=int, default=42)
    return parser


def _resolve_path(path_str: str) -> str:
    return str(resolve_path(path_str, allow_missing=True))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.metadata = _resolve_path(args.metadata)
    args.checkpoints_dir = _resolve_path(args.checkpoints_dir)
    args.logs_dir = _resolve_path(args.logs_dir)
    args.samples_dir = _resolve_path(args.samples_dir)
    args.profile_dir = _resolve_path(args.profile_dir)
    args.resume_dir = _resolve_path(args.resume_dir) if args.resume_dir else None
    args.legacy_root = [ _resolve_path(p) for p in args.legacy_root ]
    if args.lr is not None:
        args.lr_disc = args.lr
        args.lr_gen = args.lr * 2
    if args.lr_disc <= 0 or args.lr_gen <= 0:
        raise ValueError("Learning rates must be positive.")
    if args.gen_warmup_epochs < 0:
        raise ValueError("--gen-warmup-epochs cannot be negative.")
    if args.gen_warmup_epochs > args.num_epochs:
        raise ValueError("--gen-warmup-epochs cannot exceed --num-epochs.")
    if args.instance_noise:
        if args.instance_noise_sigma <= 0:
            raise ValueError("--instance-noise-sigma must be positive when instance noise is enabled.")
        if args.instance_noise_epochs <= 0:
            raise ValueError("--instance-noise-epochs must be positive when instance noise is enabled.")
    if args.profile:
        schedule_fields = [
            ("--profile-wait", args.profile_wait),
            ("--profile-warmup", args.profile_warmup),
            ("--profile-active", args.profile_active),
            ("--profile-repeat", args.profile_repeat),
            ("--profile-skip-first", args.profile_skip_first),
        ]
        for flag, value in schedule_fields:
            if value < 0:
                raise ValueError(f"{flag} cannot be negative when profiling is enabled.")
        if args.profile_active == 0:
            raise ValueError("--profile-active must be positive when profiling is enabled.")
        if args.profile_repeat == 0:
            raise ValueError("--profile-repeat must be positive when profiling is enabled.")
    return args


if __name__ == "__main__":
    main(parse_args())

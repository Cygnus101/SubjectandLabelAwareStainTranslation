#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Evaluate a CycleGAN generator checkpoint using FID and SSIM."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from PIL import Image, UnidentifiedImageError

import torch
from torch.utils.data import DataLoader
from torchvision import transforms as T

try:
    from torchmetrics.image.fid import FrechetInceptionDistance
    from torchmetrics.image.ssim import StructuralSimilarityIndexMeasure
except ImportError as exc:  # pragma: no cover - torchmetrics may not be installed
    raise SystemExit("torchmetrics is required for this script (pip install torchmetrics)") from exc

SCRIPT_DIR = Path(__file__).resolve().parent
SRC_ROOT = SCRIPT_DIR.parent
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from utils.path import ensure_project_root_on_syspath

ensure_project_root_on_syspath()

from src.models.Backbone_model.CycleGAN import UNetGenerator as Generator

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

SUPPORTED_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


class FlatImageDataset(torch.utils.data.Dataset):
    def __init__(self, root: Path, transform: T.Compose) -> None:
        self.root = root
        self.transform = transform
        if not self.root.exists():
            raise FileNotFoundError(f"Directory not found: {self.root}")
        self.paths = [p for p in self.root.rglob("*") if p.suffix.lower() in SUPPORTED_SUFFIXES]
        if not self.paths:
            raise FileNotFoundError(f"No supported images found in {self.root}")

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        attempts = 0
        while attempts < 3:
            path = self.paths[idx]
            try:
                with path.open("rb") as fh:
                    with Image.open(fh) as img:  # type: ignore[name-defined]
                        img = img.convert("RGB")
                        return self.transform(img)
            except (UnidentifiedImageError, OSError) as exc:
                logging.warning("Skipping unreadable file %s (%s)", path, type(exc).__name__)
                attempts += 1
                idx = (idx + 1) % len(self.paths)
        raise UnidentifiedImageError(f"Too many unreadable files encountered near index {idx}")


def _build_loader(root: Path, image_size: int, batch_size: int, num_workers: int) -> DataLoader:
    tfm = T.Compose(
        [
            T.Resize(image_size, interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(image_size),
            T.ToTensor(),
            T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    dataset = FlatImageDataset(root, tfm)
    return DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True)


def _denorm(x: torch.Tensor) -> torch.Tensor:
    return x * 0.5 + 0.5


def _to_uint8(x: torch.Tensor) -> torch.Tensor:
    return (x.clamp(0.0, 1.0) * 255.0).to(torch.uint8)


def compute_fid_ssim(
    generator: torch.nn.Module,
    source_loader: DataLoader,
    target_loader: DataLoader,
    device: torch.device,
) -> tuple[float, float]:
    fid = FrechetInceptionDistance(normalize=False)
    fid = fid.to(device if device.type == "cuda" else "cpu")  # mps fallback to CPU
    if device.type == "mps":
        logging.info("FID computed on CPU due to MPS float64 limitations.")
    ssim_device = device if device.type != "mps" else torch.device("cpu")
    ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(ssim_device)

    with torch.no_grad():
        for batch in target_loader:
            fid.update(_to_uint8(_denorm(batch.to(fid.device))), real=True)

    target_iter = iter(target_loader)
    generator.eval()
    with torch.no_grad():
        for source_batch in source_loader:
            source_batch = source_batch.to(device)
            fake = generator(source_batch)
            fid.update(_to_uint8(_denorm(fake.to(fid.device))), real=False)

            try:
                target_batch = next(target_iter)
            except StopIteration:
                target_iter = iter(target_loader)
                target_batch = next(target_iter)

            target_batch = target_batch.to(device)
            b = min(fake.size(0), target_batch.size(0))
            ssim.update(_denorm(fake[:b].to(ssim.device)), _denorm(target_batch[:b].to(ssim.device)))

    return float(fid.compute().cpu()), float(ssim.compute().cpu())


def load_generator(checkpoint_path: Path, device: torch.device) -> Generator:
    gen = Generator().to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    gen.load_state_dict(ckpt["state_dict"])
    logging.info("Loaded checkpoint %s", checkpoint_path)
    return gen


def _default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute FID/SSIM for a CycleGAN generator.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--target-root", type=Path, required=True)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", type=str, default=_default_device())
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    generator = load_generator(args.checkpoint, device)

    source_loader = _build_loader(args.source_root, args.image_size, args.batch_size, args.num_workers)
    target_loader = _build_loader(args.target_root, args.image_size, args.batch_size, args.num_workers)

    fid, ssim = compute_fid_ssim(generator, source_loader, target_loader, device)
    logging.info("FID: %.4f | SSIM: %.4f", fid, ssim)


if __name__ == "__main__":
    main()

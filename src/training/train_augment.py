#!/usr/bin/env python3
"""Train Augmented CycleGAN with slide-level classification + contrastive losses."""

from __future__ import annotations

import argparse
import json
import logging
import random
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import pandas as pd
from PIL import Image, UnidentifiedImageError
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
from tqdm.auto import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
SRC_ROOT = SCRIPT_DIR.parent
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from utils.path import ensure_project_root_on_syspath, get_project_root, resolve_path  # noqa: E402

ensure_project_root_on_syspath()
PROJECT_ROOT = get_project_root()

LOGGER = logging.getLogger("train_augment")
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
GRADE_PATTERN = re.compile(r"(\d+)")


try:  # pragma: no cover - fall back to the repo-local definitions
    from models.Backbone_model.CycleGANv2 import UNetGenerator as CycleGenerator  # type: ignore
    from models.Backbone_model.CycleGANv2 import Discriminator as CycleDiscriminator  # type: ignore

except:
    from models.Backbone_model.CycleGANv3 import UNetGenerator as CycleGenerator  # type: ignore
    from models.Backbone_model.CycleGANv3 import Discriminator as CycleDiscriminator  # type: ignore

try:
    from models.Feature_Extractor.Resnet50 import SimCLRModel as FeatureExtractor  # type: ignore
except Exception:  # pragma: no cover
    FeatureExtractor = None  # type: ignore

from models.Feature_Extractor import MODEL_REGISTRY as ENCODER_REGISTRY  # type: ignore
from models.Classification.abmil import ABMIL, SlideClassifier  # noqa: E402


@dataclass
class SlideRecord:
    slide_id: str
    grade: int
    he_paths: list[Path]
    ret_paths: list[Path]


def _parse_grade(value: object) -> Optional[int]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    text = str(value).strip()
    if not text:
        return None
    match = GRADE_PATTERN.search(text)
    if not match:
        return None
    grade = int(match.group(1))
    return max(0, min(3, grade))


def _build_patch_transform(image_size: int, augment: bool) -> T.Compose:
    ops: list[nn.Module] = [T.Resize(image_size), T.CenterCrop(image_size)]
    if augment:
        ops.extend([T.RandomHorizontalFlip(), T.RandomVerticalFlip(), T.RandomRotation(10)])
    ops.extend([T.ToTensor(), T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
    return T.Compose(ops)


class SlidePatchDataset(Dataset):
    """Dataset returning N patch stacks per slide for both stains."""

    def __init__(
        self,
        metadata_csv: Path,
        data_root: Optional[Path],
        patches_per_slide: int,
        image_size: int,
        augment: bool,
        seed: int,
        max_slides: Optional[int] = None,
        patch_retries: int = 8,
    ) -> None:
        self.metadata_path = resolve_path(metadata_csv)
        self.patch_root = resolve_path(data_root or self.metadata_path.parent)
        self.patches_per_slide = patches_per_slide
        self.transform = _build_patch_transform(image_size, augment)
        self.patch_retries = max(1, patch_retries)
        self.rng = random.Random(seed)
        self.entries = self._build_entries(max_slides)
        if not self.entries:
            raise RuntimeError("No slides with both H&E and Reticulin patches were found.")

    def _normalize_path(self, raw: str) -> Path:
        normalized = raw.replace("\\", "/")
        path = Path(normalized)
        if path.is_absolute():
            return path
        return (self.patch_root / path).resolve()

    def _build_entries(self, max_slides: Optional[int]) -> list[SlideRecord]:
        df = pd.read_csv(self.metadata_path)
        if {"stain_id", "type", "patch_path", "Reticulin Grade"} - set(df.columns):
            missing = {"stain_id", "type", "patch_path", "Reticulin Grade"} - set(df.columns)
            raise ValueError(f"metadata missing required columns: {sorted(missing)}")

        df["type_norm"] = df["type"].astype(str).str.lower()
        df["patch_path_norm"] = df["patch_path"].astype(str)
        ret_df = df[df["type_norm"].str.contains("reticulin", na=False)]
        he_df = df[df["type_norm"].str.contains("h&e", na=False)]
        ret_groups = ret_df.groupby("stain_id")
        he_groups = he_df.groupby("stain_id")

        entries: list[SlideRecord] = []
        for slide_id, ret_group in ret_groups:
            if slide_id not in he_groups.groups:
                continue
            grade_value = _parse_grade(ret_group["Reticulin Grade"].iloc[0])
            if grade_value is None:
                continue
            he_group = he_groups.get_group(slide_id)
            he_paths = [self._normalize_path(p) for p in he_group["patch_path_norm"].tolist()]
            ret_paths = [self._normalize_path(p) for p in ret_group["patch_path_norm"].tolist()]
            if not he_paths or not ret_paths:
                continue
            entries.append(SlideRecord(slide_id=str(slide_id), grade=grade_value, he_paths=he_paths, ret_paths=ret_paths))
            if max_slides is not None and len(entries) >= max_slides:
                break
        return entries

    def __len__(self) -> int:
        return len(self.entries)

    def _load_stack(self, paths: list[Path]) -> torch.Tensor:
        images: list[torch.Tensor] = []
        pool = list(paths)
        if not pool:
            raise RuntimeError("Slide has no available patches.")
        indices: list[int]
        if len(pool) >= self.patches_per_slide:
            indices = self.rng.sample(range(len(pool)), self.patches_per_slide)
        else:
            indices = [self.rng.randrange(len(pool)) for _ in range(self.patches_per_slide)]
        attempts = 0
        max_attempts = self.patch_retries * self.patches_per_slide
        ptr = 0
        while len(images) < self.patches_per_slide and attempts < max_attempts:
            path = pool[indices[ptr % len(indices)]]
            ptr += 1
            attempts += 1
            try:
                with Image.open(path) as img:
                    tensor = self.transform(img.convert("RGB"))
            except (FileNotFoundError, UnidentifiedImageError, OSError) as exc:
                LOGGER.warning("Failed to load patch %s (%s)", path, exc)
                continue
            images.append(tensor)
        if len(images) < self.patches_per_slide:
            raise RuntimeError(f"Unable to collect {self.patches_per_slide} patches from {paths[0].parent}")
        return torch.stack(images, dim=0)

    def __getitem__(self, idx: int):
        record = self.entries[idx]
        he_stack = self._load_stack(record.he_paths)
        ret_stack = self._load_stack(record.ret_paths)
        grade = torch.tensor(record.grade, dtype=torch.long)
        return he_stack, ret_stack, grade, record.slide_id


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


class ProjectionHead(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def freeze_module(module: nn.Module) -> nn.Module:
    module.eval()
    for param in module.parameters():
        param.requires_grad_(False)
    return module


def _load_state_dict(module: nn.Module, checkpoint_path: Path, key: Optional[str] = None) -> None:
    state = torch.load(checkpoint_path, map_location="cpu")
    if key and key in state:
        payload = state[key]
    elif isinstance(state, dict) and "state_dict" in state:
        payload = state["state_dict"]
    else:
        payload = state
    module.load_state_dict(payload, strict=False)


def build_feature_extractor(args: argparse.Namespace, device: torch.device) -> nn.Module:
    if FeatureExtractor is not None:
        model = FeatureExtractor()  # type: ignore[call-arg]
        _load_state_dict(model, resolve_path(args.pretrained_encoder))
    else:
        ctor = ENCODER_REGISTRY.get(args.encoder_backbone.lower())
        if ctor is None:
            raise ValueError(f"Unknown encoder backbone '{args.encoder_backbone}'.")
        model = ctor(
            pretrained=False,
            proj_hidden_dim=args.encoder_proj_hidden,
            proj_out_dim=args.encoder_proj_out,
        )
        state = torch.load(resolve_path(args.pretrained_encoder), map_location="cpu")
        if "model" in state:
            model.load_state_dict(state["model"])
        elif "backbone" in state:
            model.backbone.load_state_dict(state["backbone"])
        else:
            model.load_state_dict(state)
        model.projector = nn.Identity()
    return freeze_module(model).to(device)


def forward_encoder_features(encoder: nn.Module, images: torch.Tensor) -> torch.Tensor:
    outputs = encoder(images)
    if isinstance(outputs, tuple):
        return outputs[0]
    return outputs


def normalize_for_encoder(patches: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    flat = patches.view(-1, *patches.shape[2:])
    flat = (flat + 1.0) * 0.5
    return (flat - mean) / std


def extract_patch_features(
    encoder: nn.Module,
    patches: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
    require_grad: bool,
) -> torch.Tensor:
    normalized = normalize_for_encoder(patches, mean, std)
    if require_grad:
        feats = forward_encoder_features(encoder, normalized)
    else:
        with torch.no_grad():
            feats = forward_encoder_features(encoder, normalized)
    feat_dim = feats.shape[1]
    return feats.view(patches.size(0), patches.size(1), feat_dim)


def compute_classification_loss(
    z_fake: torch.Tensor,
    grades: torch.Tensor,
    abmil: ABMIL,
    classifier: SlideClassifier,
    criterion: nn.Module,
) -> tuple[torch.Tensor, torch.Tensor]:
    logits: list[torch.Tensor] = []
    for idx in range(z_fake.size(0)):
        bag = z_fake[idx]
        slide_vec, _ = abmil.pool(bag)
        logit = classifier(slide_vec.unsqueeze(0))
        logits.append(logit)
    stacked = torch.cat(logits, dim=0)
    loss = criterion(stacked, grades)
    return loss, stacked


def compute_contrastive_loss(
    z_real: torch.Tensor,
    z_fake: torch.Tensor,
    projector: ProjectionHead,
    temperature: float,
) -> torch.Tensor:
    bsz = z_real.size(0)
    e_real = z_real.mean(dim=1)
    e_fake = z_fake.mean(dim=1)
    q_real = F.normalize(projector(e_real), dim=1)
    q_fake = F.normalize(projector(e_fake), dim=1)
    embeddings = torch.cat([q_real, q_fake], dim=0)
    sim = embeddings @ embeddings.t() / temperature
    mask = torch.eye(sim.size(0), device=sim.device, dtype=torch.bool)
    sim = sim.masked_fill(mask, float("-inf"))

    losses: list[torch.Tensor] = []
    for idx in range(bsz):
        pos = idx + bsz
        losses.append(F.logsumexp(sim[idx], dim=0) - sim[idx, pos])
    for idx in range(bsz, 2 * bsz):
        pos = idx - bsz
        losses.append(F.logsumexp(sim[idx], dim=0) - sim[idx, pos])
    return torch.stack(losses).mean()


def load_cyclegan_weights(root: Path, generators: tuple[nn.Module, nn.Module], discriminators: tuple[nn.Module, nn.Module]) -> None:
    root = resolve_path(root)
    G_H2R, G_R2H = generators
    D_R, D_H = discriminators
    if root.is_file():
        checkpoint = torch.load(root, map_location="cpu")
        for key, module in {
            "G_H2R": G_H2R,
            "G_R2H": G_R2H,
            "D_R": D_R,
            "D_H": D_H,
        }.items():
            state = checkpoint.get(key)
            if state is None:
                continue
            module.load_state_dict(state)
        return

    def latest(prefix: str) -> Path:
        pattern = re.compile(rf"{re.escape(prefix)}_epoch(\d+)\.")
        best_epoch = -1
        best_path: Optional[Path] = None
        for path in root.glob(f"{prefix}_epoch*.pth*"):
            match = pattern.match(path.name)
            if not match:
                continue
            epoch = int(match.group(1))
            if epoch > best_epoch:
                best_epoch = epoch
                best_path = path
        if best_path is None:
            raise FileNotFoundError(f"No checkpoint found for {prefix} under {root}")
        return best_path

    for name, module in [("G_H2R", G_H2R), ("G_R2H", G_R2H), ("D_R", D_R), ("D_H", D_H)]:
        ckpt_path = latest(name)
        state = torch.load(ckpt_path, map_location="cpu")
        payload = state.get("state_dict", state)
        module.load_state_dict(payload)


def adversarial_loss(discriminator: nn.Module, real: torch.Tensor, fake: torch.Tensor, criterion: nn.Module) -> tuple[torch.Tensor, torch.Tensor]:
    pred_real = discriminator(real)
    pred_fake = discriminator(fake.detach())
    loss_real = criterion(pred_real, torch.ones_like(pred_real))
    loss_fake = criterion(pred_fake, torch.zeros_like(pred_fake))
    return (loss_real + loss_fake) * 0.5, pred_real


def infer_feature_dim(encoder: nn.Module, sample: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> int:
    with torch.no_grad():
        feats = forward_encoder_features(encoder, normalize_for_encoder(sample, mean, std))
    return feats.shape[1]


def save_checkpoints(run_dir: Path, epoch: int, modules: dict[str, nn.Module]) -> None:
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    for name, module in modules.items():
        path = ckpt_dir / f"{name}_epoch{epoch}.pt"
        torch.save({"state_dict": module.state_dict()}, path)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Augmented CycleGAN training")
    parser.add_argument("--metadata", type=str, default=str(PROJECT_ROOT / "metadata.csv"))
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--batch-slides", type=int, default=2)
    parser.add_argument("--patches-per-slide", type=int, default=32)
    parser.add_argument("--lambda-cycle", type=float, default=10.0)
    parser.add_argument("--lambda-identity", type=float, default=5.0)
    parser.add_argument("--lambda-cls", type=float, default=1.0)
    parser.add_argument("--lambda-con", type=float, default=1.0)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr-gen", type=float, default=2e-4)
    parser.add_argument("--lr-disc", type=float, default=2e-4)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--pretrained-cyclegan", type=str, required=True)
    parser.add_argument("--pretrained-encoder", type=str, required=True)
    parser.add_argument("--pretrained-abmil", type=str, required=True)
    parser.add_argument("--pretrained-classifier", type=str, required=True)
    parser.add_argument("--log-dir", type=str, default=str(PROJECT_ROOT / "outputs" / "logs" / "augmented_cyclegan"))
    parser.add_argument("--run-id", type=str, required=True)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--encoder-backbone", type=str, default="resnet50", choices=sorted(ENCODER_REGISTRY.keys()))
    parser.add_argument("--encoder-proj-hidden", type=int, default=2048)
    parser.add_argument("--encoder-proj-out", type=int, default=128)
    parser.add_argument("--abmil-attn-dim", type=int, default=256)
    parser.add_argument("--classifier-hidden-dim", type=int, default=512)
    parser.add_argument("--abmil-dropout", type=float, default=0.25)
    parser.add_argument("--num-classes", type=int, default=4)
    parser.add_argument("--proj-hidden-dim", type=int, default=512)
    parser.add_argument("--proj-out-dim", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--save-every", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-augment", action="store_true")
    parser.add_argument("--max-slide-count", type=int, default=None)
    parser.add_argument("--patch-retries", type=int, default=8)
    return parser.parse_args(argv)


def setup_logging(run_dir: Path) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    log_file = run_dir / "train.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file, mode="w"),
        ],
    )
    LOGGER.info("Logging to %s", log_file)


def determine_device(requested: Optional[str]) -> torch.device:
    if requested:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


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

    feature_extractor = build_feature_extractor(args, device)
    mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, -1, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=device).view(1, -1, 1, 1)

    sample_he, sample_ret, _, _ = dataset[0]
    feat_dim = infer_feature_dim(feature_extractor, sample_ret.unsqueeze(0).to(device), mean, std)
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
    load_cyclegan_weights(Path(args.pretrained_cyclegan), (G_H2R, G_R2H), (D_R, D_H))

    opt_G = torch.optim.Adam(
        list(G_H2R.parameters()) + list(G_R2H.parameters()) + list(projector.parameters()),
        lr=args.lr_gen,
        betas=(0.5, 0.999),
    )
    opt_D = torch.optim.Adam(
        list(D_R.parameters()) + list(D_H.parameters()),
        lr=args.lr_disc,
        betas=(0.5, 0.999),
    )

    criterion_gan = nn.MSELoss()
    l1_loss = nn.L1Loss()
    criterion_cls = nn.CrossEntropyLoss()

    history: list[dict[str, float]] = []

    for epoch in range(1, args.epochs + 1):
        G_H2R.train()
        G_R2H.train()
        D_R.train()
        D_H.train()
        projector.train()
        epoch_metrics = {
            "G": 0.0,
            "D": 0.0,
            "cls": 0.0,
            "con": 0.0,
            "cycle": 0.0,
            "identity": 0.0,
            "adv": 0.0,
        }
        batches = 0
        for he_batch, ret_batch, grades, slide_ids in tqdm(loader, desc=f"Epoch {epoch}"):
            batches += 1
            he_batch = he_batch.to(device)
            ret_batch = ret_batch.to(device)
            grades = grades.to(device)

            he_flat = he_batch.view(-1, *he_batch.shape[2:])
            ret_flat = ret_batch.view(-1, *ret_batch.shape[2:])

            fake_ret = G_H2R(he_flat).view_as(ret_batch)
            fake_he = G_R2H(ret_flat).view_as(he_batch)

            rec_he = G_R2H(fake_ret.view(-1, *fake_ret.shape[2:])).view_as(he_batch)
            rec_ret = G_H2R(fake_he.view(-1, *fake_he.shape[2:])).view_as(ret_batch)

            id_ret = G_H2R(ret_flat).view_as(ret_batch)
            id_he = G_R2H(he_flat).view_as(he_batch)

            pred_fake_ret = D_R(fake_ret.view(-1, *fake_ret.shape[2:]))
            pred_fake_he = D_H(fake_he.view(-1, *fake_he.shape[2:]))
            adv_loss = criterion_gan(pred_fake_ret, torch.ones_like(pred_fake_ret)) + criterion_gan(
                pred_fake_he, torch.ones_like(pred_fake_he)
            )

            cycle_loss = l1_loss(rec_he, he_batch) + l1_loss(rec_ret, ret_batch)
            id_loss = l1_loss(id_ret, ret_batch) + l1_loss(id_he, he_batch)

            z_fake = extract_patch_features(feature_extractor, fake_ret, mean, std, require_grad=True)
            z_real = extract_patch_features(feature_extractor, ret_batch, mean, std, require_grad=False)

            cls_loss, _ = compute_classification_loss(z_fake, grades, abmil, slide_classifier, criterion_cls)
            contrastive_loss = compute_contrastive_loss(z_real, z_fake, projector, args.temperature)

            loss_G = (
                adv_loss
                + args.lambda_cycle * cycle_loss
                + args.lambda_identity * id_loss
                + args.lambda_cls * cls_loss
                + args.lambda_con * contrastive_loss
            )

            opt_G.zero_grad(set_to_none=True)
            loss_G.backward()
            opt_G.step()

            D_R_loss, _ = adversarial_loss(D_R, ret_flat, fake_ret.view(-1, *fake_ret.shape[2:]), criterion_gan)
            D_H_loss, _ = adversarial_loss(D_H, he_flat, fake_he.view(-1, *fake_he.shape[2:]), criterion_gan)
            loss_D = D_R_loss + D_H_loss

            opt_D.zero_grad(set_to_none=True)
            loss_D.backward()
            opt_D.step()

            epoch_metrics["G"] += loss_G.item()
            epoch_metrics["D"] += loss_D.item()
            epoch_metrics["cls"] += cls_loss.item()
            epoch_metrics["con"] += contrastive_loss.item()
            epoch_metrics["cycle"] += cycle_loss.item()
            epoch_metrics["identity"] += id_loss.item()
            epoch_metrics["adv"] += adv_loss.item()

        for key in epoch_metrics:
            epoch_metrics[key] /= max(1, batches)

        LOGGER.info(
            "Epoch %d | G: %.4f | D: %.4f | Adv: %.4f | Cycle: %.4f | Id: %.4f | Cls: %.4f | Con: %.4f",
            epoch,
            epoch_metrics["G"],
            epoch_metrics["D"],
            epoch_metrics["adv"],
            epoch_metrics["cycle"],
            epoch_metrics["identity"],
            epoch_metrics["cls"],
            epoch_metrics["con"],
        )

        history.append({"epoch": epoch, **epoch_metrics})
        history_path = run_dir / "history.json"
        with history_path.open("w", encoding="utf-8") as fp:
            json.dump(history, fp, indent=2)

        if epoch % args.save_every == 0 or epoch == args.epochs:
            save_checkpoints(
                run_dir,
                epoch,
                {
                    "G_H2R": G_H2R,
                    "G_R2H": G_R2H,
                    "Proj": projector,
                },
            )

    LOGGER.info("Training complete. History saved to %s", run_dir / "history.json")


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    train(args)


if __name__ == "__main__":
    main()

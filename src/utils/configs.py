from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import argparse
from pathlib import Path


@dataclass
class TrainAugmentConfig:
    metadata: str
    data_root: Optional[str] = None
    batch_slides: int = 2
    patches_per_slide: int = 32
    lambda_cycle: float = 10.0
    lambda_identity: float = 5.0
    lambda_cls: float = 1.0
    lambda_con: float = 1.0
    epochs: int = 50
    lr_gen: float = 2e-4
    lr_disc: float = 2e-4
    device: Optional[str] = None
    pretrained_cyclegan: Optional[str] = None
    pretrained_encoder: Optional[str] = None
    pretrained_abmil: Optional[str] = None
    pretrained_classifier: Optional[str] = None
    log_dir: Optional[str] = None
    run_id: Optional[str] = None
    num_workers: int = 4
    image_size: int = 256
    encoder_backbone: str = "resnet50"
    encoder_proj_hidden: int = 2048
    encoder_proj_out: int = 128
    abmil_attn_dim: int = 256
    classifier_hidden_dim: int = 512
    abmil_dropout: float = 0.25
    num_classes: int = 4
    proj_hidden_dim: int = 512
    proj_out_dim: int = 128
    temperature: float = 0.1
    save_every: int = 5
    seed: int = 42
    no_augment: bool = False
    max_slide_count: Optional[int] = None
    patch_retries: int = 8
    amp: bool = False
    reticulin_embedding_dir: Optional[str] = None
    contrastive_negatives: int = 32

    def build_parser(self) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(description="Augmented CycleGAN training")
        parser.add_argument("--metadata", type=str, default=self.metadata)
        parser.add_argument("--data-root", type=str, default=self.data_root)
        parser.add_argument("--batch-slides", type=int, default=self.batch_slides)
        parser.add_argument("--patches-per-slide", type=int, default=self.patches_per_slide)
        parser.add_argument("--lambda-cycle", type=float, default=self.lambda_cycle)
        parser.add_argument("--lambda-identity", type=float, default=self.lambda_identity)
        parser.add_argument("--lambda-cls", type=float, default=self.lambda_cls)
        parser.add_argument("--lambda-con", type=float, default=self.lambda_con)
        parser.add_argument("--epochs", type=int, default=self.epochs)
        parser.add_argument("--lr-gen", type=float, default=self.lr_gen)
        parser.add_argument("--lr-disc", type=float, default=self.lr_disc)
        parser.add_argument("--device", type=str, default=self.device)
        parser.add_argument("--pretrained-cyclegan", type=str, default=self.pretrained_cyclegan)
        parser.add_argument("--pretrained-encoder", type=str, default=self.pretrained_encoder)
        parser.add_argument("--pretrained-abmil", type=str, default=self.pretrained_abmil)
        parser.add_argument("--pretrained-classifier", type=str, default=self.pretrained_classifier)
        parser.add_argument("--log-dir", type=str, default=self.log_dir)
        parser.add_argument("--run-id", type=str, default=self.run_id)
        parser.add_argument("--num-workers", type=int, default=self.num_workers)
        parser.add_argument("--image-size", type=int, default=self.image_size)
        parser.add_argument("--encoder-backbone", type=str, default=self.encoder_backbone)
        parser.add_argument("--encoder-proj-hidden", type=int, default=self.encoder_proj_hidden)
        parser.add_argument("--encoder-proj-out", type=int, default=self.encoder_proj_out)
        parser.add_argument("--abmil-attn-dim", type=int, default=self.abmil_attn_dim)
        parser.add_argument("--classifier-hidden-dim", type=int, default=self.classifier_hidden_dim)
        parser.add_argument("--abmil-dropout", type=float, default=self.abmil_dropout)
        parser.add_argument("--num-classes", type=int, default=self.num_classes)
        parser.add_argument("--proj-hidden-dim", type=int, default=self.proj_hidden_dim)
        parser.add_argument("--proj-out-dim", type=int, default=self.proj_out_dim)
        parser.add_argument("--temperature", type=float, default=self.temperature)
        parser.add_argument("--save-every", type=int, default=self.save_every)
        parser.add_argument("--seed", type=int, default=self.seed)
        parser.add_argument("--no-augment", action="store_true", default=self.no_augment)
        parser.add_argument("--max-slide-count", type=int, default=self.max_slide_count)
        parser.add_argument("--patch-retries", type=int, default=self.patch_retries)
        parser.add_argument("--amp", action="store_true", default=self.amp)
        parser.add_argument("--reticulin-embedding-dir", type=str, default=self.reticulin_embedding_dir)
        parser.add_argument("--contrastive-negatives", type=int, default=self.contrastive_negatives)
        return parser

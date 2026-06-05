
#import necessary libraries
import torch
from torchvision import transforms
from PIL import Image
import argparse
from pathlib import Path


#code to call generator
from src.models.Backbone_model.CycleGANv3 import UNetGenerator, Discriminator


#code to generate images



#code to select evaluation metric



#code to evaluate images and return score



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Model by computing score from random images.")

    parser.add_argument("--model-selection", type=Path, default=None, help = "Select model type to evaluate. Options: 'checkpoint' or 'checkpoint-root'." \
")
    parser.add_argument("--checkpoint", type=Path, default=None, help="Path to a single CycleGAN H2R checkpoint.")
    parser.add_argument("--checkpoint-root", type=Path, default=None, help="Root directory to search for G_H2R checkpoints.")
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--patch-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=_default_device())
    parser.add_argument("--lab-col", type=str, default=None, help="Column used to group slides/patients.")
    return parser.parse_args()


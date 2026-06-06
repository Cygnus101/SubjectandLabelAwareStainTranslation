
#import necessary libraries
import torch
from torchvision import transforms
from PIL import Image
import argparse
from pathlib import Path
import json
import random
import logging


#code to call generator
from src.models.Backbone_model.CycleGANv3 import UNetGenerator

#parser arguments

def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(description="Evaluate Model by computing score from random images.")

    parser.add_argument("--model-selection", type=Path, default=None, help = '''Select model type to evaluate. 
                        Options: 'CycleGAN-Baseline' or 'CycleGAN-WLoss' or 'Aug-CycleGAN' ''')
    
    parser.add_argument("--checkpoint", type=Path, default=None, help="Path to a single CycleGAN H2R checkpoint.")

    parser.add_argument("--output-dir", type=Path, default= DEFAULT_OUTPUT_DIR)

    parser.add_argument("--patch-size", type=int, default=512)

    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--device", type=str, default=_default_device())

    parser.add_argument("--split", type=str, default=None, help="Split json generated from build_augmented_dataset.py")\
    
    parser.add_argument("--num-samples", type=int, default=1000, help="Number of random samples to evaluate on.")

    parser.add_argument("--metric", type=str, default="FID", help="Evaluation metric to use. Options: 'FID' or 'KID'.")

    parser.add_argument("--slide", type=int, default=4, help="Slide json generated from build_augmented_dataset.py")    

    return parser.parse_args()


def load_generator(checkpoint_path: Path, device: torch.device) -> UNetGenerator:
    generator = UNetGenerator().to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    if "G_H2R" in ckpt:
        state = ckpt["G_H2R"]
    elif "state_dict" in ckpt:
        state = ckpt["state_dict"]
    else:
        state = ckpt        # This is there because train_cycegan.py and train_aug_cyclegan.py
                            # save checkpoints in different formats. This is to ensure compatibility with both.
    generator.load_state_dict(state)
    logging.info("Loaded generator checkpoint %s", checkpoint_path)
    return generator

def main():

    args = parse_args()

    #code to select random images from dataset
    splits = json.loads(args.split.read_text())
    slides = json.loads(args.slide.read_text())

    test_slides = [slides[i] for i in splits["test_indices"]]

    he_paths = [
        Path(patch["patch_path"])
        for slide in test_slides
        for patch in slide["he_patches"]
    ]

    rng = random.Random(args.seed)
    selected_paths = rng.sample(he_paths, min(args.num_images, len(he_paths)))

    #code to generate images
    


    #code to select evaluation metric



    #code to evaluate images and return score








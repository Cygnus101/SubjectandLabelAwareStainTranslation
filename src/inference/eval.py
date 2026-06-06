
#import necessary libraries
import torch
from torchvision import transforms
from PIL import Image
import argparse
from pathlib import Path
import json
import random
import logging

from torchvision import transforms as T
from torchmetrics.image.fid import FrechetInceptionDistance
from tqdm.auto import tqdm
from torchvision.utils import save_image

#path resolution

PROJECT_ROOT = Path(__file__).resolve().parents[2]

def resolve_patch_path(raw, data_root=None):
    path = Path(raw).expanduser()
    if path.is_absolute():
        return path
    return ((data_root or PROJECT_ROOT) / path).resolve()


#device selection

def _default_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


#function to call generator

from src.models.Backbone_model.CycleGANv3 import UNetGenerator

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


#function to generate images
    
def generate_images(
    generator,
    selected_paths,
    patch_size,
    device,
):
        
    transform = T.Compose([
        T.Resize(patch_size),
        T.CenterCrop(patch_size),
        T.ToTensor(),
        T.Normalize([0.5] * 3, [0.5] * 3),
    ])


    generator.eval()
    generated_images = []


    with torch.no_grad():
        for index, image_path in enumerate(selected_paths):
            with Image.open(image_path) as image:
                image = image.convert("RGB")
                input_tensor = transform(image).unsqueeze(0).to(device)

            generated = generator(input_tensor)

            # Convert generator output from [-1, 1] to [0, 1].
            generated = (generated * 0.5 + 0.5).clamp(0, 1)

            # Move results off the GPU before accumulating them.
            generated_images.append(generated.cpu())
        
        if not generated_images:
            return torch.empty((0, 3, patch_size, patch_size))
        
        return torch.cat(generated_images, dim=0)


#function to load real images

def load_real_images(selected_paths, patch_size):
    transform = T.Compose([
        T.Resize(patch_size),
        T.CenterCrop(patch_size),
        T.ToTensor(),
    ])

    images = []

    for image_path in selected_paths:
        with Image.open(image_path) as image:
            images.append(transform(image.convert("RGB")))

    if not images:
        return torch.empty((0, 3, patch_size, patch_size))

    return torch.stack(images)

    

#function to select evaluation metric

def select_metric(metric_name: str):
    name = metric_name.upper()

    if name == "FID":
        from src.inference.metrics.FID import evaluate
    elif name == "KID":
        from src.inference.metrics.KID import evaluate
    else:
        raise ValueError(f"Unsupported metric: {metric_name}")

    return evaluate
    
#parser arguments

def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(description="Evaluate Model by computing score from random images.")

    parser.add_argument("--model-selection", type=str, default=None, help = '''Select model type to evaluate. 
                        Options: 'CycleGAN-Baseline' or 'CycleGAN-WLoss' or 'Aug-CycleGAN' ''')
    
    parser.add_argument("--checkpoint", type=Path, default=None, required=True, help="Path to a single CycleGAN H2R checkpoint.")

    parser.add_argument("--data-root", type=Path, default=None)

    parser.add_argument("--patch-size", type=int, default=512)

    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--device", type=str, default=_default_device())

    parser.add_argument("--split", type=Path, required=True,  help="Split json generated from build_augmented_dataset.py")\
    
    parser.add_argument("--num-images", type=int, default=1000, help="Number of random images to evaluate on.")

    parser.add_argument( "--metric",type=str.upper,choices=["FID", "KID"],default="FID", help="Evaluation metric to compute. Options: 'FID' or 'KID'. Default is 'FID'.")

    parser.add_argument("--slide", type=Path, required=True, help="Slide json generated from build_augmented_dataset.py")    

    return parser.parse_args()



def main():

    args = parse_args()
    device = torch.device(args.device)

    #code to select random images from dataset
    splits = json.loads(args.split.read_text())
    slides = json.loads(args.slide.read_text())

    test_slides = [slides[i] for i in splits["test_indices"]]

    he_paths = [
        Path(patch["patch_path"], args.data_root)
        for slide in test_slides
        for patch in slide["he_patches"]
    ]

    ret_paths = [
        Path(patch["patch_path"], args.data_root)
        for slide in test_slides
        for patch in slide["ret_patches"]
    ]

    rng = random.Random(args.seed)

    selected_paths = rng.sample(he_paths, min(args.num_images, len(he_paths)))
    selected_ret_paths = rng.sample(ret_paths, min(args.num_images, len(ret_paths)))

    if not selected_paths or not selected_ret_paths:
        raise RuntimeError("No H&E or Reticulin test images were found.")
    
    #code to load generator and generate images
    generator = load_generator(args.checkpoint, device)

    generated_images = generate_images(
        generator=generator,
        selected_paths=selected_paths,
        patch_size=args.patch_size,
        device=device,
    )

    real_images = load_real_images(
    selected_paths=selected_ret_paths,
    patch_size=args.patch_size,
    )

    print(f"Generated {len(generated_images)} images")
    print(f"Loaded {len(real_images)} real images")

    #code to select metric and compute score
    metric_fn = select_metric(args.metric)

    score = metric_fn(
        real_images=real_images,
        generated_images=generated_images,
        device=device,
    )


    print(f"{args.metric.upper()}: {score}")

if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)

    main()








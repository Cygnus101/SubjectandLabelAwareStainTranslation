"""Feature extractor registry for SimCLR backbones."""

from .Resnet50 import SimCLRModel as ResNet50SimCLRModel
from .ViT import SimCLRModel as ViTSimCLRModel

MODEL_REGISTRY = {
    "resnet50": ResNet50SimCLRModel,
    "vit": ViTSimCLRModel,
}

__all__ = ["MODEL_REGISTRY", "ResNet50SimCLRModel", "ViTSimCLRModel"]

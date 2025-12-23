"""Feature extractor registry for SimCLR backbones."""

from .Resnet50 import SimCLRModel as ResNet50SimCLRModel
from .Resnet18 import SimCLRModel as ResNet18SimCLRModel
from .ViT import SimCLRModel as ViTSimCLRModel

MODEL_REGISTRY = {
    "resnet50": ResNet50SimCLRModel,
    "resnet18": ResNet18SimCLRModel,
    "vit": ViTSimCLRModel,
}
__all__ = ["MODEL_REGISTRY", "ResNet18SimCLRModel", "ResNet50SimCLRModel", "ViTSimCLRModel"]

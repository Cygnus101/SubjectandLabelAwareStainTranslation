"""
Model definitions for the project.

Convenience imports allow consumers to reach key backbones/classifiers via
`from src.models import Feature_Extractor, Classification`.
"""

from . import Feature_Extractor, Classification  # noqa: F401

__all__ = ["Feature_Extractor", "Classification"]

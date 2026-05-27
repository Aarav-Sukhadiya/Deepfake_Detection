"""Model wrappers and a registry for swapping backbones."""
from .base import DeepfakeModel
from .registry import build_model, list_models, register_model

__all__ = ["DeepfakeModel", "build_model", "register_model", "list_models"]

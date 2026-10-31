"""Abstract base for any deepfake classifier."""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from .downloader import compute_sha256


class DeepfakeModel(ABC):
    """A face-crop → fake probability classifier."""

    name: str = "base"
    input_size: int = 224

    def __init__(self, weights_path: Path | None = None, device: str = "cpu"):
        self.weights_path = weights_path
        self.device = device

    @abstractmethod
    def preprocess(self, faces: list[Any]) -> Any:
        """Stack RGB face crops into a model-ready tensor."""

    @abstractmethod
    def predict(self, batch: Any) -> list[float]:
        """Return per-sample probability that the face is FAKE."""

    @abstractmethod
    def grad_cam_target_layer(self) -> Any:
        """Return the conv layer Grad-CAM should hook into."""

    def grad_cam_model(self) -> Any:
        """Return the underlying torch ``nn.Module`` that Grad-CAM should hook.

        Wrapper classes (calibration, ensemble) can override this to point at
        the network *inside* the wrapper, so callers don't reach into private
        ``_model`` attributes. Default: assume the subclass exposes a
        ``_model`` attribute pointing at a single nn.Module.
        """
        return getattr(self, "_model", None)

    def grad_cam_preprocess(self, faces: list[Any]) -> Any:
        """Tensor input for :meth:`grad_cam_model`. Defaults to :meth:`preprocess`;
        wrappers whose ``preprocess`` doesn't produce a tensor (ensembles)
        override this to use the network Grad-CAM actually hooks."""
        return self.preprocess(faces)

    def hash(self) -> str:
        """Hex digest of the loaded weights file (or 'pretrained' fallback).

        Memoised on the instance: a 100 MB checkpoint hashed on every call
        would dominate per-file overhead. ``weights_path`` is effectively
        immutable after model construction; call :meth:`_invalidate_hash` if
        you ever swap weights at runtime.
        """
        cached = getattr(self, "_hash_cached", None)
        if cached is None:
            if self.weights_path and Path(self.weights_path).exists():
                cached = compute_sha256(Path(self.weights_path))[:16]
            else:
                cached = "pretrained"
            self._hash_cached = cached
        return cached

    def _invalidate_hash(self) -> None:
        """Drop the cached hash so the next :meth:`hash` call re-reads."""
        if hasattr(self, "_hash_cached"):
            del self._hash_cached

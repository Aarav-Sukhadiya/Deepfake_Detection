"""EfficientNet-B4 wrapper. Uses timm; weights load from ``weights_path`` if
present, are downloaded via :mod:`dfdetect.models.downloader` when a URL is
configured, and otherwise fall back to ImageNet pretraining with a loud warning."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from .base import DeepfakeModel
from .downloader import ChecksumError, ensure_weights_file
from .registry import register_model

log = logging.getLogger(__name__)

# Module-level constants so the per-batch normalisation reuses the same
# arrays instead of allocating a fresh tuple→array conversion per call.
_MEAN = np.array((0.485, 0.456, 0.406), dtype=np.float32)
_INV_STD = (1.0 / np.array((0.229, 0.224, 0.225), dtype=np.float32))


@register_model("efficientnet_b4")
class EfficientNetB4(DeepfakeModel):
    name = "efficientnet_b4"
    input_size = 224

    def __init__(
        self,
        weights_path: Path | None = None,
        device: str = "cpu",
        weights_dir: Path | None = None,
        weights_url: str | None = None,
        weights_sha256: str | None = None,
    ):
        super().__init__(weights_path=weights_path, device=device)
        self._weights_dir = weights_dir
        self._weights_url = weights_url
        self._weights_sha256 = weights_sha256
        self._model = None  # lazy-built on first use

    def _resolve_weights(self) -> Path | None:
        try:
            return ensure_weights_file(
                model_name=self.name,
                weights_dir=self._weights_dir,
                configured_path=self.weights_path,
                configured_url=self._weights_url,
                configured_sha256=self._weights_sha256,
            )
        except ChecksumError:
            raise
        except Exception as e:  # noqa: BLE001
            log.warning("Could not download weights (%s); falling back.", e)
            return None

    def _build(self):
        import timm
        import torch

        resolved = self._resolve_weights()
        if resolved is not None:
            self.weights_path = resolved

        model = timm.create_model("tf_efficientnet_b4", pretrained=False, num_classes=2)

        if self.weights_path and self.weights_path.exists():
            state = torch.load(self.weights_path, map_location="cpu")
            if isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]
            model.load_state_dict(state, strict=False)
        else:
            log.warning(
                "No deepfake weights provided. Falling back to ImageNet-pretrained backbone — "
                "scores will not be calibrated. Configure model.weights (local path), "
                "model.url (remote download), or register a default in "
                "dfdetect.models.downloader.WEIGHT_SOURCES."
            )
            model = timm.create_model("tf_efficientnet_b4", pretrained=True, num_classes=2)

        model.eval().to(self.device)
        self._model = model

    def _ensure(self):
        if self._model is None:
            self._build()

    def preprocess(self, faces: list[np.ndarray]) -> Any:
        import cv2
        import torch

        # One uint8 staging buffer for all faces — cv2.resize writes straight
        # into it via the ``dst`` arg, avoiding a fresh float allocation per
        # crop. Normalisation is then a single vectorised pass: convert,
        # rescale, mean-subtract, divide-by-std, transpose to NCHW.
        size = self.input_size
        n = len(faces)
        staging = np.empty((n, size, size, 3), dtype=np.uint8)
        for i, face in enumerate(faces):
            if face.shape[:2] == (size, size) and face.dtype == np.uint8:
                staging[i] = face
            else:
                cv2.resize(face, (size, size), dst=staging[i], interpolation=cv2.INTER_AREA)

        arr = staging.astype(np.float32, copy=False) * (1.0 / 255.0)
        arr -= _MEAN
        arr *= _INV_STD
        # NHWC → NCHW; ``ascontiguousarray`` is necessary because transpose
        # only flips strides — torch will copy if it isn't contiguous.
        arr = np.ascontiguousarray(arr.transpose(0, 3, 1, 2))
        return torch.from_numpy(arr).to(self.device)

    def predict(self, batch: Any) -> list[float]:
        import torch

        self._ensure()
        with torch.inference_mode():
            logits = self._model(batch)
            probs = torch.softmax(logits, dim=1)[:, 1]   # class 1 = FAKE by convention
        return probs.detach().cpu().tolist()

    def grad_cam_target_layer(self) -> Any:
        self._ensure()
        # tf_efficientnet_b4: last conv block before classifier head
        return self._model.conv_head

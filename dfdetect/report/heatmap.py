"""Grad-CAM overlays for the most-suspicious face crop in a result."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from ..models.base import DeepfakeModel

log = logging.getLogger(__name__)

_FAKE_CLASS = 1  # class index 1 = FAKE, matching ``EfficientNetB4.predict``


def save_grad_cam(
    model: DeepfakeModel,
    face: Any,
    out_path: Path,
) -> Path | None:
    """Render a Grad-CAM heatmap for one face crop and save it as PNG.
    Returns None on failure (heatmaps are non-fatal)."""
    try:
        import cv2
        import numpy as np
        from pytorch_grad_cam import GradCAM
        from pytorch_grad_cam.utils.image import show_cam_on_image
        from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

        target_layer = model.grad_cam_target_layer()
        torch_model = model.grad_cam_model()
        if torch_model is None:
            log.warning("Model %s does not expose a Grad-CAM torch module; skipping.", model.name)
            return None

        batch = model.grad_cam_preprocess([face])
        # The context manager removes the forward/backward hooks afterwards.
        # Without it every heatmap leaves hooks on the shared model, so all
        # later inference passes keep capturing activations (a memory leak).
        with GradCAM(model=torch_model, target_layers=[target_layer]) as cam:
            # Explain the FAKE logit specifically, not whichever class won.
            grayscale = cam(input_tensor=batch, targets=[ClassifierOutputTarget(_FAKE_CLASS)])[0]

        # Resize the heatmap onto the original face crop for nicer overlays.
        H, W = face.shape[:2]
        resized = cv2.resize(grayscale, (W, H))
        rgb_norm = face.astype(np.float32) / 255.0
        overlay = show_cam_on_image(rgb_norm, resized, use_rgb=True)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        return out_path
    except Exception:  # noqa: BLE001
        # Heatmaps are non-fatal; include traceback so debugging a missing
        # overlay doesn't require re-running with raised exceptions.
        log.exception("Grad-CAM failed for %s", out_path.name)
        return None

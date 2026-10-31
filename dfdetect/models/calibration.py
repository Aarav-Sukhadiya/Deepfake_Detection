"""Probability calibration wrapper.

Wraps any :class:`DeepfakeModel` and post-processes its raw fake probabilities
into calibrated ones. Two stages, applied in order:

1. **Temperature scaling** on the underlying logit. The raw fake probability
   ``p`` is inverted to logit ``z = log(p / (1 - p))``, divided by ``T``, and
   re-sigmoid'd. ``T > 1`` softens an over-confident model; ``T < 1`` sharpens
   an under-confident one. ``T == 1`` is a no-op.
2. **Platt-style 1-D logistic** ``sigmoid(a * p + b)`` applied after temperature
   when both ``a`` and ``b`` are set. Fit Platt on a held-out validation set if
   temperature alone leaves residual miscalibration.

Calibrated models keep ``hash()`` of the wrapped model so the run identifier
stays tied to the underlying weights, with a ``+cal`` suffix marking that
calibration is active.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

from .base import DeepfakeModel

_EPS = 1e-7


def _temperature_scale(p: float, temperature: float) -> float:
    p = min(max(p, _EPS), 1 - _EPS)
    z = math.log(p / (1 - p)) / temperature
    # Numerically stable sigmoid.
    if z >= 0:
        ez = math.exp(-z)
        return 1.0 / (1.0 + ez)
    ez = math.exp(z)
    return ez / (1.0 + ez)


def _platt(p: float, a: float, b: float) -> float:
    z = a * p + b
    if z >= 0:
        ez = math.exp(-z)
        return 1.0 / (1.0 + ez)
    ez = math.exp(z)
    return ez / (1.0 + ez)


def calibrate_score(
    p: float,
    temperature: float = 1.0,
    platt_a: float | None = None,
    platt_b: float | None = None,
) -> float:
    q = _temperature_scale(p, temperature) if temperature != 1.0 else p
    if platt_a is not None and platt_b is not None:
        q = _platt(q, platt_a, platt_b)
    return q


def calibrate_batch(
    probs: list[float] | np.ndarray,
    temperature: float = 1.0,
    platt_a: float | None = None,
    platt_b: float | None = None,
) -> list[float]:
    """Vectorised equivalent of mapping :func:`calibrate_score` over a list.

    Avoids a Python-level ``math.exp``/``math.log`` per face when batches are
    large; falls back to the input directly when calibration is a no-op.
    """
    if temperature == 1.0 and (platt_a is None or platt_b is None):
        return list(probs) if not isinstance(probs, list) else probs

    arr = np.asarray(probs, dtype=np.float64)
    if temperature != 1.0:
        np.clip(arr, _EPS, 1.0 - _EPS, out=arr)
        # Reuse a single buffer for the logit / sigmoid pair.
        logits = np.log(arr / (1.0 - arr))
        logits /= temperature
        arr = 1.0 / (1.0 + np.exp(-logits))
    if platt_a is not None and platt_b is not None:
        arr = 1.0 / (1.0 + np.exp(-(platt_a * arr + platt_b)))
    return arr.tolist()


class CalibratedModel(DeepfakeModel):
    """Decorator that calibrates the inner model's predictions."""

    def __init__(
        self,
        inner: DeepfakeModel,
        temperature: float = 1.0,
        platt_a: float | None = None,
        platt_b: float | None = None,
    ):
        super().__init__(weights_path=inner.weights_path, device=inner.device)
        self.name = inner.name
        self.input_size = inner.input_size
        self._inner = inner
        self.temperature = temperature
        self.platt_a = platt_a
        self.platt_b = platt_b

    def preprocess(self, faces: list[Any]) -> Any:
        return self._inner.preprocess(faces)

    def predict(self, batch: Any) -> list[float]:
        raw = self._inner.predict(batch)
        return calibrate_batch(raw, self.temperature, self.platt_a, self.platt_b)

    def grad_cam_target_layer(self) -> Any:
        return self._inner.grad_cam_target_layer()

    def grad_cam_model(self) -> Any:
        return self._inner.grad_cam_model()

    def grad_cam_preprocess(self, faces: list[Any]) -> Any:
        return self._inner.grad_cam_preprocess(faces)

    def hash(self) -> str:
        return f"{self._inner.hash()}+cal"

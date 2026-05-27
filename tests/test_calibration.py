"""Tests for the temperature/Platt calibration wrapper."""
from __future__ import annotations

import math
from typing import Any

import pytest

from dfdetect.models.base import DeepfakeModel
from dfdetect.models.calibration import CalibratedModel, calibrate_score


class _Stub(DeepfakeModel):
    name = "stub"

    def __init__(self, scores: list[float]):
        super().__init__()
        self._scores = scores

    def preprocess(self, faces: list[Any]) -> Any:  # type: ignore[override]
        return faces

    def predict(self, batch: Any) -> list[float]:  # type: ignore[override]
        return list(self._scores)

    def grad_cam_target_layer(self) -> Any:  # type: ignore[override]
        return None

    def hash(self) -> str:  # type: ignore[override]
        return "stub-hash"


def test_calibrate_score_identity_at_temperature_one():
    assert calibrate_score(0.73) == pytest.approx(0.73)


def test_high_temperature_pulls_scores_toward_half():
    p = 0.95
    out = calibrate_score(p, temperature=4.0)
    assert 0.5 < out < p


def test_low_temperature_sharpens():
    p = 0.6
    out = calibrate_score(p, temperature=0.5)
    assert out > p


def test_platt_scaling_applies_after_temperature():
    # With T=1 the temperature is a no-op; verify Platt sigmoid alone.
    p = 0.5
    out = calibrate_score(p, platt_a=2.0, platt_b=-1.0)
    expected = 1.0 / (1.0 + math.exp(-(2.0 * 0.5 - 1.0)))
    assert out == pytest.approx(expected)


def test_calibrated_model_wraps_predict():
    inner = _Stub([0.9, 0.1])
    calibrated = CalibratedModel(inner, temperature=2.0)
    out = calibrated.predict([None, None])
    assert len(out) == 2
    assert out[0] < 0.9
    assert out[1] > 0.1
    assert calibrated.hash() == "stub-hash+cal"


def test_calibration_rejects_non_positive_temperature():
    from dfdetect.config import CalibrationConfig

    with pytest.raises(ValueError):
        CalibrationConfig(temperature=0)

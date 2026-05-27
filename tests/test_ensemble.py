"""Tests for the EnsembleModel weighted-average wrapper."""
from __future__ import annotations

from typing import Any

import pytest

from dfdetect.models.base import DeepfakeModel
from dfdetect.models.ensemble import EnsembleModel


class _Stub(DeepfakeModel):
    def __init__(self, name: str, score: float):
        super().__init__()
        self.name = name
        self._score = score

    def preprocess(self, faces: list[Any]) -> Any:
        return faces

    def predict(self, batch: Any) -> list[float]:
        return [self._score for _ in batch]

    def grad_cam_target_layer(self) -> Any:
        return None

    def hash(self) -> str:
        return f"{self.name}-h"


def test_ensemble_weighted_average():
    a = _Stub("a", 0.8)
    b = _Stub("b", 0.2)
    ens = EnsembleModel([(a, 3.0), (b, 1.0)])  # 0.75 / 0.25
    out = ens.predict([None, None])
    assert out == pytest.approx([0.65, 0.65])


def test_ensemble_normalises_weights():
    a = _Stub("a", 1.0)
    b = _Stub("b", 0.0)
    ens = EnsembleModel([(a, 2.0), (b, 2.0)])  # both 0.5
    out = ens.predict([None])
    assert out == pytest.approx([0.5])


def test_ensemble_hash_includes_members():
    ens = EnsembleModel([(_Stub("a", 0.3), 1.0), (_Stub("b", 0.7), 1.0)])
    assert ens.hash() == "ens(a-h|b-h)"
    assert "a" in ens.name and "b" in ens.name


def test_ensemble_requires_member():
    with pytest.raises(ValueError):
        EnsembleModel([])


def test_ensemble_rejects_zero_weights():
    with pytest.raises(ValueError):
        EnsembleModel([(_Stub("a", 0.5), 0.0)])

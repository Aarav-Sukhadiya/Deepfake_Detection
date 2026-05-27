import pytest

from dfdetect.config import InferenceConfig, WeightedConfig
from dfdetect.inference.aggregator import aggregate, verdict_from_score
from dfdetect.types import Verdict


def test_mean_aggregation():
    cfg = InferenceConfig(aggregation="mean")
    assert aggregate([0.1, 0.3, 0.5], cfg) == 0.3


def test_max_aggregation():
    cfg = InferenceConfig(aggregation="max")
    assert aggregate([0.1, 0.9, 0.4], cfg) == 0.9


def test_weighted_aggregation():
    cfg = InferenceConfig(
        aggregation="weighted",
        weighted=WeightedConfig(mean=0.5, median=0.25, max=0.25),
    )
    out = aggregate([0.2, 0.4, 0.6], cfg)
    # mean=0.4, median=0.4, max=0.6 → 0.5*0.4 + 0.25*0.4 + 0.25*0.6 = 0.45
    assert abs(out - 0.45) < 1e-9


def test_empty_scores():
    assert aggregate([], InferenceConfig()) == 0.0


def test_quality_weighted_collapses_to_mean_without_weights():
    cfg = InferenceConfig(aggregation="quality_weighted")
    assert aggregate([0.2, 0.4, 0.6], cfg) == pytest.approx(0.4)


def test_quality_weighted_favors_high_quality_scores():
    # The 0.9-score face has the heavy weight; the 0.1-score face has a token weight.
    cfg = InferenceConfig(aggregation="quality_weighted")
    out = aggregate([0.9, 0.1], cfg, weights=[1.0, 0.05])
    # Pure mean = 0.5; quality-weighted should be much closer to 0.9.
    assert out > 0.8


def test_quality_weighted_handles_zero_total_weight():
    cfg = InferenceConfig(aggregation="quality_weighted")
    out = aggregate([0.2, 0.4], cfg, weights=[0.0, 0.0])
    assert out == pytest.approx(0.3)


def test_verdict_thresholds():
    band = (0.4, 0.6)
    assert verdict_from_score(0.1, 0.5, band, faces_analyzed=1) == Verdict.REAL
    assert verdict_from_score(0.5, 0.5, band, faces_analyzed=1) == Verdict.UNCERTAIN
    assert verdict_from_score(0.7, 0.5, band, faces_analyzed=1) == Verdict.FAKE
    assert verdict_from_score(0.9, 0.5, band, faces_analyzed=0) == Verdict.NO_FACE

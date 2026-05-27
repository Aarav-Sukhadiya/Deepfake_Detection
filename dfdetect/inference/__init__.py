"""Inference loop and per-file score aggregation."""
from .aggregator import aggregate, verdict_from_score
from .runner import InferenceRunner

__all__ = ["InferenceRunner", "aggregate", "verdict_from_score"]

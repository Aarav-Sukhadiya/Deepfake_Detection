"""Weighted-average ensemble over multiple :class:`DeepfakeModel` members.

Each member computes its own face preprocessing (input sizes can differ) and
its own scores. The ensemble returns the per-face weighted mean of member
scores. ``hash()`` concatenates member hashes so a run record uniquely
identifies the bag of weights used.

Constraints:

* All members must accept the same list of face crops (RGB ``H×W×3`` uint8
  ``ndarray``). Per-member preprocessing handles resize/normalize.
* Members do not share state — each holds its own backbone.
* Grad-CAM falls back to the first member, since it operates on one network.
"""
from __future__ import annotations

from typing import Any

from .base import DeepfakeModel


class EnsembleModel(DeepfakeModel):
    name = "ensemble"
    input_size = 224

    def __init__(self, members: list[tuple[DeepfakeModel, float]]):
        if not members:
            raise ValueError("EnsembleModel requires at least one member.")
        super().__init__(weights_path=None, device=members[0][0].device)
        total = sum(w for _, w in members)
        if total <= 0:
            raise ValueError("Ensemble member weights must sum to a positive value.")
        self._members: list[tuple[DeepfakeModel, float]] = [
            (m, w / total) for m, w in members
        ]
        self.name = "ensemble[" + "+".join(m.name for m, _ in self._members) + "]"

    def preprocess(self, faces: list[Any]) -> Any:
        # Defer preprocessing into predict() so each member can apply its own.
        return faces

    def predict(self, batch: Any) -> list[float]:
        faces = batch
        if not faces:
            return []
        accum: list[float] = [0.0] * len(faces)
        for member, weight in self._members:
            member_batch = member.preprocess(faces)
            member_scores = member.predict(member_batch)
            if len(member_scores) != len(faces):
                raise RuntimeError(
                    f"Ensemble member {member.name} returned {len(member_scores)} "
                    f"scores for {len(faces)} faces."
                )
            for i, s in enumerate(member_scores):
                accum[i] += weight * s
        return accum

    def grad_cam_target_layer(self) -> Any:
        return self._members[0][0].grad_cam_target_layer()

    def grad_cam_model(self) -> Any:
        # CAM operates on a single torch module; defer to the first member.
        return self._members[0][0].grad_cam_model()

    def hash(self) -> str:
        parts = [m.hash() for m, _ in self._members]
        return "ens(" + "|".join(parts) + ")"

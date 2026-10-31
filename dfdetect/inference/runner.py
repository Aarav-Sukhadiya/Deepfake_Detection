"""Batched inference loop. Owns batching strategy and OOM fallback.

The runner halves its batch on CUDA OOM and remembers the largest batch size
that survived a full chunk, so subsequent files start at the known-safe value
instead of repeating the OOM-and-halve dance every run.
"""
from __future__ import annotations

import logging

from .._cache import load_batch_size, save_batch_size
from ..models.base import DeepfakeModel
from ..types import FaceCrop

log = logging.getLogger(__name__)

_AUTOTUNE_FILE = "autotune.json"


def _is_oom(exc: BaseException) -> bool:
    return "out of memory" in str(exc).lower()


def _free_cuda_cache() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001 - best effort
        pass


class InferenceRunner:
    def __init__(
        self,
        model: DeepfakeModel,
        batch_size: int = 8,
        autotune: bool = True,
    ):
        self.model = model
        self.batch_size = batch_size
        self.autotune = autotune
        self._key = f"{model.name}:{model.hash()}"
        if autotune:
            cached = load_batch_size(_AUTOTUNE_FILE, self._key)
            if cached is not None and cached < batch_size:
                log.info(
                    "Autotune: starting batch_size=%d for %s (cached from a prior OOM).",
                    cached,
                    self._key,
                )
                self.batch_size = cached

    def run(self, faces: list[FaceCrop]) -> list[float]:
        """Return one fake-probability per FaceCrop, in input order."""
        if not faces:
            return []
        scores: list[float] = []
        bs = self.batch_size
        i = 0
        while i < len(faces):
            chunk = faces[i : i + bs]
            oom = False
            try:
                scores.extend(self.model.predict(self.model.preprocess([f.image for f in chunk])))
            except RuntimeError as e:  # torch.cuda.OutOfMemoryError subclasses RuntimeError
                if not _is_oom(e) or bs <= 1:
                    raise
                oom = True
            if oom:
                # Freed outside the ``except`` so the traceback (and the
                # tensors its frames reference) has already been released.
                _free_cuda_cache()
                bs = max(1, bs // 2)
                log.warning("CUDA OOM; halving batch size to %d and retrying.", bs)
                continue
            i += len(chunk)
        if self.autotune and bs != self.batch_size:
            save_batch_size(_AUTOTUNE_FILE, self._key, bs)
        self.batch_size = bs
        return scores

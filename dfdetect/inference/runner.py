"""Batched inference loop. Owns batching strategy and OOM fallback.

The runner halves its batch on CUDA OOM and remembers the largest batch size
that survived a full chunk, so subsequent files start at the known-safe value
instead of repeating the OOM-and-halve dance every run.
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path

from ..models.base import DeepfakeModel
from ..types import FaceCrop

log = logging.getLogger(__name__)


def _autotune_path() -> Path:
    base = os.environ.get("DFDETECT_CACHE") or os.path.expanduser("~/.cache/dfdetect")
    return Path(base) / "autotune.json"


def _load_autotune(model_key: str) -> int | None:
    p = _autotune_path()
    if not p.exists():
        return None
    try:
        data = json.loads(p.read_text())
    except Exception:  # noqa: BLE001
        return None
    val = data.get(model_key)
    return int(val) if isinstance(val, int) and val >= 1 else None


def _save_autotune(model_key: str, batch_size: int) -> None:
    p = _autotune_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    try:
        data = json.loads(p.read_text()) if p.exists() else {}
    except Exception:  # noqa: BLE001
        data = {}
    data[model_key] = int(batch_size)
    try:
        p.write_text(json.dumps(data, indent=2))
    except Exception:  # noqa: BLE001
        log.debug("Could not persist autotune state to %s", p)


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
            cached = _load_autotune(self._key)
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
            try:
                batch = self.model.preprocess([f.image for f in chunk])
                scores.extend(self.model.predict(batch))
                i += bs
            except RuntimeError as e:
                if "out of memory" in str(e).lower() and bs > 1:
                    new_bs = max(1, bs // 2)
                    log.warning("CUDA OOM at batch=%d; halving to %d and retrying.", bs, new_bs)
                    bs = new_bs
                    if self.autotune:
                        _save_autotune(self._key, bs)
                    continue
                raise
        # Persist the successful batch size so future runs skip the halving
        # dance, even if it stayed at the originally-requested size.
        if self.autotune and bs != self.batch_size:
            _save_autotune(self._key, bs)
        self.batch_size = bs
        return scores

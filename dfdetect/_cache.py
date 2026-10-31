"""Small helpers for the per-user cache directory (``~/.cache/dfdetect``).

Both the inference runner and the trainer persist an "autotuned" batch size
keyed by model/device so later runs skip the OOM-and-halve dance. The JSON
file is shared between concurrent processes, so reads and read-modify-write
cycles happen under an advisory ``flock`` (a no-op on platforms without
``fcntl``).
"""
from __future__ import annotations

import contextlib
import json
import logging
import os
from collections.abc import Iterator
from pathlib import Path

try:  # POSIX only; on other platforms locking degrades to a no-op.
    import fcntl
except ImportError:  # pragma: no cover - non-POSIX
    fcntl = None  # type: ignore[assignment]

log = logging.getLogger(__name__)


def cache_dir() -> Path:
    """Root cache directory, overridable with ``$DFDETECT_CACHE``."""
    return Path(os.environ.get("DFDETECT_CACHE") or os.path.expanduser("~/.cache/dfdetect"))


@contextlib.contextmanager
def _locked(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path.with_suffix(".lock"), "w") as f:
        if fcntl is not None:
            fcntl.flock(f, fcntl.LOCK_EX)
        try:
            yield
        finally:
            if fcntl is not None:
                fcntl.flock(f, fcntl.LOCK_UN)


def load_batch_size(filename: str, key: str) -> int | None:
    """Return the cached batch size for ``key`` or ``None`` if absent/invalid."""
    path = cache_dir() / filename
    if not path.exists():
        return None
    try:
        with _locked(path):
            data = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    val = data.get(key) if isinstance(data, dict) else None
    return val if isinstance(val, int) and val >= 1 else None


def save_batch_size(filename: str, key: str, batch_size: int) -> None:
    """Persist ``batch_size`` for ``key``. Failures are logged, never raised."""
    path = cache_dir() / filename
    try:
        with _locked(path):
            try:
                data = json.loads(path.read_text()) if path.exists() else {}
            except ValueError:
                data = {}  # corrupt file — start over rather than fail forever
            if not isinstance(data, dict):
                data = {}
            data[key] = int(batch_size)
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps(data, indent=2))
            os.replace(tmp, path)
    except OSError:
        log.debug("Could not persist autotune state to %s", path)

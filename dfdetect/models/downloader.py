"""Weight downloader with SHA-256 verification.

The model factories call :func:`ensure_weights_file` before building the network.
If a local path is configured and present we use it as-is; otherwise we try to
download from a URL (either declared by the model in :data:`WEIGHT_SOURCES` or
overridden via :class:`~dfdetect.config.ModelConfig`). Downloads stream to a
temp file, verify the SHA-256, and only then move into place — a partial
download or a hash mismatch leaves the cache untouched.
"""
from __future__ import annotations

import hashlib
import logging
import os
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)

_DOWNLOAD_CHUNK = 1 << 20  # 1 MiB


@dataclass(frozen=True)
class WeightSource:
    url: str
    sha256: str | None = None  # None disables verification (not recommended)
    filename: str | None = None  # defaults to "<model_name>.pt"


# Built-in sources for shipped models. Empty by default — users supply their own
# URL via ``ModelConfig.url`` (and optionally ``ModelConfig.sha256``) until a
# canonical public checkpoint is wired in.
WEIGHT_SOURCES: dict[str, WeightSource] = {}


class ChecksumError(RuntimeError):
    """Raised when a downloaded weights file fails SHA-256 verification."""


def register_weights(name: str, source: WeightSource) -> None:
    WEIGHT_SOURCES[name] = source


def compute_sha256(path: Path) -> str:
    # ``hashlib.file_digest`` (3.11+) is implemented in C and uses readv where
    # available, which is markedly faster than the manual chunk loop on large
    # weight files (and the project already requires Python 3.11+).
    with open(path, "rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def _stream_download(url: str, dest: Path) -> None:
    """Stream a URL to ``dest`` with a progress bar. Imports are local so the
    package stays usable in environments without ``requests``/``tqdm``."""
    import requests
    from tqdm import tqdm

    dest.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=60) as resp:
        resp.raise_for_status()
        total = int(resp.headers.get("content-length") or 0)
        bar = tqdm(
            total=total or None,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            desc=f"Downloading {dest.name}",
        )
        try:
            with open(dest, "wb") as f:
                for chunk in resp.iter_content(chunk_size=_DOWNLOAD_CHUNK):
                    if not chunk:
                        continue
                    f.write(chunk)
                    bar.update(len(chunk))
        finally:
            bar.close()


def download_file(url: str, dest: Path, expected_sha256: str | None = None) -> Path:
    """Download ``url`` to ``dest``, verifying SHA-256 when provided.

    Streams via a sibling ``.part`` file and atomically renames on success so
    interrupted downloads never look complete. A hash mismatch deletes the
    partial file and raises :class:`ChecksumError`.
    """
    dest = Path(dest)
    tmp = dest.with_suffix(dest.suffix + ".part")
    if tmp.exists():
        tmp.unlink()
    _stream_download(url, tmp)

    if expected_sha256:
        actual = compute_sha256(tmp)
        if actual.lower() != expected_sha256.lower():
            tmp.unlink(missing_ok=True)
            raise ChecksumError(
                f"SHA-256 mismatch for {url}: expected {expected_sha256}, got {actual}"
            )
    os.replace(tmp, dest)
    return dest


def ensure_weights_file(
    model_name: str,
    weights_dir: Path | None,
    configured_path: Path | None = None,
    configured_url: str | None = None,
    configured_sha256: str | None = None,
) -> Path | None:
    """Resolve the on-disk weights for ``model_name``.

    Resolution order:

    1. If ``configured_path`` exists, return it.
    2. If a URL is available (config override or :data:`WEIGHT_SOURCES`),
       download into ``weights_dir`` (or the parent of ``configured_path``)
       and return the local path. Re-uses an existing cached file as long as
       the SHA-256 still matches.
    3. Return ``None`` — caller decides whether to fall back (e.g. to an
       ImageNet-pretrained backbone) or raise.
    """
    if configured_path and configured_path.exists():
        return configured_path

    source = WEIGHT_SOURCES.get(model_name)
    url = configured_url or (source.url if source else None)
    sha256 = configured_sha256 or (source.sha256 if source else None)
    if not url:
        return None

    filename = (
        configured_path.name
        if configured_path
        else (source.filename if source and source.filename else f"{model_name}.pt")
    )
    target_dir = (
        configured_path.parent if configured_path else (weights_dir or Path.cwd())
    )
    target = Path(target_dir) / filename

    if target.exists():
        if sha256 and compute_sha256(target).lower() != sha256.lower():
            log.warning(
                "Cached weights at %s failed SHA-256 verification; re-downloading.",
                target,
            )
            target.unlink()
        else:
            return target

    log.info("Downloading %s weights from %s -> %s", model_name, url, target)
    return download_file(url, target, expected_sha256=sha256)

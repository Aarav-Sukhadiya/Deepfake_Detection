"""Header-based media type validation."""
from __future__ import annotations

from pathlib import Path

import filetype

from ..config import InputConfig
from ..types import MediaKind


def _read_kind_from_header(path: Path) -> MediaKind | None:
    kind = filetype.guess(str(path))
    if kind is None:
        return None
    if kind.mime.startswith("image/"):
        return MediaKind.IMAGE
    if kind.mime.startswith("video/"):
        return MediaKind.VIDEO
    return None


def classify(path: Path, cfg: InputConfig) -> MediaKind | None:
    """Return IMAGE / VIDEO / None. Header is authoritative; extension is a fallback."""
    if not path.is_file():
        return None
    kind = _read_kind_from_header(path)
    if kind is not None:
        return kind
    ext = path.suffix.lower()
    if ext in cfg.image_exts:
        return MediaKind.IMAGE
    if ext in cfg.video_exts:
        return MediaKind.VIDEO
    return None


def is_supported(path: Path, cfg: InputConfig) -> bool:
    return classify(path, cfg) is not None

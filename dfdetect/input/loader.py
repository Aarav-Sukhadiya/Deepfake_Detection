"""Walk inputs (files and directories) and yield validated MediaItems."""
from __future__ import annotations

import logging
from collections.abc import Iterable, Iterator
from pathlib import Path

from ..config import InputConfig
from ..types import MediaItem
from .validator import classify

log = logging.getLogger(__name__)


def _walk(root: Path, recursive: bool, follow_symlinks: bool) -> Iterator[Path]:
    if root.is_file():
        yield root
        return
    if not root.is_dir():
        log.warning("Path does not exist: %s", root)
        return
    pattern = "**/*" if recursive else "*"
    for p in root.glob(pattern):
        if p.is_symlink() and not follow_symlinks:
            continue
        if p.is_file():
            yield p


def discover(paths: Iterable[str | Path], cfg: InputConfig) -> list[MediaItem]:
    """Expand the given paths into a deduplicated list of validated MediaItems."""
    seen: set[Path] = set()
    items: list[MediaItem] = []
    for raw in paths:
        # Absolute roots so ``./a.png`` and ``/abs/a.png`` dedupe together.
        root = Path(raw).expanduser().absolute()
        for fp in _walk(root, cfg.recursive, cfg.follow_symlinks):
            # ``resolve()`` is one syscall per file and only matters for
            # symlinks (so two distinct paths that point at the same target
            # dedupe). For the common non-symlink case the path is already
            # canonical from the glob, so skip the extra stat.
            if fp.is_symlink():
                fp = fp.resolve()
            if fp in seen:
                continue
            seen.add(fp)
            try:
                kind = classify(fp, cfg)
                size = fp.stat().st_size if kind is not None else 0
            except Exception as e:  # noqa: BLE001 - unreadable/vanished file must not abort discovery
                log.warning("Cannot classify %s: %s", fp, e)
                items.append(MediaItem(path=fp, kind=None, size_bytes=0, error=str(e)))
                continue
            if kind is None:
                log.debug("Skipping unsupported file: %s", fp)
                continue
            items.append(MediaItem(path=fp, kind=kind, size_bytes=size))
    return items

"""Anti-bias dataset constraints (IMPROVEMENTS.md #27).

We refuse to start training when the user's labelled folders would obviously
bias the resulting model: empty class, single-class, extreme imbalance, or a
video-only set where one source clip dominates. The constraint layer is a
pure-function gate so it can be tested without touching torch.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path

_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
_VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}

# Defaults exposed so the CLI / Gradio layer can override per workflow.
DEFAULT_MIN_PER_CLASS = 50
DEFAULT_MIN_VIDEO_SOURCES = 5
DEFAULT_HARD_RATIO = (0.2, 0.8)   # reject outside this
DEFAULT_SOFT_RATIO = (0.4, 0.6)   # warn (but allow) outside this


@dataclass
class DatasetStats:
    real_image_count: int = 0
    fake_image_count: int = 0
    real_video_count: int = 0
    fake_video_count: int = 0
    real_unique_video_hashes: int = 0
    fake_unique_video_hashes: int = 0
    warnings: list[str] = field(default_factory=list)

    @property
    def real_count(self) -> int:
        return self.real_image_count + self.real_video_count

    @property
    def fake_count(self) -> int:
        return self.fake_image_count + self.fake_video_count

    @property
    def total(self) -> int:
        return self.real_count + self.fake_count

    @property
    def fake_ratio(self) -> float:
        return self.fake_count / self.total if self.total else 0.0


class BiasConstraintError(ValueError):
    """Raised when the dataset would obviously bias training."""


def _hash_video_head(path: Path, n_bytes: int = 1 << 20) -> str:
    """Hash the first ``n_bytes`` of a video so different clips that share a
    container header still hash differently while exact duplicates collide."""
    h = hashlib.sha1()
    with open(path, "rb") as f:
        h.update(f.read(n_bytes))
    return h.hexdigest()


def walk_media(root: Path) -> tuple[list[Path], list[Path]]:
    """Return ``(images, videos)`` found anywhere under ``root`` (recursive).

    Hidden files and symlinks are skipped to avoid loops; classification is by
    extension. Sorted for deterministic order so two runs over the same
    folder produce the same dataset.
    """
    images: list[Path] = []
    videos: list[Path] = []
    for p in sorted(root.rglob("*")):
        if not p.is_file() or p.is_symlink():
            continue
        if p.name.startswith("."):
            continue
        ext = p.suffix.lower()
        if ext in _IMAGE_EXTS:
            images.append(p)
        elif ext in _VIDEO_EXTS:
            videos.append(p)
    return images, videos


def summarise_dataset(real_dir: Path, fake_dir: Path) -> DatasetStats:
    """Walk both class folders **recursively** and build a :class:`DatasetStats`.

    Each folder may contain a mix of images and videos at any depth — every
    media file under ``real_dir`` is treated as a REAL sample, every one
    under ``fake_dir`` as a FAKE sample. Subfolders that group by source
    identity (e.g. ``real/personA/...``) are still flattened into the class.
    """
    stats = DatasetStats()
    for cls_dir, is_fake in [(real_dir, False), (fake_dir, True)]:
        if not cls_dir.is_dir():
            raise BiasConstraintError(f"Class folder is missing: {cls_dir}")
        images, videos = walk_media(cls_dir)
        video_hashes: set[str] = set()
        for p in videos:
            try:
                video_hashes.add(_hash_video_head(p))
            except OSError:
                continue
        if is_fake:
            stats.fake_image_count = len(images)
            stats.fake_video_count = len(videos)
            stats.fake_unique_video_hashes = len(video_hashes)
        else:
            stats.real_image_count = len(images)
            stats.real_video_count = len(videos)
            stats.real_unique_video_hashes = len(video_hashes)
    return stats


def enforce_constraints(
    stats: DatasetStats,
    min_per_class: int = DEFAULT_MIN_PER_CLASS,
    min_video_sources: int = DEFAULT_MIN_VIDEO_SOURCES,
    hard_ratio: tuple[float, float] = DEFAULT_HARD_RATIO,
    soft_ratio: tuple[float, float] = DEFAULT_SOFT_RATIO,
) -> DatasetStats:
    """Raise :class:`BiasConstraintError` for hard violations; append soft
    issues to ``stats.warnings``. Returns the same ``stats`` so callers can
    chain. Hard rules (any one fails → refuse to train):

    * ``real_count == 0`` or ``fake_count == 0``
    * either class < ``min_per_class``
    * ``fake_ratio`` outside ``hard_ratio``
    * a class is video-only and has fewer than ``min_video_sources`` distinct
      source-hash clips (a 1000-frame dataset cut from a single video is one
      identity, not a learnable signal)
    """
    if stats.real_count == 0:
        raise BiasConstraintError(
            "No REAL samples found. Training on a single class would yield a "
            "model that always predicts FAKE."
        )
    if stats.fake_count == 0:
        raise BiasConstraintError(
            "No FAKE samples found. Training on a single class would yield a "
            "model that always predicts REAL."
        )
    if stats.real_count < min_per_class or stats.fake_count < min_per_class:
        raise BiasConstraintError(
            f"At least {min_per_class} samples per class required "
            f"(got real={stats.real_count}, fake={stats.fake_count})."
        )
    hard_lo, hard_hi = hard_ratio
    if not (hard_lo <= stats.fake_ratio <= hard_hi):
        raise BiasConstraintError(
            f"Fake/total ratio {stats.fake_ratio:.2f} is outside the safe "
            f"range [{hard_lo}, {hard_hi}]. Balance the classes before training."
        )
    # Video-only-class check: if a class contains only videos, demand a
    # minimum number of distinct source clips.
    for label, count, vid_hashes, vid_count in [
        ("REAL", stats.real_count, stats.real_unique_video_hashes, stats.real_video_count),
        ("FAKE", stats.fake_count, stats.fake_unique_video_hashes, stats.fake_video_count),
    ]:
        if vid_count == count and vid_hashes < min_video_sources:
            raise BiasConstraintError(
                f"{label} class is video-only with only {vid_hashes} distinct "
                f"source clip(s) (need at least {min_video_sources}). The "
                "model would learn the speaker, not the manipulation."
            )

    # Soft warnings.
    soft_lo, soft_hi = soft_ratio
    if not (soft_lo <= stats.fake_ratio <= soft_hi):
        stats.warnings.append(
            f"Fake/total ratio {stats.fake_ratio:.2f} is outside the "
            f"recommended range [{soft_lo}, {soft_hi}]. Training will run, "
            "but consider class-balanced sampling or rebalancing the folders."
        )
    if stats.real_video_count and stats.real_unique_video_hashes < min_video_sources:
        stats.warnings.append(
            f"REAL class has {stats.real_unique_video_hashes} distinct video "
            f"sources (< {min_video_sources}). Per-identity leakage is likely."
        )
    if stats.fake_video_count and stats.fake_unique_video_hashes < min_video_sources:
        stats.warnings.append(
            f"FAKE class has {stats.fake_unique_video_hashes} distinct video "
            f"sources (< {min_video_sources}). Per-identity leakage is likely."
        )
    return stats

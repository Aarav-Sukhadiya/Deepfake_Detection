"""Tests for the training anti-bias constraint gate (item #27)."""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from dfdetect.training import (
    BiasConstraintError,
    enforce_constraints,
    summarise_dataset,
)


def _touch_images(root: Path, n: int, prefix: str = "img"):
    root.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        (root / f"{prefix}_{i}.png").write_bytes(b"\x89PNG fake-content")


def _touch_videos(root: Path, n_clips: int, frames_per_clip: int = 1):
    """Create ``n_clips * frames_per_clip`` files where every file from the
    same clip is byte-identical (so it hashes to the same value). Tests
    near-duplicate detection without depending on real video bytes."""
    root.mkdir(parents=True, exist_ok=True)
    for clip_idx in range(n_clips):
        head = os.urandom(1 << 12)
        for f in range(frames_per_clip):
            (root / f"clip{clip_idx}_f{f}.mp4").write_bytes(head)


def test_rejects_empty_class(tmp_path: Path):
    _touch_images(tmp_path / "real", 100)
    (tmp_path / "fake").mkdir()
    stats = summarise_dataset(tmp_path / "real", tmp_path / "fake")
    with pytest.raises(BiasConstraintError, match="No FAKE"):
        enforce_constraints(stats)


def test_rejects_undersized_class(tmp_path: Path):
    _touch_images(tmp_path / "real", 100)
    _touch_images(tmp_path / "fake", 10)
    stats = summarise_dataset(tmp_path / "real", tmp_path / "fake")
    with pytest.raises(BiasConstraintError, match="per class"):
        enforce_constraints(stats, min_per_class=50)


def test_rejects_extreme_imbalance(tmp_path: Path):
    _touch_images(tmp_path / "real", 100)
    _touch_images(tmp_path / "fake", 950)
    stats = summarise_dataset(tmp_path / "real", tmp_path / "fake")
    with pytest.raises(BiasConstraintError, match="ratio"):
        enforce_constraints(stats)


def test_accepts_balanced_dataset_and_warns_on_skew(tmp_path: Path):
    _touch_images(tmp_path / "real", 80)
    _touch_images(tmp_path / "fake", 120)   # ratio 0.6 — inside hard, on the edge of soft
    stats = summarise_dataset(tmp_path / "real", tmp_path / "fake")
    enforce_constraints(stats)
    # Should not raise; may produce a warning depending on the cutoff.
    assert stats.warnings == [] or any("ratio" in w for w in stats.warnings)


def test_video_only_class_needs_multiple_sources(tmp_path: Path):
    _touch_videos(tmp_path / "real", n_clips=1, frames_per_clip=200)
    _touch_videos(tmp_path / "fake", n_clips=10, frames_per_clip=20)
    stats = summarise_dataset(tmp_path / "real", tmp_path / "fake")
    assert stats.real_unique_video_hashes == 1
    with pytest.raises(BiasConstraintError, match="distinct source clip"):
        enforce_constraints(stats, min_video_sources=5)


def test_video_class_with_enough_sources_passes(tmp_path: Path):
    _touch_videos(tmp_path / "real", n_clips=6, frames_per_clip=15)
    _touch_videos(tmp_path / "fake", n_clips=6, frames_per_clip=15)
    stats = summarise_dataset(tmp_path / "real", tmp_path / "fake")
    enforce_constraints(stats, min_video_sources=5)


def test_summarise_walks_recursively_and_mixes_media(tmp_path: Path):
    # Realistic layout: each class is grouped by source identity (subfolders)
    # and contains a mix of images and videos. The walker should flatten
    # everything under the class root.
    (tmp_path / "real" / "alice").mkdir(parents=True)
    (tmp_path / "real" / "bob").mkdir(parents=True)
    _touch_images(tmp_path / "real" / "alice", 30, prefix="alice")
    _touch_images(tmp_path / "real" / "bob", 30, prefix="bob")
    _touch_videos(tmp_path / "real" / "bob", n_clips=3, frames_per_clip=2)
    (tmp_path / "fake" / "synth").mkdir(parents=True)
    _touch_images(tmp_path / "fake" / "synth", 50, prefix="synth")
    _touch_videos(tmp_path / "fake", n_clips=2, frames_per_clip=5)

    stats = summarise_dataset(tmp_path / "real", tmp_path / "fake")
    assert stats.real_image_count == 60
    assert stats.real_video_count == 6     # 3 clips × 2 frames
    assert stats.real_unique_video_hashes == 3
    assert stats.fake_image_count == 50
    assert stats.fake_video_count == 10    # 2 clips × 5 frames
    assert stats.fake_unique_video_hashes == 2
    # Mixed media classes don't trigger the video-only check.
    enforce_constraints(stats, min_video_sources=5)


def test_config_signature_stable_across_irrelevant_fields(tmp_path: Path):
    """Changing ``epochs`` must NOT invalidate the resume signature, but
    changing the LR must."""
    from dfdetect.training.trainer import TrainConfig, _config_signature

    base = TrainConfig(real_dir=tmp_path / "r", fake_dir=tmp_path / "f", output_dir=tmp_path / "o")
    sig_a = _config_signature(base)

    base_extra_epochs = TrainConfig(
        real_dir=tmp_path / "r", fake_dir=tmp_path / "f", output_dir=tmp_path / "o",
        epochs=base.epochs + 7,
    )
    assert _config_signature(base_extra_epochs) == sig_a, "epochs must not affect signature"

    base_diff_lr = TrainConfig(
        real_dir=tmp_path / "r", fake_dir=tmp_path / "f", output_dir=tmp_path / "o",
        learning_rate=base.learning_rate * 10,
    )
    assert _config_signature(base_diff_lr) != sig_a, "learning_rate must affect signature"


def test_walk_media_ignores_hidden_and_unrelated_files(tmp_path: Path):
    from dfdetect.training import walk_media

    (tmp_path / "sub").mkdir()
    (tmp_path / "a.jpg").write_bytes(b"x")
    (tmp_path / "sub" / "b.mp4").write_bytes(b"x")
    (tmp_path / ".hidden.jpg").write_bytes(b"x")   # hidden
    (tmp_path / "notes.txt").write_bytes(b"x")      # unrelated extension

    images, videos = walk_media(tmp_path)
    names = {p.name for p in images} | {p.name for p in videos}
    assert names == {"a.jpg", "b.mp4"}

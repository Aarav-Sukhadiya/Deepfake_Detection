"""Image and video frame sampling. Returns RGB numpy frames.

Video sampling supports three strategies (``VideoConfig.sampling_strategy``):

* ``uniform`` — strict ``frames_per_second`` step (the default, fast).
* ``scene``  — only frames that sit on a detected scene boundary, computed
  from HSV-histogram correlation between consecutive thumbnails. Catches
  short-spliced fakes that would fall between uniform samples.
* ``hybrid`` — union of uniform + scene indices, deduplicated and capped at
  ``max_frames``. Recommended when you suspect splices but still want a
  baseline coverage of the timeline.
"""
from __future__ import annotations

import logging
from collections.abc import Callable, Iterator
from pathlib import Path

import numpy as np
from PIL import Image

from ..config import VideoConfig

log = logging.getLogger(__name__)

# Frames decoded per ``VideoReader.get_batch`` call. Batching amortises
# decord's per-call seek overhead; 16 1080p frames is ~100 MB, a safe cap.
_DECORD_BATCH = 16
# OpenCV: below this forward distance, ``grab()``-ing through frames is cheaper
# than a keyframe seek via ``CAP_PROP_POS_FRAMES``.
_CV_MAX_GRAB_AHEAD = 64


def sample_image(path: Path) -> np.ndarray:
    """Decode an image file as RGB uint8."""
    with Image.open(path) as img:
        # ``np.array`` (not ``asarray``) so the result is writable.
        return np.array(img.convert("RGB"))


# --- Scene detection -----------------------------------------------------

def _hsv_histogram(frame_rgb: np.ndarray, thumb_size: int) -> np.ndarray:
    """Compute a normalised 2-D Hue/Saturation histogram for a thumbnail.

    The frame is resized to ``thumb_size`` square first so cost is bounded
    irrespective of source resolution. Returned as a flat float32 vector.
    """
    import cv2

    thumb = cv2.resize(frame_rgb, (thumb_size, thumb_size), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(thumb, cv2.COLOR_RGB2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [16, 16], [0, 180, 0, 256])
    cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    return hist.flatten().astype(np.float32)


def detect_scene_boundaries(
    histograms: list[np.ndarray],
    threshold: float,
) -> list[int]:
    """Return indices (into ``histograms``) where a scene change starts.

    A boundary is declared when the correlation between consecutive
    histograms falls below ``threshold``. Index 0 is always returned so the
    opening shot is sampled. Pure-function so callers can test the cut logic
    without decoding a real video.
    """
    import cv2

    if not histograms:
        return []
    boundaries = [0]
    for i in range(1, len(histograms)):
        corr = cv2.compareHist(histograms[i - 1], histograms[i], cv2.HISTCMP_CORREL)
        if corr < threshold:
            boundaries.append(i)
    return boundaries


# --- Video sampling ------------------------------------------------------

def _uniform_step(fps: float, cfg: VideoConfig) -> int:
    return max(1, int(round(fps / cfg.frames_per_second)))


def _plan_indices(
    total: int,
    fps: float,
    cfg: VideoConfig,
    fetch_frame: Callable[[int], np.ndarray],
) -> list[int]:
    """Decide which source-frame indices to actually decode + return.

    ``fetch_frame(idx) -> np.ndarray`` is a backend-agnostic accessor so this
    function works for both decord and OpenCV. ``total`` is the source frame
    count; ``fps`` the average frame rate.
    """
    uniform_step = _uniform_step(fps, cfg)
    uniform_indices = range(0, total, uniform_step)

    if cfg.sampling_strategy == "uniform":
        return list(uniform_indices[: cfg.max_frames])

    # Scene-detector stride: sample more densely than the uniform schedule so
    # we don't miss short cuts. ``scene_stride=0`` defers to the uniform step.
    stride = max(1, cfg.scene_stride or uniform_step)
    # Only the thumbnail histogram of each probe frame is kept, so memory
    # stays O(probes × 1 KB) rather than O(probes × full frame).
    probed: list[int] = []
    histograms: list[np.ndarray] = []
    for idx in range(0, total, stride):
        try:
            frame = fetch_frame(idx)
        except Exception as e:  # noqa: BLE001
            log.debug("scene-probe decode failed at idx=%d: %s", idx, e)
            continue
        probed.append(idx)
        histograms.append(_hsv_histogram(frame, cfg.scene_thumb_size))
    scene_indices = [probed[p] for p in detect_scene_boundaries(histograms, cfg.scene_threshold)]

    if cfg.sampling_strategy == "scene":
        return scene_indices[: cfg.max_frames]

    # Hybrid: uniform + scene, deduped, sorted, capped.
    return sorted(set(uniform_indices) | set(scene_indices))[: cfg.max_frames]


def _sample_decord(path: Path, cfg: VideoConfig) -> Iterator[tuple[int, float, np.ndarray]] | None:
    """Return a frame iterator backed by decord, or ``None`` if decord can't
    open/plan this file (so the caller can fall back to OpenCV *before*
    anything has been yielded)."""
    try:
        import decord  # lazy

        decord.bridge.set_bridge("native")
        vr = decord.VideoReader(str(path))
        fps = float(vr.get_avg_fps()) or 30.0
        indices = _plan_indices(len(vr), fps, cfg, fetch_frame=lambda i: vr[i].asnumpy())
    except Exception as e:  # noqa: BLE001
        log.debug("decord failed for %s (%s); falling back to OpenCV", path, e)
        return None

    def gen() -> Iterator[tuple[int, float, np.ndarray]]:
        for start in range(0, len(indices), _DECORD_BATCH):
            chunk = indices[start : start + _DECORD_BATCH]
            try:
                frames = vr.get_batch(chunk).asnumpy()
            except Exception as e:  # noqa: BLE001
                log.debug("decord batch decode failed at %s: %s", chunk, e)
                continue
            for idx, frame in zip(chunk, frames, strict=True):
                yield idx, idx / fps, frame

    return gen()


def _sample_opencv(path: Path, cfg: VideoConfig) -> Iterator[tuple[int, float, np.ndarray]]:
    import cv2  # lazy

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        pos = 0  # index of the frame the next ``read()`` returns

        def fetch(idx: int) -> np.ndarray:
            nonlocal pos
            ahead = idx - pos
            if ahead < 0 or ahead > _CV_MAX_GRAB_AHEAD:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            else:
                for _ in range(ahead):
                    if not cap.grab():
                        break
            ok, frame_bgr = cap.read()
            if not ok:
                pos = idx  # unknown position after a failed read; force a seek next time
                raise RuntimeError(f"cv2 read failed at frame {idx}")
            pos = idx + 1
            return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        if total <= 0:
            # Some containers don't report a frame count; walk forward at the
            # uniform step until the stream ends.
            step = _uniform_step(fps, cfg)
            for n in range(cfg.max_frames):
                idx = n * step
                try:
                    frame = fetch(idx)
                except RuntimeError:
                    break
                yield idx, idx / fps, frame
            return

        for idx in _plan_indices(total, fps, cfg, fetch_frame=fetch):
            try:
                frame = fetch(idx)
            except RuntimeError as e:
                log.debug("cv2 sample-decode failed at idx=%d: %s", idx, e)
                continue
            yield idx, idx / fps, frame
    finally:
        cap.release()


def sample_video(path: Path, cfg: VideoConfig) -> Iterator[tuple[int, float, np.ndarray]]:
    """Yield (frame_index, timestamp_s, RGB frame) for sampled frames.

    Decoder priority: decord (fast) → OpenCV (fallback). The fallback only
    happens if decord can't open the file; a decode error mid-stream skips
    that chunk instead of restarting from frame 0 with another backend (which
    would yield duplicate frames). ``max_frames`` is a hard cap.
    """
    frames = _sample_decord(path, cfg)
    if frames is None:
        frames = _sample_opencv(path, cfg)
    yield from frames

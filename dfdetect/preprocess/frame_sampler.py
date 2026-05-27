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
from collections.abc import Iterator
from pathlib import Path

import numpy as np
from PIL import Image

from ..config import VideoConfig

log = logging.getLogger(__name__)


def sample_image(path: Path) -> np.ndarray:
    """Decode an image file as RGB uint8."""
    img = Image.open(path).convert("RGB")
    return np.array(img)


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


def _scene_indices(
    decoded: list[tuple[int, np.ndarray]],
    cfg: VideoConfig,
) -> list[int]:
    """Pick decoded-frame indices that lie on a scene boundary.

    ``decoded`` is a list of ``(frame_index_in_source, frame_rgb)`` pairs
    already downsampled to the scene-detector stride. Returns the subset of
    source frame indices to actually pass into the model.
    """
    histograms = [_hsv_histogram(f, cfg.scene_thumb_size) for _, f in decoded]
    boundary_positions = detect_scene_boundaries(histograms, cfg.scene_threshold)
    return [decoded[p][0] for p in boundary_positions]


# --- Video sampling ------------------------------------------------------

def _open_video_decord(path: Path):
    import decord  # lazy

    decord.bridge.set_bridge("native")
    return decord.VideoReader(str(path))


def _plan_indices(
    total: int,
    fps: float,
    cfg: VideoConfig,
    fetch_frame,
) -> list[int]:
    """Decide which source-frame indices to actually decode + return.

    ``fetch_frame(idx) -> np.ndarray`` is a backend-agnostic accessor so this
    function works for both decord and OpenCV. ``total`` is the source frame
    count; ``fps`` the average frame rate.
    """
    uniform_step = max(1, int(round(fps / cfg.frames_per_second)))
    uniform_indices = list(range(0, total, uniform_step))

    if cfg.sampling_strategy == "uniform":
        return uniform_indices[: cfg.max_frames]

    # Scene-detector stride: sample more densely than the uniform schedule so
    # we don't miss short cuts. ``scene_stride=0`` defers to the uniform step.
    stride = cfg.scene_stride or uniform_step
    stride = max(1, stride)
    probe_indices = list(range(0, total, stride))
    decoded: list[tuple[int, np.ndarray]] = []
    for idx in probe_indices:
        try:
            frame = fetch_frame(idx)
        except Exception as e:  # noqa: BLE001
            log.debug("scene-probe decode failed at idx=%d: %s", idx, e)
            continue
        decoded.append((idx, frame))
    scene_indices = _scene_indices(decoded, cfg)

    if cfg.sampling_strategy == "scene":
        return scene_indices[: cfg.max_frames]

    # Hybrid: uniform + scene, deduped, sorted, capped.
    merged = sorted(set(uniform_indices) | set(scene_indices))
    return merged[: cfg.max_frames]


def sample_video(path: Path, cfg: VideoConfig) -> Iterator[tuple[int, float, np.ndarray]]:
    """Yield (frame_index, timestamp_s, RGB frame) for sampled frames.

    Decoder priority: decord (fast) → OpenCV (fallback). The sampling schedule
    is decided by ``cfg.sampling_strategy``; ``max_frames`` is a hard cap.
    """
    try:
        vr = _open_video_decord(path)
        fps = float(vr.get_avg_fps()) or 30.0
        total = len(vr)
        indices = _plan_indices(total, fps, cfg, fetch_frame=lambda i: vr[i].asnumpy())
        for idx in indices:
            frame = vr[idx].asnumpy()
            yield idx, idx / fps, frame
        return
    except Exception as e:  # noqa: BLE001
        log.debug("decord failed (%s); falling back to OpenCV", e)

    import cv2  # lazy

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    def _fetch_cv(idx: int) -> np.ndarray:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame_bgr = cap.read()
        if not ok:
            raise RuntimeError(f"cv2 read failed at frame {idx}")
        return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    try:
        if total <= 0:
            # Old containers sometimes lie about frame count; fall back to the
            # legacy linear walk so we still produce *something*.
            step = max(1, int(round(fps / cfg.frames_per_second)))
            yielded = 0
            idx = 0
            while yielded < cfg.max_frames:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ok, frame_bgr = cap.read()
                if not ok:
                    break
                yield idx, idx / fps, cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                yielded += 1
                idx += step
            return

        indices = _plan_indices(total, fps, cfg, fetch_frame=_fetch_cv)
        for idx in indices:
            try:
                frame_rgb = _fetch_cv(idx)
            except Exception as e:  # noqa: BLE001
                log.debug("cv2 sample-decode failed at idx=%d: %s", idx, e)
                continue
            yield idx, idx / fps, frame_rgb
    finally:
        cap.release()

"""Face detection wrappers. Currently MTCNN; RetinaFace can plug in here."""
from __future__ import annotations

import logging
import os
from typing import Protocol

import numpy as np

from ..config import PreprocessConfig
from ..types import FaceCrop

log = logging.getLogger(__name__)


class FaceDetector(Protocol):
    def detect(
        self,
        image: np.ndarray,
        frame_index: int = 0,
        timestamp_s: float = 0.0,
    ) -> list[FaceCrop]:
        ...


def expand_bbox(
    bbox: tuple[float, float, float, float],
    margin: float,
    img_shape: tuple[int, int],
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox
    w, h = x2 - x1, y2 - y1
    pad_x, pad_y = w * margin, h * margin
    H, W = img_shape[:2]
    nx1 = int(max(0, x1 - pad_x))
    ny1 = int(max(0, y1 - pad_y))
    nx2 = int(min(W, x2 + pad_x))
    ny2 = int(min(H, y2 + pad_y))
    return nx1, ny1, nx2, ny2


class MTCNNDetector:
    """facenet-pytorch MTCNN wrapper. Lazy-imported so the package is usable for tests
    without the heavy ML deps installed."""

    def __init__(self, cfg: PreprocessConfig, device: str = "cpu"):
        from facenet_pytorch import MTCNN  # lazy

        self._cfg = cfg
        # ``select_largest=True`` makes ``detect()`` return boxes sorted by
        # area (largest first). ``keep_all`` only affects ``forward()``, which
        # we don't use — ``detect()`` always returns every box.
        self._mtcnn = MTCNN(
            min_face_size=cfg.min_face_size,
            device=device,
            post_process=False,
            select_largest=True,
        )

    def detect(
        self,
        image: np.ndarray,
        frame_index: int = 0,
        timestamp_s: float = 0.0,
    ) -> list[FaceCrop]:
        # MTCNN expects RGB
        boxes, probs = self._mtcnn.detect(image)
        if boxes is None:
            return []
        if probs is None:
            probs = [1.0] * len(boxes)
        if self._cfg.largest_face_only:
            boxes, probs = boxes[:1], probs[:1]
        crops: list[FaceCrop] = []
        for box, prob in zip(boxes, probs, strict=True):
            x1, y1, x2, y2 = expand_bbox(tuple(box), self._cfg.face_margin, image.shape)
            if x2 <= x1 or y2 <= y1:
                continue
            crop = image[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            crops.append(
                FaceCrop(
                    image=crop,
                    bbox=(x1, y1, x2, y2),
                    frame_index=frame_index,
                    timestamp_s=timestamp_s,
                    detector_confidence=float(prob) if prob is not None else 1.0,
                )
            )
        return crops


class RetinaFaceDetector:
    """insightface FaceAnalysis (RetinaFace) wrapper.

    insightface eats BGR and returns ``Face`` objects with ``bbox`` and
    ``det_score`` fields; we convert from the RGB the rest of the pipeline
    passes around and apply the same margin/crop logic as :class:`MTCNNDetector`.
    """

    def __init__(self, cfg: PreprocessConfig, device: str = "cpu"):
        from insightface.app import FaceAnalysis  # lazy

        self._cfg = cfg
        ctx_id = 0 if device.startswith("cuda") else -1
        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if device.startswith("cuda")
            else ["CPUExecutionProvider"]
        )
        # ``buffalo_l`` is insightface's default pack; override with the
        # INSIGHTFACE_MODEL env var if you've downloaded a different one. Only
        # the detection module is loaded (no embeddings/landmarks).
        model_name = os.environ.get("INSIGHTFACE_MODEL", "buffalo_l")
        self._app = FaceAnalysis(
            name=model_name, allowed_modules=["detection"], providers=providers
        )
        # det_size must be a multiple of 32; 640 is the insightface default.
        self._app.prepare(ctx_id=ctx_id, det_size=(640, 640))

    def detect(
        self,
        image: np.ndarray,
        frame_index: int = 0,
        timestamp_s: float = 0.0,
    ) -> list[FaceCrop]:
        import cv2

        bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        faces = self._app.get(bgr)
        if not faces:
            return []

        # Filter sub-min-size faces and (optionally) keep only the largest one.
        filtered: list = []
        for f in faces:
            x1, y1, x2, y2 = f.bbox
            if (x2 - x1) < self._cfg.min_face_size or (y2 - y1) < self._cfg.min_face_size:
                continue
            filtered.append(f)
        if not filtered:
            return []
        if self._cfg.largest_face_only:
            filtered = [
                max(
                    filtered,
                    key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
                )
            ]

        crops: list[FaceCrop] = []
        for f in filtered:
            x1, y1, x2, y2 = expand_bbox(
                tuple(f.bbox), self._cfg.face_margin, image.shape
            )
            if x2 <= x1 or y2 <= y1:
                continue
            crop = image[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            crops.append(
                FaceCrop(
                    image=crop,
                    bbox=(x1, y1, x2, y2),
                    frame_index=frame_index,
                    timestamp_s=timestamp_s,
                    detector_confidence=float(getattr(f, "det_score", 1.0)),
                )
            )
        return crops


def build_detector(cfg: PreprocessConfig, device: str = "cpu") -> FaceDetector:
    if cfg.face_detector == "mtcnn":
        return MTCNNDetector(cfg, device=device)
    if cfg.face_detector == "retinaface":
        try:
            import insightface  # noqa: F401
        except ImportError as e:
            raise RuntimeError(
                "RetinaFace requested but `insightface` is not installed. "
                "Install it with `pip install insightface onnxruntime` "
                "(or `onnxruntime-gpu` for CUDA) and retry, or fall back to mtcnn."
            ) from e
        return RetinaFaceDetector(cfg, device=device)
    raise ValueError(f"Unknown face detector: {cfg.face_detector}")

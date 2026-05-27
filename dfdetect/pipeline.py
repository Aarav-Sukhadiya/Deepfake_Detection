"""End-to-end orchestration: paths in, FileResults out."""
from __future__ import annotations

import logging
import queue
import threading
import time
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .config import Config
from .inference.aggregator import aggregate, derive_segments, verdict_from_score
from .inference.runner import InferenceRunner
from .input.loader import discover
from .models.registry import build_model
from .preprocess.face_detect import build_detector
from .preprocess.frame_sampler import sample_image, sample_video
from .preprocess.tracker import assign_tracks
from .report.heatmap import save_grad_cam
from .storage.db import ResultsStore
from .types import FaceCrop, FileResult, FrameScore, MediaItem, MediaKind, Verdict

log = logging.getLogger(__name__)


def _resolve_device(device: str) -> str:
    if device != "auto":
        return device
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:  # noqa: BLE001
        return "cpu"


class Pipeline:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.device = _resolve_device(cfg.device)
        self.model = build_model(cfg.model, device=self.device, weights_dir=cfg.storage.models_dir)
        self.detector = build_detector(cfg.preprocess, device=self.device)
        self.runner = InferenceRunner(self.model, batch_size=cfg.batch_size)
        self.store = ResultsStore(cfg.storage.db_path)

    def _faces_from_item(self, item: MediaItem) -> list[FaceCrop]:
        if item.kind == MediaKind.IMAGE:
            img = sample_image(item.path)
            return self.detector.detect(img, frame_index=0, timestamp_s=0.0)
        # video
        all_faces: list[FaceCrop] = []
        for idx, t, frame in sample_video(item.path, self.cfg.video):
            faces = self.detector.detect(frame, frame_index=idx, timestamp_s=t)
            all_faces.extend(faces)

        # If multi-face detection is on, group faces into per-identity tracks
        # so the aggregator can produce one score per identity (and the
        # per-file score is the max across identities by default). The
        # tracker is a no-op for single-face runs since every face will land
        # in the same track.
        if not self.cfg.preprocess.largest_face_only and all_faces:
            assign_tracks(all_faces)
        return all_faces

    def _preprocess_item(self, item: MediaItem) -> tuple[list[FaceCrop], FileResult | None, float]:
        """Pre-extract faces. Returns (faces, early_result, t_start).

        ``early_result`` is non-None when preprocessing failed or produced
        zero faces — in those cases the caller skips inference and uses the
        early result directly.
        """
        start = time.perf_counter()
        try:
            faces = self._faces_from_item(item)
        except Exception as e:  # noqa: BLE001
            log.exception("Preprocess failed for %s", item.path)
            return [], FileResult(
                path=item.path,
                kind=item.kind,
                verdict=Verdict.ERROR,
                score=0.0,
                processing_time_s=time.perf_counter() - start,
                model_name=self.model.name,
                model_hash=self.model.hash(),
                error=f"preprocess: {e}",
            ), start
        if not faces:
            return [], FileResult(
                path=item.path,
                kind=item.kind,
                verdict=Verdict.NO_FACE,
                score=0.0,
                processing_time_s=time.perf_counter() - start,
                model_name=self.model.name,
                model_hash=self.model.hash(),
            ), start
        return faces, None, start

    def _score_item(
        self, item: MediaItem, faces: list[FaceCrop], start: float
    ) -> FileResult:
        try:
            scores = self.runner.run(faces)
        except Exception as e:  # noqa: BLE001
            log.exception("Inference failed for %s", item.path)
            return FileResult(
                path=item.path,
                kind=item.kind,
                verdict=Verdict.ERROR,
                score=0.0,
                faces_analyzed=len(faces),
                processing_time_s=time.perf_counter() - start,
                model_name=self.model.name,
                model_hash=self.model.hash(),
                error=f"inference: {e}",
            )

        frame_scores = [
            FrameScore(frame_index=f.frame_index, timestamp_s=f.timestamp_s, score=s)
            for f, s in zip(faces, scores)
        ]
        weights = [f.quality_weight() for f in faces]

        # Per-identity aggregation: if the tracker assigned track_ids, compute
        # a score per track first, then collapse the tracks. Otherwise fall
        # back to a single aggregation across all faces.
        track_agg = self.cfg.inference.track_aggregation
        has_tracks = any(f.track_id is not None for f in faces)
        if has_tracks and track_agg != "off":
            per_track: dict[int, tuple[list[float], list[float]]] = {}
            for f, s, w in zip(faces, scores, weights):
                tid = f.track_id if f.track_id is not None else -1
                bucket = per_track.setdefault(tid, ([], []))
                bucket[0].append(s)
                bucket[1].append(w)
            track_scores = [
                aggregate(s_list, self.cfg.inference, weights=w_list)
                for s_list, w_list in per_track.values()
            ]
            agg = max(track_scores) if track_agg == "max" else (
                sum(track_scores) / len(track_scores)
            )
        else:
            agg = aggregate(scores, self.cfg.inference, weights=weights)

        verdict = verdict_from_score(
            agg, self.cfg.threshold, self.cfg.uncertain_band, len(faces)
        )

        heatmap_path: Path | None = None
        if self.cfg.report.heatmaps:
            top = max(zip(faces, scores), key=lambda fs: fs[1])[0]
            heatmap_path = save_grad_cam(
                self.model,
                top.image,
                self.cfg.report.output_dir / "heatmaps" / f"{item.path.stem}.png",
            )

        # Segments only make sense for videos with multi-frame coverage.
        segments = (
            derive_segments(frame_scores, self.cfg.threshold)
            if item.kind == MediaKind.VIDEO
            else []
        )

        return FileResult(
            path=item.path,
            kind=item.kind,
            verdict=verdict,
            score=agg,
            faces_analyzed=len(faces),
            frame_scores=frame_scores,
            segments=segments,
            processing_time_s=time.perf_counter() - start,
            model_name=self.model.name,
            model_hash=self.model.hash(),
            heatmap_path=heatmap_path,
        )

    def _result_for_item(self, item: MediaItem) -> FileResult:
        faces, early, start = self._preprocess_item(item)
        if early is not None:
            return early
        return self._score_item(item, faces, start)

    def _run_parallel(
        self,
        items: list[MediaItem],
        workers: int,
        progress,
    ) -> list[FileResult]:
        """Preprocess in a thread pool, run inference serially.

        The model isn't safely shared across threads (especially CUDA), so we
        keep ``runner.run`` on the main thread. Preprocessing (decode + face
        detect) is the CPU-bound step that benefits most from threading. A
        bounded queue back-pressures the producer when inference can't keep
        up.
        """
        prefetch = max(1, self.cfg.pipeline.prefetch)
        out_queue: queue.Queue[tuple[int, MediaItem, list[FaceCrop], FileResult | None, float]] = (
            queue.Queue(maxsize=prefetch)
        )
        stop_event = threading.Event()
        results: list[FileResult | None] = [None] * len(items)

        def produce():
            """Submit at most ``workers + prefetch`` items into the pool at any
            time, in input order. Bounds memory growth on very large runs and
            keeps the futures-in-flight window small enough that ``stop_event``
            actually stops new work promptly."""
            in_flight: int = workers + prefetch
            with ThreadPoolExecutor(max_workers=workers) as pool:
                pending: list[tuple[int, MediaItem, "Future"]] = []  # type: ignore[name-defined]
                it = enumerate(items)
                # Prime the window.
                for _ in range(in_flight):
                    try:
                        idx, item = next(it)
                    except StopIteration:
                        break
                    pending.append((idx, item, pool.submit(self._preprocess_item, item)))
                while pending:
                    if stop_event.is_set():
                        return
                    idx, item, fut = pending.pop(0)
                    faces, early, start = fut.result()
                    out_queue.put((idx, item, faces, early, start))
                    # Refill the window so the next file is already being
                    # preprocessed by the time we finish scoring this one.
                    try:
                        nidx, nitem = next(it)
                        pending.append((nidx, nitem, pool.submit(self._preprocess_item, nitem)))
                    except StopIteration:
                        pass
            out_queue.put((-1, None, [], None, 0.0))  # sentinel

        thread = threading.Thread(target=produce, daemon=True)
        thread.start()
        try:
            done = 0
            while True:
                idx, item, faces, early, start = out_queue.get()
                if idx == -1:
                    break
                if progress is not None:
                    progress(done, len(items), item.path.name)
                if early is not None:
                    results[idx] = early
                else:
                    results[idx] = self._score_item(item, faces, start)
                done += 1
        finally:
            stop_event.set()
            thread.join(timeout=1.0)
        if progress is not None:
            progress(len(items), len(items), "done")
        return [r for r in results if r is not None]

    def run(
        self,
        paths: Iterable[str | Path],
        progress=None,
    ) -> list[FileResult]:
        items = discover(paths, self.cfg.input)
        workers = self.cfg.pipeline.workers
        if workers <= 0 or len(items) <= 1:
            results: list[FileResult] = []
            for i, item in enumerate(items):
                if progress is not None:
                    progress(i, len(items), item.path.name)
                results.append(self._result_for_item(item))
            if progress is not None:
                progress(len(items), len(items), "done")
        else:
            results = self._run_parallel(items, workers, progress)
        self.store.save_run(
            results,
            model_name=self.model.name,
            model_hash=self.model.hash(),
            config_dict=self.cfg.model_dump(mode="json"),
        )
        return results

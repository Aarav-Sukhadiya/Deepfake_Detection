"""Minimal fine-tuning loop. Lazy-imports torch so the rest of the package
remains usable on machines that only have CPU-only inference deps.

Both class folders are walked **recursively** via
:func:`dfdetect.training.walk_media` and may contain images and videos at any
depth. Videos are subsampled at ``frames_per_second`` and contribute one face
crop per sampled frame; images contribute one crop each.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from pathlib import Path

from ..models.downloader import compute_sha256
from .constraints import DatasetStats, enforce_constraints, summarise_dataset, walk_media

log = logging.getLogger(__name__)


def _config_signature(cfg: "TrainConfig") -> str:
    """Fingerprint the parts of ``cfg`` that, if changed, would invalidate a
    resume — i.e. anything that affects model architecture, dataset, or
    sample selection. ``epochs`` is *intentionally excluded* so a user can
    extend a run by re-invoking with more epochs and have it pick up where
    the last epoch finished."""
    payload = {
        "real_dir": str(Path(cfg.real_dir).resolve()),
        "fake_dir": str(Path(cfg.fake_dir).resolve()),
        "backbone": cfg.backbone,
        "batch_size": cfg.batch_size,
        "learning_rate": cfg.learning_rate,
        "val_split": cfg.val_split,
        "seed": cfg.seed,
        "augment": cfg.augment,
        "video_fps": cfg.video_fps,
        "max_frames_per_video": cfg.max_frames_per_video,
        "face_detection": cfg.face_detection,
    }
    return hashlib.sha1(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


def _atomic_save_torch(obj, path: Path) -> None:  # pragma: no cover - touches ML stack
    """Save via a tmp file + rename so a Ctrl-C mid-write doesn't corrupt the
    resume checkpoint."""
    import torch

    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def _train_autotune_path() -> Path:
    base = os.environ.get("DFDETECT_CACHE") or os.path.expanduser("~/.cache/dfdetect")
    return Path(base) / "autotune_train.json"


def _train_autotune_key(cfg: "TrainConfig", device_name: str) -> str:
    """The OOM-safe batch size depends on backbone, input size, AMP, and GPU
    memory. Same key collides only on identical model/device combos."""
    return f"{cfg.backbone}:224:amp{int(cfg.amp)}:{device_name.replace(' ', '_')}"


def _load_train_autotune(key: str) -> int | None:
    p = _train_autotune_path()
    if not p.exists():
        return None
    try:
        data = json.loads(p.read_text())
    except Exception:  # noqa: BLE001
        return None
    val = data.get(key)
    return int(val) if isinstance(val, int) and val >= 1 else None


def _save_train_autotune(key: str, batch_size: int) -> None:
    p = _train_autotune_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    try:
        data = json.loads(p.read_text()) if p.exists() else {}
    except Exception:  # noqa: BLE001
        data = {}
    data[key] = int(batch_size)
    try:
        p.write_text(json.dumps(data, indent=2))
    except Exception:  # noqa: BLE001
        log.debug("Could not persist training-autotune state to %s", p)


def _resolve_device(requested: str):  # pragma: no cover - touches ML stack
    """Map ``"auto"`` / ``"cuda"`` / ``"cpu"`` (or explicit ``"cuda:0"``) to a
    :class:`torch.device`. Falls back to CPU with a warning if the user asks
    for CUDA on a machine that doesn't have it."""
    import torch

    req = (requested or "auto").lower()
    if req == "auto":
        return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    if req.startswith("cuda"):
        if not torch.cuda.is_available():
            log.warning("CUDA requested but not available; falling back to CPU.")
            return torch.device("cpu")
        return torch.device(req)
    return torch.device("cpu")


@dataclass
class TrainConfig:
    real_dir: Path
    fake_dir: Path
    output_dir: Path
    epochs: int = 5
    learning_rate: float = 1e-4
    batch_size: int = 16
    val_split: float = 0.1
    seed: int = 42
    augment: bool = True
    backbone: str = "tf_efficientnet_b4"
    # Video sampling for training data — we don't need every frame, and
    # frames inside one clip are heavily correlated.
    video_fps: float = 1.0
    max_frames_per_video: int = 32
    # Set ``False`` for already-face-cropped datasets (e.g. FFHQ / StyleGAN
    # 256² faces, the Kaggle 140K Real-and-Fake set). MTCNN on a pre-cropped
    # face is wasted Python loop time — the whole image already IS the face,
    # so we skip detection and just resize. Leaves the inference-time face
    # detector untouched.
    face_detection: bool = True
    # Resume from ``output_dir/resume.pt`` (written after each epoch) when
    # set, allowing Ctrl-C to be safe: re-running the same command picks up
    # from the last completed epoch. Disable for clean-slate retrains.
    resume: bool = True
    # ``auto`` → CUDA if available, else CPU. Override with ``cuda`` /
    # ``cuda:0`` / ``cpu``. PyTorch only sees NVIDIA / supported-vendor GPUs
    # via CUDA — an Intel/AMD iGPU is invisible here regardless of how the
    # OS exposes it.
    device: str = "auto"
    # Mixed-precision (float16 autocast + GradScaler). Roughly 2x faster on
    # any post-Volta NVIDIA GPU with no accuracy loss for this workload.
    # Auto-enabled on CUDA, no-op on CPU. Disable for strict-fp32 debugging.
    amp: bool = True


def _detect_face(rgb, mtcnn):  # pragma: no cover - touches ML stack
    """Largest face crop from one RGB frame, or ``None`` if MTCNN finds nothing."""
    box, _ = mtcnn.detect(rgb)
    if box is None or len(box) == 0:
        return None
    x1, y1, x2, y2 = [int(v) for v in box[0]]
    if x2 <= x1 or y2 <= y1:
        return None
    crop = rgb[max(0, y1):y2, max(0, x1):x2]
    return crop if crop.size > 0 else None


def _image_crops(path: Path, mtcnn) -> Iterator:  # pragma: no cover - touches ML stack
    """Yield the largest face crop from an image, if any.

    When ``mtcnn`` is ``None`` we skip detection and yield the whole image —
    used for already-face-cropped datasets where the image *is* the crop.
    """
    import cv2

    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        return
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    if mtcnn is None:
        yield rgb
        return
    crop = _detect_face(rgb, mtcnn)
    if crop is not None:
        yield crop


def _video_crops(  # pragma: no cover - touches ML stack
    path: Path,
    mtcnn,
    fps: float,
    max_frames: int,
) -> Iterator:
    """Yield face crops from sampled frames of a video.

    When ``mtcnn`` is ``None`` we yield each sampled frame as-is (whole-frame
    training, no detection)."""
    """Yield one face crop per sampled frame from a video.

    Sampling matches the inference path: every ``round(source_fps / fps)``
    frames, capped at ``max_frames``. ``decord`` first (fast), OpenCV fallback.
    """
    import cv2

    indices: list[int]
    rgb_frames: list = []
    try:
        import decord

        decord.bridge.set_bridge("native")
        vr = decord.VideoReader(str(path))
        src_fps = float(vr.get_avg_fps()) or 30.0
        step = max(1, int(round(src_fps / max(fps, 1e-3))))
        indices = list(range(0, len(vr), step))[:max_frames]
        for idx in indices:
            rgb_frames.append(vr[idx].asnumpy())
    except Exception:  # noqa: BLE001
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            return
        src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        step = max(1, int(round(src_fps / max(fps, 1e-3))))
        try:
            i = 0
            yielded = 0
            while yielded < max_frames:
                cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                ok, bgr = cap.read()
                if not ok:
                    break
                rgb_frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
                yielded += 1
                i += step
        finally:
            cap.release()

    for rgb in rgb_frames:
        if mtcnn is None:
            yield rgb
            continue
        crop = _detect_face(rgb, mtcnn)
        if crop is not None:
            yield crop


@dataclass
class _Sample:
    """One training example, identified by path + (for videos) a frame index.

    Decoded lazily inside the Dataset's ``__getitem__`` so a 100K-image dataset
    doesn't have to fit in RAM (256² uint8 × 100K ≈ 20 GB)."""
    path: Path
    label: int
    frame_index: int | None = None  # ``None`` for still images


def _video_frame_indices(  # pragma: no cover - touches ML stack
    path: Path, fps: float, max_frames: int
) -> list[int]:
    """Compute the frame indices we'd sample from this video.

    Performed eagerly so the Dataset has a fixed length up front. The cost is
    one cheap header read per video — full decoding stays in ``__getitem__``.
    """
    import cv2

    try:
        import decord

        decord.bridge.set_bridge("native")
        vr = decord.VideoReader(str(path))
        src_fps = float(vr.get_avg_fps()) or 30.0
        step = max(1, int(round(src_fps / max(fps, 1e-3))))
        return list(range(0, len(vr), step))[:max_frames]
    except Exception:  # noqa: BLE001
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            return []
        src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        cap.release()
        if total <= 0:
            return []
        step = max(1, int(round(src_fps / max(fps, 1e-3))))
        return list(range(0, total, step))[:max_frames]


def _collect_samples(  # pragma: no cover - filesystem-bound
    class_dir: Path,
    label: int,
    fps: float,
    max_frames: int,
    progress=None,
) -> list[_Sample]:
    """Walk a class folder and return a list of :class:`_Sample` pointers.

    No image bytes are decoded here — we only record (path, optional frame
    index, label) so the Dataset can decode on demand during training.
    """
    images, videos = walk_media(class_dir)
    samples: list[_Sample] = [_Sample(path=p, label=label) for p in images]
    total = len(images) + len(videos)
    for i, p in enumerate(videos):
        if progress is not None:
            progress(len(images) + i, total, p.name)
        for idx in _video_frame_indices(p, fps=fps, max_frames=max_frames):
            samples.append(_Sample(path=p, label=label, frame_index=idx))
    if progress is not None:
        progress(total, total, "done")
    return samples


# Module-level MTCNN — created once on first use so we don't pay the
# constructor cost per Dataset item.
_mtcnn_singleton = None


def _get_mtcnn():  # pragma: no cover - touches ML stack
    global _mtcnn_singleton
    if _mtcnn_singleton is None:
        from facenet_pytorch import MTCNN

        _mtcnn_singleton = MTCNN(
            keep_all=False, post_process=False, device="cpu", select_largest=True
        )
    return _mtcnn_singleton


def _decode_sample_rgb(sample: _Sample):  # pragma: no cover - touches ML stack
    """Return the RGB frame for ``sample`` (image or video frame) or None."""
    import cv2

    if sample.frame_index is None:
        bgr = cv2.imread(str(sample.path), cv2.IMREAD_COLOR)
        if bgr is None:
            return None
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    # Video frame — try decord then fall back to OpenCV.
    try:
        import decord

        decord.bridge.set_bridge("native")
        vr = decord.VideoReader(str(sample.path))
        return vr[sample.frame_index].asnumpy()
    except Exception:  # noqa: BLE001
        cap = cv2.VideoCapture(str(sample.path))
        if not cap.isOpened():
            return None
        try:
            cap.set(cv2.CAP_PROP_POS_FRAMES, sample.frame_index)
            ok, bgr = cap.read()
            if not ok:
                return None
            return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        finally:
            cap.release()


def train(cfg: TrainConfig, progress=None) -> Path:  # pragma: no cover - touches ML stack
    """Train and write ``cfg.output_dir/model.pt`` plus a ``train.json`` manifest.

    Hard anti-bias constraints are enforced *before* any GPU/CPU work so
    misconfigured folders fail fast. ``progress(phase, done, total, label)``
    is called repeatedly so the Gradio UI can stream status updates; pass
    ``None`` for the CLI path. Returns the new checkpoint path.
    """
    stats = summarise_dataset(cfg.real_dir, cfg.fake_dir)
    enforce_constraints(stats)
    for warn in stats.warnings:
        log.warning("[training] %s", warn)

    import cv2
    import numpy as np
    import timm
    import torch
    from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

    _MEAN_ARR = np.array((0.485, 0.456, 0.406), dtype=np.float32)
    _STD_ARR = np.array((0.229, 0.224, 0.225), dtype=np.float32)

    class _LazyCropDataset(Dataset):
        """Decode + (optional) detect + resize + normalise per ``__getitem__``.

        Returns ``None`` when face detection fails on a sample; a custom
        ``collate_fn`` filters those out so the DataLoader yields variable-size
        batches instead of crashing."""

        def __init__(self, samples: list[_Sample], train: bool):
            self.samples = samples
            self.train = train

        def __len__(self) -> int:
            return len(self.samples)

        def __getitem__(self, idx: int):
            sample = self.samples[idx]
            rgb = _decode_sample_rgb(sample)
            if rgb is None:
                return None
            if cfg.face_detection:
                rgb = _detect_face(rgb, _get_mtcnn())
                if rgb is None:
                    return None
            crop = cv2.resize(rgb, (224, 224), interpolation=cv2.INTER_AREA)
            if self.train and cfg.augment and np.random.rand() < 0.5:
                crop = crop[:, ::-1].copy()
            x = crop.astype(np.float32, copy=False) * (1.0 / 255.0)
            x -= _MEAN_ARR
            x /= _STD_ARR
            x = np.ascontiguousarray(x.transpose(2, 0, 1))
            return torch.from_numpy(x), torch.tensor(sample.label, dtype=torch.long)

    def _collate_drop_none(batch):
        batch = [b for b in batch if b is not None]
        if not batch:
            return None
        xs = torch.stack([b[0] for b in batch])
        ys = torch.stack([b[1] for b in batch])
        return xs, ys

    def _phase(name):
        if progress is None:
            return lambda *a, **k: None
        return lambda done, total, label: progress(name, done, total, label)

    log.info("Indexing real folder: %s", cfg.real_dir)
    samples: list[_Sample] = []
    samples += _collect_samples(
        cfg.real_dir, label=0,
        fps=cfg.video_fps, max_frames=cfg.max_frames_per_video,
        progress=_phase("index-real"),
    )
    log.info("Indexing fake folder: %s", cfg.fake_dir)
    samples += _collect_samples(
        cfg.fake_dir, label=1,
        fps=cfg.video_fps, max_frames=cfg.max_frames_per_video,
        progress=_phase("index-fake"),
    )
    if not samples:
        raise RuntimeError(
            "No samples found in the labelled folders. Check the paths and "
            "that the files have supported image/video extensions."
        )

    rng = np.random.default_rng(cfg.seed)
    rng.shuffle(samples)
    val_n = max(1, int(len(samples) * cfg.val_split))
    val, train_split = samples[:val_n], samples[val_n:]

    # Class-balanced sampler: weight each row by ``1 / class_count`` so a
    # batch is balanced even when one class has more samples than the other.
    counts = {0: 0, 1: 0}
    for s in train_split:
        counts[s.label] += 1
    weights = [1.0 / max(1, counts[s.label]) for s in train_split]
    sampler = WeightedRandomSampler(weights, num_samples=len(train_split), replacement=True)

    log.info(
        "Dataset: %d total samples — train=%d (real=%d, fake=%d), val=%d. "
        "face_detection=%s, batch_size=%d, epochs=%d, lr=%g.",
        len(samples), len(train_split), counts[0], counts[1], len(val),
        cfg.face_detection, cfg.batch_size, cfg.epochs, cfg.learning_rate,
    )

    device = _resolve_device(cfg.device)
    use_cuda = device.type == "cuda"
    use_amp = bool(cfg.amp) and use_cuda
    if use_cuda:
        # cuDNN's autotuner picks the fastest conv algo for the (input,
        # weight) shapes we'll see. Inputs are fixed at 224x224x3, so the
        # tuning is one-off.
        torch.backends.cudnn.benchmark = True
        gpu_name = torch.cuda.get_device_name(device)
        gpu_mem = torch.cuda.get_device_properties(device).total_memory / 1e9
        log.info(
            "Training device: %s (%s, %.1f GB). AMP=%s, cudnn.benchmark=True.",
            device, gpu_name, gpu_mem, use_amp,
        )
    else:
        log.info("Training device: %s. AMP disabled (CPU).", device)

    model = timm.create_model(cfg.backbone, pretrained=True, num_classes=2).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate)
    loss_fn = torch.nn.CrossEntropyLoss().to(device)
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    # --- Auto-tune the actual per-step (micro-)batch size ---------------
    # The user's ``cfg.batch_size`` is the *effective* batch. We probe to
    # find the largest size that actually fits in VRAM, then use gradient
    # accumulation to preserve the effective batch's gradient semantics.
    requested_bs = max(1, cfg.batch_size)
    autotune_key = _train_autotune_key(
        cfg, torch.cuda.get_device_name(device) if use_cuda else "cpu"
    )
    cached_bs = _load_train_autotune(autotune_key) if use_cuda else None
    micro_bs = min(requested_bs, cached_bs) if cached_bs else requested_bs

    def _probe(bs: int) -> bool:
        """Try a few real training steps at ``bs`` to detect OOM.

        A single forward+backward isn't enough — AdamW only materialises
        its 2× param-size optimizer state after the *first* ``step()``, and
        cuDNN may pick a heavier-workspace conv algorithm after a handful
        of warm-up iterations. We run 3 full steps (forward + backward +
        optimizer step) so the peak memory has actually been exercised.

        The probe perturbs ``model`` and ``opt`` slightly (it's 3 real
        training updates), but they happen on real training data and would
        have been taken in the normal training loop anyway — net effect is
        ~0.05% of an epoch."""
        if not use_cuda or not train_split:
            return True
        n_steps = min(3, max(1, len(train_split) // max(1, bs)))
        try:
            probe_samples = train_split[: bs * n_steps]
            probe_loader = DataLoader(
                _LazyCropDataset(probe_samples, train=True),
                batch_size=bs, num_workers=0,
                collate_fn=_collate_drop_none, pin_memory=True,
            )
            steps = 0
            for batch in probe_loader:
                if batch is None:
                    continue
                xb, yb = batch
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
                opt.zero_grad(set_to_none=True)
                with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                    logits = model(xb)
                    loss = loss_fn(logits, yb)
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.step(opt)
                    scaler.update()
                else:
                    loss.backward()
                    opt.step()
                steps += 1
                if steps >= n_steps:
                    break
            opt.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            return True
        except torch.cuda.OutOfMemoryError:
            opt.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            return False

    while micro_bs > 1 and not _probe(micro_bs):
        log.warning("CUDA OOM probing batch_size=%d; halving.", micro_bs)
        micro_bs = max(1, micro_bs // 2)
    if not _probe(micro_bs):
        # Even bs=1 OOMs — nothing we can do.
        raise RuntimeError(
            f"Even batch_size=1 OOMs on {device}. EfficientNet-B4 needs ~6 GB "
            "in fp16. Free GPU memory (close other processes) or use --device cpu."
        )

    grad_accum = max(1, requested_bs // micro_bs)
    if use_cuda:
        _save_train_autotune(autotune_key, micro_bs)
    if micro_bs != requested_bs:
        log.info(
            "Auto-tuned batch_size: requested=%d, per-step micro_bs=%d, "
            "grad_accum=%d (effective batch=%d). Persisted to %s.",
            requested_bs, micro_bs, grad_accum, micro_bs * grad_accum,
            _train_autotune_path() if use_cuda else "(not persisted)",
        )

    def _build_loaders(bs: int):
        return (
            DataLoader(
                _LazyCropDataset(train_split, train=True),
                batch_size=bs, sampler=sampler, num_workers=0,
                collate_fn=_collate_drop_none, pin_memory=use_cuda,
            ),
            DataLoader(
                _LazyCropDataset(val, train=False),
                batch_size=bs, num_workers=0,
                collate_fn=_collate_drop_none, pin_memory=use_cuda,
            ),
        )

    train_loader, val_loader = _build_loaders(micro_bs)

    best_val_loss = float("inf")
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = cfg.output_dir / "model.pt"
    resume_path = cfg.output_dir / "resume.pt"

    # Resume: only honour a previous checkpoint if the config signature still
    # matches. A changed dataset / LR / backbone would invalidate the optimizer
    # state, so we'd silently train on stale momentum — better to fail loud.
    cfg_sig = _config_signature(cfg)
    start_epoch = 0
    if cfg.resume and resume_path.exists():
        try:
            # ``map_location=device`` lands all tensors on the training device
            # immediately, so the optimizer state matches the model location.
            state = torch.load(resume_path, map_location=device)
        except Exception:  # noqa: BLE001
            log.warning("Resume file %s is unreadable; starting fresh.", resume_path)
            state = None
        if state is None:
            pass
        elif state.get("config_signature") != cfg_sig:
            log.warning(
                "Resume file %s exists but the config has changed since it "
                "was written. Starting fresh; delete the file to silence "
                "this message.",
                resume_path,
            )
        elif state.get("next_epoch", 0) >= cfg.epochs:
            log.info(
                "Resume file says all %d epoch(s) already completed. "
                "Nothing to do — re-run with --epochs N (>%d) to extend.",
                cfg.epochs, state["next_epoch"],
            )
            return ckpt_path
        else:
            model.load_state_dict(state["model"])
            opt.load_state_dict(state["optimizer"])
            if scaler is not None and state.get("scaler") is not None:
                scaler.load_state_dict(state["scaler"])
            start_epoch = int(state["next_epoch"])
            best_val_loss = float(state.get("best_val_loss", float("inf")))
            log.info(
                "Resuming from %s: starting epoch %d/%d (best_val_loss=%.4f).",
                resume_path, start_epoch + 1, cfg.epochs, best_val_loss,
            )

    from tqdm import tqdm

    interrupted = False
    last_completed_epoch = start_epoch  # nothing completed yet in *this* run
    def _run_epoch(epoch: int) -> None:
        """Run one full epoch (train + val + checkpoint + resume save).

        Captures variables from the enclosing scope (model, opt, etc.). Any
        :class:`torch.cuda.OutOfMemoryError` propagates so the caller can
        halve ``micro_bs`` and retry.
        """
        nonlocal best_val_loss, last_completed_epoch
        model.train()
        running_loss = 0.0
        running_n = 0
        seen_batches = 0
        pbar = tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1}/{cfg.epochs} [train]",
            total=len(train_loader),
            leave=False,
        )
        opt.zero_grad(set_to_none=True)
        accum_step = 0
        try:
            for batch_idx, batch in enumerate(pbar):
                if batch is None:
                    continue
                xb, yb = batch
                xb = xb.to(device, non_blocking=use_cuda)
                yb = yb.to(device, non_blocking=use_cuda)
                with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                    logits = model(xb)
                    loss = loss_fn(logits, yb)
                scaled = loss / grad_accum if grad_accum > 1 else loss
                if scaler is not None:
                    scaler.scale(scaled).backward()
                else:
                    scaled.backward()

                accum_step += 1
                if accum_step >= grad_accum:
                    if scaler is not None:
                        scaler.step(opt)
                        scaler.update()
                    else:
                        opt.step()
                    opt.zero_grad(set_to_none=True)
                    accum_step = 0

                running_loss += loss.item() * len(xb)
                running_n += len(xb)
                seen_batches += 1
                if seen_batches % 20 == 0:
                    pbar.set_postfix(loss=f"{running_loss / running_n:.4f}")
                if progress is not None and seen_batches % 50 == 0:
                    progress(
                        "batch", batch_idx + 1, len(train_loader),
                        f"epoch={epoch + 1} loss={running_loss / running_n:.4f}",
                    )
            if accum_step > 0:
                if scaler is not None:
                    scaler.step(opt)
                    scaler.update()
                else:
                    opt.step()
                opt.zero_grad(set_to_none=True)
        finally:
            pbar.close()

        if running_n:
            log.info(
                "Epoch %d/%d train_loss=%.4f (over %d samples)",
                epoch + 1, cfg.epochs, running_loss / running_n, running_n,
            )

        # Validation.
        model.eval()
        val_loss = 0.0
        n_val = 0
        vbar = tqdm(
            val_loader,
            desc=f"Epoch {epoch + 1}/{cfg.epochs} [val]",
            total=len(val_loader),
            leave=False,
        )
        try:
            with torch.inference_mode(), torch.amp.autocast(
                device_type=device.type, enabled=use_amp
            ):
                for batch in vbar:
                    if batch is None:
                        continue
                    xb, yb = batch
                    xb = xb.to(device, non_blocking=use_cuda)
                    yb = yb.to(device, non_blocking=use_cuda)
                    logits = model(xb)
                    val_loss += loss_fn(logits, yb).item() * len(xb)
                    n_val += len(xb)
        finally:
            vbar.close()
        val_loss /= max(1, n_val)
        log.info("Epoch %d/%d val_loss=%.4f", epoch + 1, cfg.epochs, val_loss)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), ckpt_path)
            log.info("Saved new best checkpoint -> %s (val_loss=%.4f)", ckpt_path, val_loss)
        if progress is not None:
            progress("train", epoch + 1, cfg.epochs, f"val_loss={val_loss:.4f}")

        _atomic_save_torch(
            {
                "model": model.state_dict(),
                "optimizer": opt.state_dict(),
                "scaler": scaler.state_dict() if scaler is not None else None,
                "next_epoch": epoch + 1,
                "best_val_loss": best_val_loss,
                "config_signature": cfg_sig,
            },
            resume_path,
        )
        last_completed_epoch = epoch + 1

    try:
        for epoch in range(start_epoch, cfg.epochs):
            while True:
                try:
                    _run_epoch(epoch)
                    break
                except torch.cuda.OutOfMemoryError:
                    if micro_bs <= 1:
                        log.error("CUDA OOM even at micro_bs=1; cannot continue.")
                        raise
                    log.warning(
                        "CUDA OOM at micro_bs=%d during epoch %d; halving and "
                        "retrying the epoch (this is wasted work — the next "
                        "run will start at the new size).",
                        micro_bs, epoch + 1,
                    )
                    torch.cuda.empty_cache()
                    micro_bs = max(1, micro_bs // 2)
                    grad_accum = max(1, requested_bs // micro_bs)
                    _save_train_autotune(autotune_key, micro_bs)
                    train_loader, val_loader = _build_loaders(micro_bs)
                    opt.zero_grad(set_to_none=True)
    except KeyboardInterrupt:
        interrupted = True
        log.warning(
            "Interrupted (Ctrl-C). Last completed epoch: %d. "
            "Re-run the same command to resume from epoch %d "
            "(or pass --no-resume to start over).",
            last_completed_epoch,
            last_completed_epoch + 1,
        )

    manifest = {
        "backbone": cfg.backbone,
        "epochs": cfg.epochs,
        "completed_epochs": last_completed_epoch,
        "learning_rate": cfg.learning_rate,
        "batch_size": cfg.batch_size,
        "val_split": cfg.val_split,
        "best_val_loss": best_val_loss,
        "interrupted": interrupted,
        "stats": asdict(stats),
        "ckpt_sha256": compute_sha256(ckpt_path) if ckpt_path.exists() else None,
    }
    manifest_path = cfg.output_dir / "train.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))

    if interrupted:
        log.info(
            "Partial run saved. best_val_loss=%.4f after %d/%d epoch(s). "
            "Checkpoint=%s, manifest=%s, resume_state=%s",
            best_val_loss, last_completed_epoch, cfg.epochs,
            ckpt_path, manifest_path, resume_path,
        )
        return ckpt_path

    # Full success: delete the resume sentinel so a future run with the same
    # output_dir doesn't think it has unfinished work.
    if resume_path.exists():
        resume_path.unlink()
    log.info(
        "Training complete. best_val_loss=%.4f, checkpoint=%s, manifest=%s",
        best_val_loss, ckpt_path, manifest_path,
    )
    return ckpt_path

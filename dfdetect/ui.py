"""Gradio web UI: an Analyze tab for drag-and-drop detection and a Train tab
that fine-tunes on user-selected real/fake folders.

Gradio doesn't ship a reliable native folder picker, so the Train tab takes
absolute folder paths in textboxes. Both folders are walked **recursively**
and may contain images and videos at any depth (see
:func:`dfdetect.training.walk_media`)."""
from __future__ import annotations

import tempfile
import threading
import traceback
from pathlib import Path
from typing import Any

from .config import Config, load_config
from .types import FileResult


def _results_to_table(results: list[FileResult]) -> list[list]:
    return [
        [r.path.name, r.verdict.value, round(r.score, 3), r.faces_analyzed, round(r.processing_time_s, 2)]
        for r in results
    ]


def _format_stats_md(stats) -> str:
    """Render a DatasetStats summary as Markdown for the Validate panel."""
    lines = [
        "### Dataset summary",
        "",
        f"- **Real:** {stats.real_count} samples "
        f"({stats.real_image_count} images, {stats.real_video_count} videos, "
        f"{stats.real_unique_video_hashes} distinct video sources)",
        f"- **Fake:** {stats.fake_count} samples "
        f"({stats.fake_image_count} images, {stats.fake_video_count} videos, "
        f"{stats.fake_unique_video_hashes} distinct video sources)",
        f"- **Fake / total ratio:** {stats.fake_ratio:.2f}",
    ]
    if stats.warnings:
        lines += ["", "**Warnings:**"]
        lines += [f"- {w}" for w in stats.warnings]
    return "\n".join(lines)


def launch(
    config_path: Path | None = None,
    host: str = "127.0.0.1",
    port: int = 7860,
    share: bool = False,
):
    import gradio as gr

    from .pipeline import Pipeline
    from .report.exporter import write_html
    from .training import (
        BiasConstraintError,
        enforce_constraints,
        summarise_dataset,
    )

    cfg: Config = load_config(config_path)
    pipeline = Pipeline(cfg)

    # ---------- Analyze tab handler --------------------------------------
    def analyze(files, threshold: float, fps: float, heatmaps: bool):
        if not files:
            return [], None, "No files."
        cfg.threshold = float(threshold)
        cfg.video.frames_per_second = float(fps)
        cfg.report.heatmaps = bool(heatmaps)
        paths = [Path(f.name if hasattr(f, "name") else f) for f in files]
        results = pipeline.run(paths)
        tmp = Path(tempfile.mkdtemp(prefix="dfdetect-")) / "report.html"
        write_html(
            results,
            tmp,
            title="dfdetect run",
            threshold=cfg.threshold,
            uncertain_band=cfg.uncertain_band,
        )
        return _results_to_table(results), str(tmp), f"Analyzed {len(results)} files."

    # ---------- Train tab handlers --------------------------------------
    def _validate(real_dir: str, fake_dir: str):
        """Run :func:`summarise_dataset` + :func:`enforce_constraints` and
        report what would happen if the user pressed Train. Hard-violation
        errors are returned as a red status line; warnings stack under the
        stats summary so the user can decide whether to proceed."""
        try:
            real, fake = Path(real_dir).expanduser(), Path(fake_dir).expanduser()
            if not real.is_dir() or not fake.is_dir():
                return "**Error:** both folders must exist.", ""
            stats = summarise_dataset(real, fake)
        except Exception as e:  # noqa: BLE001
            return f"**Error:** {e}", ""
        try:
            enforce_constraints(stats)
        except BiasConstraintError as e:
            return f"**Cannot train:** {e}", _format_stats_md(stats)
        return "**Dataset OK — ready to train.**", _format_stats_md(stats)

    # Training runs on a background thread so the UI stays responsive. The
    # progress callback writes the latest status line into ``_train_state``;
    # the "Refresh status" button reads it.
    _train_state: dict[str, Any] = {
        "status": "Idle.",
        "thread": None,
        "result": None,
        "error": None,
    }
    _state_lock = threading.Lock()

    def _train(
        real_dir: str,
        fake_dir: str,
        output_dir: str,
        epochs: int,
        learning_rate: float,
        batch_size: int,
        val_split: float,
        video_fps: float,
        max_frames_per_video: int,
        face_detection: bool,
        resume: bool,
        device: str,
        amp: bool,
    ):
        from .training.trainer import TrainConfig, train

        # Refuse to start a second run while one is in flight.
        with _state_lock:
            t = _train_state["thread"]
            if t is not None and t.is_alive():
                return "Training already in progress."
            _train_state.update(status="Validating dataset...", result=None, error=None)

        def runner():
            try:
                tcfg = TrainConfig(
                    real_dir=Path(real_dir).expanduser(),
                    fake_dir=Path(fake_dir).expanduser(),
                    output_dir=Path(output_dir).expanduser(),
                    epochs=int(epochs),
                    learning_rate=float(learning_rate),
                    batch_size=int(batch_size),
                    val_split=float(val_split),
                    video_fps=float(video_fps),
                    max_frames_per_video=int(max_frames_per_video),
                    face_detection=bool(face_detection),
                    resume=bool(resume),
                    device=str(device),
                    amp=bool(amp),
                )

                def cb(phase: str, done: int, total: int, label: str):
                    pct = (done / total * 100) if total else 0.0
                    with _state_lock:
                        _train_state["status"] = (
                            f"[{phase}] {done}/{total} ({pct:.0f}%) — {label}"
                        )

                ckpt = train(tcfg, progress=cb)
                with _state_lock:
                    _train_state["status"] = f"Done. Checkpoint: {ckpt}"
                    _train_state["result"] = str(ckpt)
            except Exception as e:  # noqa: BLE001
                with _state_lock:
                    _train_state["status"] = f"Failed: {e}"
                    _train_state["error"] = traceback.format_exc()

        # Check-and-start under one lock acquisition; otherwise two quick
        # clicks could both pass the "already running?" check above.
        with _state_lock:
            t = _train_state["thread"]
            if t is not None and t.is_alive():
                return "Training already in progress."
            th = threading.Thread(target=runner, daemon=True)
            _train_state["thread"] = th
            th.start()
        return "Training started. Press Refresh to update."

    def _train_status_md():
        with _state_lock:
            status = _train_state["status"]
            err = _train_state["error"]
        if err:
            return f"### Training status\n\n{status}\n\n<details><summary>Traceback</summary>\n\n```\n{err}\n```\n</details>"
        return f"### Training status\n\n{status}"

    # ---------- Layout ---------------------------------------------------
    with gr.Blocks(title="dfdetect — local deepfake detection") as demo:
        gr.Markdown("# dfdetect\nLocal deepfake detection. Files never leave this machine.")

        with gr.Tabs():
            # ------------- Analyze tab ------------------------------------
            with gr.Tab("Analyze"):
                with gr.Row():
                    with gr.Column(scale=2):
                        files = gr.File(file_count="multiple", label="Drop images or videos")
                    with gr.Column(scale=1):
                        threshold = gr.Slider(0.0, 1.0, value=cfg.threshold, label="Decision threshold")
                        fps = gr.Slider(
                            0.1, 5.0, value=cfg.video.frames_per_second, step=0.1,
                            label="Video frames / second",
                        )
                        heatmaps = gr.Checkbox(
                            value=cfg.report.heatmaps, label="Generate Grad-CAM heatmaps"
                        )
                        run_btn = gr.Button("Analyze", variant="primary")
                msg = gr.Markdown()
                table = gr.Dataframe(
                    headers=["file", "verdict", "score", "faces", "time_s"],
                    datatype=["str", "str", "number", "number", "number"],
                    interactive=False,
                )
                report_file = gr.File(label="Download HTML report", interactive=False)

                run_btn.click(
                    analyze,
                    inputs=[files, threshold, fps, heatmaps],
                    outputs=[table, report_file, msg],
                )

            # ------------- Train tab -------------------------------------
            with gr.Tab("Train"):
                gr.Markdown(
                    "Fine-tune EfficientNet-B4 on your own labelled data.\n\n"
                    "**Both folders are searched recursively** and may contain "
                    "images **and** videos at any depth. Subfolders that group "
                    "by source identity (e.g. `real/personA/...`) are flattened "
                    "into the class.\n\n"
                    "Anti-bias rules (training refuses to start otherwise):\n"
                    "- ≥ 50 samples per class\n"
                    "- Fake/total ratio in `[0.2, 0.8]`\n"
                    "- Video-only classes need ≥ 5 distinct source clips"
                )
                with gr.Row():
                    real_dir = gr.Textbox(
                        label="Real folder (images and/or videos, recursive)",
                        placeholder="/absolute/path/to/labels/real",
                    )
                    fake_dir = gr.Textbox(
                        label="Fake folder (images and/or videos, recursive)",
                        placeholder="/absolute/path/to/labels/fake",
                    )
                output_dir = gr.Textbox(
                    label="Output folder for `model.pt` + `train.json`",
                    value="./checkpoints",
                )
                with gr.Row():
                    epochs = gr.Slider(1, 50, value=5, step=1, label="Epochs")
                    learning_rate = gr.Number(value=1e-4, label="Learning rate")
                    batch_size = gr.Slider(2, 64, value=16, step=2, label="Batch size")
                    val_split = gr.Slider(0.05, 0.4, value=0.1, step=0.05, label="Validation split")
                with gr.Row():
                    video_fps = gr.Slider(
                        0.5, 5.0, value=1.0, step=0.5,
                        label="Video sample fps",
                    )
                    max_frames_per_video = gr.Slider(
                        4, 128, value=32, step=4,
                        label="Max frames per video",
                    )
                face_detection_cb = gr.Checkbox(
                    value=True,
                    label="Run MTCNN face detection (uncheck for already-face-cropped datasets like FFHQ/StyleGAN)",
                )
                resume_cb = gr.Checkbox(
                    value=True,
                    label="Resume from <output>/resume.pt if it exists and the config matches",
                )
                with gr.Row():
                    device_box = gr.Textbox(
                        value="auto",
                        label="Device (auto | cuda | cuda:0 | cpu)",
                    )
                    amp_cb = gr.Checkbox(
                        value=True,
                        label="Mixed precision (AMP). 2x faster on Volta+ GPUs, no-op on CPU.",
                    )
                with gr.Row():
                    validate_btn = gr.Button("Validate dataset")
                    train_btn = gr.Button("Train", variant="primary")
                    refresh_btn = gr.Button("Refresh status")
                validate_msg = gr.Markdown()
                validate_stats = gr.Markdown()
                train_msg = gr.Markdown()
                status = gr.Markdown(value="### Training status\n\nIdle.")

                validate_btn.click(
                    _validate,
                    inputs=[real_dir, fake_dir],
                    outputs=[validate_msg, validate_stats],
                )
                train_btn.click(
                    _train,
                    inputs=[
                        real_dir, fake_dir, output_dir,
                        epochs, learning_rate, batch_size, val_split,
                        video_fps, max_frames_per_video,
                        face_detection_cb,
                        resume_cb,
                        device_box,
                        amp_cb,
                    ],
                    outputs=[train_msg],
                )
                refresh_btn.click(_train_status_md, outputs=[status])

    demo.launch(server_name=host, server_port=port, share=share, inbrowser=False)

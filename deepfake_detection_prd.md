# Deepfake Detection System — Product Requirements Document

**Author:** Aarav
**Status:** Draft v1.0
**Last updated:** May 2026

---

## 1. Overview

A locally-run deepfake detection tool written in Python that analyzes images and videos to determine the likelihood of AI-generated or manipulated face content. The system processes batches of media files entirely on the user's machine — no uploads, no cloud APIs, no telemetry — and produces per-file authenticity scores plus optional per-frame visualizations.

The tool targets two primary use cases: (a) verifying media authenticity for journalism, OSINT, or personal vetting, and (b) serving as a learning/research project for understanding face-forgery detection pipelines.

---

## 2. Goals and Non-Goals

### Goals

- Detect face-swap, face-reenactment, and fully synthetic (GAN/diffusion) face content in images and videos.
- Support batch input — multiple files in a single run.
- Run 100% offline after the initial model download.
- Return a calibrated confidence score (0.0 – 1.0) per file plus a binary verdict at a configurable threshold.
- Provide both a CLI and a local web UI (Gradio).
- Persist results to a local database for later review.

### Non-Goals

- Real-time webcam analysis (out of scope for v1).
- Detection of non-face manipulations (object insertion, scene editing, audio deepfakes).
- A mobile app or a hosted web service.
- Beating SOTA benchmarks — using strong open-source pretrained weights is sufficient.

---

## 3. User Stories

1. *As a user,* I drop 20 short videos into a folder, run one command, and get a CSV with verdicts for all of them.
2. *As a user,* I open a local web UI, upload 5 images, and see confidence scores with face crops highlighted.
3. *As a user,* I run the tool on a long video and get a timeline showing which segments look manipulated.
4. *As a user,* I re-run analysis later and pull historical results from the local database without reprocessing.
5. *As a user,* I tune the decision threshold and get updated verdicts without re-running inference.

---

## 4. Functional Requirements

### 4.1 Input handling
- Accept image formats: `.jpg`, `.jpeg`, `.png`, `.webp`, `.bmp`.
- Accept video formats: `.mp4`, `.mov`, `.avi`, `.mkv`, `.webm`.
- Accept inputs as: a single file, a list of files, or a directory (recursive flag).
- Validate file headers (not just extensions) using `python-magic` or `filetype`.
- Skip and log corrupt or unreadable files without crashing the batch.

### 4.2 Preprocessing
- Detect faces in each image / sampled video frame using MTCNN or RetinaFace.
- For videos, sample frames at a configurable rate (default: 1 frame per second, capped at 64 frames per video).
- Crop face regions with a configurable margin (default 30%) and resize to model input size (e.g., 224×224 or 299×299).
- Skip files where no face is detected and report this clearly.

### 4.3 Inference
- Run a pretrained classifier on each face crop.
- For videos, aggregate per-frame scores into a single video-level score (mean, median, and max — final score is a weighted blend, default = mean).
- Support GPU (CUDA) inference with automatic CPU fallback.
- Configurable batch size based on available VRAM.

### 4.4 Output
- Per-file results containing: filename, verdict (REAL / FAKE / UNCERTAIN / NO_FACE), confidence score, number of faces analyzed, processing time, model used.
- For videos: per-frame timeline of scores.
- Export formats: JSON, CSV, and a human-readable HTML report.
- Optional Grad-CAM heatmap overlays saved as images.

### 4.5 Storage
- SQLite database (`results.db`) storing every analysis run with timestamps, inputs, outputs, and model version.
- Allow querying past results without re-processing.

### 4.6 Interfaces
- **CLI:** `dfdetect analyze <paths...> [--output csv|json|html] [--threshold 0.5] [--gpu] [--frames-per-second 1]`
- **Web UI (Gradio):** drag-and-drop multiple files, live progress bar, results table, downloadable report.

---

## 5. Non-Functional Requirements

| Category | Requirement |
|---|---|
| Privacy | All processing local. No outbound network calls at runtime. |
| Performance (target) | Image: < 1s per face on RTX 5070 Max-Q. Video: < 1× realtime at 1 fps sampling. |
| Memory | Inference fits in 8 GB VRAM with batch size ≥ 8 at 224×224. |
| Reliability | A single bad file must not crash a batch run. |
| Extensibility | Adding a new model = drop weights + register in a config file. |
| Reproducibility | Pin versions in `requirements.txt`; log model hash with every result. |
| Cross-platform | Linux primary (Arch/Omarchy), Windows and macOS best-effort. |

---

## 6. Technical Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   Input Layer   │────▶│  Preprocessing   │────▶│   Inference     │
│ (CLI / Gradio)  │     │ (face detect +   │     │ (classifier +   │
│                 │     │  frame sample)   │     │  aggregation)   │
└─────────────────┘     └──────────────────┘     └────────┬────────┘
                                                          │
                        ┌─────────────────────────────────┤
                        ▼                                 ▼
                ┌──────────────┐                  ┌───────────────┐
                │  Reporter    │                  │   Storage     │
                │ (JSON/CSV/   │                  │  (SQLite)     │
                │  HTML/Heatmap)│                 │               │
                └──────────────┘                  └───────────────┘
```

Modular pipeline — each stage is swappable. A central `Pipeline` orchestrator wires components based on a YAML config.

---

## 7. Tech Stack

| Layer | Choice | Reasoning |
|---|---|---|
| Language | Python 3.11+ | Standard for ML, good async support |
| ML framework | PyTorch 2.x | CUDA support, large model ecosystem, your RTX 5070 works well with it |
| Face detection | `facenet-pytorch` (MTCNN) or `insightface` (RetinaFace) | Battle-tested, GPU-accelerated |
| Video I/O | OpenCV + `decord` | `decord` is faster for frame sampling |
| Image I/O | Pillow + OpenCV | Standard combo |
| Classifier backbone | EfficientNet-B4 or Xception (timm) | Fits in 8 GB VRAM, strong on FaceForensics++ |
| Visualization | `pytorch-grad-cam` | Heatmap overlays |
| Web UI | Gradio | Single-file local UI, drag-and-drop, no JS required |
| CLI | Typer + Rich | Modern CLI with progress bars |
| DB | SQLite via SQLAlchemy | Zero-setup, file-based |
| Config | YAML via `pydantic-settings` | Validated config |
| Packaging | `uv` or `pipx` | Reproducible installs |
| Testing | pytest + sample mini-dataset | Regression checks |

---

## 8. Models

Use pretrained weights from public deepfake-detection releases. Candidates:

1. **EfficientNet-B4 (DFDC-trained)** — primary. ~75 MB, ~75% accuracy on out-of-distribution deepfakes.
2. **Xception (FaceForensics++ trained)** — secondary, for ensemble option.
3. **MesoNet (small)** — lightweight fallback for low-resource mode.

Optional v2: simple ensemble averaging EfficientNet-B4 + Xception for higher robustness.

Model files live in `~/.cache/dfdetect/models/` and are downloaded on first run with hash verification.

---

## 9. Project Structure

```
dfdetect/
├── pyproject.toml
├── README.md
├── config/
│   └── default.yaml
├── dfdetect/
│   ├── __init__.py
│   ├── cli.py
│   ├── ui.py                  # Gradio app
│   ├── pipeline.py
│   ├── input/
│   │   ├── loader.py
│   │   └── validator.py
│   ├── preprocess/
│   │   ├── face_detect.py
│   │   └── frame_sampler.py
│   ├── models/
│   │   ├── base.py
│   │   ├── efficientnet.py
│   │   └── registry.py
│   ├── inference/
│   │   ├── runner.py
│   │   └── aggregator.py
│   ├── report/
│   │   ├── exporter.py
│   │   └── heatmap.py
│   └── storage/
│       └── db.py
├── tests/
└── samples/                    # small set of REAL + FAKE test files
```

---

## 10. Data Flow Example (Video)

1. User runs `dfdetect analyze movie.mp4 --output html`.
2. Loader validates the file and probes metadata.
3. Frame sampler extracts 1 frame/sec, capped at 64 frames.
4. Face detector finds the largest face per frame; non-face frames discarded.
5. Each face crop is resized, normalized, and pushed into a batch.
6. EfficientNet-B4 outputs a probability per crop.
7. Aggregator computes mean / median / max scores plus per-frame timeline.
8. Reporter writes results to SQLite and renders an HTML report with the timeline graph.
9. CLI prints summary table.

---

## 11. Milestones

| Milestone | Deliverable | Estimate |
|---|---|---|
| M1 | Project scaffold, config loader, CLI skeleton | 2 days |
| M2 | Face detection + frame sampling working on sample files | 3 days |
| M3 | EfficientNet-B4 inference + per-image scoring | 3 days |
| M4 | Video aggregation + per-frame timeline | 2 days |
| M5 | SQLite storage + JSON/CSV exporters | 2 days |
| M6 | Gradio UI with batch upload | 2 days |
| M7 | HTML report + Grad-CAM heatmaps | 2 days |
| M8 | Tests + README + packaging | 2 days |

Total: ~18 working days for v1.

---

## 12. Risks and Mitigations

| Risk | Mitigation |
|---|---|
| Pretrained model generalizes poorly to new deepfakes | Document accuracy clearly; allow ensemble; expose threshold |
| 8 GB VRAM not enough at high batch size | Auto-tune batch size based on available VRAM; CPU fallback |
| MTCNN misses faces in low-light video | Add RetinaFace as fallback detector |
| Long videos blow up processing time | Hard cap on sampled frames; configurable sampling rate |
| Model weights download blocked offline | Allow manual placement of weight files; print exact expected hash |
| False positives mislead users | Always present score + confidence band, never just a binary verdict in the UI |

---

## 13. Future Enhancements (v2+)

- Audio deepfake detection (separate model branch).
- Real-time webcam mode.
- Active learning loop — flag uncertain samples for user labeling, fine-tune locally.
- Browser extension that sends files to the local server for analysis.
- Model card auto-generation per run.
- Support for diffusion-generated full-image fakes (not just face-swaps).

---

## 14. Open Questions

1. Should v1 ship with one model or an ensemble by default? *(Leaning: one model, ensemble as opt-in flag.)*
2. Should the Gradio UI persist between runs as a system service, or launch on demand? *(Leaning: launch on demand via `dfdetect ui`.)*
3. How aggressive should the "uncertain" band be? Default proposal: 0.4 – 0.6 confidence = UNCERTAIN.

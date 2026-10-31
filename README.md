# dfdetect

Local deepfake detection for images and videos. Runs entirely on your machine — no uploads, no cloud APIs, no telemetry.

dfdetect analyzes face content in images and short videos and emits a fake-probability per file plus a verdict (REAL / FAKE / UNCERTAIN / NO_FACE / ERROR). Scores are only as calibrated as the weights and calibration you configure — see [Caveats](#caveats). It ships with a CLI, a Gradio web UI, and a SQLite-backed history store.

> Status: v0.1 reference implementation. See [docs/USAGE.md](docs/USAGE.md) for the full guide.

## Features

- Batch images and videos in a single run.
- Pretrained EfficientNet-B4 backbone via timm; pluggable model registry.
- **Weight downloader with SHA-256 verification** — point the config at a URL and the first run fetches + caches + checksums the file atomically.
- **Probability calibration** — built-in temperature scaling and optional Platt rescaling so scores are real probabilities, not raw softmax.
- **Weighted-average ensembles** — list two or more backbones under `model.members` and the pipeline averages their predictions.
- MTCNN or **RetinaFace** (via `insightface`) face detection.
- **Three video sampling strategies** — uniform / scene-change / hybrid (scene cuts caught via OpenCV HSV-histogram correlation, no extra dep).
- **Per-identity tracks** for multi-face videos via a best-IoU-first tracker (gap tolerance scales with the sampling rate), so one fake speaker isn't averaged out by real co-presenters.
- **Per-face quality weighting** in aggregation — tiny background faces don't drag the verdict.
- **Timestamped fake-region segments** — JSON + HTML emit `[{t_start, t_end, score_peak, score_mean}, ...]` for every video.
- **Parallel preprocessing** — opt-in thread pool runs face detection ahead of inference (`pipeline.workers > 0`).
- **Auto-tuning batch size** — the runner persists the last OOM-survived batch so subsequent runs skip the halving dance.
- Outputs: JSON, CSV, and a self-contained HTML report with **per-file SVG timelines** and shaded fake segments.
- Optional Grad-CAM heatmaps of the FAKE class for the most-suspicious face — works through calibration and ensemble wrappers.
- **Markdown model card** (`--model-card`) summarising model, calibration, decision policy, and limitations.
- SQLite persistence (normalised per-frame / per-segment tables, indexed) with a **`schema_version` table** + automatic in-place migrations; **`dfdetect query`** filters results without re-running inference; **`reverdict`** persists re-thresholded verdicts as derived runs.
- **`dfdetect train`** fine-tunes a backbone on labelled `--real`/`--fake` folders with anti-bias constraints (min 50 per class, ratio in [0.2, 0.8], ≥5 distinct video sources per video-only class), a **leak-free train/val split by source file**, inference-identical face crops, resumable epochs, and OOM-aware batch auto-tuning.
- `python -m dfdetect` entry point; ruff + mypy in the `dev` extras.
- Gradio drag-and-drop UI.
- 100% offline at runtime; weights cached to `~/.cache/dfdetect/`.

## Install

Requires Python 3.11+. PyTorch is installed via the standard wheel; for CUDA you may want to install torch separately first per the official [PyTorch instructions](https://pytorch.org).

```bash
git clone https://github.com/Aarav-Sukhadiya/Deepfake_Detection.git
cd Deepfake_Detection
python -m venv .venv && source .venv/bin/activate
pip install -e .
```

For development (tests + linter + type checker):

```bash
pip install -e '.[dev]'
pytest            # torch-dependent smoke test is skipped if torch isn't installed
ruff check .
mypy
```

## Quick start

```bash
# Analyze one image
dfdetect analyze ./photo.jpg

# Batch a directory recursively, write CSV
dfdetect analyze ./media -r --output csv --output-dir ./reports

# Use your own trained checkpoint (see docs/CONFIGURATION.md)
dfdetect analyze ./media -c ./my-config.yaml

# Force CPU and tune the threshold
dfdetect analyze ./videos --cpu --threshold 0.6

# Launch the Gradio UI on http://127.0.0.1:7860
dfdetect ui
```

If `model.url` is configured, the first run downloads the weights into `~/.cache/dfdetect/models/` (SHA-256 verified); with no weights configured, timm fetches ImageNet weights once. After that, everything runs offline. Set `DFDETECT_CACHE` to relocate the cache directory.

## Output

Every run writes a report (default JSON) and persists results to `~/.cache/dfdetect/results.db`.

```json
[
  {
    "path": "/home/me/clip.mp4",
    "kind": "video",
    "verdict": "FAKE",
    "score": 0.873,
    "faces_analyzed": 28,
    "frame_scores": [{"frame": 0, "t": 0.0, "score": 0.81}, ...],
    "segments": [{"t_start": 4.0, "t_end": 9.0, "score_peak": 0.97, "score_mean": 0.9, "n_frames": 6}],
    "processing_time_s": 4.21,
    "model_name": "efficientnet_b4",
    "model_hash": "9f3a...",
    "error": null,
    "heatmap_path": null,
    "created_at": "2026-10-31T12:00:00+00:00"
  }
]
```

## Project structure

```
dfdetect/
├── cli.py                  # Typer entry point
├── ui.py                   # Gradio app
├── pipeline.py             # Orchestrates input → infer → report → store
├── config.py               # YAML + pydantic config
├── types.py                # Shared dataclasses
├── _cache.py               # ~/.cache/dfdetect helpers (locked autotune state)
├── input/                  # File discovery + header validation
├── preprocess/             # Face detection, frame sampling, identity tracker
├── models/                 # base, registry, efficientnet, calibration, ensemble, downloader
├── inference/              # batched runner + aggregator
├── report/                 # JSON / CSV / HTML / Grad-CAM / model card
├── storage/                # SQLite via SQLAlchemy
├── training/               # Anti-bias constraints + fine-tuning loop
└── templates/              # Jinja HTML report
config/default.yaml         # All tunables live here
docs/                       # Usage, configuration, architecture, models
scripts/                    # DFD dataset prep + training launcher
tests/                      # pytest suite
```

## Caveats

- The bundled EfficientNet-B4 falls back to **ImageNet-pretrained weights** when no `model.weights` path and no `model.url` are configured. Scores are not calibrated in that mode — see [docs/MODELS.md](docs/MODELS.md) for how to wire in DFDC- or FaceForensics++-trained checkpoints (locally or via the SHA-256-verified downloader).
- Calibration (`model.calibration.temperature`, `platt_a`, `platt_b`) needs to be fit against held-out data — the defaults are a no-op. See [docs/MODELS.md](docs/MODELS.md#calibration).
- Detection accuracy on novel deepfake families is bounded by the quality of those public weights. Always present scores alongside verdicts.
- v1 does not handle audio deepfakes, real-time webcam input, or non-face manipulations.

## Documentation

- [Usage guide](docs/USAGE.md) — CLI, UI, common workflows
- [Configuration reference](docs/CONFIGURATION.md) — every YAML key explained
- [Architecture](docs/ARCHITECTURE.md) — how the pipeline fits together
- [Models](docs/MODELS.md) — how to add or swap a backbone
- [Training scripts](scripts/README.md) — end-to-end DFD training recipe
- [Changelog](CHANGELOG.md) — notable fixes and behaviour changes
- [Original PRD](deepfake_detection_prd.md)

## License

MIT.

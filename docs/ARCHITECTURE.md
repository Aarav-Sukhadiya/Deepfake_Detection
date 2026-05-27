# Architecture

dfdetect is a strictly linear pipeline with one orchestrator. Each stage is a small module exposing a narrow interface, so any one of them can be swapped without touching the others.

```
┌──────────────┐    ┌────────────────┐    ┌──────────────┐
│  CLI / UI    │───▶│  Pipeline      │───▶│ Reporter +   │
│              │    │  (orchestrator)│    │ SQLite       │
└──────────────┘    └────┬───────────┘    └──────────────┘
                         │
       ┌─────────────────┼────────────────────┐
       ▼                 ▼                    ▼
┌────────────┐    ┌─────────────┐    ┌──────────────┐
│  Input     │    │ Preprocess  │    │  Inference   │
│  loader +  │    │ frame samp. │    │  runner +    │
│  validator │    │ + face det. │    │  aggregator  │
└────────────┘    └─────────────┘    └──────────────┘
```

---

## Modules

### `dfdetect/cli.py`
Typer entry point. Parses CLI flags, applies overrides on top of the loaded `Config`, and hands the work to `Pipeline`. Renders results with Rich.

### `dfdetect/ui.py`
Gradio app. Same `Pipeline`, different driver. Exposes the threshold/fps/heatmap controls as live widgets.

### `dfdetect/pipeline.py`
The orchestrator. Owns:
- the `Config` (validated pydantic object),
- the chosen `DeepfakeModel`,
- the chosen face `FaceDetector`,
- the `InferenceRunner`,
- the `ResultsStore`.

Per file: discover → sample → detect faces → batch → infer → aggregate → verdict → store.

### `dfdetect/input/`
- `validator.py` — header-based MIME detection via `filetype`, with extension fallback.
- `loader.py` — walks the user-supplied paths into a deduplicated list of `MediaItem`s.

### `dfdetect/preprocess/`
- `frame_sampler.py` — opens a video with `decord` (fast) or OpenCV (fallback) and yields `(frame_index, timestamp_s, RGB_frame)`. Three sampling strategies: `uniform`, `scene` (boundary-only, HSV-histogram cut detection), and `hybrid` (union of both). Backend-agnostic scheduling lives in `_plan_indices`; the cut logic is split into the pure-function `detect_scene_boundaries` for testability.
- `tracker.py` — `assign_tracks(faces)` performs a cheap greedy IoU + centroid match across frames to group faces into per-identity tracks. Tracks idle for `max_gap_frames` are closed. Used when `largest_face_only=False` so multi-face videos get one score per identity.
- `face_detect.py` — `MTCNNDetector` wraps `facenet-pytorch`; `RetinaFaceDetector` wraps `insightface.app.FaceAnalysis` (detection-only) and shares the same margin/min-size logic.

### `dfdetect/models/`
- `base.py` — `DeepfakeModel` abstract: `preprocess`, `predict`, `grad_cam_target_layer`, `hash`.
- `downloader.py` — streaming weight downloader with SHA-256 verification, atomic rename via `.part` sidecar, and a `WEIGHT_SOURCES` registry. `ensure_weights_file` resolves to a local path or `None` so each model decides whether to use ImageNet fallback.
- `registry.py` — `@register_model("name")` decorator + `build_model(cfg)`. `build_model` is now the single place that composes the model graph: it builds the base model, applies `CalibratedModel` when `model.calibration` is non-trivial, and wraps everything in `EnsembleModel` when `model.ensemble` is set.
- `efficientnet.py` — concrete `tf_efficientnet_b4` via `timm`. Calls `ensure_weights_file` so a configured `url` (or registered `WeightSource`) is fetched on first use. Falls back to ImageNet pretraining only when no local file *and* no URL is available.
- `calibration.py` — `CalibratedModel` decorator: temperature scaling on the underlying logit, optional Platt rescaling on the post-temperature probability.
- `ensemble.py` — `EnsembleModel`: weighted-mean over multiple `DeepfakeModel` members, each running its own `preprocess`/`predict` so members can have different input sizes. Grad-CAM defers to the first member.

### `dfdetect/inference/`
- `runner.py` — batched inference with CUDA OOM auto-halving. Persists the largest surviving batch size to `~/.cache/dfdetect/autotune.json` keyed by `(model_name, model_hash)` so subsequent runs skip the OOM-and-halve dance.
- `aggregator.py` — collapses per-face scores (`mean` / `median` / `max` / `weighted` / `quality_weighted`) and applies the threshold + uncertain band to produce a `Verdict`. Also exposes `derive_segments` which run-length-encodes high-scoring frames into `Segment(t_start, t_end, score_peak, score_mean, n_frames)` with a configurable merge gap.

### `dfdetect/report/`
- `exporter.py` — `write_json`, `write_csv`, `write_html` (Jinja). `write_html` takes `threshold` and `uncertain_band` so the rendered SVG can draw them.
- `heatmap.py` — Grad-CAM PNG overlays. Calls `model.grad_cam_model()` (added on `DeepfakeModel`) so it works through `CalibratedModel` and `EnsembleModel` wrappers without reaching into private attributes.
- `model_card.py` — `render_model_card(cfg, results, out_path)` writes a markdown summary (model, calibration, decision policy, run breakdown, known limitations). Wired via `dfdetect analyze --model-card`.

### `dfdetect/storage/`
- `db.py` — SQLAlchemy 2 declarative models (`Run`, `Result`, `SchemaVersion`) and a `ResultsStore` façade. SQLite file at `~/.cache/dfdetect/results.db`. `Run` carries `kind` (`analyze` / `reverdict`) and an optional `parent_run_id` so derived runs from `dfdetect reverdict` are first-class history rows. `Result` carries `segments`. `_migrate_in_place` performs additive `ALTER TABLE ADD COLUMN` upgrades and stamps the DB with the current `SCHEMA_VERSION`. `ResultsStore.query(...)` filters by run / verdict / path / score range and powers `dfdetect query`.

### `dfdetect/training/`
- `constraints.py` — pure-function gate: `summarise_dataset(real_dir, fake_dir)` walks the folders (image vs. video via extension, video-source dedup via head-byte hash) into a `DatasetStats`; `enforce_constraints(stats)` raises `BiasConstraintError` on hard rule violations (empty/undersized class, ratio outside `[0.2, 0.8]`, video-only class with too few distinct sources) and appends warnings for softer issues.
- `trainer.py` — lazy-imports torch/timm. One-epoch loop with a balanced sampler over face crops, validates, writes the best-loss checkpoint + a `train.json` manifest.

### `dfdetect/audio/`, `dfdetect/active_learning.py`, `dfdetect/provenance.py`
Scaffolds for IMPROVEMENTS.md #22 (audio detection), #23 (active learning), and #24 (C2PA provenance). They expose the shape a real implementation must satisfy but raise/return defaults today so callers can stub out integration.

### `dfdetect/templates/`
- `report.html.j2` — self-contained HTML report. Inlined CSS, no external assets.

---

## Data types

All inter-stage communication uses the dataclasses in [`dfdetect/types.py`](../dfdetect/types.py):

- `MediaItem` — a discovered file (path + kind + size).
- `FaceCrop` — one ready-to-classify face crop with bbox + frame info.
- `FrameScore` — `(frame_index, timestamp_s, score)` for a single frame.
- `FileResult` — the final per-file artifact: verdict, score, frame timeline, model metadata, processing time, optional error and heatmap path.
- `Verdict` — enum: `REAL / FAKE / UNCERTAIN / NO_FACE / ERROR`.

---

## Failure model

A single bad file must never abort a batch. Each per-file step (`_faces_from_item` and `runner.run`) is wrapped in a try/except in [`pipeline.py`](../dfdetect/pipeline.py); failures become a `FileResult` with `Verdict.ERROR` and an `error` string. The batch continues.

CUDA OOM during inference is non-fatal: the runner halves its batch size and retries until it succeeds or hits batch=1.

---

## Extension points

| To add… | Touch… |
|---|---|
| A new face detector | Implement the `FaceDetector` Protocol in `preprocess/face_detect.py`, branch in `build_detector`. |
| A new model backbone | New file under `models/` decorated with `@register_model("name")`; subclass `DeepfakeModel`. Accept the optional `weights_url` / `weights_sha256` kwargs and call `ensure_weights_file` in your `_build`. |
| A new calibration scheme | Subclass `CalibratedModel` or add a wrapper module and compose it inside `registry.build_model`. |
| A new aggregation strategy | Add a branch in `aggregator.aggregate` and extend the `aggregation` literal in `config.InferenceConfig`. |
| A new report format | New `write_*` function in `report/exporter.py`; wire it into `cli.analyze`. |
| New persisted columns | Add a `Mapped[...]` field to `storage.db.Result`/`Run` and a matching entry in `ResultsStore._migrate_in_place` so existing DBs upgrade additively. |

The pipeline orchestrator (`Pipeline`) only knows the abstract interfaces, so extensions slot in without changes there.

---

## Lazy imports

Heavy ML deps (`torch`, `timm`, `facenet_pytorch`, `decord`, `cv2`, `pytorch_grad_cam`) are imported inside the functions that use them, not at module top. This means:

- The CLI starts fast.
- The exporter / config / storage layers are unit-testable without torch installed.
- Optional features (e.g. `retinaface`) only error out if you actually try to use them.

---

## Testing strategy

Unit tests cover the deterministic, dependency-light layers: config parsing, input validation/discovery, aggregator math, storage round-trip, and report exporters. The model + face detector layers are integration concerns — verify them on the bundled `samples/` set after install.

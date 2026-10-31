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
Typer entry point. Parses CLI flags, applies overrides on top of the loaded `Config`, and hands the work to `Pipeline`. Renders results with Rich. Every override option defaults to `None` so a flag only replaces a config value when it is actually passed, and all arguments (output format, threshold, fps, batch size, verdict filter) are validated *before* any model loads.

### `dfdetect/ui.py`
Gradio app. Same `Pipeline`, different driver. Exposes the threshold/fps/heatmap controls as live widgets.

### `dfdetect/pipeline.py`
The orchestrator. Owns:
- the `Config` (validated pydantic object),
- the chosen `DeepfakeModel`,
- the chosen face `FaceDetector`,
- the `InferenceRunner`,
- the `ResultsStore`.

Per file: discover → sample → detect faces → (track) → batch → infer → aggregate → verdict → store.

With `pipeline.workers > 0`, `_run_parallel` runs `_preprocess_item` (decode + face detection) in a thread pool while inference stays on the calling thread. A bounded queue applies back-pressure; at most `workers + prefetch` files are in flight. The producer's queue writes give up once the consumer stops, and unstarted work is cancelled, so an exception mid-batch can't leave a thread blocked forever.

### `dfdetect/input/`
- `validator.py` — header-based MIME detection via `filetype`, with extension fallback.
- `loader.py` — walks the user-supplied paths into a deduplicated list of `MediaItem`s. Roots are made absolute so `./a.png` and `/abs/a.png` dedupe. A file that can't be read becomes a `MediaItem` with `kind=None` and an `error`, which the pipeline turns into an `ERROR` result instead of aborting discovery.

### `dfdetect/preprocess/`
- `frame_sampler.py` — opens a video with `decord` (fast) or OpenCV (fallback) and yields `(frame_index, timestamp_s, RGB_frame)`. Three sampling strategies: `uniform`, `scene` (boundary-only, HSV-histogram cut detection), and `hybrid` (union of both). Backend-agnostic scheduling lives in `_plan_indices`; the cut logic is split into the pure-function `detect_scene_boundaries` for testability.
  - The OpenCV fallback is used only if decord can't *open/plan* the file. A decode error later skips that chunk rather than restarting with another backend, which would yield duplicate frames.
  - Scene probing keeps only each probe frame's ~1 KB thumbnail histogram, never the full frame.
  - decord decodes in batches of 16 (`get_batch`). OpenCV `grab()`s forward up to 64 frames instead of seeking, and seeks only for larger jumps.
- `tracker.py` — `assign_tracks(faces, iou_threshold, max_gap_frames)` groups faces into per-identity tracks. Within each frame, all (face, open track) pairs above the IoU threshold are matched best-IoU-first, so detection order can't let one face steal another's track. Tracks idle for more than `max_gap_frames` *source* frames are closed. The pipeline sets that gap to `max(30, 3 × sampling step)` (`pipeline._track_gap`), so low-fps sampling of high-fps video doesn't split every identity. Only used when `largest_face_only=False`.
- `face_detect.py` — `MTCNNDetector` wraps `facenet-pytorch`; `RetinaFaceDetector` wraps `insightface.app.FaceAnalysis` (detection-only). Both use the shared `expand_bbox` margin logic, which the trainer reuses so training and inference crops match. Note that MTCNN's `detect()` returns every face regardless of `keep_all`, so the wrapper truncates to the largest box itself when `largest_face_only=True`.

### `dfdetect/models/`
- `base.py` — `DeepfakeModel` abstract: `preprocess`, `predict`, `grad_cam_target_layer`, plus overridable `grad_cam_model` / `grad_cam_preprocess` and a memoised `hash`.
- `downloader.py` — streaming weight downloader with SHA-256 verification, atomic rename via `.part` sidecar, and a `WEIGHT_SOURCES` registry. `ensure_weights_file` resolves to a local path or `None` so each model decides whether to use ImageNet fallback.
- `registry.py` — `@register_model("name")` decorator + `build_model(cfg)`. `build_model` is now the single place that composes the model graph: it builds the base model, applies `CalibratedModel` when `model.calibration` is non-trivial, and wraps everything in `EnsembleModel` when `model.ensemble` is set.
- `efficientnet.py` — concrete `tf_efficientnet_b4` via `timm`. Calls `ensure_weights_file` so a configured `url` (or registered `WeightSource`) is fetched on first use. Falls back to ImageNet pretraining only when no local file *and* no URL is available. Checkpoints load with `torch.load(weights_only=True)` (no arbitrary pickle execution). `module.` / `model.` prefixes are stripped, and loading **raises** if backbone parameters are missing, instead of silently running a randomly initialised network.
- `calibration.py` — `CalibratedModel` decorator: temperature scaling on the underlying logit, optional Platt rescaling on the post-temperature probability.
- `ensemble.py` — `EnsembleModel`: weighted-mean over multiple `DeepfakeModel` members, each running its own `preprocess`/`predict` so members can have different input sizes. Grad-CAM (`grad_cam_model` *and* `grad_cam_preprocess`) defers to the first member.

### `dfdetect/inference/`
- `runner.py` — batched inference with CUDA OOM auto-halving. The CUDA allocator cache is freed after each OOM, outside the `except` block so the traceback's tensors are already released. The surviving batch size is persisted to `<cache>/autotune.json`, keyed by `model_name:model_hash`, so later runs skip the OOM-and-halve dance.
- `aggregator.py` — collapses per-face scores (`mean` / `median` / `max` / `weighted` / `quality_weighted`) and applies the threshold + uncertain band to produce a `Verdict`.
  - `reverdict(result, threshold, band)` re-derives a stored verdict. `ERROR` results are preserved, because their 0.0 score is a placeholder.
  - `derive_segments` run-length-encodes high-scoring frames into `Segment(t_start, t_end, score_peak, score_mean, n_frames)`, with a configurable merge gap.

### `dfdetect/_cache.py`
`cache_dir()` (honours `$DFDETECT_CACHE`, default `~/.cache/dfdetect`) and `load_batch_size` / `save_batch_size`, shared by the runner and the trainer. Reads and read-modify-writes happen under an advisory `flock` (a no-op where `fcntl` is unavailable), and writes are atomic (temp file + `os.replace`). Concurrent runs therefore never corrupt the file.

### `dfdetect/report/`
- `exporter.py` — `write_json`, `write_csv`, `write_html` (Jinja, UTF-8). `write_html` takes `threshold` and `uncertain_band` so the rendered SVG can draw them. The template environment is built once and has `autoescape=True` explicitly: extension-based `select_autoescape` does not match `report.html.j2`, which would leave filenames unescaped.
- `heatmap.py` — Grad-CAM PNG overlays of the **FAKE** logit (`ClassifierOutputTarget(1)`). It goes through `model.grad_cam_model()` / `grad_cam_preprocess()`, so it works through `CalibratedModel` and `EnsembleModel` wrappers. `GradCAM` is used as a context manager so its hooks are removed after each call; otherwise every later inference pass would keep capturing activations. Files are named `<stem>-<sha1(path)[:8]>.png` so same-named files from different folders don't collide.
- `model_card.py` — `render_model_card(cfg, results, out_path)` writes a markdown summary (model, calibration, decision policy, run breakdown, known limitations). Wired via `dfdetect analyze --model-card`.

### `dfdetect/storage/`
- `db.py` — SQLAlchemy 2 declarative models and a `ResultsStore` façade. SQLite file at `~/.cache/dfdetect/results.db`. Schema v3:

  | Table | Contents |
  |---|---|
  | `runs` | One row per `analyze` / `reverdict`. `kind` and an optional `parent_run_id` make derived runs first-class history rows. |
  | `results` | One row per file (indexed on `run_id`). `kind` is `''` for files that couldn't be classified. |
  | `frame_scores` | Per-frame scores (indexed on `result_id`). |
  | `segments` | Fake-region segments (indexed on `result_id`). |
  | `schema_version` | Single row holding the current `SCHEMA_VERSION`. |

  `_migrate_in_place` checks the live schema rather than trusting the version number alone, and every step is idempotent, so an interrupted migration resumes on the next open:
  - **v1 → v2:** adds `runs.parent_run_id` / `runs.kind`.
  - **v2 → v3:** moves the legacy JSON `frame_scores` / `segments` columns into the child tables (rename → create → bulk copy via `executemany` → drop).

  Reads eager-load children and the parent run in a fixed three queries (`selectinload` / `joinedload`). `list_runs` counts results in SQL instead of loading them. `ResultsStore.query(...)` filters by run / verdict / path / score range and powers `dfdetect query`.

### `dfdetect/training/`
- `constraints.py` — pure-function gate: `summarise_dataset(real_dir, fake_dir)` walks the folders (image vs. video via extension, video-source dedup via head-byte hash) into a `DatasetStats`; `enforce_constraints(stats)` raises `BiasConstraintError` on hard rule violations (empty/undersized class, ratio outside `[0.2, 0.8]`, video-only class with too few distinct sources) and appends warnings for softer issues.
- `trainer.py` — lazy-imports torch/timm. Indexes samples as `(path, frame_index)` pointers and decodes lazily in the `Dataset`.
  - **Split:** train/val is split **by source file**. Hardlinks are grouped via `(st_dev, st_ino)`, so near-duplicate frames of one clip can't land on both sides.
  - **Crops:** the same margin as inference (`face_margin`, default 0.30) via `expand_bbox`.
  - **Speed:** face boxes are cached after the first visit, so later epochs skip MTCNN. A small LRU of open decord readers avoids reopening a video for every frame.
  - **Memory:** a class-balanced `WeightedRandomSampler`, plus AMP on CUDA. The micro-batch size is auto-tuned by a probe, and gradient accumulation preserves the requested effective batch.
  - **Checkpoints:** the best checkpoint is written atomically. An epoch with no usable validation samples saves the latest weights instead of recording a bogus 0.0 loss. A `resume.pt` is written after each epoch (keyed by a config signature), and `train.json` records the run.

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

A single bad file must never abort a batch. Unreadable inputs, preprocessing (`_faces_from_item`) and inference (`runner.run`) failures each become a `FileResult` with `Verdict.ERROR` and an `error` string prefixed `input:` / `preprocess:` / `inference:` (see [`pipeline.py`](../dfdetect/pipeline.py)). Heatmap failures are logged and leave `heatmap_path` empty. The batch continues, and saving to SQLite handles results whose media kind is unknown.

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
| New persisted columns | Add a `Mapped[...]` field to `storage.db.Result`/`Run`, bump `SCHEMA_VERSION`, and add an idempotent step to `ResultsStore._migrate_in_place` (check the live schema via `PRAGMA table_info` before altering). Add a test that opens a DB in the old shape — see `tests/test_regressions.py::test_v2_db_migrates_json_into_tables`. |

The pipeline orchestrator (`Pipeline`) only knows the abstract interfaces, so extensions slot in without changes there.

---

## Lazy imports

Heavy ML deps (`torch`, `timm`, `facenet_pytorch`, `decord`, `cv2`, `pytorch_grad_cam`) are imported inside the functions that use them, not at module top. This means:

- The CLI starts fast.
- The exporter / config / storage layers are unit-testable without torch installed.
- Optional features (e.g. `retinaface`) only error out if you actually try to use them.

---

## Testing strategy

Unit tests cover the deterministic, dependency-light layers: config parsing, input validation/discovery, frame-index planning, tracking, aggregator math, storage round-trip and migration, the CLI's argument validation, the trainer's train/val split, and the report exporters. `tests/test_regressions.py` pins each fixed bug with a test that fails on the old behaviour.

The model and face-detector layers are integration concerns. `tests/test_smoke_pipeline.py` runs the full pipeline on `samples/` when torch, timm and facenet-pytorch are installed, and is skipped otherwise.

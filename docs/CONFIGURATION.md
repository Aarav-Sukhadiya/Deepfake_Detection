# Configuration Reference

Every tunable lives in [`config/default.yaml`](../config/default.yaml). Override by passing `--config path/to/yours.yaml` to any CLI command, or by editing the bundled defaults.

Config is validated with pydantic — bad values fail fast at load time with a precise error message.

---

## Top-level

| Key | Default | Description |
|---|---|---|
| `device` | `auto` | `auto`, `cuda`, or `cpu`. `auto` → CUDA if available, else CPU. |
| `batch_size` | `8` | Inference batch size. Auto-halves on CUDA OOM. |
| `threshold` | `0.5` | FAKE if score ≥ threshold (after the uncertain band). |
| `uncertain_band` | `[0.4, 0.6]` | Scores inside this range collapse to UNCERTAIN. |

---

## `input`

Controls file discovery and validation.

| Key | Default | Description |
|---|---|---|
| `recursive` | `false` | Descend into subdirectories. |
| `follow_symlinks` | `false` | Follow symlinked files/dirs during discovery. |
| `image_exts` | `.jpg .jpeg .png .webp .bmp` | Hint extensions for files where header probing returns nothing. Header detection takes precedence. |
| `video_exts` | `.mp4 .mov .avi .mkv .webm` | Same, for video. |

---

## `preprocess`

| Key | Default | Description |
|---|---|---|
| `face_detector` | `mtcnn` | `mtcnn` (default) or `retinaface`. `retinaface` requires `pip install insightface onnxruntime` (or `onnxruntime-gpu` on CUDA); pack name overridable via `INSIGHTFACE_MODEL` (default `buffalo_l`). |
| `face_margin` | `0.30` | Fractional padding around the detected face bbox before cropping. |
| `input_size` | `224` | Square pixel size fed to the model. Match the model's training resolution. |
| `min_face_size` | `40` | Minimum face height/width in pixels for the detector. Lower = catch tiny faces, more false positives. |
| `largest_face_only` | `true` | If `true`, keep only the largest face per frame (ideal for talking-head videos). |

---

## `video`

| Key | Default | Description |
|---|---|---|
| `frames_per_second` | `1.0` | Uniform sampling rate. `1.0` = one frame per second of source video. |
| `max_frames` | `64` | Hard cap per video. Long videos won't blow up runtime. |
| `sampling_strategy` | `uniform` | `uniform` (today's behaviour), `scene` (only frames on a detected cut), or `hybrid` (union of both, deduped, capped). |
| `scene_threshold` | `0.6` | HSV-histogram correlation between consecutive scene-detector thumbnails. **Lower = stricter** — more cuts flagged. Range `[0, 1]`. |
| `scene_thumb_size` | `64` | Thumbnail edge length for the cut detector. Bigger = more sensitive, slower. |
| `scene_stride` | `0` | Probe every Nth source frame for cuts. `0` reuses the uniform step; set lower (e.g. `1` or `5`) to catch sub-second splices. |

`hybrid` is the most defensive choice when you don't trust the source: you keep `frames_per_second` coverage of the timeline and pick up cut frames that fall between samples.

---

## `inference`

| Key | Default | Description |
|---|---|---|
| `aggregation` | `mean` | How to collapse per-face scores: `mean`, `median`, `max`, `weighted`, or `quality_weighted`. |
| `weighted.mean` | `0.6` | Weight when `aggregation: weighted`. |
| `weighted.median` | `0.2` | " |
| `weighted.max` | `0.2` | " |
| `track_aggregation` | `max` | How to collapse per-identity (track) scores into a single per-file score for multi-face videos. `max` flags a clip if *any* identity is fake; `mean` averages; `off` skips track-aware aggregation. |

`weighted` is useful when you want the average to dominate but be sensitive to a single very-fake frame. `quality_weighted` uses each face's `quality_weight()` (face area × detector confidence) so tiny background faces don't drag the score.

---

## `model`

| Key | Default | Description |
|---|---|---|
| `name` | `efficientnet_b4` | Registered model name. See `dfdetect models`. |
| `weights` | `null` | Local path to a `.pt`/`.pth` checkpoint. Wins over `url` if present. |
| `url` | `null` | Remote source for the weights file. Downloaded on first run into `storage.models_dir`. |
| `sha256` | `null` | Hex digest used to verify the downloaded file. Strongly recommended whenever `url` is set. |
| `num_classes` | `2` | Output classes. Class 1 = FAKE for the standard binary setup. |
| `ensemble` | `false` | If `true`, the pipeline builds each entry of `members` and averages their (per-face) predictions. |
| `members` | `[]` | List of member specs (`name`, `weights`, `url`, `sha256`, `weight`, `calibration`). Required when `ensemble: true`. |
| `calibration.temperature` | `1.0` | Logit temperature applied to raw softmax fake probability. `T > 1` softens an over-confident model. |
| `calibration.platt_a` / `calibration.platt_b` | `null` | If **both** are set, a 1-D logistic `sigmoid(a·p + b)` is applied after temperature. |

Resolution order for weights, per model:

1. `weights` (local file). If it exists, it's used as-is.
2. `url` (+ `sha256`). Downloaded into `storage.models_dir`, atomically renamed only after the hash matches. Subsequent runs reuse the cached file as long as the digest still matches.
3. None of the above → ImageNet-pretrained backbone with a warning. Verdicts are *uncalibrated*.

Example ensemble of two EfficientNet checkpoints with per-member calibration:

```yaml
model:
  ensemble: true
  members:
    - name: efficientnet_b4
      weights: ~/.cache/dfdetect/models/dfdc_b4.pt
      weight: 2.0
      calibration: { temperature: 1.5 }
    - name: efficientnet_b4
      weights: ~/.cache/dfdetect/models/ff_b4.pt
      weight: 1.0
      calibration: { temperature: 1.2, platt_a: 1.1, platt_b: -0.1 }
  calibration:
    temperature: 1.0    # outer no-op; inner per-member calibration already applied
```

See [MODELS.md](MODELS.md) for adding a new model and fitting calibration.

---

## `storage`

| Key | Default | Description |
|---|---|---|
| `db_path` | `~/.cache/dfdetect/results.db` | SQLite database for run history. Auto-created. |
| `models_dir` | `~/.cache/dfdetect/models` | Where downloaded weights are cached. |

---

## `report`

| Key | Default | Description |
|---|---|---|
| `default_format` | `json` | Used when CLI omits `--output`. |
| `heatmaps` | `false` | Render Grad-CAM PNGs for the most-suspicious face per file. |
| `output_dir` | `./reports` | Where reports and `heatmaps/` are written. |

---

## `logging`

| Key | Default | Description |
|---|---|---|
| `level` | `INFO` | Standard Python log levels. |

---

## `pipeline`

| Key | Default | Description |
|---|---|---|
| `workers` | `0` | Threads used to pre-extract face crops while the model is busy on the previous file. `0` = single-threaded (today's behaviour). |
| `prefetch` | `2` | Maximum number of files preprocessed ahead of inference. Backpressure protects against memory blow-ups when preprocessing is much faster than inference. |

The model itself stays on the main thread regardless of `workers` — CUDA contexts aren't safely shared.

---

## Override precedence

For a single field, the highest-priority source wins:

1. CLI flag (`--threshold 0.7`)
2. `--config <path>` file
3. Bundled `config/default.yaml`
4. Code defaults in [`dfdetect/config.py`](../dfdetect/config.py)

---

## Example: a high-recall config for journalism

```yaml
threshold: 0.4
uncertain_band: [0.35, 0.55]
preprocess:
  largest_face_only: false   # multi-person interviews
  min_face_size: 24
video:
  frames_per_second: 2.0
  max_frames: 128
inference:
  aggregation: weighted
  weighted: { mean: 0.4, median: 0.2, max: 0.4 }
report:
  heatmaps: true
```

This biases toward catching forgeries (lower threshold, higher sampling, max-aware aggregation) at the cost of more false positives — which the heatmaps help triage.

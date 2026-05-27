# Usage Guide

This guide walks through analyzing media with `dfdetect`, both from the CLI and the Gradio web UI.

---

## 1. Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .
```

Verify the install:

```bash
dfdetect --help
dfdetect models       # lists registered model backbones
```

The first call to `analyze` or `ui` will lazy-build the model and detector. Model weights are cached at `~/.cache/dfdetect/models/`, results at `~/.cache/dfdetect/results.db`.

---

## 2. CLI

### 2.1 `analyze` — run the pipeline

```
dfdetect analyze <paths...> [options]
```

| Option | Default | Description |
|---|---|---|
| `-c, --config PATH` | bundled `config/default.yaml` | Override any config field. |
| `-o, --output FORMAT` | `json` | `json`, `csv`, or `html`. |
| `--output-dir PATH` | `./reports` | Where to write the report file. |
| `-t, --threshold FLOAT` | `0.5` | FAKE if score ≥ threshold (after the uncertain band). |
| `-r, --recursive` | off | Descend into subdirectories. |
| `--gpu` / `--cpu` | auto-detect | Force device. |
| `-f, --frames-per-second FLOAT` | `1.0` | Video sampling rate. |
| `-b, --batch-size INT` | `8` | VRAM-tunable batch size. Auto-halves on CUDA OOM. |
| `--heatmaps` | off | Render Grad-CAM PNGs into `<output-dir>/heatmaps/`. |
| `--model-card` | off | Render a markdown model card alongside the report. |
| `--log-level` | `INFO` | `DEBUG`, `INFO`, `WARNING`, etc. |

**Inputs** can be any combination of:

- a single file (`./photo.jpg`)
- multiple files (`a.png b.mp4 c.webp`)
- a directory (uses non-recursive globbing unless `-r` is set)

Unsupported files are silently skipped. Header detection (via `filetype`) takes precedence over file extensions, so files with wrong extensions are still classified correctly.

#### Examples

```bash
# 1. Single image, pretty CLI table + JSON report:
dfdetect analyze ./photo.jpg

# 2. Whole folder of videos, CSV output, custom threshold:
dfdetect analyze ./clips -r --output csv --threshold 0.65

# 3. HTML report with heatmaps, sampling 2 fps:
dfdetect analyze ./batch --output html --heatmaps -f 2

# 4. Force CPU on a laptop without CUDA:
dfdetect analyze ./media --cpu

# 5. Use a custom config file (your own model + tuning):
dfdetect analyze ./media -c ./my-config.yaml
```

### 2.2 `ui` — Gradio web app

```bash
dfdetect ui                           # http://127.0.0.1:7860
dfdetect ui --port 9000 --host 0.0.0.0
dfdetect ui --share                   # public Gradio tunnel (still local processing)
```

Drop files into the upload area, set the threshold and sampling rate, and click **Analyze**. The results table and downloadable HTML report update in place.

### 2.3 `runs` — list past runs

Every `analyze` invocation is persisted. Inspect history without rerunning anything:

```bash
dfdetect runs                  # 20 most recent
dfdetect runs --limit 100
```

### 2.4 `reverdict` — re-threshold an existing run

```bash
dfdetect reverdict 17 --threshold 0.7
dfdetect reverdict 17 -t 0.65 --uncertain-lo 0.55 --uncertain-hi 0.70
dfdetect reverdict 17 --threshold 0.7 --dry-run
```

Re-applies the new threshold (and optional uncertain band) to the stored scores from run `17` and prints the updated verdicts. **No inference is re-run.**

By default, the new verdicts are **persisted** as a *derived run* that records `parent_run_id = 17`, `kind = "reverdict"`, and the threshold + band used. `dfdetect runs` will then show both the original and the derived row, and `dfdetect query` (or any future read) can target the derived run by its id. Pass `--dry-run` to preview without writing.

### 2.5 `models` — list registered backbones

```bash
dfdetect models
```

### 2.6 `query` — filter persisted results

```bash
dfdetect query --verdict FAKE --min-score 0.8
dfdetect query --run 17 --path '%clip%.mp4'
dfdetect query --verdict UNCERTAIN --max-score 0.55 --limit 50
```

Pulls rows directly from the SQLite store without re-running inference. `--path` uses SQL `LIKE` syntax (`%` wildcard). Combines with any of `--run`, `--verdict`, `--min-score`, `--max-score`, `--limit`.

### 2.7 `train` — fine-tune on labelled folders

```bash
dfdetect train --real ./labels/real --fake ./labels/fake \
  --output-dir ./checkpoints --epochs 5 --lr 1e-4

# Run only the anti-bias checks without training:
dfdetect train --real ./labels/real --fake ./labels/fake --check-only
```

Hard anti-bias constraints fire **before** any GPU work:

- Both classes must have ≥ 50 samples (default; configurable in code).
- Fake/total ratio must be in `[0.2, 0.8]`.
- Each video-only class must include ≥ 5 distinct source clips (hashed by file head, so a 1000-frame dataset cut from one video is rejected).

Soft warnings are logged for ratios outside `[0.4, 0.6]` or low video-source counts. Output is `model.pt` + a `train.json` manifest (dataset stats, hash, best val loss) in `--output-dir`. Wire the checkpoint back into inference via `model.weights: ./checkpoints/model.pt` in your config.

---

## 3. Web UI

`dfdetect ui` launches a local Gradio app with two tabs.

### Analyze tab

- **Upload area:** drag-and-drop multiple images or videos.
- **Threshold slider:** decision boundary for FAKE vs REAL.
- **Frames/sec slider:** how aggressively to sample videos (more = slower, more thorough).
- **Heatmaps checkbox:** generate Grad-CAM overlays for the most-suspicious face per file.
- **Download:** a self-contained HTML report you can open in any browser.

### Train tab

Fine-tune EfficientNet-B4 on your own labelled data without leaving the UI.

- **Real folder** / **Fake folder**: absolute paths. **Both are searched recursively** and may contain images **and** videos at any depth. Subfolders that group by source identity (`real/personA/...`) are flattened into the class.
- **Output folder:** where `model.pt` + `train.json` go.
- **Epochs / LR / Batch size / Val split:** standard knobs.
- **Video sample fps / Max frames per video:** how aggressively to subsample video clips into face crops.
- **Validate dataset** runs the constraint gate (≥ 50/class, ratio ∈ [0.2, 0.8], ≥ 5 distinct video sources for video-only classes) and reports counts before any training starts.
- **Train** launches the run on a background thread; **Refresh status** updates the progress line (face extraction → epoch → done).

Files never leave the host — Gradio is bound to `127.0.0.1` by default. `--share` opens a public tunnel for *access*; the actual model still runs locally.

---

## 4. Output formats

### JSON

```jsonc
[
  {
    "path": "/home/me/clip.mp4",
    "kind": "video",
    "verdict": "FAKE",
    "score": 0.873,
    "faces_analyzed": 28,
    "frame_scores": [
      {"frame": 0,  "t": 0.0, "score": 0.81},
      {"frame": 30, "t": 1.0, "score": 0.92}
    ],
    "processing_time_s": 4.21,
    "model_name": "efficientnet_b4",
    "model_hash": "9f3a..."
  }
]
```

### CSV

Flat one-row-per-file with columns: `path, kind, verdict, score, faces_analyzed, processing_time_s, model_name, model_hash, error`. Per-frame timelines are dropped — use JSON or HTML if you need them.

### HTML

A static, self-contained report with score bars and a per-frame timeline strip (green → red gradient). Open in any browser.

### Grad-CAM heatmaps

When `--heatmaps` is set, the most-suspicious face crop per file is saved as a PNG with a heatmap overlay under `<output-dir>/heatmaps/`. The HTML report does not yet inline these — see them on disk, or open them next to the report.

### Per-file SVG timelines (HTML)

The HTML report now renders a per-file detail block below the summary table for every video with frame scores. The detail block contains an SVG line chart of score vs. time with:

- A dashed horizontal line at the **threshold**.
- A shaded band for the **uncertain band**.
- Translucent red boxes over **fake segments** (run-length-encoded high-score frames; see "Segments" below).
- Per-frame markers with tooltips (`title="t=... score=..."`).

A bullet list under the chart enumerates each segment with its `t_start → t_end`, peak score, mean score, and frame count.

### Segments

The JSON output now includes a `segments` field per result:

```jsonc
"segments": [
  {"t_start": 4.2, "t_end": 7.8, "score_peak": 0.94, "score_mean": 0.86, "n_frames": 4}
]
```

Segments are computed by run-length-encoding frames where `score >= threshold`. Short gaps (≤ 2s by default) are merged so a single dip doesn't split what should be one continuous fake region. Segments are also persisted in SQLite.

---

## 5. Verdict semantics

| Verdict | Meaning |
|---|---|
| `REAL` | Score is below the threshold and outside the uncertain band. |
| `FAKE` | Score ≥ threshold and outside the uncertain band. |
| `UNCERTAIN` | Score falls inside `uncertain_band` (default `[0.4, 0.6]`). |
| `NO_FACE` | The face detector found nothing analyzable. |
| `ERROR` | Preprocessing or inference failed for this file (see `error` field). The batch continues. |

A single bad file never aborts a batch — the failure is captured per file.

---

## 6. Common workflows

### Verifying a single suspicious clip

```bash
dfdetect analyze ./suspicious.mp4 --output html --heatmaps -f 2
xdg-open ./reports/dfdetect-report.html
```

### Bulk vetting a directory

```bash
dfdetect analyze ./incoming -r --output csv --threshold 0.6
# Then sort the CSV in your spreadsheet of choice.
```

### Catching short splices in a video

Uniform sampling can miss fakes shorter than the sampling interval. Switch to scene/hybrid:

```yaml
# my-config.yaml
video:
  frames_per_second: 1.0
  sampling_strategy: hybrid    # uniform + scene boundaries
  scene_threshold: 0.55        # lower = pickier about cuts
```

```bash
dfdetect analyze suspicious.mp4 -c my-config.yaml --output html
```

### Re-thresholding without re-running

```bash
dfdetect runs                # find run id
dfdetect reverdict 42 --threshold 0.7
```

### Programmatic use

```python
from dfdetect.config import load_config
from dfdetect.pipeline import Pipeline

cfg = load_config()
cfg.threshold = 0.6
cfg.report.heatmaps = True

pipeline = Pipeline(cfg)
results = pipeline.run(["./media"])
for r in results:
    print(r.path, r.verdict.value, r.score)
```

See [ARCHITECTURE.md](ARCHITECTURE.md) for the dataflow.

---

## 7. Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `RuntimeError: CUDA out of memory` | Batch too large for VRAM | The runner auto-halves the batch — if it still fails, lower `batch_size` in the config or pass `-b 2`. |
| All files reported as `NO_FACE` | Tiny faces or low resolution | Lower `preprocess.min_face_size` in the config (default 40 px). |
| Scores look uncalibrated (everything ~0.5) | Using ImageNet fallback weights | Wire in a deepfake-trained checkpoint — see [MODELS.md](MODELS.md). |
| Scores systematically over- or under-confident | Calibration not fit | Set `model.calibration.temperature` (>1 softens, <1 sharpens) — see [MODELS.md](MODELS.md#calibration). |
| `ChecksumError: SHA-256 mismatch for ...` | Downloaded weights don't match the expected hash | Re-check the published `sha256`. The partial download is auto-deleted, so a corrected config will retry cleanly. |
| `decord` decode errors | Unsupported codec | The pipeline auto-falls back to OpenCV; if that also fails, transcode the file (e.g. `ffmpeg -i in.mkv -c copy out.mp4`). |
| `RetinaFace requested but insightface is not installed` | Switched detector without the optional dep | `pip install insightface onnxruntime` (or `onnxruntime-gpu` on CUDA), or set `face_detector: mtcnn`. |
| Gradio UI port in use | `--port 7860` already taken | Pass `--port 9000` (or any free port). |

For more, run any command with `--log-level DEBUG`.

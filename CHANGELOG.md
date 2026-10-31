# Changelog

## Unreleased — correctness & performance pass

### ⚠️ Action needed

- **Retrain checkpoints made with earlier versions of `dfdetect train`.**
  - Validation was split per *frame*, so frames of one clip landed in both train and val. An epoch with no usable val samples was also scored as loss `0.0`.
  - Either way `best_val_loss` could hit ~0. After that no later epoch could "improve", and `model.pt` stopped updating, often after epoch 1.
  - Training crops also lacked the 30% margin inference uses.
- Existing `resume.pt` files won't be resumed, because the config signature now includes `face_margin`.
- `model.weights` must now point at a **state dict** (`dfdetect train` output qualifies). Pickled whole-model files are refused (`weights_only=True`). A checkpoint for a different architecture now raises an error instead of silently producing random scores.

### Fixed — detection

- `preprocess.largest_face_only: true` was ignored with MTCNN: `detect()` returns every face, so all faces were scored.
- Multi-face tracking:
  - Tracks were retired between samples when the sampling step exceeded 30 source frames (e.g. 1 fps on 60 fps video). The gap now scales with the step.
  - Matching was order-dependent and could swap identities. It is now best-IoU-first.
- Video sampling:
  - A decord error mid-video fell back to OpenCV and re-yielded frames from the start.
  - Scene/hybrid probing held every probed full-resolution frame in memory.
- Grad-CAM:
  - Hooks were never removed, so every later inference pass kept capturing activations (memory leak and slowdown).
  - Heatmaps now explain the FAKE class rather than whichever class won.
  - Heatmaps now work for ensembles.
  - Same-named files in different folders overwrote each other's heatmap.
- Unreadable or unclassifiable inputs are now reported as `ERROR` (`input: …`). Before, they were pushed through the video decoder, and saving them crashed the run *after* analysis finished.

### Fixed — CLI, storage, reports

- `analyze`: `--heatmaps`, `--recursive` and `--output-dir` always overrode the config, even when not passed. `report.default_format` was never used. An invalid `--output` was only rejected after the whole analysis.
- `reverdict` turned `ERROR` results into `REAL`/`NO_FACE`, and ignored the configured threshold.
- `--config` with a missing file silently ran on defaults; it is now an error. `query --verdict` is validated and case-insensitive. Lower-case `--log-level` values work.
- Config validation: `video.frames_per_second > 0`, `video.max_frames >= 1`, `pipeline.workers >= 0`, `pipeline.prefetch >= 1`.
- **HTML report XSS:** autoescaping was off, because `select_autoescape` doesn't match `report.html.j2`. Crafted filenames could inject markup.
- Schema v3 migration rewritten:
  - Idempotent, and resumes if interrupted.
  - Fresh databases no longer get a stray JSON column.
  - Adds the missing indexes on `frame_scores.result_id`, `segments.result_id` and `results.run_id`.
  - Rows are copied with `executemany`.
- Results read back from the DB now carry `model_name` / `model_hash`.
- The parallel pipeline's producer thread could block forever on a full queue after the consumer stopped.
- Training: a successful batch-size probe was run twice. The best checkpoint is now written atomically. A Ctrl-C mid-save no longer leaves a corrupt `model.pt`.
- `scripts/prepare_dfd.py` named links with Python's per-process-salted `hash()`, so every "idempotent" re-run relinked and pruned everything.

### Performance

- Training caches face boxes after the first visit, so MTCNN runs once per sample instead of once per epoch. It also reuses open decord readers instead of reopening the video for every frame.
- Video decode: decord batches frames via `get_batch`, and OpenCV reads forward instead of seeking for every sample.
- `runs` counts results in SQL instead of loading every result row. Result queries use a fixed three queries rather than per-row lazy loads.
- Ensemble averaging is vectorised. The HTML template environment is built once.
- After a CUDA OOM, the runner frees the allocator cache before retrying.

### Internal

- Autotune state helpers (file locking, atomic writes) are de-duplicated into `dfdetect/_cache.py`. Locking degrades to a no-op without `fcntl`.
- New `DeepfakeModel.grad_cam_preprocess()` hook. `expand_bbox` is now public and shared by inference and training.
- `inference.aggregator.reverdict()` holds the re-thresholding rule. `MediaItem.kind` / `FileResult.kind` are typed `MediaKind | None`.
- Removed dead training helpers (`_image_crops`, `_video_crops`).
- `ruff` and `mypy` are clean. `tests/test_regressions.py` adds a test for each fix; each one fails on the previous code.

# Training scripts

End-to-end recipe for training `dfdetect` on the
[Kaggle DFD mirror](https://www.kaggle.com/datasets/sanikatiwarekar/deep-fake-detection-dfd-entire-original-dataset/data)
(Google's original Deep Fake Detection dataset).

## 1. Unzip the Kaggle download

Drop the unzipped dataset anywhere under the repo root, e.g.:

```
Deepfake_Detection/
└── DFD/
    ├── original_videos/    # or Real/ or Original sequences/ ...
    └── manipulated_videos/ # or Fake/ or DeepFakes/ ...
```

The prep script doesn't care about the exact folder names — it autodetects
by keyword (`real|original|pristine|authentic` vs
`fake|manipulated|deepfake|swapped`).

## 2. Prepare a train-ready tree

```bash
python scripts/prepare_dfd.py \
    --source ./DFD \
    --out ./data/dfd_train \
    --check
```

What it does:

- Walks the source tree, classifying every video by its containing folder.
- Subsamples the fake class (stratified by the leading actor id in the
  filename) so `fake/total ≤ 0.78`, under the trainer's 0.8 hard cap.
- Builds `./data/dfd_train/{real,fake}/` as **hardlinks** to the originals
  — no copy, no extra disk. The trainer's `walk_media` skips symlinks but
  follows hardlinks, so this is the path that actually works.
- `--check` runs the same bias-constraint gate the trainer runs, so you
  catch problems before kicking off the GPU.

Re-running is idempotent: existing hardlinks are kept, stale ones pruned.

## 3. Train

```bash
./scripts/train_dfd.sh
```

Defaults (overridable via env vars):

| Variable | Default | Notes |
| --- | --- | --- |
| `DATA_DIR` | `./data/dfd_train` | Prepared dataset root. |
| `OUT_DIR`  | `./checkpoints/dfd_efficientnet_b4` | Where `model.pt` + `train.json` land. |
| `EPOCHS`   | `5` | DFD is small + correlated; 3–7 is the sweet spot. |
| `BATCH_SIZE` | `16` | RTX 5070 Laptop (8 GB) handles 16 at 224 px with AMP. Drop to 8 on OOM. |
| `LR`       | `1e-4` | Standard fine-tune LR for EfficientNet-B4. |
| `VIDEO_FPS`| `1.0` | One frame per second of source video. |
| `MAX_FRAMES` | `32` | Hard cap per clip; prevents long videos from dominating. |
| `VAL_SPLIT`| `0.1` | Random split. **Caveat below.** |

The launcher runs the bias check first, then training with `--device cuda --amp`.

## 4. Wire the checkpoint into inference

Edit `config/default.yaml`:

```yaml
model:
  name: efficientnet_b4
  weights: ./checkpoints/dfd_efficientnet_b4/model.pt
```

Then:

```bash
dfdetect analyze ./samples
```

## Caveats specific to DFD

- **Single manipulation family** (face-swap). Models trained on DFD alone
  generalize poorly to reenactment / lip-sync / diffusion fakes. Blend
  with FaceForensics++ originals or DFDC for a stronger checkpoint.
- **Only 28 actors** and the same actor appears in both real and fake.
  The trainer currently splits train/val *by file*, not by actor, so the
  validation score will look optimistic. If you care about generalization,
  hold out a few actor ids manually before running `prepare_dfd.py`
  (e.g. move 4–5 actors' files into a separate `holdout/` folder and run
  evaluation against that).
- After training, fit calibration (temperature scaling) against held-out
  data and set `model.calibration.temperature` in the config — see
  [docs/MODELS.md](../docs/MODELS.md#calibration).

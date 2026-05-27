#!/usr/bin/env bash
# Train EfficientNet-B4 on the prepared DFD dataset.
#
# Run scripts/prepare_dfd.py first to build ./data/dfd_train/{real,fake}.
# Tuned for a single 8 GB CUDA GPU (RTX 5070 Laptop). Bump --batch-size on
# bigger cards or drop to 8 if you hit OOM.

set -euo pipefail

cd "$(dirname "$0")/.."

DATA_DIR="${DATA_DIR:-./data/dfd_train}"
OUT_DIR="${OUT_DIR:-./checkpoints/dfd_efficientnet_b4}"
EPOCHS="${EPOCHS:-5}"
BATCH_SIZE="${BATCH_SIZE:-16}"
LR="${LR:-1e-4}"
VIDEO_FPS="${VIDEO_FPS:-1.0}"
MAX_FRAMES="${MAX_FRAMES:-32}"
VAL_SPLIT="${VAL_SPLIT:-0.1}"

if [ ! -d "$DATA_DIR/real" ] || [ ! -d "$DATA_DIR/fake" ]; then
  echo "ERROR: $DATA_DIR/{real,fake} not found." >&2
  echo "Run scripts/prepare_dfd.py first, e.g.:" >&2
  echo "  python scripts/prepare_dfd.py --source ./DFD --out $DATA_DIR --check" >&2
  exit 1
fi

mkdir -p "$OUT_DIR"

echo "=== Pre-flight bias check ==="
dfdetect train \
  --real "$DATA_DIR/real" \
  --fake "$DATA_DIR/fake" \
  --check-only

echo
echo "=== Training (epochs=$EPOCHS, batch=$BATCH_SIZE, lr=$LR) ==="
dfdetect train \
  --real "$DATA_DIR/real" \
  --fake "$DATA_DIR/fake" \
  --output-dir "$OUT_DIR" \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --lr "$LR" \
  --val-split "$VAL_SPLIT" \
  --video-fps "$VIDEO_FPS" \
  --max-frames-per-video "$MAX_FRAMES" \
  --device cuda \
  --amp

echo
echo "Done. Checkpoint: $OUT_DIR/model.pt"
echo "To use it for inference, edit config/default.yaml:"
echo "  model.weights: $OUT_DIR/model.pt"

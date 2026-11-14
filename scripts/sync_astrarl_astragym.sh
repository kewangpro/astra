#!/usr/bin/env bash
# EMERGENCY ONLY — prefer: pip install -e "$ASTRA_ROOT" into Mini finetune-env.
# Transitional finetune/astrarl + finetune/astragym mirrors were removed (Epoch 10 / Phase 118).
# This script recreates them only if you explicitly need a cwd-relative fallback.
set -euo pipefail
ASTRA_ROOT="${ASTRA_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
ENSEMBLE_ROOT="${ENSEMBLE_ROOT:-$(cd "$ASTRA_ROOT/../ensemble" && pwd)}"
DEST_RL="$ENSEMBLE_ROOT/finetune/astrarl"
DEST_GYM="$ENSEMBLE_ROOT/finetune/astragym"
DEST_TUNE="$ENSEMBLE_ROOT/finetune/astratune"

echo "WARNING: recreating transitional mirrors under $ENSEMBLE_ROOT/finetune/"
echo "Preferred path: ~/finetune-env/bin/pip install -e \"$ASTRA_ROOT\""
echo "ASTRA_ROOT=$ASTRA_ROOT"
echo "ENSEMBLE_ROOT=$ENSEMBLE_ROOT"

rsync -a --delete \
  --exclude '__pycache__' --exclude '*.pyc' \
  "$ASTRA_ROOT/astrarl/" "$DEST_RL/"
rsync -a --delete \
  --exclude '__pycache__' --exclude '*.pyc' \
  "$ASTRA_ROOT/astragym/" "$DEST_GYM/"
rsync -a --delete \
  --exclude '__pycache__' --exclude '*.pyc' \
  "$ASTRA_ROOT/astratune/" "$DEST_TUNE/"

echo "Synced astrarl → $DEST_RL"
echo "Synced astragym → $DEST_GYM"
echo "Synced astratune → $DEST_TUNE"
echo "Mini: ~/finetune-env/bin/pip install -e \"$ASTRA_ROOT\""

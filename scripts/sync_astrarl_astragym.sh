#!/usr/bin/env bash
# Sync canonical astrarl/ + astragym/ into Ensemble finetune/ mirrors (transitional).
# Prefer: pip install -e "$ASTRA_ROOT" into Mini finetune-env, then delete mirrors.
set -euo pipefail
ASTRA_ROOT="${ASTRA_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
ENSEMBLE_ROOT="${ENSEMBLE_ROOT:-$(cd "$ASTRA_ROOT/../ensemble" && pwd)}"
DEST_RL="$ENSEMBLE_ROOT/finetune/astrarl"
DEST_GYM="$ENSEMBLE_ROOT/finetune/astragym"

echo "ASTRA_ROOT=$ASTRA_ROOT"
echo "ENSEMBLE_ROOT=$ENSEMBLE_ROOT"

rsync -a --delete \
  --exclude '__pycache__' --exclude '*.pyc' \
  "$ASTRA_ROOT/astrarl/" "$DEST_RL/"
rsync -a --delete \
  --exclude '__pycache__' --exclude '*.pyc' \
  "$ASTRA_ROOT/astragym/" "$DEST_GYM/"

echo "Synced astrarl → $DEST_RL"
echo "Synced astragym → $DEST_GYM"
echo "Mini long-term: ~/finetune-env/bin/pip install -e \"$ASTRA_ROOT\""

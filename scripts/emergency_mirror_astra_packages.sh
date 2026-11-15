#!/usr/bin/env bash
# Emergency: mirror astrarl/astragym/astratune into a product workspace.
# Prefer pip install -e ~/astra-pkg. Set PRODUCT_ROOT to the product repo root.
set -euo pipefail
ASTRA_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
if [[ -z "${PRODUCT_ROOT:-}" ]]; then
  echo "Set PRODUCT_ROOT to the product repository root" >&2
  exit 1
fi
DEST_RL="$PRODUCT_ROOT/finetune/astrarl"
DEST_GYM="$PRODUCT_ROOT/finetune/astragym"
DEST_TUNE="$PRODUCT_ROOT/finetune/astratune"

echo "WARNING: recreating transitional mirrors under $PRODUCT_ROOT/finetune/"
echo "ASTRA_ROOT=$ASTRA_ROOT"
echo "PRODUCT_ROOT=$PRODUCT_ROOT"
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

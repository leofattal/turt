#!/usr/bin/env bash
#
# Colab training driver (see docs/CLOUD.md): restores the C4 corpus from a
# Drive tarball (or builds and stashes it on the first run), then starts —
# or, after a session reset, resumes — pretraining via gpu-pretrain-loop.sh.
#
# Defaults target the free-tier T4: a ~33M-param model on ~0.7B C4 tokens.
# Every knob is env-overridable, e.g.:
#   !STEPS=160000 BATCH=16 bash /content/turt/scripts/colab-train.sh

set -euo pipefail
cd "$(dirname "$0")/.."

TARGET_GB=${TARGET_GB:-3}
DATA_TAR=${DATA_TAR:-/content/drive/MyDrive/turt-data-big.tar}

if [ ! -f data-big/train.bin ]; then
  if [ -f "$DATA_TAR" ]; then
    echo "=== restoring corpus from $DATA_TAR ==="
    tar xf "$DATA_TAR"
  else
    echo "=== building ${TARGET_GB}GB C4 corpus (first run only, ~1h) ==="
    pnpm prepare-data-big --target-gb "$TARGET_GB"
    if [ -d "$(dirname "$DATA_TAR")" ]; then
      echo "=== stashing corpus to $DATA_TAR so restarts skip the build ==="
      tar cf "$DATA_TAR" data-big/tokenizer.json data-big/meta.json data-big/train.bin data-big/val.bin
    fi
  fi
fi

LAYERS=${LAYERS:-8} HEADS=${HEADS:-8} EMBD=${EMBD:-512} BLOCK=${BLOCK:-512} \
BATCH=${BATCH:-8} STEPS=${STEPS:-160000} DATA=data-big OUT=${OUT:-turt-c4-33m.bin} \
  bash scripts/gpu-pretrain-loop.sh

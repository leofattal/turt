#!/usr/bin/env bash
#
# Supervisor for long GPU pretraining runs: launches scripts/gpu-pretrain.ts and,
# if it exits non-zero (a WebGPU driver hiccup over a multi-day run), resumes from
# the last checkpoint and keeps going. Exits 0 only when training reaches --steps.
#
# Usage:
#   LAYERS=8 HEADS=8 EMBD=512 BLOCK=256 BATCH=24 STEPS=80000 LR=6e-4 \
#     ARCH=modern OUT=gpt-big.bin bash scripts/gpu-pretrain-loop.sh
#
# ARCH is only passed on a fresh start; a resume adopts the checkpoint's own
# architecture (gpu-pretrain.ts ignores --arch when --resume is set).
#
# All knobs have defaults (see below). Progress streams to stdout; redirect to a
# log when running in the background.

set -u
cd "$(dirname "$0")/.."

LAYERS=${LAYERS:-8}
HEADS=${HEADS:-8}
EMBD=${EMBD:-512}
BLOCK=${BLOCK:-256}
BATCH=${BATCH:-24}
STEPS=${STEPS:-80000}
LR=${LR:-6e-4}
ARCH=${ARCH:-modern}
OUT=${OUT:-gpt-big.bin}
DATA=${DATA:-data}

DENO_ARGS=(run --unstable-webgpu --unstable-sloppy-imports --allow-read --allow-write scripts/gpu-pretrain.ts)
COMMON=(--layers "$LAYERS" --heads "$HEADS" --embd "$EMBD" --block "$BLOCK" --batch "$BATCH" --steps "$STEPS" --lr "$LR" --data "$DATA" --out "$OUT")

attempt=0
while true; do
  attempt=$((attempt + 1))
  if [[ -f "checkpoints/$OUT" ]]; then
    echo "=== attempt $attempt: resuming from checkpoints/$OUT ==="
    deno "${DENO_ARGS[@]}" "${COMMON[@]}" --resume "$OUT"
  else
    echo "=== attempt $attempt: fresh start (arch $ARCH) ==="
    deno "${DENO_ARGS[@]}" "${COMMON[@]}" --arch "$ARCH"
  fi
  code=$?
  if [[ $code -eq 0 ]]; then
    echo "=== training finished cleanly (exit 0) after $attempt attempt(s) ==="
    break
  fi
  echo "=== gpu-pretrain exited $code; resuming in 15s (attempt $attempt) ==="
  sleep 15
done

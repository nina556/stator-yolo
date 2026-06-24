#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${1:?Usage: bash export/export_engine.sh /path/to/best.pt}"
IMGSZ="${IMGSZ:-640}"
DEVICE="${DEVICE:-0}"

yolo export \
  model="${MODEL_PATH}" \
  format=engine \
  imgsz="${IMGSZ}" \
  device="${DEVICE}" \
  half=True \
  simplify=True

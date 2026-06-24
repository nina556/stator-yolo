#!/usr/bin/env bash
set -euo pipefail

MODEL="${1:-yolov8n.pt}"
DATASET="${2:-data/dataset.yaml}"
PROJECT="${3:-runs}"
NAME="${4:-stator_yolov8}"
EPOCHS="${EPOCHS:-100}"
IMGSZ="${IMGSZ:-640}"
BATCH="${BATCH:-16}"
DEVICE="${DEVICE:-0}"

yolo detect train \
  model="${MODEL}" \
  data="${DATASET}" \
  project="${PROJECT}" \
  name="${NAME}" \
  epochs="${EPOCHS}" \
  imgsz="${IMGSZ}" \
  batch="${BATCH}" \
  device="${DEVICE}" \
  pretrained=True \
  cache=False \
  workers=4 \
  degrees=10 \
  translate=0.05 \
  scale=0.1 \
  fliplr=0.0

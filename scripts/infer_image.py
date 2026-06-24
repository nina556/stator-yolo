#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run YOLO inference on one image.")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("runs/infer/result.jpg"))
    parser.add_argument("--conf", type=float, default=0.25)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = YOLO(str(args.model))
    results = model.predict(source=str(args.image), conf=args.conf, verbose=False)
    annotated = results[0].plot()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(args.output), annotated):
        raise SystemExit(f"Failed to write output image to {args.output}")
    print(f"Saved inference result to {args.output}")


if __name__ == "__main__":
    main()

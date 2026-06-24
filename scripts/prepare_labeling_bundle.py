#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare a flat image bundle for labeling.")
    parser.add_argument("--frames-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    copied = 0
    for image_path in sorted(args.frames_dir.rglob("*")):
        if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        target_path = args.output_dir / image_path.name
        if target_path.exists():
            raise SystemExit(f"Duplicate image filename: {image_path.name}")
        shutil.copy2(image_path, target_path)
        copied += 1

    print(f"Prepared {copied} images in {args.output_dir}")


if __name__ == "__main__":
    main()

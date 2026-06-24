#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate YOLO detection labels.")
    parser.add_argument("--images-dir", type=Path, required=True)
    parser.add_argument("--labels-dir", type=Path, required=True)
    return parser.parse_args()


def list_images(images_dir: Path) -> list[Path]:
    return sorted(
        path for path in images_dir.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def validate_label_file(label_path: Path) -> list[str]:
    errors: list[str] = []
    for line_number, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), start=1):
        parts = line.split()
        if len(parts) != 5:
            errors.append(f"{label_path}: line {line_number} has {len(parts)} fields")
            continue
        try:
            class_id = int(parts[0])
            values = [float(value) for value in parts[1:]]
        except ValueError:
            errors.append(f"{label_path}: line {line_number} contains non-numeric values")
            continue
        if class_id < 0:
            errors.append(f"{label_path}: line {line_number} has negative class id")
        for value in values:
            if not 0.0 <= value <= 1.0:
                errors.append(f"{label_path}: line {line_number} value {value} is out of [0, 1]")
    return errors


def main() -> None:
    args = parse_args()
    images = list_images(args.images_dir)
    if not images:
        raise SystemExit(f"No images found in {args.images_dir}")

    errors: list[str] = []
    for image_path in images:
        label_path = args.labels_dir / f"{image_path.stem}.txt"
        if not label_path.exists():
            errors.append(f"Missing label file for {image_path.name}")
            continue
        errors.extend(validate_label_file(label_path))

    if errors:
        for error in errors:
            print(error)
        raise SystemExit(f"Validation failed with {len(errors)} issue(s)")

    print(f"Validated {len(images)} images and labels successfully.")


if __name__ == "__main__":
    main()

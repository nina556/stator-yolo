#!/usr/bin/env python3
from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Split YOLO dataset into train/val/test folders.")
    parser.add_argument("--images-dir", type=Path, required=True)
    parser.add_argument("--labels-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def ensure_ratios(train_ratio: float, val_ratio: float, test_ratio: float) -> None:
    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 1e-6:
        raise SystemExit(f"Ratios must sum to 1.0, got {total}")


def list_images(images_dir: Path) -> list[Path]:
    return sorted(
        path for path in images_dir.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def copy_pair(image_path: Path, label_path: Path, split: str, output_dir: Path) -> None:
    image_target = output_dir / "images" / split / image_path.name
    label_target = output_dir / "labels" / split / label_path.name
    image_target.parent.mkdir(parents=True, exist_ok=True)
    label_target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(image_path, image_target)
    shutil.copy2(label_path, label_target)


def main() -> None:
    args = parse_args()
    ensure_ratios(args.train_ratio, args.val_ratio, args.test_ratio)

    images = list_images(args.images_dir)
    if not images:
        raise SystemExit(f"No images found in {args.images_dir}")

    dataset: list[tuple[Path, Path]] = []
    for image_path in images:
        label_path = args.labels_dir / f"{image_path.stem}.txt"
        if not label_path.exists():
            raise SystemExit(f"Missing label file for {image_path}")
        dataset.append((image_path, label_path))

    random.seed(args.seed)
    random.shuffle(dataset)

    total = len(dataset)
    train_end = int(total * args.train_ratio)
    val_end = train_end + int(total * args.val_ratio)

    splits = {
        "train": dataset[:train_end],
        "val": dataset[train_end:val_end],
        "test": dataset[val_end:],
    }

    for split_name, items in splits.items():
        for image_path, label_path in items:
            copy_pair(image_path, label_path, split_name, args.output_dir)
        print(f"{split_name}: {len(items)} samples")


if __name__ == "__main__":
    main()

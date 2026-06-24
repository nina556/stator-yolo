#!/usr/bin/env python3
from __future__ import annotations

import argparse
import random
from pathlib import Path

import cv2
import numpy as np

try:
    import albumentations as A
except ModuleNotFoundError:
    A = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Augment YOLO training images in place.")
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--copies-per-image", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_yolo_bboxes(label_path: Path) -> list[list[float]]:
    boxes: list[list[float]] = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        class_id, x_center, y_center, width, height = line.split()
        boxes.append([float(x_center), float(y_center), float(width), float(height), int(class_id)])
    return boxes


def save_yolo_bboxes(label_path: Path, boxes: list[list[float]]) -> None:
    lines = []
    for x_center, y_center, width, height, class_id in boxes:
        lines.append(
            f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
        )
    label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def build_transform() -> A.Compose:
    if A is None:
        raise RuntimeError("albumentations is not available")
    return A.Compose(
        [
            A.RandomBrightnessContrast(p=0.5),
            A.GaussNoise(p=0.3),
            A.MotionBlur(blur_limit=5, p=0.2),
            A.GaussianBlur(blur_limit=(3, 5), p=0.2),
            A.Rotate(limit=20, border_mode=cv2.BORDER_REPLICATE, p=0.4),
            A.CLAHE(p=0.2),
        ],
        bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"], min_visibility=0.3),
    )


def augment_image_without_bbox_change(image: np.ndarray) -> np.ndarray:
    augmented = image.copy()

    alpha = random.uniform(0.85, 1.2)
    beta = random.uniform(-25, 25)
    augmented = cv2.convertScaleAbs(augmented, alpha=alpha, beta=beta)

    if random.random() < 0.35:
        noise = np.random.normal(0, random.uniform(4, 12), augmented.shape).astype(np.int16)
        augmented = np.clip(augmented.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    if random.random() < 0.3:
        augmented = cv2.GaussianBlur(augmented, (3, 3), 0)

    if random.random() < 0.25:
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        augmented = cv2.filter2D(augmented, -1, kernel)

    return augmented


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    images_dir = args.dataset_dir / "images" / "train"
    labels_dir = args.dataset_dir / "labels" / "train"
    image_paths = sorted(path for path in images_dir.glob("*") if path.is_file())
    transform = build_transform() if A is not None else None

    if not image_paths:
        raise SystemExit(f"No training images found in {images_dir}")

    if transform is None:
        print("albumentations not installed; using OpenCV-safe image-only augmentation.")

    created = 0
    for image_path in image_paths:
        label_path = labels_dir / f"{image_path.stem}.txt"
        if not label_path.exists():
            continue

        image = cv2.imread(str(image_path))
        if image is None:
            raise SystemExit(f"Failed to read image {image_path}")

        boxes = load_yolo_bboxes(label_path)
        if not boxes:
            continue

        bbox_values = [box[:4] for box in boxes]
        class_labels = [box[4] for box in boxes]

        for index in range(args.copies_per_image):
            if transform is not None:
                augmented = transform(image=image, bboxes=bbox_values, class_labels=class_labels)
                if not augmented["bboxes"]:
                    continue
                aug_image = augmented["image"]
                aug_boxes = [
                    [bbox[0], bbox[1], bbox[2], bbox[3], class_id]
                    for bbox, class_id in zip(augmented["bboxes"], augmented["class_labels"], strict=True)
                ]
            else:
                aug_image = augment_image_without_bbox_change(image)
                aug_boxes = boxes

            aug_image_path = images_dir / f"{image_path.stem}_aug{index}{image_path.suffix}"
            aug_label_path = labels_dir / f"{image_path.stem}_aug{index}.txt"

            ok = cv2.imwrite(str(aug_image_path), aug_image)
            if not ok:
                raise SystemExit(f"Failed to write augmented image {aug_image_path}")

            save_yolo_bboxes(aug_label_path, aug_boxes)
            created += 1

    print(f"Created {created} augmented training samples.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Minimal stator detector for ONNX Runtime deployments."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort


CLASS_NAMES = ["stator"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run stator detection with ONNX Runtime.")
    parser.add_argument("--model", type=Path, default=Path(__file__).with_name("best.onnx"))
    parser.add_argument("--source", required=True, help="Image, video path, or camera index such as 0.")
    parser.add_argument("--output", type=Path, help="Optional annotated image/video output path.")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--imgsz", type=int, default=960, help="Fallback size for a dynamic ONNX model.")
    parser.add_argument("--show", action="store_true", help="Display the annotated stream.")
    return parser.parse_args()


def letterbox(image: np.ndarray, width: int, height: int) -> tuple[np.ndarray, float, int, int]:
    source_height, source_width = image.shape[:2]
    scale = min(width / source_width, height / source_height)
    resized_width = round(source_width * scale)
    resized_height = round(source_height * scale)
    resized = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)
    pad_x = (width - resized_width) // 2
    pad_y = (height - resized_height) // 2
    canvas = np.full((height, width, 3), 114, dtype=np.uint8)
    canvas[pad_y:pad_y + resized_height, pad_x:pad_x + resized_width] = resized
    return canvas, scale, pad_x, pad_y


class StatorDetector:
    def __init__(self, model: Path, confidence: float, iou: float, fallback_size: int) -> None:
        if not model.is_file():
            raise FileNotFoundError(f"ONNX model not found: {model}")
        providers = ["CPUExecutionProvider"]
        if "CUDAExecutionProvider" in ort.get_available_providers():
            providers.insert(0, "CUDAExecutionProvider")
        self.session = ort.InferenceSession(str(model), providers=providers)
        self.input = self.session.get_inputs()[0]
        shape = self.input.shape
        self.height = shape[2] if isinstance(shape[2], int) else fallback_size
        self.width = shape[3] if isinstance(shape[3], int) else fallback_size
        self.confidence = confidence
        self.iou = iou
        print(f"Model: {model}")
        print(f"Provider: {self.session.get_providers()[0]}")
        print(f"Input: {self.width}x{self.height}")

    def predict(self, image: np.ndarray) -> list[tuple[int, float, tuple[int, int, int, int]]]:
        prepared, scale, pad_x, pad_y = letterbox(image, self.width, self.height)
        tensor = cv2.cvtColor(prepared, cv2.COLOR_BGR2RGB)
        tensor = tensor.transpose(2, 0, 1)[None].astype(np.float32) / 255.0
        output = self.session.run(None, {self.input.name: tensor})[0]
        predictions = np.squeeze(output)
        if predictions.shape[0] < predictions.shape[1]:
            predictions = predictions.T

        boxes: list[list[int]] = []
        scores: list[float] = []
        class_ids: list[int] = []
        image_height, image_width = image.shape[:2]
        for row in predictions:
            class_scores = row[4:]
            class_id = int(np.argmax(class_scores))
            score = float(class_scores[class_id])
            if score < self.confidence:
                continue
            center_x, center_y, width, height = row[:4]
            x1 = int(round((center_x - width / 2 - pad_x) / scale))
            y1 = int(round((center_y - height / 2 - pad_y) / scale))
            x2 = int(round((center_x + width / 2 - pad_x) / scale))
            y2 = int(round((center_y + height / 2 - pad_y) / scale))
            x1 = max(0, min(image_width - 1, x1))
            y1 = max(0, min(image_height - 1, y1))
            x2 = max(0, min(image_width - 1, x2))
            y2 = max(0, min(image_height - 1, y2))
            if x2 <= x1 or y2 <= y1:
                continue
            boxes.append([x1, y1, x2 - x1, y2 - y1])
            scores.append(score)
            class_ids.append(class_id)

        kept = cv2.dnn.NMSBoxes(boxes, scores, self.confidence, self.iou)
        return [(class_ids[index], scores[index], tuple(boxes[index])) for index in kept]


def annotate(image: np.ndarray, detections: list[tuple[int, float, tuple[int, int, int, int]]]) -> np.ndarray:
    result = image.copy()
    for class_id, score, (x, y, width, height) in detections:
        label = f"{CLASS_NAMES[class_id]} {score:.2f}"
        cv2.rectangle(result, (x, y), (x + width, y + height), (60, 220, 80), 2)
        cv2.putText(
            result,
            label,
            (x, max(20, y - 7)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (60, 220, 80),
            2,
            cv2.LINE_AA,
        )
    return result


def image_inference(detector: StatorDetector, source: Path, output: Path | None, show: bool) -> None:
    image = cv2.imread(str(source))
    if image is None:
        raise RuntimeError(f"Failed to read image: {source}")
    detections = detector.predict(image)
    result = annotate(image, detections)
    print(f"Detections: {len(detections)}")
    for class_id, score, box in detections:
        print(f"  {CLASS_NAMES[class_id]} confidence={score:.3f} box={box}")
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(output), result):
            raise RuntimeError(f"Failed to write image: {output}")
        print(f"Saved: {output}")
    if show:
        cv2.imshow("Stator ONNX", result)
        cv2.waitKey(0)


def stream_inference(
    detector: StatorDetector,
    source: str,
    output: Path | None,
    show: bool,
) -> None:
    capture_source: int | str = int(source) if source.isdigit() else source
    capture = cv2.VideoCapture(capture_source)
    if not capture.isOpened():
        raise RuntimeError(f"Failed to open source: {source}")
    writer = None
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            result = annotate(frame, detector.predict(frame))
            if output and writer is None:
                output.parent.mkdir(parents=True, exist_ok=True)
                fps = capture.get(cv2.CAP_PROP_FPS) or 25
                writer = cv2.VideoWriter(
                    str(output),
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    fps,
                    (result.shape[1], result.shape[0]),
                )
            if writer:
                writer.write(result)
            if show:
                cv2.imshow("Stator ONNX", result)
                if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                    break
    finally:
        capture.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()


def main() -> None:
    args = parse_args()
    detector = StatorDetector(args.model, args.conf, args.iou, args.imgsz)
    source_path = Path(args.source)
    if source_path.is_file() and source_path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
        image_inference(detector, source_path, args.output, args.show)
    else:
        stream_inference(detector, args.source, args.output, args.show)


if __name__ == "__main__":
    main()

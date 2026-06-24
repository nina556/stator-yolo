#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2


VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".m4v"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract sampled frames from videos.")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=Path("data/manifests/frame_manifest.csv"),
    )
    parser.add_argument(
        "--sample-fps",
        type=float,
        default=2.0,
        help="Target frame extraction rate per second.",
    )
    parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=95,
        help="OpenCV JPEG quality from 0 to 100.",
    )
    return parser.parse_args()


def iter_videos(input_dir: Path) -> list[Path]:
    return sorted(
        path for path in input_dir.rglob("*") if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
    )


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def extract_video(
    video_path: Path,
    output_dir: Path,
    writer: csv.DictWriter,
    sample_fps: float,
    jpeg_quality: int,
) -> None:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    source_fps = capture.get(cv2.CAP_PROP_FPS) or 0.0
    source_fps = source_fps if source_fps > 0 else sample_fps
    frame_interval = max(int(round(source_fps / sample_fps)), 1)

    relative_stem = video_path.stem
    frame_index = 0
    saved_count = 0

    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if frame_index % frame_interval != 0:
            frame_index += 1
            continue

        filename = f"{relative_stem}_f{frame_index:06d}.jpg"
        output_path = output_dir / filename
        ensure_parent(output_path)
        success = cv2.imwrite(
            str(output_path),
            frame,
            [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality],
        )
        if not success:
            raise RuntimeError(f"Failed to write frame: {output_path}")

        writer.writerow(
            {
                "video_path": str(video_path),
                "frame_index": frame_index,
                "timestamp_sec": frame_index / source_fps if source_fps else 0.0,
                "image_path": str(output_path),
                "width": frame.shape[1],
                "height": frame.shape[0],
            }
        )
        saved_count += 1
        frame_index += 1

    capture.release()
    print(f"Saved {saved_count} frames from {video_path}")


def main() -> None:
    args = parse_args()
    videos = iter_videos(args.input_dir)
    if not videos:
        raise SystemExit(f"No videos found in {args.input_dir}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    ensure_parent(args.manifest_path)

    with args.manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "video_path",
                "frame_index",
                "timestamp_sec",
                "image_path",
                "width",
                "height",
            ],
        )
        writer.writeheader()
        for video_path in videos:
            extract_video(
                video_path=video_path,
                output_dir=args.output_dir,
                writer=writer,
                sample_fps=args.sample_fps,
                jpeg_quality=args.jpeg_quality,
            )


if __name__ == "__main__":
    main()

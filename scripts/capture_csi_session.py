#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import time
from datetime import datetime
from pathlib import Path

import cv2


SUPPORTED_MODES = {
    (3280, 2464): 21,
    (3280, 1848): 28,
    (1920, 1080): 30,
    (1640, 1232): 30,
    (1280, 720): 60,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Capture one CSI camera session for stator YOLO labeling."
    )
    parser.add_argument("--sensor-id", type=int, default=0, choices=[0, 1])
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--flip-method", type=int, default=2)
    parser.add_argument("--duration-sec", type=float, default=60.0)
    parser.add_argument("--sample-fps", type=float, default=2.0)
    parser.add_argument(
        "--session-id",
        default=None,
        help="Defaults to csi{sensor_id}_YYYYmmdd_HHMMSS.",
    )
    parser.add_argument("--scene", default="")
    parser.add_argument("--lighting", default="")
    parser.add_argument("--background", default="")
    parser.add_argument("--pose-group", default="")
    parser.add_argument("--occlusion-level", default="")
    parser.add_argument("--robot-state", default="")
    parser.add_argument("--notes", default="")
    parser.add_argument("--raw-video-dir", type=Path, default=Path("data/raw_videos"))
    parser.add_argument("--frames-dir", type=Path, default=Path("data/frames/raw"))
    parser.add_argument(
        "--frame-manifest",
        type=Path,
        default=Path("data/manifests/frame_manifest.csv"),
    )
    parser.add_argument(
        "--session-manifest",
        type=Path,
        default=Path("data/manifests/session_manifest.csv"),
    )
    parser.add_argument("--no-preview", action="store_true")
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--jpeg-quality", type=int, default=95)
    return parser.parse_args()


def validate_mode(width: int, height: int, fps: int) -> None:
    max_fps = SUPPORTED_MODES.get((width, height))
    if max_fps is None:
        modes = ", ".join(f"{w}x{h}@{mode_fps}" for (w, h), mode_fps in SUPPORTED_MODES.items())
        raise SystemExit(f"Unsupported mode {width}x{height}. Supported modes: {modes}")
    if fps > max_fps:
        raise SystemExit(f"{width}x{height} supports up to {max_fps} FPS, got {fps} FPS")


def build_pipeline(sensor_id: int, width: int, height: int, fps: int, flip_method: int) -> str:
    return (
        f"nvarguscamerasrc sensor-id={sensor_id} ! "
        f"video/x-raw(memory:NVMM), width={width}, height={height}, "
        f"format=(string)NV12, framerate=(fraction){fps}/1 ! "
        f"nvvidconv flip-method={flip_method} ! "
        f"video/x-raw, width={width}, height={height}, format=(string)BGRx ! "
        "videoconvert ! video/x-raw, format=(string)BGR ! appsink"
    )


def append_csv_row(path: Path, fieldnames: list[str], row: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def make_session_id(sensor_id: int) -> str:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"csi{sensor_id}_{stamp}"


def main() -> None:
    args = parse_args()
    validate_mode(args.width, args.height, args.fps)
    if args.sample_fps <= 0:
        raise SystemExit("--sample-fps must be > 0")
    if args.duration_sec <= 0:
        raise SystemExit("--duration-sec must be > 0")

    session_id = args.session_id or make_session_id(args.sensor_id)
    session_frames_dir = args.frames_dir / session_id
    session_frames_dir.mkdir(parents=True, exist_ok=True)
    args.raw_video_dir.mkdir(parents=True, exist_ok=True)

    video_path = args.raw_video_dir / f"{session_id}.mp4"
    pipeline = build_pipeline(
        sensor_id=args.sensor_id,
        width=args.width,
        height=args.height,
        fps=args.fps,
        flip_method=args.flip_method,
    )
    capture = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
    if not capture.isOpened():
        raise SystemExit(f"Failed to open CSI camera with pipeline:\n{pipeline}")

    writer = None
    if not args.no_video:
        writer = cv2.VideoWriter(
            str(video_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            args.fps,
            (args.width, args.height),
        )
        if not writer.isOpened():
            capture.release()
            raise SystemExit(f"Failed to open video writer: {video_path}")

    frame_fields = [
        "session_id",
        "sensor_id",
        "frame_index",
        "timestamp_sec",
        "image_path",
        "width",
        "height",
    ]
    session_fields = [
        "session_id",
        "video_file",
        "scene",
        "lighting",
        "background",
        "pose_group",
        "occlusion_level",
        "robot_state",
        "camera_id",
        "notes",
    ]

    start_time = time.monotonic()
    next_sample_time = 0.0
    frame_index = 0
    saved_frames = 0

    print(f"Capturing {session_id} from sensor {args.sensor_id}")
    print(f"Pipeline: {pipeline}")
    print("Press q to stop early.")

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break

            elapsed = time.monotonic() - start_time
            if elapsed >= args.duration_sec:
                break

            if writer is not None:
                writer.write(frame)

            if elapsed >= next_sample_time:
                image_path = session_frames_dir / f"{session_id}_f{frame_index:06d}.jpg"
                success = cv2.imwrite(
                    str(image_path),
                    frame,
                    [int(cv2.IMWRITE_JPEG_QUALITY), args.jpeg_quality],
                )
                if not success:
                    raise SystemExit(f"Failed to write frame: {image_path}")
                append_csv_row(
                    args.frame_manifest,
                    frame_fields,
                    {
                        "session_id": session_id,
                        "sensor_id": args.sensor_id,
                        "frame_index": frame_index,
                        "timestamp_sec": f"{elapsed:.3f}",
                        "image_path": str(image_path),
                        "width": args.width,
                        "height": args.height,
                    },
                )
                saved_frames += 1
                next_sample_time += 1.0 / args.sample_fps

            if not args.no_preview:
                fps_text = f"sampled {saved_frames} | {elapsed:.1f}s"
                cv2.putText(
                    frame,
                    fps_text,
                    (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2,
                )
                cv2.imshow(f"capture {session_id}", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            frame_index += 1
    finally:
        capture.release()
        if writer is not None:
            writer.release()
        if not args.no_preview:
            cv2.destroyAllWindows()

    append_csv_row(
        args.session_manifest,
        session_fields,
        {
            "session_id": session_id,
            "video_file": "" if args.no_video else str(video_path),
            "scene": args.scene,
            "lighting": args.lighting,
            "background": args.background,
            "pose_group": args.pose_group,
            "occlusion_level": args.occlusion_level,
            "robot_state": args.robot_state,
            "camera_id": args.sensor_id,
            "notes": args.notes,
        },
    )
    print(f"Saved {saved_frames} sampled frames to {session_frames_dir}")
    if not args.no_video:
        print(f"Saved video to {video_path}")


if __name__ == "__main__":
    main()

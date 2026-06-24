#!/usr/bin/env python
# encoding: utf-8
import argparse
import time

import cv2 as cv


def parse_args():
    parser = argparse.ArgumentParser(description="Preview one Jetson CSI camera.")
    parser.add_argument("--sensor-id", type=int, default=1, choices=[0, 1])
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument(
        "--flip-method",
        type=int,
        default=2,
        help="nvvidconv flip-method. 2 rotates 180 degrees for upside-down cameras.",
    )
    return parser.parse_args()


def build_pipeline(sensor_id, width, height, fps, flip_method):
    return (
        f"nvarguscamerasrc sensor-id={sensor_id} ! "
        f"video/x-raw(memory:NVMM), width={width}, height={height}, "
        f"format=(string)NV12, framerate=(fraction){fps}/1 ! "
        f"nvvidconv flip-method={flip_method} ! "
        f"video/x-raw, width={width}, height={height}, format=(string)BGRx ! "
        "videoconvert ! video/x-raw, format=(string)BGR ! appsink"
    )


def validate_mode(width, height, fps):
    supported_modes = {
        (3280, 2464): 21,
        (3280, 1848): 28,
        (1920, 1080): 30,
        (1640, 1232): 30,
        (1280, 720): 60,
    }
    max_fps = supported_modes.get((width, height))
    if max_fps is None:
        modes = ", ".join(f"{w}x{h}@{mode_fps}" for (w, h), mode_fps in supported_modes.items())
        raise SystemExit(f"Unsupported mode {width}x{height}. Supported modes: {modes}")
    if fps > max_fps:
        raise SystemExit(f"{width}x{height} supports up to {max_fps} FPS, got {fps} FPS")


def main():
    args = parse_args()
    validate_mode(args.width, args.height, args.fps)
    pipeline = build_pipeline(
        sensor_id=args.sensor_id,
        width=args.width,
        height=args.height,
        fps=args.fps,
        flip_method=args.flip_method,
    )
    capture = cv.VideoCapture(pipeline, cv.CAP_GSTREAMER)

    if not capture.isOpened():
        raise SystemExit(f"Failed to open CSI camera with pipeline:\n{pipeline}")

    print("pipeline:", pipeline)
    print("capture get FPS:", capture.get(cv.CAP_PROP_FPS))
    while capture.isOpened():
        start = time.time()
        ret, frame = capture.read()
        if not ret:
            break

        end = time.time()
        fps = 1 / max(end - start, 1e-6)
        text = "FPS : " + str(int(fps))
        cv.putText(frame, text, (20, 30), cv.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 1)
        cv.imshow(f"camera{args.sensor_id}", frame)
        if cv.waitKey(1) & 0xFF == ord("q"):
            break

    capture.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()

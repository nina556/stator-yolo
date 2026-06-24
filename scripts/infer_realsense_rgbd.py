#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import cv2
import numpy as np
import pyrealsense2 as rs
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run YOLO on RealSense RGB and report depth/3D camera coordinates.")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--color-width", type=int, default=640)
    parser.add_argument("--color-height", type=int, default=480)
    parser.add_argument("--color-fps", type=int, default=30)
    parser.add_argument("--depth-width", type=int, default=640)
    parser.add_argument("--depth-height", type=int, default=480)
    parser.add_argument("--depth-fps", type=int, default=30)
    parser.add_argument("--frames", type=int, default=0, help="0 means run until q/Ctrl-C.")
    parser.add_argument("--frame-timeout-ms", type=int, default=5000)
    parser.add_argument("--max-timeouts", type=int, default=3)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/realsense_yolo"))
    parser.add_argument("--save-video", action="store_true")
    parser.add_argument("--save-csv", action="store_true")
    parser.add_argument("--no-preview", action="store_true")
    parser.add_argument("--device", default=None, help="Ultralytics device, e.g. 0 or cpu.")
    parser.add_argument(
        "--flip",
        choices=["none", "vertical", "horizontal", "both"],
        default="none",
    )
    parser.add_argument("--depth-window", type=int, default=5, help="Median depth window around bbox center, odd pixels.")
    parser.add_argument("--min-depth-m", type=float, default=0.05)
    parser.add_argument("--max-depth-m", type=float, default=6.0)
    return parser.parse_args()


def flip_frame(frame: np.ndarray, mode: str) -> np.ndarray:
    if mode == "vertical":
        return cv2.flip(frame, 0)
    if mode == "horizontal":
        return cv2.flip(frame, 1)
    if mode == "both":
        return cv2.flip(frame, -1)
    return frame


def robust_depth_m(depth: np.ndarray, cx: int, cy: int, depth_scale: float, window: int, min_depth_m: float, max_depth_m: float) -> float:
    radius = max(0, window // 2)
    y0 = max(0, cy - radius)
    y1 = min(depth.shape[0], cy + radius + 1)
    x0 = max(0, cx - radius)
    x1 = min(depth.shape[1], cx + radius + 1)
    patch = depth[y0:y1, x0:x1].astype(np.float32) * depth_scale
    valid = patch[(patch >= min_depth_m) & (patch <= max_depth_m)]
    if valid.size == 0:
        return 0.0
    return float(np.median(valid))


def deproject(rs_module, intrinsics, cx: int, cy: int, depth_m: float) -> tuple[float, float, float]:
    if depth_m <= 0:
        return 0.0, 0.0, 0.0
    x, y, z = rs_module.rs2_deproject_pixel_to_point(intrinsics, [float(cx), float(cy)], depth_m)
    return float(x), float(y), float(z)


def annotate_detection(
    image: np.ndarray,
    xyxy: np.ndarray,
    cls_id: int,
    conf: float,
    name: str,
    center: tuple[int, int],
    depth_m: float,
    point_xyz: tuple[float, float, float],
) -> None:
    x1, y1, x2, y2 = [int(v) for v in xyxy]
    cx, cy = center
    color = (40, 220, 80) if depth_m > 0 else (0, 165, 255)
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
    cv2.circle(image, (cx, cy), 4, (0, 0, 255), -1)

    if depth_m > 0:
        label = f"{name} {conf:.2f} z={depth_m:.3f}m x={point_xyz[0]:.3f} y={point_xyz[1]:.3f}"
    else:
        label = f"{name} {conf:.2f} z=invalid"
    y_text = max(20, y1 - 8)
    cv2.putText(image, label, (x1, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(image, label, (x1, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 1, cv2.LINE_AA)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, args.color_width, args.color_height, rs.format.bgr8, args.color_fps)
    config.enable_stream(rs.stream.depth, args.depth_width, args.depth_height, rs.format.z16, args.depth_fps)
    align = rs.align(rs.stream.color)

    try:
        profile = pipeline.start(config)
    except RuntimeError as exc:
        raise SystemExit(
            "Failed to start RealSense stream. Use 640x480@30 first, or run "
            "`python3 scripts/test_realsense_sdk_stream.py --list-profiles`.\n"
            f"RealSense error: {exc}"
        ) from exc

    depth_scale = float(profile.get_device().first_depth_sensor().get_depth_scale())
    color_intrinsics = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
    model = YOLO(str(args.model))

    csv_file = None
    writer = None
    if args.save_csv:
        csv_path = args.output_dir / "detections.csv"
        csv_file = csv_path.open("w", newline="")
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "frame_index",
                "timestamp_ms",
                "class_id",
                "class_name",
                "confidence",
                "x1",
                "y1",
                "x2",
                "y2",
                "center_x",
                "center_y",
                "depth_m",
                "camera_x_m",
                "camera_y_m",
                "camera_z_m",
            ],
        )
        writer.writeheader()

    video_writer = None
    if args.save_video:
        output_video = args.output_dir / "realsense_yolo.mp4"
        video_writer = cv2.VideoWriter(
            str(output_video),
            cv2.VideoWriter_fourcc(*"mp4v"),
            float(args.color_fps),
            (args.color_width, args.color_height),
        )

    frame_count = 0
    timeout_count = 0
    start_time = time.monotonic()
    print(
        "Started RealSense YOLO:",
        f"color={args.color_width}x{args.color_height}@{args.color_fps}",
        f"depth={args.depth_width}x{args.depth_height}@{args.depth_fps}",
        f"depth_scale={depth_scale:.6f}",
    )

    try:
        while args.frames <= 0 or frame_count < args.frames:
            try:
                frames = pipeline.wait_for_frames(timeout_ms=args.frame_timeout_ms)
                timeout_count = 0
            except RuntimeError as exc:
                timeout_count += 1
                print(f"Frame wait timeout {timeout_count}/{args.max_timeouts}: {exc}")
                if timeout_count >= args.max_timeouts:
                    break
                continue

            frames = align.process(frames)
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()
            if not color_frame or not depth_frame:
                continue

            color = np.asanyarray(color_frame.get_data())
            depth = np.asanyarray(depth_frame.get_data())
            color = flip_frame(color, args.flip)
            depth = flip_frame(depth, args.flip)

            result = model.predict(
                source=color,
                conf=args.conf,
                iou=args.iou,
                imgsz=args.imgsz,
                device=args.device,
                verbose=False,
            )[0]
            annotated = color.copy()

            boxes = result.boxes
            detection_count = 0
            if boxes is not None:
                for box in boxes:
                    xyxy = box.xyxy[0].detach().cpu().numpy()
                    cls_id = int(box.cls[0].item())
                    conf = float(box.conf[0].item())
                    x1, y1, x2, y2 = xyxy
                    cx = int(round((x1 + x2) * 0.5))
                    cy = int(round((y1 + y2) * 0.5))
                    cx = min(max(cx, 0), depth.shape[1] - 1)
                    cy = min(max(cy, 0), depth.shape[0] - 1)
                    depth_m = robust_depth_m(depth, cx, cy, depth_scale, args.depth_window, args.min_depth_m, args.max_depth_m)
                    point_xyz = deproject(rs, color_intrinsics, cx, cy, depth_m)
                    class_name = result.names.get(cls_id, str(cls_id))
                    annotate_detection(annotated, xyxy, cls_id, conf, class_name, (cx, cy), depth_m, point_xyz)
                    detection_count += 1

                    if writer is not None:
                        writer.writerow(
                            {
                                "frame_index": frame_count,
                                "timestamp_ms": f"{frames.get_timestamp():.3f}",
                                "class_id": cls_id,
                                "class_name": class_name,
                                "confidence": f"{conf:.6f}",
                                "x1": f"{x1:.2f}",
                                "y1": f"{y1:.2f}",
                                "x2": f"{x2:.2f}",
                                "y2": f"{y2:.2f}",
                                "center_x": cx,
                                "center_y": cy,
                                "depth_m": f"{depth_m:.6f}",
                                "camera_x_m": f"{point_xyz[0]:.6f}",
                                "camera_y_m": f"{point_xyz[1]:.6f}",
                                "camera_z_m": f"{point_xyz[2]:.6f}",
                            }
                        )

            elapsed = max(time.monotonic() - start_time, 1e-6)
            status = f"frame={frame_count} detections={detection_count} fps={frame_count / elapsed:.1f}"
            cv2.putText(annotated, status, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(annotated, status, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1, cv2.LINE_AA)

            if video_writer is not None:
                video_writer.write(annotated)

            if not args.no_preview:
                cv2.imshow("RealSense YOLO RGB-D", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            frame_count += 1
    except KeyboardInterrupt:
        print("\nStopped by Ctrl-C.")
    finally:
        pipeline.stop()
        if video_writer is not None:
            video_writer.release()
        if csv_file is not None:
            csv_file.close()
        if not args.no_preview:
            cv2.destroyAllWindows()

    elapsed = max(time.monotonic() - start_time, 1e-6)
    print(f"Frames: {frame_count}, measured_fps={frame_count / elapsed:.2f}")
    print(f"Output dir: {args.output_dir}")


if __name__ == "__main__":
    main()

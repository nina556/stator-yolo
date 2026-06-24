#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Read synchronized RealSense color/depth frames with pyrealsense2."
    )
    parser.add_argument("--width", type=int, default=640, help="Legacy shortcut used for both color/depth width.")
    parser.add_argument("--height", type=int, default=480, help="Legacy shortcut used for both color/depth height.")
    parser.add_argument("--fps", type=int, default=30, help="Legacy shortcut used for both color/depth FPS.")
    parser.add_argument("--color-width", type=int, default=None)
    parser.add_argument("--color-height", type=int, default=None)
    parser.add_argument("--color-fps", type=int, default=None)
    parser.add_argument("--depth-width", type=int, default=None)
    parser.add_argument("--depth-height", type=int, default=None)
    parser.add_argument("--depth-fps", type=int, default=None)
    parser.add_argument("--frames", type=int, default=0, help="0 means run until q/Ctrl-C.")
    parser.add_argument("--frame-timeout-ms", type=int, default=5000)
    parser.add_argument("--max-timeouts", type=int, default=3)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/realsense_sdk_probe"))
    parser.add_argument("--save-every", type=int, default=30, help="Save one RGB-D pair every N frames. 0 disables saving.")
    parser.add_argument("--no-preview", action="store_true", help="Disable cv2.imshow preview.")
    parser.add_argument("--no-align", action="store_true", help="Do not align depth to color.")
    parser.add_argument("--list-profiles", action="store_true", help="List SDK stream profiles and exit.")
    parser.add_argument(
        "--flip",
        choices=["none", "vertical", "horizontal", "both"],
        default="none",
        help="Flip color/depth frames after capture.",
    )
    return parser.parse_args()


def import_realsense():
    try:
        import pyrealsense2 as rs
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Missing dependency: pyrealsense2.\n"
            "This camera is detected as Intel RealSense D455, so install librealsense "
            "and its Python binding first. On Jetson, prefer the vendor/JetPack-compatible "
            "librealsense package instead of a random pip wheel."
        ) from exc
    return rs


def import_runtime_dependencies() -> None:
    global cv2
    global np

    import cv2 as cv2_module
    import numpy as np_module

    cv2 = cv2_module
    np = np_module


def selected_profiles(args: argparse.Namespace) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    color = (
        args.color_width if args.color_width is not None else args.width,
        args.color_height if args.color_height is not None else args.height,
        args.color_fps if args.color_fps is not None else args.fps,
    )
    depth = (
        args.depth_width if args.depth_width is not None else args.width,
        args.depth_height if args.depth_height is not None else args.height,
        args.depth_fps if args.depth_fps is not None else args.fps,
    )
    return color, depth


def list_profiles(rs) -> None:
    context = rs.context()
    devices = context.query_devices()
    if len(devices) == 0:
        print("No RealSense device found.")
        return

    for device in devices:
        name = device.get_info(rs.camera_info.name) if device.supports(rs.camera_info.name) else "unknown"
        serial = device.get_info(rs.camera_info.serial_number) if device.supports(rs.camera_info.serial_number) else "unknown"
        usb = device.get_info(rs.camera_info.usb_type_descriptor) if device.supports(rs.camera_info.usb_type_descriptor) else "unknown"
        print(f"Device: {name}, serial={serial}, usb={usb}")
        for sensor in device.query_sensors():
            print(f"\nSensor: {sensor.get_info(rs.camera_info.name)}")
            print("  stream    resolution    format    fps")
            for profile in sensor.get_stream_profiles():
                video_profile = profile.as_video_stream_profile()
                stream_name = str(profile.stream_type()).replace("stream.", "")
                fmt = str(profile.format()).replace("format.", "")
                print(
                    f"  {stream_name:<9} "
                    f"{video_profile.width()}x{video_profile.height():<7} "
                    f"{fmt:<9} "
                    f"{profile.fps()}"
                )


def print_profile_hint(color: tuple[int, int, int], depth: tuple[int, int, int]) -> None:
    print("\nRequested profile could not be started.")
    print(f"  color: {color[0]}x{color[1]}@{color[2]} BGR8")
    print(f"  depth: {depth[0]}x{depth[1]}@{depth[2]} Z16")
    print("\nKnown working commands from this D455 profile list:")
    print("  30 FPS RGB-D:")
    print(
        "    python3 scripts/test_realsense_sdk_stream.py "
        "--color-width 640 --color-height 480 --color-fps 30 "
        "--depth-width 640 --depth-height 480 --depth-fps 30 "
        "--frames 300 --save-every 30 --no-preview"
    )
    print("  60 FPS low-resolution RGB-D:")
    print(
        "    python3 scripts/test_realsense_sdk_stream.py "
        "--color-width 424 --color-height 240 --color-fps 60 "
        "--depth-width 480 --depth-height 270 --depth-fps 60 "
        "--frames 300 --save-every 30 --no-preview"
    )
    print("  Higher color resolution may need a USB3 connection and should be tested separately.")
    print("\nUse --list-profiles to print all SDK profiles.")


def flip_frame(frame: np.ndarray, mode: str) -> np.ndarray:
    if mode == "vertical":
        return cv2.flip(frame, 0)
    if mode == "horizontal":
        return cv2.flip(frame, 1)
    if mode == "both":
        return cv2.flip(frame, -1)
    return frame


def depth_preview(depth_image: np.ndarray) -> np.ndarray:
    nonzero = depth_image[depth_image > 0]
    if nonzero.size:
        low, high = np.percentile(nonzero, [2, 98])
        if high <= low:
            high = low + 1
        scaled = np.clip((depth_image.astype(np.float32) - low) * 255.0 / (high - low), 0, 255)
    else:
        scaled = np.zeros_like(depth_image, dtype=np.float32)
    return cv2.applyColorMap(scaled.astype(np.uint8), cv2.COLORMAP_JET)


def write_sample(output_dir: Path, index: int, color: np.ndarray, depth: np.ndarray, preview: np.ndarray) -> tuple[Path, Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    color_path = output_dir / f"frame_{index:06d}_color.jpg"
    depth_path = output_dir / f"frame_{index:06d}_depth_z16.png"
    preview_path = output_dir / f"frame_{index:06d}_depth_preview.png"
    cv2.imwrite(str(color_path), color)
    cv2.imwrite(str(depth_path), depth)
    cv2.imwrite(str(preview_path), preview)
    return color_path, depth_path, preview_path


def print_device_info(rs, pipeline_profile) -> float:
    device = pipeline_profile.get_device()
    print("RealSense device:")
    for label, key in (
        ("name", rs.camera_info.name),
        ("serial_number", rs.camera_info.serial_number),
        ("firmware_version", rs.camera_info.firmware_version),
        ("product_line", rs.camera_info.product_line),
        ("usb_type", rs.camera_info.usb_type_descriptor),
    ):
        if device.supports(key):
            print(f"  {label}: {device.get_info(key)}")

    depth_sensor = device.first_depth_sensor()
    depth_scale = float(depth_sensor.get_depth_scale())
    print(f"  depth_scale_m_per_unit: {depth_scale:.8f}")

    color_stream = pipeline_profile.get_stream(rs.stream.color).as_video_stream_profile()
    depth_stream = pipeline_profile.get_stream(rs.stream.depth).as_video_stream_profile()
    print("Color intrinsics:", color_stream.get_intrinsics())
    print("Depth intrinsics:", depth_stream.get_intrinsics())
    return depth_scale


def main() -> None:
    args = parse_args()
    rs = import_realsense()

    if args.list_profiles:
        list_profiles(rs)
        return

    import_runtime_dependencies()
    color_profile, depth_profile = selected_profiles(args)

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, color_profile[0], color_profile[1], rs.format.bgr8, color_profile[2])
    config.enable_stream(rs.stream.depth, depth_profile[0], depth_profile[1], rs.format.z16, depth_profile[2])
    align = None if args.no_align else rs.align(rs.stream.color)

    manifest_path = args.output_dir / "manifest.csv"
    args.output_dir.mkdir(parents=True, exist_ok=True)

    try:
        profile = pipeline.start(config)
    except RuntimeError as exc:
        print_profile_hint(color_profile, depth_profile)
        raise SystemExit(f"\nRealSense pipeline.start failed: {exc}") from exc
    depth_scale = print_device_info(rs, profile)
    print(
        "Streaming:",
        f"color={color_profile[0]}x{color_profile[1]}@{color_profile[2]}",
        f"depth={depth_profile[0]}x{depth_profile[1]}@{depth_profile[2]}",
        f"align_depth_to_color={not args.no_align}",
        f"preview={not args.no_preview}",
    )
    print("Press q in the preview window to stop.")

    start_time = time.monotonic()
    saved_count = 0
    frame_count = 0
    manifest_file = manifest_path.open("w", newline="")
    writer = csv.DictWriter(
        manifest_file,
        fieldnames=[
            "frame_index",
            "timestamp_ms",
            "color_path",
            "depth_path",
            "depth_preview_path",
            "center_depth_m",
            "min_depth_m",
            "max_depth_m",
        ],
    )
    writer.writeheader()

    try:
        timeout_count = 0
        while args.frames <= 0 or frame_count < args.frames:
            try:
                frames = pipeline.wait_for_frames(timeout_ms=args.frame_timeout_ms)
                timeout_count = 0
            except RuntimeError as exc:
                timeout_count += 1
                print(f"Frame wait timeout {timeout_count}/{args.max_timeouts}: {exc}")
                if timeout_count >= args.max_timeouts:
                    print("Stopping because no frames arrived. Try a lower resolution/FPS or another USB port/cable.")
                    break
                continue
            if align is not None:
                frames = align.process(frames)

            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()
            if not color_frame or not depth_frame:
                continue

            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = flip_frame(color_image, args.flip)
            depth_image = flip_frame(depth_image, args.flip)
            preview = depth_preview(depth_image)

            center_y = depth_image.shape[0] // 2
            center_x = depth_image.shape[1] // 2
            center_depth_m = float(depth_image[center_y, center_x]) * depth_scale
            valid_depth = depth_image[depth_image > 0]
            min_depth_m = float(valid_depth.min()) * depth_scale if valid_depth.size else 0.0
            max_depth_m = float(valid_depth.max()) * depth_scale if valid_depth.size else 0.0

            color_path = depth_path = preview_path = ""
            if args.save_every > 0 and frame_count % args.save_every == 0:
                color_path, depth_path, preview_path = write_sample(args.output_dir, frame_count, color_image, depth_image, preview)
                saved_count += 1

            writer.writerow(
                {
                    "frame_index": frame_count,
                    "timestamp_ms": f"{frames.get_timestamp():.3f}",
                    "color_path": str(color_path),
                    "depth_path": str(depth_path),
                    "depth_preview_path": str(preview_path),
                    "center_depth_m": f"{center_depth_m:.6f}",
                    "min_depth_m": f"{min_depth_m:.6f}",
                    "max_depth_m": f"{max_depth_m:.6f}",
                }
            )

            if not args.no_preview:
                combined = np.hstack((color_image, preview))
                cv2.imshow("RealSense RGB | Depth", combined)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            frame_count += 1
    except KeyboardInterrupt:
        print("\nStopped by Ctrl-C.")
    finally:
        elapsed = max(time.monotonic() - start_time, 1e-6)
        manifest_file.close()
        pipeline.stop()
        if not args.no_preview:
            cv2.destroyAllWindows()

    print(f"Frames: {frame_count}, measured_fps={frame_count / elapsed:.2f}")
    print(f"Saved RGB-D pairs: {saved_count}")
    print(f"Output: {args.output_dir}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()

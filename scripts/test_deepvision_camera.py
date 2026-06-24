#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np


SDK_MODULES = [
    "pyrealsense2",
    "pyorbbecsdk",
    "openni",
    "openni2",
    "depthai",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Probe a DeepVision/3D camera through SDK imports and V4L2/OpenCV."
    )
    parser.add_argument(
        "--devices",
        nargs="*",
        default=None,
        help="Video devices to test, e.g. /dev/video8. Defaults to all /dev/video*.",
    )
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--frames", type=int, default=30)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/deepvision_probe"))
    parser.add_argument("--list-only", action="store_true")
    parser.add_argument(
        "--fourcc",
        default=None,
        help="Optional V4L2 fourcc to request, e.g. YUYV, MJPG, Z16.",
    )
    parser.add_argument(
        "--raw-v4l2",
        action="store_true",
        help="Use v4l2-ctl raw streaming instead of OpenCV. Useful for Z16 depth.",
    )
    parser.add_argument(
        "--no-convert-rgb",
        action="store_true",
        help="Disable OpenCV RGB conversion. Useful when probing raw depth formats.",
    )
    return parser.parse_args()


def run_command(command: list[str]) -> tuple[int, str]:
    try:
        process = subprocess.run(
            command,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
    except FileNotFoundError:
        return 127, f"command not found: {command[0]}"
    return process.returncode, process.stdout.strip()


def normalize_fourcc(value: str | None) -> str | None:
    if value is None:
        return None
    if value.upper() == "Z16":
        return "Z16 "
    return value


def print_sdk_status() -> None:
    print("SDK import status:")
    for module_name in SDK_MODULES:
        try:
            __import__(module_name)
        except Exception as exc:
            print(f"  {module_name}: missing ({exc.__class__.__name__})")
        else:
            print(f"  {module_name}: available")


def list_v4l2_devices() -> None:
    print("\nV4L2 devices:")
    code, output = run_command(["v4l2-ctl", "--list-devices"])
    if output:
        print(output)
    if code != 0:
        print(f"v4l2-ctl --list-devices exited with {code}")


def list_device_formats(device: str) -> None:
    print(f"\nFormats for {device}:")
    code, output = run_command(["v4l2-ctl", "-d", device, "--list-formats-ext"])
    if output:
        print(output)
    if code != 0:
        print(f"v4l2-ctl format query exited with {code}")


def discover_video_devices() -> list[str]:
    return [str(path) for path in sorted(Path("/dev").glob("video*"))]


def frame_stats(frame: np.ndarray) -> str:
    if frame is None:
        return "none"
    return (
        f"shape={frame.shape} dtype={frame.dtype} "
        f"min={int(np.min(frame))} max={int(np.max(frame))}"
    )


def save_frame(output_dir: Path, device: str, frame: np.ndarray, suffix: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    device_name = Path(device).name
    output_path = output_dir / f"{device_name}_{suffix}.png"
    if frame.ndim == 2:
        image = frame
    elif frame.ndim == 3 and frame.shape[2] == 1:
        image = frame[:, :, 0]
    else:
        image = frame
    ok = cv2.imwrite(str(output_path), image)
    if not ok:
        raise RuntimeError(f"failed to write {output_path}")
    return output_path


def save_z16_outputs(output_dir: Path, device: str, frame: np.ndarray) -> None:
    device_name = Path(device).name
    npy_path = output_dir / f"{device_name}_depth_z16.npy"
    png_path = output_dir / f"{device_name}_depth_z16.png"
    preview_path = output_dir / f"{device_name}_depth_preview.png"

    np.save(npy_path, frame)
    cv2.imwrite(str(png_path), frame)

    nonzero = frame[frame > 0]
    if nonzero.size:
        low, high = np.percentile(nonzero, [2, 98])
        if high <= low:
            high = low + 1
        scaled = np.clip((frame.astype(np.float32) - low) * 255.0 / (high - low), 0, 255)
    else:
        scaled = np.zeros_like(frame, dtype=np.float32)
    preview = cv2.applyColorMap(scaled.astype(np.uint8), cv2.COLORMAP_JET)
    cv2.imwrite(str(preview_path), preview)

    print(f"  saved depth npy: {npy_path}")
    print(f"  saved depth png: {png_path}")
    print(f"  saved depth preview: {preview_path}")


def probe_device_raw_v4l2(args: argparse.Namespace, device: str) -> bool:
    fourcc = normalize_fourcc(args.fourcc)
    if not fourcc:
        print(f"\nRaw V4L2 probe for {device}: skipped, --fourcc is required")
        return False

    args.output_dir.mkdir(parents=True, exist_ok=True)
    device_name = Path(device).name
    raw_path = args.output_dir / f"{device_name}_{fourcc.strip().lower() or 'raw'}.raw"
    command = [
        "v4l2-ctl",
        "-d",
        device,
        f"--set-fmt-video=width={args.width},height={args.height},pixelformat={fourcc}",
        "--stream-mmap",
        f"--stream-count={args.frames}",
        f"--stream-to={raw_path}",
    ]

    print(f"\nRaw V4L2 probe for {device}")
    print(
        "  requested:",
        f"{args.width}x{args.height}@driver-default",
        f"fourcc={fourcc!r}",
    )
    code, output = run_command(command)
    if output:
        print(f"  v4l2-ctl: {output}")
    if code != 0:
        print(f"  result: failed, exit code {code}")
        return False
    if not raw_path.exists() or raw_path.stat().st_size == 0:
        print("  result: no raw data written")
        return False

    raw_size = raw_path.stat().st_size
    print(f"  raw saved: {raw_path} ({raw_size} bytes)")

    if fourcc == "Z16 ":
        frame_bytes = args.width * args.height * 2
        if raw_size < frame_bytes:
            print(f"  result: raw file smaller than one Z16 frame ({frame_bytes} bytes)")
            return False
        frame = np.fromfile(raw_path, dtype=np.uint16, count=args.width * args.height)
        frame = frame.reshape((args.height, args.width))
        print(f"  first depth frame: {frame_stats(frame)}")
        save_z16_outputs(args.output_dir, device, frame)

    return True


def probe_device(args: argparse.Namespace, device: str) -> bool:
    print(f"\nProbing {device}")
    capture = cv2.VideoCapture(device, cv2.CAP_V4L2)
    if not capture.isOpened():
        print("  open: failed")
        return False

    capture.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    capture.set(cv2.CAP_PROP_FPS, args.fps)
    if args.fourcc:
        fourcc_text = normalize_fourcc(args.fourcc)
        fourcc = cv2.VideoWriter_fourcc(*fourcc_text)
        capture.set(cv2.CAP_PROP_FOURCC, fourcc)
    if args.no_convert_rgb:
        capture.set(cv2.CAP_PROP_CONVERT_RGB, 0)

    print(
        "  requested:",
        f"{args.width}x{args.height}@{args.fps}",
        f"fourcc={args.fourcc or 'default'}",
        f"convert_rgb={not args.no_convert_rgb}",
    )
    print(
        "  opened:",
        f"{int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))}x"
        f"{int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))}@"
        f"{capture.get(cv2.CAP_PROP_FPS):.1f}",
    )

    first_frame = None
    ok_count = 0
    start_time = time.monotonic()
    for _index in range(args.frames):
        ok, frame = capture.read()
        if not ok:
            continue
        ok_count += 1
        if first_frame is None:
            first_frame = frame.copy()
    elapsed = max(time.monotonic() - start_time, 1e-6)
    capture.release()

    print(f"  read: {ok_count}/{args.frames} frames, measured_fps={ok_count / elapsed:.2f}")
    if first_frame is None:
        print("  result: no frame")
        return False

    print(f"  first frame: {frame_stats(first_frame)}")
    output_path = save_frame(args.output_dir, device, first_frame, "frame")
    print(f"  saved: {output_path}")
    return True


def main() -> None:
    args = parse_args()
    args.fourcc = normalize_fourcc(args.fourcc)
    print_sdk_status()
    list_v4l2_devices()

    devices = args.devices if args.devices else discover_video_devices()
    if not devices:
        raise SystemExit("No /dev/video* devices found")

    for device in devices:
        list_device_formats(device)

    if args.list_only:
        return

    print("\nOpenCV capture probe:")
    successes = []
    for device in devices:
        if args.raw_v4l2:
            success = probe_device_raw_v4l2(args, device)
        else:
            success = probe_device(args, device)
        if success:
            successes.append(device)

    print("\nSummary:")
    if successes:
        print("  readable devices:", ", ".join(successes))
        print(f"  sample frames saved in: {args.output_dir}")
    else:
        print("  no readable OpenCV/V4L2 device found")


if __name__ == "__main__":
    main()

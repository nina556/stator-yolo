from __future__ import annotations

import importlib.util
import platform
import subprocess
import sys
from pathlib import Path

from .paths import ensure_project_dirs, repo_root


def command_output(command: list[str]) -> str:
    try:
        process = subprocess.run(
            command,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
    except FileNotFoundError:
        return f"missing command: {command[0]}"
    return process.stdout.strip()


def module_status(module_name: str) -> str:
    return "ok" if importlib.util.find_spec(module_name) else "missing"


def main() -> None:
    ensure_project_dirs()
    root = repo_root()
    print("Stator YOLO environment")
    print(f"  root: {root}")
    print(f"  python: {sys.version.split()[0]}")
    print(f"  platform: {platform.platform()}")
    print(f"  machine: {platform.machine()}")
    print()
    print("Python modules")
    for module_name in ("cv2", "numpy", "PIL", "ultralytics", "yaml", "tqdm", "albumentations", "pyrealsense2"):
        print(f"  {module_name}: {module_status(module_name)}")
    print()
    print("Project files")
    for relative in (
        "run_web.py",
        "data/dataset.yaml",
        "runs/stator_yolov8/weights/best.pt",
        "runs/stator_yolov8/weights/best.engine",
    ):
        path = root / relative
        state = "ok" if path.exists() else "missing"
        print(f"  {relative}: {state}")
    print()
    print("RealSense/V4L2")
    print("  rs-enumerate-devices:")
    for line in command_output(["rs-enumerate-devices"]).splitlines()[:18]:
        print(f"    {line}")
    print("  v4l2 devices:")
    for line in command_output(["v4l2-ctl", "--list-devices"]).splitlines()[:30]:
        print(f"    {line}")


if __name__ == "__main__":
    main()

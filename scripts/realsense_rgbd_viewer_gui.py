#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import queue
import threading
import time
import tkinter as tk
from dataclasses import dataclass
from pathlib import Path
from tkinter import ttk

import cv2
import numpy as np
import pyrealsense2 as rs
from PIL import Image, ImageTk


@dataclass(frozen=True)
class StreamConfig:
    color_width: int
    color_height: int
    color_fps: int
    depth_width: int
    depth_height: int
    depth_fps: int
    flip: str
    point_step: int
    max_depth_m: float
    min_depth_m: float
    view_yaw_deg: float
    view_pitch_deg: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RealSense RGB-D GUI viewer.")
    parser.add_argument("--color-width", type=int, default=640)
    parser.add_argument("--color-height", type=int, default=480)
    parser.add_argument("--color-fps", type=int, default=30)
    parser.add_argument("--depth-width", type=int, default=640)
    parser.add_argument("--depth-height", type=int, default=480)
    parser.add_argument("--depth-fps", type=int, default=30)
    parser.add_argument("--point-step", type=int, default=4)
    parser.add_argument("--min-depth-m", type=float, default=0.15)
    parser.add_argument("--max-depth-m", type=float, default=4.0)
    parser.add_argument("--view-yaw-deg", type=float, default=-25.0)
    parser.add_argument("--view-pitch-deg", type=float, default=-18.0)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/realsense_rgbd_viewer"))
    parser.add_argument("--frame-timeout-ms", type=int, default=5000)
    parser.add_argument("--queue-size", type=int, default=2)
    parser.add_argument("--auto-start", action="store_true", help="Start streaming immediately.")
    parser.add_argument("--auto-close-sec", type=float, default=0.0, help="Close the GUI automatically after N seconds.")
    parser.add_argument(
        "--flip",
        choices=["none", "vertical", "horizontal", "both"],
        default="none",
    )
    return parser.parse_args()


def flip_frame(frame: np.ndarray, mode: str) -> np.ndarray:
    if mode == "vertical":
        return cv2.flip(frame, 0)
    if mode == "horizontal":
        return cv2.flip(frame, 1)
    if mode == "both":
        return cv2.flip(frame, -1)
    return frame


def resize_to_panel(image: np.ndarray, panel_size: tuple[int, int]) -> np.ndarray:
    panel_width, panel_height = panel_size
    height, width = image.shape[:2]
    scale = min(panel_width / width, panel_height / height)
    new_width = max(1, int(width * scale))
    new_height = max(1, int(height * scale))
    resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    canvas = np.full((panel_height, panel_width, 3), 18, dtype=np.uint8)
    x0 = (panel_width - new_width) // 2
    y0 = (panel_height - new_height) // 2
    canvas[y0 : y0 + new_height, x0 : x0 + new_width] = resized
    return canvas


def depth_colormap(depth: np.ndarray, min_depth_m: float, max_depth_m: float, depth_scale: float) -> np.ndarray:
    depth_m = depth.astype(np.float32) * depth_scale
    valid = (depth_m >= min_depth_m) & (depth_m <= max_depth_m)
    scaled = np.zeros(depth.shape, dtype=np.uint8)
    if np.any(valid):
        clipped = np.clip(depth_m, min_depth_m, max_depth_m)
        scaled = ((1.0 - (clipped - min_depth_m) / (max_depth_m - min_depth_m)) * 255.0).astype(np.uint8)
        scaled[~valid] = 0
    return cv2.applyColorMap(scaled, cv2.COLORMAP_TURBO)


def make_point_cloud_view(
    color: np.ndarray,
    depth: np.ndarray,
    intrinsics,
    depth_scale: float,
    config: StreamConfig,
    output_size: tuple[int, int],
) -> np.ndarray:
    out_width, out_height = output_size
    canvas = np.full((out_height, out_width, 3), 15, dtype=np.uint8)

    step = max(1, config.point_step)
    rows = np.arange(0, depth.shape[0], step)
    cols = np.arange(0, depth.shape[1], step)
    grid_x, grid_y = np.meshgrid(cols, rows)
    z = depth[grid_y, grid_x].astype(np.float32) * depth_scale
    valid = (z >= config.min_depth_m) & (z <= config.max_depth_m)
    if not np.any(valid):
        cv2.putText(canvas, "No valid depth points", (24, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (230, 230, 230), 2)
        return canvas

    x = (grid_x.astype(np.float32) - intrinsics.ppx) / intrinsics.fx * z
    y = (grid_y.astype(np.float32) - intrinsics.ppy) / intrinsics.fy * z
    points = np.stack((x[valid], -y[valid], z[valid]), axis=1)
    colors = color[grid_y[valid], grid_x[valid]]

    yaw = math.radians(config.view_yaw_deg)
    pitch = math.radians(config.view_pitch_deg)
    yaw_matrix = np.array(
        [
            [math.cos(yaw), 0.0, math.sin(yaw)],
            [0.0, 1.0, 0.0],
            [-math.sin(yaw), 0.0, math.cos(yaw)],
        ],
        dtype=np.float32,
    )
    pitch_matrix = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, math.cos(pitch), -math.sin(pitch)],
            [0.0, math.sin(pitch), math.cos(pitch)],
        ],
        dtype=np.float32,
    )
    rotated = points @ (yaw_matrix @ pitch_matrix).T

    distance = 2.2
    projected_z = rotated[:, 2] + distance
    in_front = projected_z > 0.05
    rotated = rotated[in_front]
    colors = colors[in_front]
    projected_z = projected_z[in_front]
    if rotated.size == 0:
        return canvas

    focal = min(out_width, out_height) * 0.78
    u = (rotated[:, 0] * focal / projected_z + out_width * 0.5).astype(np.int32)
    v = (-rotated[:, 1] * focal / projected_z + out_height * 0.56).astype(np.int32)
    inside = (u >= 0) & (u < out_width) & (v >= 0) & (v < out_height)
    u = u[inside]
    v = v[inside]
    colors = colors[inside]
    projected_z = projected_z[inside]

    order = np.argsort(projected_z)[::-1]
    for px, py, color_value in zip(u[order], v[order], colors[order]):
        cv2.circle(canvas, (int(px), int(py)), 1, tuple(int(c) for c in color_value), -1)

    cv2.line(canvas, (24, out_height - 32), (144, out_height - 32), (75, 75, 75), 2)
    cv2.putText(canvas, "3D depth point view", (24, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (235, 235, 235), 2)
    cv2.putText(
        canvas,
        f"points={len(u)}  range={config.min_depth_m:.2f}-{config.max_depth_m:.1f}m",
        (24, 64),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.54,
        (210, 210, 210),
        1,
    )
    return canvas


class RealSenseWorker(threading.Thread):
    def __init__(
        self,
        config: StreamConfig,
        output_queue: queue.Queue,
        stop_event: threading.Event,
        frame_timeout_ms: int,
    ) -> None:
        super().__init__(daemon=True)
        self.config = config
        self.output_queue = output_queue
        self.stop_event = stop_event
        self.frame_timeout_ms = frame_timeout_ms

    def run(self) -> None:
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(
            rs.stream.color,
            self.config.color_width,
            self.config.color_height,
            rs.format.bgr8,
            self.config.color_fps,
        )
        config.enable_stream(
            rs.stream.depth,
            self.config.depth_width,
            self.config.depth_height,
            rs.format.z16,
            self.config.depth_fps,
        )
        align = rs.align(rs.stream.color)

        try:
            started = False
            profile = pipeline.start(config)
            started = True
            depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
            color_stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
            intrinsics = color_stream.get_intrinsics()
            self._put(("status", f"Started RealSense RGB-D, depth_scale={depth_scale:.6f}"))

            while not self.stop_event.is_set():
                try:
                    frames = pipeline.wait_for_frames(timeout_ms=self.frame_timeout_ms)
                except RuntimeError as exc:
                    self._put(("status", f"Frame timeout: {exc}"))
                    continue

                frames = align.process(frames)
                color_frame = frames.get_color_frame()
                depth_frame = frames.get_depth_frame()
                if not color_frame or not depth_frame:
                    continue

                color = np.asanyarray(color_frame.get_data())
                depth = np.asanyarray(depth_frame.get_data())
                color = flip_frame(color, self.config.flip)
                depth = flip_frame(depth, self.config.flip)
                timestamp_ms = frames.get_timestamp()
                self._put(("frame", color, depth, intrinsics, depth_scale, timestamp_ms))
        except Exception as exc:
            self._put(("error", str(exc)))
        finally:
            if "started" in locals() and started:
                pipeline.stop()
            self._put(("status", "RealSense stream stopped"))

    def _put(self, item: tuple) -> None:
        while not self.stop_event.is_set():
            try:
                self.output_queue.put_nowait(item)
                return
            except queue.Full:
                try:
                    self.output_queue.get_nowait()
                except queue.Empty:
                    pass


class RealSenseRgbdViewer:
    def __init__(self, root: tk.Tk, args: argparse.Namespace) -> None:
        self.root = root
        self.args = args
        self.root.title("RealSense RGB-D Viewer")
        self.root.geometry("1320x760")
        self.root.minsize(980, 620)

        self.frame_queue: queue.Queue = queue.Queue(maxsize=args.queue_size)
        self.stop_event = threading.Event()
        self.worker: RealSenseWorker | None = None
        self.last_color: np.ndarray | None = None
        self.last_depth: np.ndarray | None = None
        self.last_depth_preview: np.ndarray | None = None
        self.left_photo: ImageTk.PhotoImage | None = None
        self.right_photo: ImageTk.PhotoImage | None = None
        self.frame_count = 0
        self.start_time = time.monotonic()

        self.status_var = tk.StringVar(value="Stopped")
        self.fps_var = tk.StringVar(value="FPS: --")
        self.depth_var = tk.StringVar(value="Depth: --")
        self.mode_var = tk.StringVar(value="3d")
        self.flip_var = tk.StringVar(value=args.flip)
        self.point_step_var = tk.IntVar(value=args.point_step)
        self.max_depth_var = tk.DoubleVar(value=args.max_depth_m)
        self.min_depth_var = tk.DoubleVar(value=args.min_depth_m)
        self.yaw_var = tk.DoubleVar(value=args.view_yaw_deg)
        self.pitch_var = tk.DoubleVar(value=args.view_pitch_deg)

        self._build_ui()
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.root.after(50, self.process_queue)
        if args.auto_start:
            self.root.after(200, self.start_stream)
        if args.auto_close_sec > 0:
            self.root.after(int(args.auto_close_sec * 1000), self.close)

    def _build_ui(self) -> None:
        style = ttk.Style()
        style.configure("Title.TLabel", font=("Arial", 15, "bold"))
        style.configure("Metric.TLabel", font=("Arial", 10))

        top = ttk.Frame(self.root, padding=(12, 10))
        top.pack(fill=tk.X)
        ttk.Label(top, text="RealSense RGB-D Viewer", style="Title.TLabel").pack(side=tk.LEFT)
        ttk.Label(top, textvariable=self.status_var).pack(side=tk.RIGHT)

        controls = ttk.Frame(self.root, padding=(12, 0, 12, 8))
        controls.pack(fill=tk.X)
        ttk.Button(controls, text="Start", command=self.start_stream).pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(controls, text="Stop", command=self.stop_stream).pack(side=tk.LEFT, padx=6)
        ttk.Button(controls, text="Save Snapshot", command=self.save_snapshot).pack(side=tk.LEFT, padx=6)
        ttk.Label(controls, textvariable=self.fps_var, style="Metric.TLabel").pack(side=tk.LEFT, padx=(18, 6))
        ttk.Label(controls, textvariable=self.depth_var, style="Metric.TLabel").pack(side=tk.LEFT, padx=6)

        ttk.Label(controls, text="Right View").pack(side=tk.LEFT, padx=(20, 4))
        ttk.Combobox(
            controls,
            textvariable=self.mode_var,
            values=("3d", "depth"),
            width=8,
            state="readonly",
        ).pack(side=tk.LEFT)
        ttk.Label(controls, text="Flip").pack(side=tk.LEFT, padx=(14, 4))
        ttk.Combobox(
            controls,
            textvariable=self.flip_var,
            values=("none", "vertical", "horizontal", "both"),
            width=10,
            state="readonly",
        ).pack(side=tk.LEFT)

        sliders = ttk.Frame(self.root, padding=(12, 0, 12, 8))
        sliders.pack(fill=tk.X)
        self._add_slider(sliders, "Point Step", self.point_step_var, 1, 12, 0)
        self._add_slider(sliders, "Min Depth", self.min_depth_var, 0.05, 1.5, 1)
        self._add_slider(sliders, "Max Depth", self.max_depth_var, 0.5, 8.0, 2)
        self._add_slider(sliders, "Yaw", self.yaw_var, -90.0, 90.0, 3)
        self._add_slider(sliders, "Pitch", self.pitch_var, -70.0, 30.0, 4)

        body = ttk.Frame(self.root, padding=(12, 0, 12, 12))
        body.pack(fill=tk.BOTH, expand=True)
        body.columnconfigure(0, weight=1)
        body.columnconfigure(1, weight=1)
        body.rowconfigure(1, weight=1)

        ttk.Label(body, text="2D RGB Stream", style="Title.TLabel").grid(row=0, column=0, sticky=tk.W, pady=(0, 6))
        ttk.Label(body, text="3D Depth View", style="Title.TLabel").grid(row=0, column=1, sticky=tk.W, pady=(0, 6))

        self.left_label = ttk.Label(body)
        self.left_label.grid(row=1, column=0, sticky=tk.NSEW, padx=(0, 6))
        self.right_label = ttk.Label(body)
        self.right_label.grid(row=1, column=1, sticky=tk.NSEW, padx=(6, 0))

    def _add_slider(self, parent: ttk.Frame, label: str, variable: tk.Variable, start: float, end: float, column: int) -> None:
        frame = ttk.Frame(parent)
        frame.grid(row=0, column=column, sticky=tk.EW, padx=(0, 14))
        parent.columnconfigure(column, weight=1)
        ttk.Label(frame, text=label).pack(anchor=tk.W)
        ttk.Scale(frame, from_=start, to=end, variable=variable, orient=tk.HORIZONTAL).pack(fill=tk.X)

    def current_config(self) -> StreamConfig:
        return StreamConfig(
            color_width=self.args.color_width,
            color_height=self.args.color_height,
            color_fps=self.args.color_fps,
            depth_width=self.args.depth_width,
            depth_height=self.args.depth_height,
            depth_fps=self.args.depth_fps,
            flip=self.flip_var.get(),
            point_step=int(self.point_step_var.get()),
            min_depth_m=float(self.min_depth_var.get()),
            max_depth_m=float(self.max_depth_var.get()),
            view_yaw_deg=float(self.yaw_var.get()),
            view_pitch_deg=float(self.pitch_var.get()),
        )

    def start_stream(self) -> None:
        if self.worker and self.worker.is_alive():
            return
        self.stop_event.clear()
        self.frame_count = 0
        self.start_time = time.monotonic()
        self.worker = RealSenseWorker(
            self.current_config(),
            self.frame_queue,
            self.stop_event,
            self.args.frame_timeout_ms,
        )
        self.worker.start()
        self.status_var.set("Starting stream...")

    def stop_stream(self) -> None:
        self.stop_event.set()
        self.status_var.set("Stopping...")

    def process_queue(self) -> None:
        latest_frame = None
        try:
            while True:
                item = self.frame_queue.get_nowait()
                if item[0] == "frame":
                    latest_frame = item
                elif item[0] == "status":
                    self.status_var.set(item[1])
                elif item[0] == "error":
                    self.status_var.set(f"Error: {item[1]}")
        except queue.Empty:
            pass

        if latest_frame is not None:
            _, color, depth, intrinsics, depth_scale, timestamp_ms = latest_frame
            self.render_frame(color, depth, intrinsics, depth_scale, timestamp_ms)

        self.root.after(33, self.process_queue)

    def render_frame(self, color: np.ndarray, depth: np.ndarray, intrinsics, depth_scale: float, timestamp_ms: float) -> None:
        config = self.current_config()
        self.last_color = color.copy()
        self.last_depth = depth.copy()
        self.last_depth_preview = depth_colormap(depth, config.min_depth_m, config.max_depth_m, depth_scale)

        panel_width = max(320, self.left_label.winfo_width())
        panel_height = max(260, self.left_label.winfo_height())
        color_panel = resize_to_panel(color, (panel_width, panel_height))
        if self.mode_var.get() == "depth":
            right = self.last_depth_preview
        else:
            right = make_point_cloud_view(color, depth, intrinsics, depth_scale, config, (panel_width, panel_height))
        right_panel = resize_to_panel(right, (panel_width, panel_height))

        self.left_photo = self.to_photo(color_panel)
        self.right_photo = self.to_photo(right_panel)
        self.left_label.configure(image=self.left_photo)
        self.right_label.configure(image=self.right_photo)

        self.frame_count += 1
        elapsed = max(time.monotonic() - self.start_time, 1e-6)
        self.fps_var.set(f"FPS: {self.frame_count / elapsed:.1f}")

        valid = depth[depth > 0]
        if valid.size:
            center = depth[depth.shape[0] // 2, depth.shape[1] // 2] * depth_scale
            self.depth_var.set(f"Center: {center:.3f} m  Timestamp: {timestamp_ms:.0f} ms")

    def to_photo(self, bgr: np.ndarray) -> ImageTk.PhotoImage:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return ImageTk.PhotoImage(Image.fromarray(rgb))

    def save_snapshot(self) -> None:
        if self.last_color is None or self.last_depth is None or self.last_depth_preview is None:
            self.status_var.set("No frame to save")
            return
        self.args.output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        color_path = self.args.output_dir / f"{timestamp}_color.jpg"
        depth_path = self.args.output_dir / f"{timestamp}_depth_z16.png"
        preview_path = self.args.output_dir / f"{timestamp}_depth_preview.png"
        cv2.imwrite(str(color_path), self.last_color)
        cv2.imwrite(str(depth_path), self.last_depth)
        cv2.imwrite(str(preview_path), self.last_depth_preview)
        self.status_var.set(f"Saved snapshot: {color_path}")

    def close(self) -> None:
        self.stop_event.set()
        self.root.after(100, self.root.destroy)


def main() -> None:
    args = parse_args()
    root = tk.Tk()
    RealSenseRgbdViewer(root, args)
    root.mainloop()


if __name__ == "__main__":
    main()

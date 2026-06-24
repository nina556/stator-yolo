#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import queue
import threading
import time
import tkinter as tk
from dataclasses import dataclass
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import cv2
import numpy as np
import pyrealsense2 as rs
from PIL import Image, ImageTk


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RealSense YOLO RGB-D detection GUI.")
    parser.add_argument("--auto-start", action="store_true", help="Start detection after opening the GUI.")
    parser.add_argument("--auto-close-sec", type=float, default=0.0, help="Close the GUI automatically after N seconds.")
    return parser.parse_args()


def flip_frame(frame: np.ndarray, mode: str) -> np.ndarray:
    if mode == "vertical":
        return cv2.flip(frame, 0)
    if mode == "horizontal":
        return cv2.flip(frame, 1)
    if mode == "both":
        return cv2.flip(frame, -1)
    return frame


def robust_depth_m(
    depth: np.ndarray,
    cx: int,
    cy: int,
    depth_scale: float,
    window: int,
    min_depth_m: float,
    max_depth_m: float,
) -> float:
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


def resize_to_panel(image: np.ndarray, width: int, height: int) -> np.ndarray:
    src_h, src_w = image.shape[:2]
    scale = min(width / src_w, height / src_h)
    dst_w = max(1, int(src_w * scale))
    dst_h = max(1, int(src_h * scale))
    resized = cv2.resize(image, (dst_w, dst_h), interpolation=cv2.INTER_AREA)
    canvas = np.full((height, width, 3), 17, dtype=np.uint8)
    x0 = (width - dst_w) // 2
    y0 = (height - dst_h) // 2
    canvas[y0 : y0 + dst_h, x0 : x0 + dst_w] = resized
    return canvas


@dataclass(frozen=True)
class DetectionConfig:
    model_path: Path
    conf: float
    iou: float
    imgsz: int
    device: str
    color_width: int
    color_height: int
    color_fps: int
    depth_width: int
    depth_height: int
    depth_fps: int
    flip: str
    min_depth_m: float
    max_depth_m: float
    depth_window: int
    save_csv: bool
    save_video: bool
    output_dir: Path


class RealSenseYoloWorker(threading.Thread):
    def __init__(
        self,
        cfg: DetectionConfig,
        frame_queue: queue.Queue,
        log_queue: queue.Queue,
        stop_event: threading.Event,
    ) -> None:
        super().__init__(daemon=True)
        self.cfg = cfg
        self.frame_queue = frame_queue
        self.log_queue = log_queue
        self.stop_event = stop_event

    def log(self, message: str) -> None:
        self.log_queue.put(message)

    def put_frame(self, frame: np.ndarray) -> None:
        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
            except queue.Empty:
                break
        self.frame_queue.put(frame)

    def run(self) -> None:
        pipeline = rs.pipeline()
        csv_file = None
        writer = None
        video_writer = None
        started = False

        try:
            if not self.cfg.model_path.exists():
                raise RuntimeError(f"model not found: {self.cfg.model_path}")

            from ultralytics import YOLO

            self.log(f"Loading model: {self.cfg.model_path}")
            model = YOLO(str(self.cfg.model_path))

            rs_config = rs.config()
            rs_config.enable_stream(
                rs.stream.color,
                self.cfg.color_width,
                self.cfg.color_height,
                rs.format.bgr8,
                self.cfg.color_fps,
            )
            rs_config.enable_stream(
                rs.stream.depth,
                self.cfg.depth_width,
                self.cfg.depth_height,
                rs.format.z16,
                self.cfg.depth_fps,
            )
            align = rs.align(rs.stream.color)

            self.log(
                "Starting RealSense: "
                f"color={self.cfg.color_width}x{self.cfg.color_height}@{self.cfg.color_fps}, "
                f"depth={self.cfg.depth_width}x{self.cfg.depth_height}@{self.cfg.depth_fps}"
            )
            profile = pipeline.start(rs_config)
            started = True
            depth_scale = float(profile.get_device().first_depth_sensor().get_depth_scale())
            color_intrinsics = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
            self.log(f"Depth scale: {depth_scale:.6f} m/unit")

            self.cfg.output_dir.mkdir(parents=True, exist_ok=True)
            if self.cfg.save_csv:
                csv_file = (self.cfg.output_dir / "detections.csv").open("w", newline="")
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
                self.log(f"Writing CSV: {csv_file.name}")

            if self.cfg.save_video:
                video_path = self.cfg.output_dir / "realsense_yolo_gui.mp4"
                video_writer = cv2.VideoWriter(
                    str(video_path),
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    float(self.cfg.color_fps),
                    (self.cfg.color_width, self.cfg.color_height),
                )
                self.log(f"Writing video: {video_path}")

            frames_seen = 0
            start_time = time.monotonic()
            while not self.stop_event.is_set():
                frames = pipeline.wait_for_frames(timeout_ms=5000)
                frames = align.process(frames)
                color_frame = frames.get_color_frame()
                depth_frame = frames.get_depth_frame()
                if not color_frame or not depth_frame:
                    continue

                color = np.asanyarray(color_frame.get_data())
                depth = np.asanyarray(depth_frame.get_data())
                color = flip_frame(color, self.cfg.flip)
                depth = flip_frame(depth, self.cfg.flip)

                predict_kwargs = {
                    "source": color,
                    "conf": self.cfg.conf,
                    "iou": self.cfg.iou,
                    "imgsz": self.cfg.imgsz,
                    "verbose": False,
                }
                if self.cfg.device.strip():
                    predict_kwargs["device"] = self.cfg.device.strip()
                result = model.predict(**predict_kwargs)[0]

                annotated = color.copy()
                detections = 0
                if result.boxes is not None:
                    for box in result.boxes:
                        xyxy = box.xyxy[0].detach().cpu().numpy()
                        cls_id = int(box.cls[0].item())
                        confidence = float(box.conf[0].item())
                        class_name = result.names.get(cls_id, str(cls_id))
                        x1, y1, x2, y2 = xyxy
                        cx = int(round((x1 + x2) * 0.5))
                        cy = int(round((y1 + y2) * 0.5))
                        cx = min(max(cx, 0), depth.shape[1] - 1)
                        cy = min(max(cy, 0), depth.shape[0] - 1)
                        depth_m = robust_depth_m(
                            depth,
                            cx,
                            cy,
                            depth_scale,
                            self.cfg.depth_window,
                            self.cfg.min_depth_m,
                            self.cfg.max_depth_m,
                        )
                        if depth_m > 0:
                            point_xyz = rs.rs2_deproject_pixel_to_point(
                                color_intrinsics,
                                [float(cx), float(cy)],
                                depth_m,
                            )
                            point_xyz = tuple(float(v) for v in point_xyz)
                        else:
                            point_xyz = (0.0, 0.0, 0.0)

                        self.draw_detection(
                            annotated,
                            xyxy,
                            class_name,
                            confidence,
                            (cx, cy),
                            depth_m,
                            point_xyz,
                        )
                        detections += 1

                        if writer is not None:
                            writer.writerow(
                                {
                                    "frame_index": frames_seen,
                                    "timestamp_ms": f"{frames.get_timestamp():.3f}",
                                    "class_id": cls_id,
                                    "class_name": class_name,
                                    "confidence": f"{confidence:.6f}",
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

                frames_seen += 1
                elapsed = max(time.monotonic() - start_time, 1e-6)
                status = f"FPS {frames_seen / elapsed:.1f} | detections {detections}"
                cv2.putText(annotated, status, (14, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(annotated, status, (14, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 1, cv2.LINE_AA)

                if video_writer is not None:
                    video_writer.write(annotated)
                self.put_frame(annotated)

                if frames_seen % 30 == 0:
                    self.log(status)

            self.log("Detection stopped")
        except Exception as exc:
            self.log(f"ERROR: {exc}")
        finally:
            if started:
                pipeline.stop()
            if video_writer is not None:
                video_writer.release()
            if csv_file is not None:
                csv_file.close()

    @staticmethod
    def draw_detection(
        image: np.ndarray,
        xyxy: np.ndarray,
        class_name: str,
        confidence: float,
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
            label = f"{class_name} {confidence:.2f} z={depth_m:.3f}m x={point_xyz[0]:.3f} y={point_xyz[1]:.3f}"
        else:
            label = f"{class_name} {confidence:.2f} z=invalid"
        y_text = max(22, y1 - 8)
        cv2.putText(image, label, (x1, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.54, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(image, label, (x1, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.54, color, 1, cv2.LINE_AA)


class RealSenseYoloGui:
    def __init__(self, root: tk.Tk, args: argparse.Namespace) -> None:
        self.root = root
        self.args = args
        self.root.title("RealSense YOLO RGB-D Detection")
        self.root.geometry("1250x820")
        self.root.minsize(1050, 720)
        self.base_dir = repo_root()

        default_model = self.base_dir / "runs/stator_yolov8/weights/best.pt"
        self.model_var = tk.StringVar(value=str(default_model))
        self.output_dir_var = tk.StringVar(value=str(self.base_dir / "runs/realsense_yolo_gui"))
        self.conf_var = tk.DoubleVar(value=0.25)
        self.iou_var = tk.DoubleVar(value=0.7)
        self.imgsz_var = tk.IntVar(value=640)
        self.device_var = tk.StringVar(value="0")
        self.color_width_var = tk.IntVar(value=640)
        self.color_height_var = tk.IntVar(value=480)
        self.color_fps_var = tk.IntVar(value=30)
        self.depth_width_var = tk.IntVar(value=640)
        self.depth_height_var = tk.IntVar(value=480)
        self.depth_fps_var = tk.IntVar(value=30)
        self.flip_var = tk.StringVar(value="none")
        self.min_depth_var = tk.DoubleVar(value=0.05)
        self.max_depth_var = tk.DoubleVar(value=6.0)
        self.depth_window_var = tk.IntVar(value=5)
        self.save_csv_var = tk.BooleanVar(value=True)
        self.save_video_var = tk.BooleanVar(value=False)
        self.status_var = tk.StringVar(value="Stopped")

        self.frame_queue: queue.Queue = queue.Queue()
        self.log_queue: queue.Queue = queue.Queue()
        self.stop_event = threading.Event()
        self.worker: RealSenseYoloWorker | None = None
        self.preview_photo: ImageTk.PhotoImage | None = None

        self._build_ui()
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.root.after(40, self.process_queues)
        if args.auto_start:
            self.root.after(300, self.start_detection)
        if args.auto_close_sec > 0:
            self.root.after(int(args.auto_close_sec * 1000), self.close)

    def _build_ui(self) -> None:
        style = ttk.Style()
        style.configure("Header.TLabel", font=("Arial", 16, "bold"))
        style.configure("Primary.TButton", font=("Arial", 10, "bold"))
        style.configure("Danger.TButton", font=("Arial", 10, "bold"))

        header = ttk.Frame(self.root, padding=(12, 10))
        header.pack(fill=tk.X)
        ttk.Label(header, text="RealSense YOLO RGB-D Detection", style="Header.TLabel").pack(side=tk.LEFT)
        ttk.Label(header, textvariable=self.status_var).pack(side=tk.RIGHT)

        body = ttk.Frame(self.root, padding=12)
        body.pack(fill=tk.BOTH, expand=True)
        body.columnconfigure(1, weight=1)
        body.rowconfigure(0, weight=1)

        controls = ttk.Frame(body)
        controls.grid(row=0, column=0, sticky=tk.NS, padx=(0, 12))
        preview = ttk.Frame(body)
        preview.grid(row=0, column=1, sticky=tk.NSEW)
        preview.columnconfigure(0, weight=1)
        preview.rowconfigure(1, weight=1)

        model_box = ttk.Labelframe(controls, text="Model", padding=10)
        model_box.grid(row=0, column=0, sticky=tk.EW)
        model_box.columnconfigure(1, weight=1)
        self._path_row(model_box, "Model", self.model_var, 0, file_mode=True)
        self._row_entry(model_box, "Confidence", self.conf_var, 1)
        self._row_entry(model_box, "IOU", self.iou_var, 2)
        self._row_entry(model_box, "Image Size", self.imgsz_var, 3)
        self._row_entry(model_box, "Device", self.device_var, 4)

        camera_box = ttk.Labelframe(controls, text="RealSense RGB-D", padding=10)
        camera_box.grid(row=1, column=0, sticky=tk.EW, pady=(10, 0))
        camera_box.columnconfigure(1, weight=1)
        self._row_entry(camera_box, "Color Width", self.color_width_var, 0)
        self._row_entry(camera_box, "Color Height", self.color_height_var, 1)
        self._row_entry(camera_box, "Color FPS", self.color_fps_var, 2)
        self._row_entry(camera_box, "Depth Width", self.depth_width_var, 3)
        self._row_entry(camera_box, "Depth Height", self.depth_height_var, 4)
        self._row_entry(camera_box, "Depth FPS", self.depth_fps_var, 5)
        ttk.Label(camera_box, text="Flip").grid(row=6, column=0, sticky=tk.W, pady=2)
        ttk.Combobox(
            camera_box,
            textvariable=self.flip_var,
            values=("none", "vertical", "horizontal", "both"),
            state="readonly",
            width=16,
        ).grid(row=6, column=1, sticky=tk.EW, pady=2)

        depth_box = ttk.Labelframe(controls, text="Depth / Output", padding=10)
        depth_box.grid(row=2, column=0, sticky=tk.EW, pady=(10, 0))
        depth_box.columnconfigure(1, weight=1)
        self._row_entry(depth_box, "Min Depth m", self.min_depth_var, 0)
        self._row_entry(depth_box, "Max Depth m", self.max_depth_var, 1)
        self._row_entry(depth_box, "Depth Window", self.depth_window_var, 2)
        self._path_row(depth_box, "Output", self.output_dir_var, 3, file_mode=False)
        ttk.Checkbutton(depth_box, text="Save CSV", variable=self.save_csv_var).grid(row=4, column=0, sticky=tk.W, pady=(8, 0))
        ttk.Checkbutton(depth_box, text="Save Video", variable=self.save_video_var).grid(row=4, column=1, sticky=tk.W, pady=(8, 0))

        actions = ttk.Frame(controls)
        actions.grid(row=3, column=0, sticky=tk.EW, pady=(12, 0))
        actions.columnconfigure(0, weight=1)
        actions.columnconfigure(1, weight=1)
        self.start_button = ttk.Button(actions, text="Start Detection", command=self.start_detection, style="Primary.TButton")
        self.start_button.grid(row=0, column=0, sticky=tk.EW, padx=(0, 5))
        self.stop_button = ttk.Button(actions, text="Stop", command=self.stop_detection, state=tk.DISABLED, style="Danger.TButton")
        self.stop_button.grid(row=0, column=1, sticky=tk.EW, padx=(5, 0))

        ttk.Label(preview, text="Realtime RGB-D Detection View", style="Header.TLabel").grid(row=0, column=0, sticky=tk.W, pady=(0, 8))
        self.preview_label = tk.Label(preview, bg="#111827", fg="#e5e7eb", text="Detection not running")
        self.preview_label.grid(row=1, column=0, sticky=tk.NSEW)
        self.log_text = tk.Text(preview, height=10)
        self.log_text.grid(row=2, column=0, sticky=tk.EW, pady=(8, 0))

    def _row_entry(self, parent, label: str, variable, row: int) -> None:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky=tk.W, pady=2)
        ttk.Entry(parent, textvariable=variable, width=18).grid(row=row, column=1, sticky=tk.EW, pady=2)

    def _path_row(self, parent, label: str, variable: tk.StringVar, row: int, file_mode: bool) -> None:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky=tk.W, pady=2)
        ttk.Entry(parent, textvariable=variable, width=42).grid(row=row, column=1, sticky=tk.EW, pady=2)
        command = self.browse_file if file_mode else self.browse_dir
        ttk.Button(parent, text="Browse", command=lambda: command(variable)).grid(row=row, column=2, padx=(4, 0), pady=2)

    def browse_file(self, variable: tk.StringVar) -> None:
        initial = Path(variable.get()).parent if variable.get() else self.base_dir
        selected = filedialog.askopenfilename(initialdir=str(initial))
        if selected:
            variable.set(selected)

    def browse_dir(self, variable: tk.StringVar) -> None:
        selected = filedialog.askdirectory(initialdir=variable.get() or str(self.base_dir))
        if selected:
            variable.set(selected)

    def read_config(self) -> DetectionConfig:
        return DetectionConfig(
            model_path=Path(self.model_var.get()),
            conf=float(self.conf_var.get()),
            iou=float(self.iou_var.get()),
            imgsz=int(self.imgsz_var.get()),
            device=self.device_var.get(),
            color_width=int(self.color_width_var.get()),
            color_height=int(self.color_height_var.get()),
            color_fps=int(self.color_fps_var.get()),
            depth_width=int(self.depth_width_var.get()),
            depth_height=int(self.depth_height_var.get()),
            depth_fps=int(self.depth_fps_var.get()),
            flip=self.flip_var.get(),
            min_depth_m=float(self.min_depth_var.get()),
            max_depth_m=float(self.max_depth_var.get()),
            depth_window=int(self.depth_window_var.get()),
            save_csv=bool(self.save_csv_var.get()),
            save_video=bool(self.save_video_var.get()),
            output_dir=Path(self.output_dir_var.get()),
        )

    def start_detection(self) -> None:
        if self.worker and self.worker.is_alive():
            return
        try:
            cfg = self.read_config()
        except Exception as exc:
            messagebox.showerror("Invalid settings", str(exc))
            return
        self.stop_event.clear()
        self.worker = RealSenseYoloWorker(cfg, self.frame_queue, self.log_queue, self.stop_event)
        self.worker.start()
        self.start_button.configure(state=tk.DISABLED)
        self.stop_button.configure(state=tk.NORMAL)
        self.status_var.set("Running")
        self.log("Detection worker started")

    def stop_detection(self) -> None:
        self.stop_event.set()
        self.start_button.configure(state=tk.NORMAL)
        self.stop_button.configure(state=tk.DISABLED)
        self.status_var.set("Stopping")

    def process_queues(self) -> None:
        try:
            while True:
                self.log(self.log_queue.get_nowait())
        except queue.Empty:
            pass

        latest_frame = None
        try:
            while True:
                latest_frame = self.frame_queue.get_nowait()
        except queue.Empty:
            pass

        if latest_frame is not None:
            self.show_frame(latest_frame)

        if self.worker and not self.worker.is_alive() and self.status_var.get() in {"Running", "Stopping"}:
            self.start_button.configure(state=tk.NORMAL)
            self.stop_button.configure(state=tk.DISABLED)
            self.status_var.set("Stopped")

        self.root.after(40, self.process_queues)

    def show_frame(self, frame: np.ndarray) -> None:
        width = max(640, self.preview_label.winfo_width())
        height = max(420, self.preview_label.winfo_height())
        panel = resize_to_panel(frame, width, height)
        rgb = cv2.cvtColor(panel, cv2.COLOR_BGR2RGB)
        self.preview_photo = ImageTk.PhotoImage(Image.fromarray(rgb))
        self.preview_label.configure(image=self.preview_photo, text="")

    def log(self, message: str) -> None:
        timestamp = time.strftime("%H:%M:%S")
        self.log_text.insert(tk.END, f"[{timestamp}] {message}\n")
        self.log_text.see(tk.END)

    def close(self) -> None:
        self.stop_event.set()
        self.root.after(150, self.root.destroy)


def main() -> None:
    args = parse_args()
    root = tk.Tk()
    RealSenseYoloGui(root, args)
    root.mainloop()


if __name__ == "__main__":
    main()

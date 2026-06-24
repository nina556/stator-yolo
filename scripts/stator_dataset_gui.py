#!/usr/bin/env python3
from __future__ import annotations

import csv
import queue
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
import tkinter as tk

import cv2
import numpy as np
from PIL import Image, ImageTk


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
SUPPORTED_MODES = {
    (3280, 2464): 21,
    (3280, 1848): 28,
    (1920, 1080): 30,
    (1640, 1232): 30,
    (1280, 720): 60,
}


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def build_pipeline(sensor_id: int, width: int, height: int, fps: int, flip_method: int) -> str:
    return (
        f"nvarguscamerasrc sensor-id={sensor_id} ! "
        f"video/x-raw(memory:NVMM), width={width}, height={height}, "
        f"format=(string)NV12, framerate=(fraction){fps}/1 ! "
        f"nvvidconv flip-method={flip_method} ! "
        f"video/x-raw, width={width}, height={height}, format=(string)BGRx ! "
        "videoconvert ! video/x-raw, format=(string)BGR ! appsink"
    )


def validate_mode(width: int, height: int, fps: int) -> None:
    max_fps = SUPPORTED_MODES.get((width, height))
    if max_fps is None:
        modes = ", ".join(f"{w}x{h}@{mode_fps}" for (w, h), mode_fps in SUPPORTED_MODES.items())
        raise ValueError(f"Unsupported mode {width}x{height}. Supported modes: {modes}")
    if fps > max_fps:
        raise ValueError(f"{width}x{height} supports up to {max_fps} FPS, got {fps} FPS")


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


def make_realsense_session_id() -> str:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"realsense_{stamp}"


@dataclass
class CaptureConfig:
    sensor_id: int
    width: int
    height: int
    fps: int
    flip_method: int
    duration_sec: float
    sample_fps: float
    session_id: str
    scene: str
    lighting: str
    background: str
    pose_group: str
    occlusion_level: str
    robot_state: str
    notes: str
    raw_video_dir: Path
    frames_dir: Path
    frame_manifest: Path
    session_manifest: Path
    record_video: bool
    jpeg_quality: int


@dataclass
class RealSenseCaptureConfig:
    color_width: int
    color_height: int
    color_fps: int
    depth_width: int
    depth_height: int
    depth_fps: int
    flip_mode: str
    duration_sec: float
    sample_fps: float
    session_id: str
    scene: str
    lighting: str
    background: str
    pose_group: str
    occlusion_level: str
    robot_state: str
    notes: str
    raw_video_dir: Path
    frames_dir: Path
    depth_dir: Path
    frame_manifest: Path
    session_manifest: Path
    record_video: bool
    jpeg_quality: int


class PreviewWorker(threading.Thread):
    def __init__(
        self,
        sensor_id: int,
        width: int,
        height: int,
        fps: int,
        flip_method: int,
        frame_queue: queue.Queue,
        log_queue: queue.Queue,
        stop_event: threading.Event,
    ) -> None:
        super().__init__(daemon=True)
        self.sensor_id = sensor_id
        self.width = width
        self.height = height
        self.fps = fps
        self.flip_method = flip_method
        self.frame_queue = frame_queue
        self.log_queue = log_queue
        self.stop_event = stop_event

    def log(self, message: str) -> None:
        self.log_queue.put(message)

    def put_frame(self, frame) -> None:
        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
            except queue.Empty:
                break
        self.frame_queue.put(frame)

    def run(self) -> None:
        try:
            validate_mode(self.width, self.height, self.fps)
            pipeline = build_pipeline(
                self.sensor_id,
                self.width,
                self.height,
                self.fps,
                self.flip_method,
            )
            self.log(f"Opening preview: {pipeline}")
            capture = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
            if not capture.isOpened():
                raise RuntimeError("failed to open CSI camera preview")

            start_time = time.monotonic()
            try:
                while not self.stop_event.is_set():
                    ok, frame = capture.read()
                    if not ok:
                        self.log("Preview read returned no frame")
                        break
                    elapsed = time.monotonic() - start_time
                    cv2.putText(
                        frame,
                        f"LIVE PREVIEW sensor={self.sensor_id} t={elapsed:.1f}s",
                        (20, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 200, 255),
                        2,
                    )
                    self.put_frame(frame)
            finally:
                capture.release()
            self.log("Preview stopped")
        except Exception as exc:
            self.log(f"ERROR: {exc}")


class CaptureWorker(threading.Thread):
    def __init__(
        self,
        config: CaptureConfig,
        frame_queue: queue.Queue,
        log_queue: queue.Queue,
        stop_event: threading.Event,
    ) -> None:
        super().__init__(daemon=True)
        self.config = config
        self.frame_queue = frame_queue
        self.log_queue = log_queue
        self.stop_event = stop_event

    def log(self, message: str) -> None:
        self.log_queue.put(message)

    def put_frame(self, frame) -> None:
        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
            except queue.Empty:
                break
        self.frame_queue.put(frame)

    def run(self) -> None:
        cfg = self.config
        try:
            validate_mode(cfg.width, cfg.height, cfg.fps)
            if cfg.sample_fps <= 0:
                raise ValueError("sample FPS must be > 0")
            if cfg.duration_sec <= 0:
                raise ValueError("duration must be > 0")

            session_frames_dir = cfg.frames_dir / cfg.session_id
            session_frames_dir.mkdir(parents=True, exist_ok=True)
            cfg.raw_video_dir.mkdir(parents=True, exist_ok=True)
            video_path = cfg.raw_video_dir / f"{cfg.session_id}.mp4"

            pipeline = build_pipeline(
                cfg.sensor_id,
                cfg.width,
                cfg.height,
                cfg.fps,
                cfg.flip_method,
            )
            self.log(f"Opening camera: {pipeline}")
            capture = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
            if not capture.isOpened():
                raise RuntimeError("failed to open CSI camera")

            writer = None
            if cfg.record_video:
                writer = cv2.VideoWriter(
                    str(video_path),
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    cfg.fps,
                    (cfg.width, cfg.height),
                )
                if not writer.isOpened():
                    capture.release()
                    raise RuntimeError(f"failed to open video writer: {video_path}")

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
            self.log(f"Started session {cfg.session_id}")

            try:
                while not self.stop_event.is_set():
                    ok, frame = capture.read()
                    if not ok:
                        self.log("Camera read returned no frame")
                        break

                    elapsed = time.monotonic() - start_time
                    if elapsed >= cfg.duration_sec:
                        break

                    if writer is not None:
                        writer.write(frame)

                    if elapsed >= next_sample_time:
                        image_path = session_frames_dir / f"{cfg.session_id}_f{frame_index:06d}.jpg"
                        success = cv2.imwrite(
                            str(image_path),
                            frame,
                            [int(cv2.IMWRITE_JPEG_QUALITY), cfg.jpeg_quality],
                        )
                        if not success:
                            raise RuntimeError(f"failed to write frame: {image_path}")
                        append_csv_row(
                            cfg.frame_manifest,
                            frame_fields,
                            {
                                "session_id": cfg.session_id,
                                "sensor_id": cfg.sensor_id,
                                "frame_index": frame_index,
                                "timestamp_sec": f"{elapsed:.3f}",
                                "image_path": str(image_path),
                                "width": cfg.width,
                                "height": cfg.height,
                            },
                        )
                        saved_frames += 1
                        next_sample_time += 1.0 / cfg.sample_fps

                    preview = frame.copy()
                    cv2.putText(
                        preview,
                        f"{cfg.session_id} sampled={saved_frames} t={elapsed:.1f}s",
                        (20, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 0, 255),
                        2,
                    )
                    self.put_frame(preview)
                    frame_index += 1
            finally:
                capture.release()
                if writer is not None:
                    writer.release()

            append_csv_row(
                cfg.session_manifest,
                session_fields,
                {
                    "session_id": cfg.session_id,
                    "video_file": str(video_path) if cfg.record_video else "",
                    "scene": cfg.scene,
                    "lighting": cfg.lighting,
                    "background": cfg.background,
                    "pose_group": cfg.pose_group,
                    "occlusion_level": cfg.occlusion_level,
                    "robot_state": cfg.robot_state,
                    "camera_id": cfg.sensor_id,
                    "notes": cfg.notes,
                },
            )
            self.log(f"Stopped session {cfg.session_id}; saved {saved_frames} sampled frames")
        except Exception as exc:
            self.log(f"ERROR: {exc}")


class RealSensePreviewWorker(threading.Thread):
    def __init__(
        self,
        color_width: int,
        color_height: int,
        color_fps: int,
        depth_width: int,
        depth_height: int,
        depth_fps: int,
        flip_mode: str,
        frame_queue: queue.Queue,
        log_queue: queue.Queue,
        stop_event: threading.Event,
    ) -> None:
        super().__init__(daemon=True)
        self.color_width = color_width
        self.color_height = color_height
        self.color_fps = color_fps
        self.depth_width = depth_width
        self.depth_height = depth_height
        self.depth_fps = depth_fps
        self.flip_mode = flip_mode
        self.frame_queue = frame_queue
        self.log_queue = log_queue
        self.stop_event = stop_event

    def log(self, message: str) -> None:
        self.log_queue.put(message)

    def put_frame(self, frame) -> None:
        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
            except queue.Empty:
                break
        self.frame_queue.put(frame)

    @staticmethod
    def flip_frame(frame, mode: str):
        if mode == "vertical":
            return cv2.flip(frame, 0)
        if mode == "horizontal":
            return cv2.flip(frame, 1)
        if mode == "both":
            return cv2.flip(frame, -1)
        return frame

    def run(self) -> None:
        pipeline = None
        try:
            import pyrealsense2 as rs

            pipeline = rs.pipeline()
            config = rs.config()
            config.enable_stream(rs.stream.color, self.color_width, self.color_height, rs.format.bgr8, self.color_fps)
            config.enable_stream(rs.stream.depth, self.depth_width, self.depth_height, rs.format.z16, self.depth_fps)
            align = rs.align(rs.stream.color)
            self.log(
                "Opening RealSense preview: "
                f"color={self.color_width}x{self.color_height}@{self.color_fps}, "
                f"depth={self.depth_width}x{self.depth_height}@{self.depth_fps}"
            )
            profile = pipeline.start(config)
            depth_scale = float(profile.get_device().first_depth_sensor().get_depth_scale())
            start_time = time.monotonic()
            while not self.stop_event.is_set():
                frameset = pipeline.wait_for_frames(timeout_ms=5000)
                frameset = align.process(frameset)
                color_frame = frameset.get_color_frame()
                depth_frame = frameset.get_depth_frame()
                if not color_frame or not depth_frame:
                    continue
                color = np.asanyarray(color_frame.get_data())
                depth = np.asanyarray(depth_frame.get_data())
                color = self.flip_frame(color, self.flip_mode)
                depth = self.flip_frame(depth, self.flip_mode)
                center_depth = float(depth[depth.shape[0] // 2, depth.shape[1] // 2]) * depth_scale
                elapsed = time.monotonic() - start_time
                cv2.putText(
                    color,
                    f"REALSENSE PREVIEW t={elapsed:.1f}s center={center_depth:.3f}m",
                    (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.75,
                    (0, 200, 255),
                    2,
                )
                self.put_frame(color)
            self.log("RealSense preview stopped")
        except Exception as exc:
            self.log(f"ERROR: {exc}")
        finally:
            if pipeline is not None:
                try:
                    pipeline.stop()
                except Exception:
                    pass


class RealSenseCaptureWorker(threading.Thread):
    def __init__(
        self,
        config: RealSenseCaptureConfig,
        frame_queue: queue.Queue,
        log_queue: queue.Queue,
        stop_event: threading.Event,
    ) -> None:
        super().__init__(daemon=True)
        self.config = config
        self.frame_queue = frame_queue
        self.log_queue = log_queue
        self.stop_event = stop_event

    def log(self, message: str) -> None:
        self.log_queue.put(message)

    def put_frame(self, frame) -> None:
        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
            except queue.Empty:
                break
        self.frame_queue.put(frame)

    @staticmethod
    def flip_frame(frame, mode: str):
        if mode == "vertical":
            return cv2.flip(frame, 0)
        if mode == "horizontal":
            return cv2.flip(frame, 1)
        if mode == "both":
            return cv2.flip(frame, -1)
        return frame

    def run(self) -> None:
        cfg = self.config
        pipeline = None
        writer = None
        try:
            if cfg.sample_fps <= 0:
                raise ValueError("sample FPS must be > 0")
            if cfg.duration_sec <= 0:
                raise ValueError("duration must be > 0")

            import pyrealsense2 as rs

            session_frames_dir = cfg.frames_dir / cfg.session_id
            session_depth_dir = cfg.depth_dir / cfg.session_id
            session_frames_dir.mkdir(parents=True, exist_ok=True)
            session_depth_dir.mkdir(parents=True, exist_ok=True)
            cfg.raw_video_dir.mkdir(parents=True, exist_ok=True)
            video_path = cfg.raw_video_dir / f"{cfg.session_id}_color.mp4"

            pipeline = rs.pipeline()
            config = rs.config()
            config.enable_stream(rs.stream.color, cfg.color_width, cfg.color_height, rs.format.bgr8, cfg.color_fps)
            config.enable_stream(rs.stream.depth, cfg.depth_width, cfg.depth_height, rs.format.z16, cfg.depth_fps)
            align = rs.align(rs.stream.color)
            self.log(
                "Opening RealSense capture: "
                f"color={cfg.color_width}x{cfg.color_height}@{cfg.color_fps}, "
                f"depth={cfg.depth_width}x{cfg.depth_height}@{cfg.depth_fps}"
            )
            profile = pipeline.start(config)
            depth_scale = float(profile.get_device().first_depth_sensor().get_depth_scale())

            if cfg.record_video:
                writer = cv2.VideoWriter(
                    str(video_path),
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    cfg.color_fps,
                    (cfg.color_width, cfg.color_height),
                )
                if not writer.isOpened():
                    raise RuntimeError(f"failed to open video writer: {video_path}")

            frame_fields = [
                "session_id",
                "camera_type",
                "frame_index",
                "timestamp_sec",
                "image_path",
                "depth_path",
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
            self.log(f"Started RealSense session {cfg.session_id}; depth_scale={depth_scale:.6f}")

            while not self.stop_event.is_set():
                frameset = pipeline.wait_for_frames(timeout_ms=5000)
                frameset = align.process(frameset)
                color_frame = frameset.get_color_frame()
                depth_frame = frameset.get_depth_frame()
                if not color_frame or not depth_frame:
                    continue
                color = np.asanyarray(color_frame.get_data())
                depth = np.asanyarray(depth_frame.get_data())
                color = self.flip_frame(color, cfg.flip_mode)
                depth = self.flip_frame(depth, cfg.flip_mode)

                elapsed = time.monotonic() - start_time
                if elapsed >= cfg.duration_sec:
                    break
                if writer is not None:
                    writer.write(color)

                if elapsed >= next_sample_time:
                    image_path = session_frames_dir / f"{cfg.session_id}_f{frame_index:06d}.jpg"
                    depth_path = session_depth_dir / f"{cfg.session_id}_f{frame_index:06d}_depth.png"
                    if not cv2.imwrite(str(image_path), color, [int(cv2.IMWRITE_JPEG_QUALITY), cfg.jpeg_quality]):
                        raise RuntimeError(f"failed to write frame: {image_path}")
                    if not cv2.imwrite(str(depth_path), depth):
                        raise RuntimeError(f"failed to write depth: {depth_path}")
                    append_csv_row(
                        cfg.frame_manifest,
                        frame_fields,
                        {
                            "session_id": cfg.session_id,
                            "camera_type": "realsense",
                            "frame_index": frame_index,
                            "timestamp_sec": f"{elapsed:.3f}",
                            "image_path": str(image_path),
                            "depth_path": str(depth_path),
                            "width": cfg.color_width,
                            "height": cfg.color_height,
                        },
                    )
                    saved_frames += 1
                    next_sample_time += 1.0 / cfg.sample_fps

                preview = color.copy()
                cv2.putText(
                    preview,
                    f"{cfg.session_id} sampled={saved_frames} t={elapsed:.1f}s",
                    (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.75,
                    (0, 0, 255),
                    2,
                )
                self.put_frame(preview)
                frame_index += 1

            append_csv_row(
                cfg.session_manifest,
                session_fields,
                {
                    "session_id": cfg.session_id,
                    "video_file": str(video_path) if cfg.record_video else "",
                    "scene": cfg.scene,
                    "lighting": cfg.lighting,
                    "background": cfg.background,
                    "pose_group": cfg.pose_group,
                    "occlusion_level": cfg.occlusion_level,
                    "robot_state": cfg.robot_state,
                    "camera_id": "realsense",
                    "notes": cfg.notes,
                },
            )
            self.log(f"Stopped RealSense session {cfg.session_id}; saved {saved_frames} sampled frames")
        except Exception as exc:
            self.log(f"ERROR: {exc}")
        finally:
            if writer is not None:
                writer.release()
            if pipeline is not None:
                try:
                    pipeline.stop()
                except Exception:
                    pass


class DetectionWorker(threading.Thread):
    def __init__(
        self,
        model_path: Path,
        sensor_id: int,
        width: int,
        height: int,
        fps: int,
        flip_method: int,
        conf: float,
        frame_queue: queue.Queue,
        log_queue: queue.Queue,
        stop_event: threading.Event,
    ) -> None:
        super().__init__(daemon=True)
        self.model_path = model_path
        self.sensor_id = sensor_id
        self.width = width
        self.height = height
        self.fps = fps
        self.flip_method = flip_method
        self.conf = conf
        self.frame_queue = frame_queue
        self.log_queue = log_queue
        self.stop_event = stop_event

    def log(self, message: str) -> None:
        self.log_queue.put(message)

    def put_frame(self, frame) -> None:
        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
            except queue.Empty:
                break
        self.frame_queue.put(frame)

    def run(self) -> None:
        try:
            validate_mode(self.width, self.height, self.fps)
            if not self.model_path.exists():
                raise RuntimeError(f"model not found: {self.model_path}")

            from ultralytics import YOLO

            self.log(f"Loading model: {self.model_path}")
            model = YOLO(str(self.model_path))
            pipeline = build_pipeline(
                self.sensor_id,
                self.width,
                self.height,
                self.fps,
                self.flip_method,
            )
            self.log(f"Opening detection camera: {pipeline}")
            capture = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
            if not capture.isOpened():
                raise RuntimeError("failed to open CSI camera for detection")

            frames = 0
            start_time = time.monotonic()
            try:
                while not self.stop_event.is_set():
                    ok, frame = capture.read()
                    if not ok:
                        self.log("Detection read returned no frame")
                        break
                    result = model.predict(source=frame, conf=self.conf, verbose=False)[0]
                    annotated = result.plot()
                    frames += 1
                    elapsed = max(time.monotonic() - start_time, 1e-6)
                    fps_text = f"DETECT FPS {frames / elapsed:.1f}"
                    cv2.putText(
                        annotated,
                        fps_text,
                        (20, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 0, 255),
                        2,
                    )
                    self.put_frame(annotated)
            finally:
                capture.release()
            self.log("Detection stopped")
        except Exception as exc:
            self.log(f"ERROR: {exc}")


class RealSenseDetectionWorker(threading.Thread):
    def __init__(
        self,
        model_path: Path,
        color_width: int,
        color_height: int,
        color_fps: int,
        depth_width: int,
        depth_height: int,
        depth_fps: int,
        flip_mode: str,
        conf: float,
        frame_queue: queue.Queue,
        log_queue: queue.Queue,
        stop_event: threading.Event,
    ) -> None:
        super().__init__(daemon=True)
        self.model_path = model_path
        self.color_width = color_width
        self.color_height = color_height
        self.color_fps = color_fps
        self.depth_width = depth_width
        self.depth_height = depth_height
        self.depth_fps = depth_fps
        self.flip_mode = flip_mode
        self.conf = conf
        self.frame_queue = frame_queue
        self.log_queue = log_queue
        self.stop_event = stop_event

    def log(self, message: str) -> None:
        self.log_queue.put(message)

    def put_frame(self, frame) -> None:
        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
            except queue.Empty:
                break
        self.frame_queue.put(frame)

    @staticmethod
    def flip_frame(frame, mode: str):
        if mode == "vertical":
            return cv2.flip(frame, 0)
        if mode == "horizontal":
            return cv2.flip(frame, 1)
        if mode == "both":
            return cv2.flip(frame, -1)
        return frame

    @staticmethod
    def robust_depth_m(depth, cx: int, cy: int, depth_scale: float) -> float:
        radius = 2
        y0 = max(0, cy - radius)
        y1 = min(depth.shape[0], cy + radius + 1)
        x0 = max(0, cx - radius)
        x1 = min(depth.shape[1], cx + radius + 1)
        patch = depth[y0:y1, x0:x1].astype(np.float32) * depth_scale
        valid = patch[(patch > 0.05) & (patch < 6.0)]
        if valid.size == 0:
            return 0.0
        return float(np.median(valid))

    def run(self) -> None:
        pipeline = None
        try:
            if not self.model_path.exists():
                raise RuntimeError(f"model not found: {self.model_path}")

            import pyrealsense2 as rs
            from ultralytics import YOLO

            self.log(f"Loading model: {self.model_path}")
            model = YOLO(str(self.model_path))

            pipeline = rs.pipeline()
            config = rs.config()
            config.enable_stream(rs.stream.color, self.color_width, self.color_height, rs.format.bgr8, self.color_fps)
            config.enable_stream(rs.stream.depth, self.depth_width, self.depth_height, rs.format.z16, self.depth_fps)
            align = rs.align(rs.stream.color)

            self.log(
                "Opening RealSense: "
                f"color={self.color_width}x{self.color_height}@{self.color_fps}, "
                f"depth={self.depth_width}x{self.depth_height}@{self.depth_fps}"
            )
            profile = pipeline.start(config)
            depth_scale = float(profile.get_device().first_depth_sensor().get_depth_scale())
            color_intrinsics = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
            self.log(f"RealSense started; depth_scale={depth_scale:.6f}")

            frames = 0
            start_time = time.monotonic()
            while not self.stop_event.is_set():
                frameset = pipeline.wait_for_frames(timeout_ms=5000)
                frameset = align.process(frameset)
                color_frame = frameset.get_color_frame()
                depth_frame = frameset.get_depth_frame()
                if not color_frame or not depth_frame:
                    continue

                color = np.asanyarray(color_frame.get_data())
                depth = np.asanyarray(depth_frame.get_data())
                color = self.flip_frame(color, self.flip_mode)
                depth = self.flip_frame(depth, self.flip_mode)

                result = model.predict(source=color, conf=self.conf, verbose=False)[0]
                annotated = color.copy()
                detections = 0
                if result.boxes is not None:
                    for box in result.boxes:
                        xyxy = box.xyxy[0].detach().cpu().numpy()
                        cls_id = int(box.cls[0].item())
                        score = float(box.conf[0].item())
                        name = result.names.get(cls_id, str(cls_id))
                        x1, y1, x2, y2 = xyxy
                        cx = int(round((x1 + x2) * 0.5))
                        cy = int(round((y1 + y2) * 0.5))
                        cx = min(max(cx, 0), depth.shape[1] - 1)
                        cy = min(max(cy, 0), depth.shape[0] - 1)
                        depth_m = self.robust_depth_m(depth, cx, cy, depth_scale)
                        if depth_m > 0:
                            point = rs.rs2_deproject_pixel_to_point(color_intrinsics, [float(cx), float(cy)], depth_m)
                            xyz = tuple(float(v) for v in point)
                        else:
                            xyz = (0.0, 0.0, 0.0)
                        color_box = (40, 220, 80) if depth_m > 0 else (0, 165, 255)
                        cv2.rectangle(annotated, (int(x1), int(y1)), (int(x2), int(y2)), color_box, 2)
                        cv2.circle(annotated, (cx, cy), 4, (0, 0, 255), -1)
                        label = f"{name} {score:.2f} z={depth_m:.3f} x={xyz[0]:.3f} y={xyz[1]:.3f}"
                        text_y = max(22, int(y1) - 8)
                        cv2.putText(annotated, label, (int(x1), text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.54, (0, 0, 0), 3)
                        cv2.putText(annotated, label, (int(x1), text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.54, color_box, 1)
                        detections += 1

                frames += 1
                elapsed = max(time.monotonic() - start_time, 1e-6)
                status = f"REALSENSE YOLO FPS {frames / elapsed:.1f} DET {detections}"
                cv2.putText(annotated, status, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 3)
                cv2.putText(annotated, status, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 1)
                self.put_frame(annotated)
            self.log("RealSense detection stopped")
        except Exception as exc:
            self.log(f"ERROR: {exc}")
        finally:
            if pipeline is not None:
                try:
                    pipeline.stop()
                except Exception:
                    pass


class StatorDatasetGui:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Stator Dataset Workflow")
        self.root.geometry("1200x820")
        self.base_dir = repo_root()
        self.frame_queue: queue.Queue = queue.Queue(maxsize=1)
        self.log_queue: queue.Queue = queue.Queue()
        self.preview_stop = threading.Event()
        self.preview_worker: PreviewWorker | None = None
        self.capture_stop = threading.Event()
        self.capture_worker: CaptureWorker | None = None
        self.preview_photo = None
        self.status_var = tk.StringVar(value="Idle")
        self.train_process: subprocess.Popen | None = None
        self.train_thread: threading.Thread | None = None
        self.train_results_csv: Path | None = None
        self.train_plot_job = None
        self.export_process: subprocess.Popen | None = None
        self.test_stop = threading.Event()
        self.test_worker: DetectionWorker | None = None
        self.test_frame_queue: queue.Queue = queue.Queue(maxsize=1)
        self.test_log_queue: queue.Queue = queue.Queue()
        self.test_photo = None

        self.label_images: list[Path] = []
        self.label_index = 0
        self.label_image_path: Path | None = None
        self.label_image = None
        self.label_photo = None
        self.label_boxes: list[tuple[float, float, float, float]] = []
        self.drag_start: tuple[int, int] | None = None
        self.drag_rect = None
        self.image_scale = 1.0
        self.image_offset = (0, 0)

        self._configure_style()
        self._build_ui()
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.root.after(50, self._poll_queues)

    def _configure_style(self) -> None:
        style = ttk.Style()
        style.theme_use("clam")
        style.configure(".", font=("DejaVu Sans", 10))
        style.configure("TFrame", background="#f4f6f8")
        style.configure("TLabel", background="#f4f6f8", foreground="#1f2933")
        style.configure("Header.TLabel", font=("DejaVu Sans", 13, "bold"))
        style.configure("Status.TLabel", font=("DejaVu Sans", 11, "bold"), foreground="#0f766e")
        style.configure("TButton", padding=(10, 6))
        style.configure("Primary.TButton", padding=(12, 7))
        style.configure("Danger.TButton", padding=(12, 7))
        style.configure("TLabelframe", background="#f4f6f8")
        style.configure("TLabelframe.Label", background="#f4f6f8", foreground="#1f2933")
        style.configure("TNotebook", background="#e5e7eb", borderwidth=0)
        style.configure("TNotebook.Tab", padding=(16, 8))

    def _build_ui(self) -> None:
        header = ttk.Frame(self.root, padding=(12, 10))
        header.pack(fill=tk.X)
        ttk.Label(header, text="Stator Dataset Workflow", style="Header.TLabel").pack(side=tk.LEFT)
        ttk.Label(header, textvariable=self.status_var, style="Status.TLabel").pack(side=tk.RIGHT)

        notebook = ttk.Notebook(self.root)
        notebook.pack(fill=tk.BOTH, expand=True)

        self.capture_tab = ttk.Frame(notebook)
        self.label_tab = ttk.Frame(notebook)
        self.dataset_tab = ttk.Frame(notebook)
        self.train_tab = ttk.Frame(notebook)
        self.test_tab = ttk.Frame(notebook)
        notebook.add(self.capture_tab, text="Capture")
        notebook.add(self.label_tab, text="Label")
        notebook.add(self.dataset_tab, text="Dataset")
        notebook.add(self.train_tab, text="Train")
        notebook.add(self.test_tab, text="Test")

        self._build_capture_tab()
        self._build_label_tab()
        self._build_dataset_tab()
        self._build_train_tab()
        self._build_test_tab()

    def _build_capture_tab(self) -> None:
        self.capture_tab.columnconfigure(1, weight=1)
        self.capture_tab.rowconfigure(0, weight=1)
        controls = ttk.Frame(self.capture_tab, padding=12)
        controls.grid(row=0, column=0, sticky=tk.NS)
        preview_frame = ttk.Frame(self.capture_tab, padding=12)
        preview_frame.grid(row=0, column=1, sticky=tk.NSEW)
        preview_frame.columnconfigure(0, weight=1)
        preview_frame.rowconfigure(1, weight=1)

        self.capture_camera_type_var = tk.StringVar(value="RealSense")
        self.sensor_var = tk.IntVar(value=0)
        self.width_var = tk.IntVar(value=640)
        self.height_var = tk.IntVar(value=480)
        self.fps_var = tk.IntVar(value=30)
        self.depth_width_var = tk.IntVar(value=640)
        self.depth_height_var = tk.IntVar(value=480)
        self.depth_fps_var = tk.IntVar(value=30)
        self.flip_var = tk.IntVar(value=2)
        self.realsense_flip_var = tk.StringVar(value="none")
        self.duration_var = tk.DoubleVar(value=60.0)
        self.sample_fps_var = tk.DoubleVar(value=2.0)
        self.record_video_var = tk.BooleanVar(value=True)
        self.scene_var = tk.StringVar(value="first_stator_test")
        self.lighting_var = tk.StringVar(value="normal")
        self.background_var = tk.StringVar(value="workbench")
        self.pose_group_var = tk.StringVar(value="mixed")
        self.occlusion_var = tk.StringVar(value="low")
        self.robot_state_var = tk.StringVar(value="static")
        self.notes_var = tk.StringVar(value="")

        camera_box = ttk.Labelframe(controls, text="Camera", padding=10)
        camera_box.grid(row=0, column=0, sticky=tk.EW)
        camera_box.columnconfigure(1, weight=1)
        ttk.Label(camera_box, text="Camera").grid(row=0, column=0, sticky=tk.W, pady=2)
        ttk.Combobox(
            camera_box,
            textvariable=self.capture_camera_type_var,
            values=("RealSense", "CSI"),
            state="readonly",
            width=16,
        ).grid(row=0, column=1, sticky=tk.EW, pady=2)
        self._row_entry(camera_box, "CSI Sensor", self.sensor_var, 1)
        self._row_entry(camera_box, "Color Width", self.width_var, 2)
        self._row_entry(camera_box, "Color Height", self.height_var, 3)
        self._row_entry(camera_box, "Color FPS", self.fps_var, 4)
        self._row_entry(camera_box, "Depth Width", self.depth_width_var, 5)
        self._row_entry(camera_box, "Depth Height", self.depth_height_var, 6)
        self._row_entry(camera_box, "Depth FPS", self.depth_fps_var, 7)
        self._row_entry(camera_box, "CSI Flip", self.flip_var, 8)
        ttk.Label(camera_box, text="RS Flip").grid(row=9, column=0, sticky=tk.W, pady=2)
        ttk.Combobox(
            camera_box,
            textvariable=self.realsense_flip_var,
            values=("none", "vertical", "horizontal", "both"),
            state="readonly",
            width=16,
        ).grid(row=9, column=1, sticky=tk.EW, pady=2)
        self.preview_button = ttk.Button(
            camera_box,
            text="Start Preview",
            command=self.start_preview,
            style="Primary.TButton",
        )
        self.preview_button.grid(row=10, column=0, sticky=tk.EW, pady=(10, 4))
        self.stop_preview_button = ttk.Button(
            camera_box,
            text="Stop Preview",
            command=self.stop_preview,
            state=tk.DISABLED,
        )
        self.stop_preview_button.grid(row=10, column=1, sticky=tk.EW, pady=(10, 4), padx=(6, 0))

        capture_box = ttk.Labelframe(controls, text="Sampling", padding=10)
        capture_box.grid(row=1, column=0, sticky=tk.EW, pady=(12, 0))
        capture_box.columnconfigure(1, weight=1)
        self._row_entry(capture_box, "Duration sec", self.duration_var, 0)
        self._row_entry(capture_box, "Sample FPS", self.sample_fps_var, 1)
        ttk.Checkbutton(capture_box, text="Record video", variable=self.record_video_var).grid(
            row=2, column=0, columnspan=2, sticky=tk.W, pady=4
        )
        self.start_button = ttk.Button(
            capture_box,
            text="Start Capture",
            command=self.start_capture,
            style="Primary.TButton",
        )
        self.start_button.grid(row=3, column=0, sticky=tk.EW, pady=(10, 4))
        self.stop_button = ttk.Button(
            capture_box,
            text="Stop Capture",
            command=self.stop_capture,
            state=tk.DISABLED,
            style="Danger.TButton",
        )
        self.stop_button.grid(row=3, column=1, sticky=tk.EW, pady=(10, 4), padx=(6, 0))

        metadata_box = ttk.Labelframe(controls, text="Session Metadata", padding=10)
        metadata_box.grid(row=2, column=0, sticky=tk.EW, pady=(12, 0))
        metadata_box.columnconfigure(1, weight=1)
        self._row_entry(metadata_box, "Scene", self.scene_var, 0)
        self._row_entry(metadata_box, "Lighting", self.lighting_var, 1)
        self._row_entry(metadata_box, "Background", self.background_var, 2)
        self._row_entry(metadata_box, "Pose", self.pose_group_var, 3)
        self._row_entry(metadata_box, "Occlusion", self.occlusion_var, 4)
        self._row_entry(metadata_box, "Robot", self.robot_state_var, 5)
        self._row_entry(metadata_box, "Notes", self.notes_var, 6)

        ttk.Label(preview_frame, text="Live Camera View", style="Header.TLabel").grid(
            row=0, column=0, sticky=tk.W, pady=(0, 8)
        )
        self.preview_label = tk.Label(preview_frame, bg="#111827", fg="#e5e7eb", text="Preview not running")
        self.preview_label.grid(row=1, column=0, sticky=tk.NSEW)
        self.log_text = tk.Text(preview_frame, height=9)
        self.log_text.grid(row=2, column=0, sticky=tk.EW, pady=(8, 0))

    def _build_label_tab(self) -> None:
        top = ttk.Frame(self.label_tab, padding=8)
        top.pack(fill=tk.X)
        body = ttk.Frame(self.label_tab, padding=8)
        body.pack(fill=tk.BOTH, expand=True)

        self.label_source_var = tk.StringVar(value=str(self.base_dir / "data/frames/raw"))
        self.export_images_var = tk.StringVar(value=str(self.base_dir / "data/labeling/export/images"))
        self.export_labels_var = tk.StringVar(value=str(self.base_dir / "data/labeling/export/labels"))

        self._path_row(top, "Frames", self.label_source_var, 0)
        self._path_row(top, "Export images", self.export_images_var, 1)
        self._path_row(top, "Export labels", self.export_labels_var, 2)

        actions = ttk.Frame(top)
        actions.grid(row=3, column=0, columnspan=3, sticky=tk.W, pady=(8, 0))
        ttk.Button(actions, text="Load Frames", command=self.load_label_images).pack(side=tk.LEFT, padx=(0, 4))
        ttk.Button(actions, text="Prev", command=self.prev_label_image).pack(side=tk.LEFT, padx=4)
        ttk.Button(actions, text="Next", command=self.next_label_image).pack(side=tk.LEFT, padx=4)
        ttk.Button(actions, text="Undo Box", command=self.undo_box).pack(side=tk.LEFT, padx=4)
        ttk.Button(actions, text="Clear Boxes", command=self.clear_boxes).pack(side=tk.LEFT, padx=4)
        ttk.Button(actions, text="Save Label", command=self.save_label).pack(side=tk.LEFT, padx=4)
        ttk.Button(actions, text="Save + Next", command=self.save_label_and_next).pack(side=tk.LEFT, padx=4)

        self.label_canvas = tk.Canvas(body, bg="#202020", highlightthickness=0)
        self.label_canvas.pack(fill=tk.BOTH, expand=True)
        self.label_canvas.bind("<ButtonPress-1>", self.on_label_press)
        self.label_canvas.bind("<B1-Motion>", self.on_label_drag)
        self.label_canvas.bind("<ButtonRelease-1>", self.on_label_release)
        self.label_canvas.bind("<Configure>", lambda _event: self.redraw_label_image())
        self.label_status = ttk.Label(self.label_tab, text="Load frames to start labeling.")
        self.label_status.pack(fill=tk.X, padx=8, pady=(0, 8))

    def _build_dataset_tab(self) -> None:
        controls = ttk.Frame(self.dataset_tab, padding=8)
        controls.pack(side=tk.LEFT, fill=tk.Y)
        log_frame = ttk.Frame(self.dataset_tab, padding=8)
        log_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.dataset_images_var = tk.StringVar(value=str(self.base_dir / "data/labeling/export/images"))
        self.dataset_labels_var = tk.StringVar(value=str(self.base_dir / "data/labeling/export/labels"))
        self.dataset_output_var = tk.StringVar(value=str(self.base_dir / "data/dataset"))
        self.train_ratio_var = tk.DoubleVar(value=0.8)
        self.val_ratio_var = tk.DoubleVar(value=0.1)
        self.test_ratio_var = tk.DoubleVar(value=0.1)
        self.augment_copies_var = tk.IntVar(value=2)

        self._path_row(controls, "Images", self.dataset_images_var, 0)
        self._path_row(controls, "Labels", self.dataset_labels_var, 1)
        self._path_row(controls, "Dataset", self.dataset_output_var, 2)
        self._row_entry(controls, "Train ratio", self.train_ratio_var, 3)
        self._row_entry(controls, "Val ratio", self.val_ratio_var, 4)
        self._row_entry(controls, "Test ratio", self.test_ratio_var, 5)
        self._row_entry(controls, "Aug copies", self.augment_copies_var, 6)
        ttk.Button(controls, text="Validate Labels", command=self.run_validate).grid(
            row=7, column=0, columnspan=3, sticky=tk.EW, pady=(12, 4)
        )
        ttk.Button(controls, text="Split Dataset", command=self.run_split).grid(
            row=8, column=0, columnspan=3, sticky=tk.EW, pady=4
        )
        ttk.Button(controls, text="Augment Train Split", command=self.run_augment).grid(
            row=9, column=0, columnspan=3, sticky=tk.EW, pady=4
        )

        self.dataset_log_text = tk.Text(log_frame)
        self.dataset_log_text.pack(fill=tk.BOTH, expand=True)

    def _build_train_tab(self) -> None:
        self.train_tab.columnconfigure(1, weight=1)
        self.train_tab.rowconfigure(0, weight=1)
        controls = ttk.Frame(self.train_tab, padding=12)
        controls.grid(row=0, column=0, sticky=tk.NS)
        monitor = ttk.Frame(self.train_tab, padding=12)
        monitor.grid(row=0, column=1, sticky=tk.NSEW)
        monitor.columnconfigure(0, weight=1)
        monitor.rowconfigure(0, weight=3)
        monitor.rowconfigure(1, weight=2)

        self.train_model_var = tk.StringVar(value="yolov8n.pt")
        self.train_dataset_var = tk.StringVar(value=str(self.base_dir / "data/dataset.yaml"))
        self.train_project_var = tk.StringVar(value=str(self.base_dir / "runs"))
        self.train_name_var = tk.StringVar(value="stator_yolov8")
        self.train_epochs_var = tk.IntVar(value=100)
        self.train_imgsz_var = tk.IntVar(value=640)
        self.train_batch_var = tk.IntVar(value=16)
        self.train_device_var = tk.StringVar(value="0")
        self.export_model_var = tk.StringVar(
            value=str(self.base_dir / "runs/stator_yolov8/weights/best.pt")
        )
        self.export_imgsz_var = tk.IntVar(value=640)
        self.export_device_var = tk.StringVar(value="0")

        train_box = ttk.Labelframe(controls, text="Fine-tune YOLO", padding=10)
        train_box.grid(row=0, column=0, sticky=tk.EW)
        train_box.columnconfigure(1, weight=1)
        self._row_entry(train_box, "Model", self.train_model_var, 0)
        self._path_row(train_box, "Dataset yaml", self.train_dataset_var, 1, file_mode=True)
        self._path_row(train_box, "Project", self.train_project_var, 2)
        self._row_entry(train_box, "Run name", self.train_name_var, 3)
        self._row_entry(train_box, "Epochs", self.train_epochs_var, 4)
        self._row_entry(train_box, "Image size", self.train_imgsz_var, 5)
        self._row_entry(train_box, "Batch", self.train_batch_var, 6)
        self._row_entry(train_box, "Device", self.train_device_var, 7)

        self.train_start_button = ttk.Button(
            train_box,
            text="Start Training",
            command=self.start_training,
            style="Primary.TButton",
        )
        self.train_start_button.grid(row=8, column=0, sticky=tk.EW, pady=(12, 4))
        self.train_stop_button = ttk.Button(
            train_box,
            text="Stop Training",
            command=self.stop_training,
            state=tk.DISABLED,
            style="Danger.TButton",
        )
        self.train_stop_button.grid(row=8, column=1, sticky=tk.EW, padx=(6, 0), pady=(12, 4))
        ttk.Button(train_box, text="Refresh Curves", command=self.refresh_training_plot).grid(
            row=9, column=0, columnspan=2, sticky=tk.EW, pady=4
        )

        export_box = ttk.Labelframe(controls, text="Export TensorRT", padding=10)
        export_box.grid(row=1, column=0, sticky=tk.EW, pady=(12, 0))
        export_box.columnconfigure(1, weight=1)
        self._path_row(export_box, "PT model", self.export_model_var, 0, file_mode=True)
        self._row_entry(export_box, "Image size", self.export_imgsz_var, 1)
        self._row_entry(export_box, "Device", self.export_device_var, 2)
        self.export_button = ttk.Button(
            export_box,
            text="Export Engine",
            command=self.start_export_engine,
            style="Primary.TButton",
        )
        self.export_button.grid(row=3, column=0, columnspan=2, sticky=tk.EW, pady=(10, 4))

        plots = ttk.Frame(monitor)
        plots.grid(row=0, column=0, sticky=tk.NSEW)
        plots.columnconfigure(0, weight=1)
        plots.columnconfigure(1, weight=1)
        plots.rowconfigure(0, weight=1)
        self.loss_canvas = tk.Canvas(plots, bg="#ffffff", highlightthickness=1, highlightbackground="#cbd5e1")
        self.metric_canvas = tk.Canvas(plots, bg="#ffffff", highlightthickness=1, highlightbackground="#cbd5e1")
        self.loss_canvas.grid(row=0, column=0, sticky=tk.NSEW, padx=(0, 6))
        self.metric_canvas.grid(row=0, column=1, sticky=tk.NSEW, padx=(6, 0))

        self.train_log_text = tk.Text(monitor, height=12)
        self.train_log_text.grid(row=1, column=0, sticky=tk.NSEW, pady=(8, 0))

    def _build_test_tab(self) -> None:
        self.test_tab.columnconfigure(1, weight=1)
        self.test_tab.rowconfigure(0, weight=1)
        controls = ttk.Frame(self.test_tab, padding=12)
        controls.grid(row=0, column=0, sticky=tk.NS)
        preview = ttk.Frame(self.test_tab, padding=12)
        preview.grid(row=0, column=1, sticky=tk.NSEW)
        preview.columnconfigure(0, weight=1)
        preview.rowconfigure(1, weight=1)

        preferred_model = self.base_dir / "runs/stator_yolov8/weights/best.engine"
        if not preferred_model.exists():
            preferred_model = self.base_dir / "runs/stator_yolov8/weights/best.pt"
        self.test_model_var = tk.StringVar(value=str(preferred_model))
        self.test_camera_type_var = tk.StringVar(value="RealSense")
        self.test_sensor_var = tk.IntVar(value=0)
        self.test_width_var = tk.IntVar(value=640)
        self.test_height_var = tk.IntVar(value=480)
        self.test_fps_var = tk.IntVar(value=30)
        self.test_depth_width_var = tk.IntVar(value=640)
        self.test_depth_height_var = tk.IntVar(value=480)
        self.test_depth_fps_var = tk.IntVar(value=30)
        self.test_flip_var = tk.StringVar(value="none")
        self.test_conf_var = tk.DoubleVar(value=0.25)

        test_box = ttk.Labelframe(controls, text="Realtime Detection", padding=10)
        test_box.grid(row=0, column=0, sticky=tk.EW)
        test_box.columnconfigure(1, weight=1)
        self._path_row(test_box, "Model", self.test_model_var, 0, file_mode=True)
        ttk.Label(test_box, text="Camera").grid(row=1, column=0, sticky=tk.W, pady=2)
        ttk.Combobox(
            test_box,
            textvariable=self.test_camera_type_var,
            values=("RealSense", "CSI"),
            state="readonly",
            width=16,
        ).grid(row=1, column=1, sticky=tk.EW, pady=2)
        self._row_entry(test_box, "CSI Sensor", self.test_sensor_var, 2)
        self._row_entry(test_box, "Color Width", self.test_width_var, 3)
        self._row_entry(test_box, "Color Height", self.test_height_var, 4)
        self._row_entry(test_box, "Color FPS", self.test_fps_var, 5)
        self._row_entry(test_box, "Depth Width", self.test_depth_width_var, 6)
        self._row_entry(test_box, "Depth Height", self.test_depth_height_var, 7)
        self._row_entry(test_box, "Depth FPS", self.test_depth_fps_var, 8)
        ttk.Label(test_box, text="Flip").grid(row=9, column=0, sticky=tk.W, pady=2)
        ttk.Combobox(
            test_box,
            textvariable=self.test_flip_var,
            values=("none", "vertical", "horizontal", "both", "2"),
            state="readonly",
            width=16,
        ).grid(row=9, column=1, sticky=tk.EW, pady=2)
        self._row_entry(test_box, "Confidence", self.test_conf_var, 10)
        self.test_start_button = ttk.Button(
            test_box,
            text="Start Detection",
            command=self.start_detection,
            style="Primary.TButton",
        )
        self.test_start_button.grid(row=11, column=0, sticky=tk.EW, pady=(12, 4))
        self.test_stop_button = ttk.Button(
            test_box,
            text="Stop Detection",
            command=self.stop_detection,
            state=tk.DISABLED,
            style="Danger.TButton",
        )
        self.test_stop_button.grid(row=11, column=1, sticky=tk.EW, padx=(6, 0), pady=(12, 4))

        ttk.Label(preview, text="Realtime Detection View", style="Header.TLabel").grid(
            row=0, column=0, sticky=tk.W, pady=(0, 8)
        )
        self.test_preview_label = tk.Label(preview, bg="#111827", fg="#e5e7eb", text="Detection not running")
        self.test_preview_label.grid(row=1, column=0, sticky=tk.NSEW)
        self.test_log_text = tk.Text(preview, height=9)
        self.test_log_text.grid(row=2, column=0, sticky=tk.EW, pady=(8, 0))

    def _row_entry(self, parent, label: str, variable, row: int) -> None:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky=tk.W, pady=2)
        ttk.Entry(parent, textvariable=variable, width=18).grid(row=row, column=1, sticky=tk.EW, pady=2)

    def _path_row(self, parent, label: str, variable: tk.StringVar, row: int, file_mode: bool = False) -> None:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky=tk.W, pady=2)
        ttk.Entry(parent, textvariable=variable, width=62).grid(row=row, column=1, sticky=tk.EW, pady=2)
        browse_command = self.browse_file if file_mode else self.browse_dir
        ttk.Button(parent, text="Browse", command=lambda: browse_command(variable)).grid(
            row=row, column=2, sticky=tk.EW, padx=(4, 0), pady=2
        )

    def browse_dir(self, variable: tk.StringVar) -> None:
        selected = filedialog.askdirectory(initialdir=variable.get() or str(self.base_dir))
        if selected:
            variable.set(selected)

    def browse_file(self, variable: tk.StringVar) -> None:
        selected = filedialog.askopenfilename(initialdir=str(Path(variable.get()).parent or self.base_dir))
        if selected:
            variable.set(selected)

    def log(self, message: str) -> None:
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)

    def dataset_log(self, message: str) -> None:
        self.dataset_log_text.insert(tk.END, message + "\n")
        self.dataset_log_text.see(tk.END)

    def read_camera_settings(self) -> tuple[int, int, int, int, int]:
        sensor_id = int(self.sensor_var.get())
        width = int(self.width_var.get())
        height = int(self.height_var.get())
        fps = int(self.fps_var.get())
        flip_method = int(self.flip_var.get())
        validate_mode(width, height, fps)
        return sensor_id, width, height, fps, flip_method

    def read_realsense_settings(self) -> tuple[int, int, int, int, int, int, str]:
        color_width = int(self.width_var.get())
        color_height = int(self.height_var.get())
        color_fps = int(self.fps_var.get())
        depth_width = int(self.depth_width_var.get())
        depth_height = int(self.depth_height_var.get())
        depth_fps = int(self.depth_fps_var.get())
        flip_mode = self.realsense_flip_var.get()
        return color_width, color_height, color_fps, depth_width, depth_height, depth_fps, flip_mode

    def start_preview(self) -> None:
        if self.test_worker and self.test_worker.is_alive():
            messagebox.showinfo("Camera busy", "Stop detection before starting preview.")
            return
        if self.capture_worker and self.capture_worker.is_alive():
            messagebox.showinfo("Camera busy", "Stop capture before starting preview.")
            return
        if self.preview_worker and self.preview_worker.is_alive():
            return
        try:
            camera_type = self.capture_camera_type_var.get()
            if camera_type == "CSI":
                sensor_id, width, height, fps, flip_method = self.read_camera_settings()
            else:
                color_width, color_height, color_fps, depth_width, depth_height, depth_fps, flip_mode = (
                    self.read_realsense_settings()
                )
        except Exception as exc:
            messagebox.showerror("Preview configuration error", str(exc))
            return
        self.preview_stop.clear()
        if camera_type == "CSI":
            self.preview_worker = PreviewWorker(
                sensor_id,
                width,
                height,
                fps,
                flip_method,
                self.frame_queue,
                self.log_queue,
                self.preview_stop,
            )
        else:
            self.preview_worker = RealSensePreviewWorker(
                color_width,
                color_height,
                color_fps,
                depth_width,
                depth_height,
                depth_fps,
                flip_mode,
                self.frame_queue,
                self.log_queue,
                self.preview_stop,
            )
        self.preview_worker.start()
        self.preview_button.configure(state=tk.DISABLED)
        self.stop_preview_button.configure(state=tk.NORMAL)
        self.start_button.configure(state=tk.NORMAL)
        self.status_var.set("Preview")
        self.log("Preview requested")

    def stop_preview(self) -> None:
        self.preview_stop.set()
        if self.preview_worker and self.preview_worker.is_alive():
            self.preview_worker.join(timeout=2.0)
        self.preview_button.configure(state=tk.NORMAL)
        self.stop_preview_button.configure(state=tk.DISABLED)
        if not (self.capture_worker and self.capture_worker.is_alive()):
            self.status_var.set("Idle")
        self.log("Preview stop requested")

    def start_capture(self) -> None:
        if self.capture_worker and self.capture_worker.is_alive():
            return
        if self.test_worker and self.test_worker.is_alive():
            messagebox.showinfo("Camera busy", "Stop detection before starting capture.")
            return
        if self.preview_worker and self.preview_worker.is_alive():
            self.stop_preview()
            self.preview_worker.join(timeout=1.5)
        try:
            camera_type = self.capture_camera_type_var.get()
            if camera_type == "CSI":
                sensor_id, width, height, fps, flip_method = self.read_camera_settings()
                session_id = make_session_id(sensor_id)
                cfg = CaptureConfig(
                    sensor_id=sensor_id,
                    width=width,
                    height=height,
                    fps=fps,
                    flip_method=flip_method,
                    duration_sec=float(self.duration_var.get()),
                    sample_fps=float(self.sample_fps_var.get()),
                    session_id=session_id,
                    scene=self.scene_var.get(),
                    lighting=self.lighting_var.get(),
                    background=self.background_var.get(),
                    pose_group=self.pose_group_var.get(),
                    occlusion_level=self.occlusion_var.get(),
                    robot_state=self.robot_state_var.get(),
                    notes=self.notes_var.get(),
                    raw_video_dir=self.base_dir / "data/raw_videos",
                    frames_dir=self.base_dir / "data/frames/raw",
                    frame_manifest=self.base_dir / "data/manifests/frame_manifest.csv",
                    session_manifest=self.base_dir / "data/manifests/session_manifest.csv",
                    record_video=bool(self.record_video_var.get()),
                    jpeg_quality=95,
                )
            else:
                color_width, color_height, color_fps, depth_width, depth_height, depth_fps, flip_mode = (
                    self.read_realsense_settings()
                )
                session_id = make_realsense_session_id()
                cfg = RealSenseCaptureConfig(
                    color_width=color_width,
                    color_height=color_height,
                    color_fps=color_fps,
                    depth_width=depth_width,
                    depth_height=depth_height,
                    depth_fps=depth_fps,
                    flip_mode=flip_mode,
                    duration_sec=float(self.duration_var.get()),
                    sample_fps=float(self.sample_fps_var.get()),
                    session_id=session_id,
                    scene=self.scene_var.get(),
                    lighting=self.lighting_var.get(),
                    background=self.background_var.get(),
                    pose_group=self.pose_group_var.get(),
                    occlusion_level=self.occlusion_var.get(),
                    robot_state=self.robot_state_var.get(),
                    notes=self.notes_var.get(),
                    raw_video_dir=self.base_dir / "data/raw_videos",
                    frames_dir=self.base_dir / "data/frames/raw",
                    depth_dir=self.base_dir / "data/depth/raw",
                    frame_manifest=self.base_dir / "data/manifests/frame_manifest.csv",
                    session_manifest=self.base_dir / "data/manifests/session_manifest.csv",
                    record_video=bool(self.record_video_var.get()),
                    jpeg_quality=95,
                )
        except Exception as exc:
            messagebox.showerror("Capture configuration error", str(exc))
            return

        self.capture_stop.clear()
        if camera_type == "CSI":
            self.capture_worker = CaptureWorker(cfg, self.frame_queue, self.log_queue, self.capture_stop)
        else:
            self.capture_worker = RealSenseCaptureWorker(cfg, self.frame_queue, self.log_queue, self.capture_stop)
        self.capture_worker.start()
        self.start_button.configure(state=tk.DISABLED)
        self.stop_button.configure(state=tk.NORMAL)
        self.preview_button.configure(state=tk.DISABLED)
        self.stop_preview_button.configure(state=tk.DISABLED)
        self.status_var.set("Capture")
        self.log(f"Starting capture session {cfg.session_id}")

    def stop_capture(self) -> None:
        self.capture_stop.set()
        if self.capture_worker and self.capture_worker.is_alive():
            self.capture_worker.join(timeout=2.0)
        self.stop_button.configure(state=tk.DISABLED)
        self.start_button.configure(state=tk.NORMAL)
        self.preview_button.configure(state=tk.NORMAL)
        self.status_var.set("Idle")
        self.log("Stop requested")

    def _poll_queues(self) -> None:
        while True:
            try:
                message = self.log_queue.get_nowait()
            except queue.Empty:
                break
            self.log(message)
            if message.startswith("Stopped") or message.startswith("ERROR"):
                self.start_button.configure(state=tk.NORMAL)
                self.stop_button.configure(state=tk.DISABLED)
                self.preview_button.configure(state=tk.NORMAL)
                self.stop_preview_button.configure(state=tk.DISABLED)
                self.status_var.set("Idle")
            elif message == "Preview stopped":
                self.preview_button.configure(state=tk.NORMAL)
                self.stop_preview_button.configure(state=tk.DISABLED)
                if not (self.capture_worker and self.capture_worker.is_alive()):
                    self.status_var.set("Idle")

        try:
            frame = self.frame_queue.get_nowait()
        except queue.Empty:
            frame = None
        if frame is not None:
            self.show_preview_frame(frame)

        while True:
            try:
                message = self.test_log_queue.get_nowait()
            except queue.Empty:
                break
            self.test_log(message)
            if message.startswith("Detection stopped") or message.startswith("ERROR"):
                self.test_start_button.configure(state=tk.NORMAL)
                self.test_stop_button.configure(state=tk.DISABLED)
                self.status_var.set("Idle")

        try:
            test_frame = self.test_frame_queue.get_nowait()
        except queue.Empty:
            test_frame = None
        if test_frame is not None:
            self.show_test_frame(test_frame)
        self.root.after(50, self._poll_queues)

    def show_preview_frame(self, frame) -> None:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)
        width = max(self.preview_label.winfo_width(), 320)
        height = max(self.preview_label.winfo_height(), 240)
        image.thumbnail((width, height))
        self.preview_photo = ImageTk.PhotoImage(image)
        self.preview_label.configure(image=self.preview_photo)

    def load_label_images(self) -> None:
        source_dir = Path(self.label_source_var.get())
        self.label_images = sorted(
            path
            for path in source_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        self.label_index = 0
        if not self.label_images:
            self.label_status.configure(text=f"No images found in {source_dir}")
            return
        self.load_current_label_image()

    def load_current_label_image(self) -> None:
        if not self.label_images:
            return
        self.label_image_path = self.label_images[self.label_index]
        self.label_image = Image.open(self.label_image_path).convert("RGB")
        self.label_boxes = self.load_existing_boxes(self.label_image_path)
        self.redraw_label_image()
        self.label_status.configure(
            text=f"{self.label_index + 1}/{len(self.label_images)} {self.label_image_path}"
        )

    def load_existing_boxes(self, image_path: Path) -> list[tuple[float, float, float, float]]:
        label_path = Path(self.export_labels_var.get()) / f"{image_path.stem}.txt"
        if not label_path.exists():
            return []
        boxes = []
        for line in label_path.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) != 5:
                continue
            class_id, x_center, y_center, width, height = parts
            if class_id != "0":
                continue
            boxes.append((float(x_center), float(y_center), float(width), float(height)))
        return boxes

    def redraw_label_image(self) -> None:
        self.label_canvas.delete("all")
        if self.label_image is None:
            return
        canvas_w = max(self.label_canvas.winfo_width(), 320)
        canvas_h = max(self.label_canvas.winfo_height(), 240)
        img_w, img_h = self.label_image.size
        scale = min(canvas_w / img_w, canvas_h / img_h)
        draw_w = max(int(img_w * scale), 1)
        draw_h = max(int(img_h * scale), 1)
        offset_x = (canvas_w - draw_w) // 2
        offset_y = (canvas_h - draw_h) // 2
        self.image_scale = scale
        self.image_offset = (offset_x, offset_y)
        display_image = self.label_image.resize((draw_w, draw_h))
        self.label_photo = ImageTk.PhotoImage(display_image)
        self.label_canvas.create_image(offset_x, offset_y, image=self.label_photo, anchor=tk.NW)
        for box in self.label_boxes:
            x1, y1, x2, y2 = self.norm_box_to_canvas(box)
            self.label_canvas.create_rectangle(x1, y1, x2, y2, outline="#00ff66", width=2)
            self.label_canvas.create_text(x1 + 4, y1 + 12, text="stator", fill="#00ff66", anchor=tk.W)

    def norm_box_to_canvas(self, box: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
        x_center, y_center, width, height = box
        img_w, img_h = self.label_image.size
        x1 = (x_center - width / 2.0) * img_w
        y1 = (y_center - height / 2.0) * img_h
        x2 = (x_center + width / 2.0) * img_w
        y2 = (y_center + height / 2.0) * img_h
        offset_x, offset_y = self.image_offset
        return (
            offset_x + x1 * self.image_scale,
            offset_y + y1 * self.image_scale,
            offset_x + x2 * self.image_scale,
            offset_y + y2 * self.image_scale,
        )

    def canvas_to_image_xy(self, x: int, y: int) -> tuple[float, float]:
        offset_x, offset_y = self.image_offset
        img_w, img_h = self.label_image.size
        img_x = min(max((x - offset_x) / self.image_scale, 0), img_w)
        img_y = min(max((y - offset_y) / self.image_scale, 0), img_h)
        return img_x, img_y

    def on_label_press(self, event) -> None:
        if self.label_image is None:
            return
        self.drag_start = (event.x, event.y)
        self.drag_rect = self.label_canvas.create_rectangle(
            event.x,
            event.y,
            event.x,
            event.y,
            outline="#ffcc00",
            width=2,
        )

    def on_label_drag(self, event) -> None:
        if self.drag_start is None or self.drag_rect is None:
            return
        x0, y0 = self.drag_start
        self.label_canvas.coords(self.drag_rect, x0, y0, event.x, event.y)

    def on_label_release(self, event) -> None:
        if self.drag_start is None or self.label_image is None:
            return
        x0, y0 = self.drag_start
        x1, y1 = self.canvas_to_image_xy(x0, y0)
        x2, y2 = self.canvas_to_image_xy(event.x, event.y)
        left, right = sorted((x1, x2))
        top, bottom = sorted((y1, y2))
        img_w, img_h = self.label_image.size
        if right - left >= 3 and bottom - top >= 3:
            x_center = ((left + right) / 2.0) / img_w
            y_center = ((top + bottom) / 2.0) / img_h
            width = (right - left) / img_w
            height = (bottom - top) / img_h
            self.label_boxes.append((x_center, y_center, width, height))
        self.drag_start = None
        self.drag_rect = None
        self.redraw_label_image()

    def prev_label_image(self) -> None:
        if not self.label_images:
            return
        self.label_index = max(self.label_index - 1, 0)
        self.load_current_label_image()

    def next_label_image(self) -> None:
        if not self.label_images:
            return
        self.label_index = min(self.label_index + 1, len(self.label_images) - 1)
        self.load_current_label_image()

    def undo_box(self) -> None:
        if self.label_boxes:
            self.label_boxes.pop()
            self.redraw_label_image()

    def clear_boxes(self) -> None:
        self.label_boxes = []
        self.redraw_label_image()

    def save_label(self) -> None:
        if self.label_image_path is None:
            return
        export_images = Path(self.export_images_var.get())
        export_labels = Path(self.export_labels_var.get())
        export_images.mkdir(parents=True, exist_ok=True)
        export_labels.mkdir(parents=True, exist_ok=True)
        target_image = export_images / self.label_image_path.name
        target_label = export_labels / f"{self.label_image_path.stem}.txt"
        shutil.copy2(self.label_image_path, target_image)
        lines = [
            f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
            for x_center, y_center, width, height in self.label_boxes
        ]
        target_label.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        self.label_status.configure(text=f"Saved {target_label}")

    def save_label_and_next(self) -> None:
        self.save_label()
        self.next_label_image()

    def run_validate(self) -> None:
        self.run_command(
            [
                sys.executable,
                "scripts/check_yolo_labels.py",
                "--images-dir",
                self.dataset_images_var.get(),
                "--labels-dir",
                self.dataset_labels_var.get(),
            ]
        )

    def run_split(self) -> None:
        self.run_command(
            [
                sys.executable,
                "scripts/split_dataset.py",
                "--images-dir",
                self.dataset_images_var.get(),
                "--labels-dir",
                self.dataset_labels_var.get(),
                "--output-dir",
                self.dataset_output_var.get(),
                "--train-ratio",
                str(self.train_ratio_var.get()),
                "--val-ratio",
                str(self.val_ratio_var.get()),
                "--test-ratio",
                str(self.test_ratio_var.get()),
            ]
        )

    def run_augment(self) -> None:
        self.run_command(
            [
                sys.executable,
                "scripts/augment_dataset.py",
                "--dataset-dir",
                self.dataset_output_var.get(),
                "--copies-per-image",
                str(self.augment_copies_var.get()),
            ]
        )

    def train_log(self, message: str) -> None:
        self.train_log_text.insert(tk.END, message + "\n")
        self.train_log_text.see(tk.END)

    def start_training(self) -> None:
        if self.train_process and self.train_process.poll() is None:
            messagebox.showinfo("Training running", "Stop the current training first.")
            return

        dataset_yaml = Path(self.train_dataset_var.get())
        if not dataset_yaml.exists():
            messagebox.showerror("Training config error", f"Dataset yaml not found: {dataset_yaml}")
            return

        self.train_results_csv = Path(self.train_project_var.get()) / self.train_name_var.get() / "results.csv"
        command = [
            "yolo",
            "detect",
            "train",
            f"model={self.train_model_var.get()}",
            f"data={self.train_dataset_var.get()}",
            f"project={self.train_project_var.get()}",
            f"name={self.train_name_var.get()}",
            f"epochs={int(self.train_epochs_var.get())}",
            f"imgsz={int(self.train_imgsz_var.get())}",
            f"batch={int(self.train_batch_var.get())}",
            f"device={self.train_device_var.get()}",
            "pretrained=True",
            "cache=False",
            "workers=4",
            "degrees=10",
            "translate=0.05",
            "scale=0.1",
            "fliplr=0.0",
        ]
        self.train_log("$ " + " ".join(command))
        self.status_var.set("Training")
        self.train_start_button.configure(state=tk.DISABLED)
        self.train_stop_button.configure(state=tk.NORMAL)

        def worker() -> None:
            try:
                self.train_process = subprocess.Popen(
                    command,
                    cwd=self.base_dir,
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    bufsize=1,
                )
                assert self.train_process.stdout is not None
                for line in self.train_process.stdout:
                    self.root.after(0, self.train_log, line.rstrip())
                return_code = self.train_process.wait()
                self.root.after(0, self.train_log, f"exit code: {return_code}")
            except Exception as exc:
                self.root.after(0, self.train_log, f"ERROR: {exc}")
            finally:
                self.root.after(0, self._finish_training)

        self.train_thread = threading.Thread(target=worker, daemon=True)
        self.train_thread.start()
        self.schedule_training_plot_refresh()

    def stop_training(self) -> None:
        if self.train_process and self.train_process.poll() is None:
            self.train_process.terminate()
            self.train_log("Training stop requested")

    def start_export_engine(self) -> None:
        if self.export_process and self.export_process.poll() is None:
            messagebox.showinfo("Export running", "TensorRT export is already running.")
            return
        model_path = Path(self.export_model_var.get())
        if not model_path.exists():
            messagebox.showerror("Export config error", f"PT model not found: {model_path}")
            return
        if model_path.suffix.lower() != ".pt":
            messagebox.showerror("Export config error", "TensorRT export expects a .pt model.")
            return

        command = [
            "yolo",
            "export",
            f"model={self.export_model_var.get()}",
            "format=engine",
            f"imgsz={int(self.export_imgsz_var.get())}",
            f"device={self.export_device_var.get()}",
            "half=True",
            "simplify=True",
        ]
        self.train_log("$ " + " ".join(command))
        self.export_button.configure(state=tk.DISABLED)
        self.status_var.set("Exporting")

        def worker() -> None:
            try:
                self.export_process = subprocess.Popen(
                    command,
                    cwd=self.base_dir,
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    bufsize=1,
                )
                assert self.export_process.stdout is not None
                for line in self.export_process.stdout:
                    self.root.after(0, self.train_log, line.rstrip())
                return_code = self.export_process.wait()
                self.root.after(0, self.train_log, f"export exit code: {return_code}")
                engine_path = model_path.with_suffix(".engine")
                if return_code == 0 and engine_path.exists():
                    self.root.after(0, self.test_model_var.set, str(engine_path))
                    self.root.after(0, self.train_log, f"Engine ready: {engine_path}")
            except Exception as exc:
                self.root.after(0, self.train_log, f"ERROR: {exc}")
            finally:
                self.root.after(0, self._finish_export_engine)

        threading.Thread(target=worker, daemon=True).start()

    def _finish_export_engine(self) -> None:
        self.export_button.configure(state=tk.NORMAL)
        self.status_var.set("Idle")

    def _finish_training(self) -> None:
        self.status_var.set("Idle")
        self.train_start_button.configure(state=tk.NORMAL)
        self.train_stop_button.configure(state=tk.DISABLED)
        self.refresh_training_plot()

    def schedule_training_plot_refresh(self) -> None:
        if self.train_plot_job is not None:
            self.root.after_cancel(self.train_plot_job)
        self.train_plot_job = self.root.after(2000, self.refresh_training_plot)

    def refresh_training_plot(self) -> None:
        results_csv = self.train_results_csv
        if results_csv is None or not results_csv.exists():
            self.draw_empty_plot(self.loss_canvas, "Loss")
            self.draw_empty_plot(self.metric_canvas, "Metrics")
            if self.train_process and self.train_process.poll() is None:
                self.schedule_training_plot_refresh()
            return

        try:
            rows = self.read_results_csv(results_csv)
        except Exception as exc:
            self.train_log(f"Could not read training results: {exc}")
            rows = []

        self.draw_training_plot(
            self.loss_canvas,
            "Loss",
            rows,
            [
                ("train/box_loss", "#2563eb"),
                ("val/box_loss", "#dc2626"),
                ("train/cls_loss", "#16a34a"),
                ("val/cls_loss", "#ca8a04"),
                ("train/dfl_loss", "#7c3aed"),
                ("val/dfl_loss", "#0891b2"),
            ],
        )
        self.draw_training_plot(
            self.metric_canvas,
            "Metrics",
            rows,
            [
                ("metrics/precision(B)", "#2563eb"),
                ("metrics/recall(B)", "#16a34a"),
                ("metrics/mAP50(B)", "#dc2626"),
                ("metrics/mAP50-95(B)", "#7c3aed"),
            ],
        )
        if self.train_process and self.train_process.poll() is None:
            self.schedule_training_plot_refresh()

    def read_results_csv(self, results_csv: Path) -> list[dict[str, float]]:
        rows: list[dict[str, float]] = []
        with results_csv.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                parsed: dict[str, float] = {}
                for key, value in row.items():
                    if key is None or value is None or value == "":
                        continue
                    try:
                        parsed[key.strip()] = float(value)
                    except ValueError:
                        continue
                if parsed:
                    rows.append(parsed)
        return rows

    def draw_empty_plot(self, canvas: tk.Canvas, title: str) -> None:
        canvas.delete("all")
        width = max(canvas.winfo_width(), 300)
        height = max(canvas.winfo_height(), 220)
        canvas.create_text(width // 2, 24, text=title, fill="#111827", font=("DejaVu Sans", 11, "bold"))
        canvas.create_text(width // 2, height // 2, text="Waiting for results.csv", fill="#64748b")

    def draw_training_plot(
        self,
        canvas: tk.Canvas,
        title: str,
        rows: list[dict[str, float]],
        series: list[tuple[str, str]],
    ) -> None:
        canvas.delete("all")
        width = max(canvas.winfo_width(), 300)
        height = max(canvas.winfo_height(), 220)
        pad_left = 54
        pad_right = 18
        pad_top = 42
        pad_bottom = 42
        plot_w = width - pad_left - pad_right
        plot_h = height - pad_top - pad_bottom
        canvas.create_text(width // 2, 18, text=title, fill="#111827", font=("DejaVu Sans", 11, "bold"))

        if not rows:
            canvas.create_text(width // 2, height // 2, text="No data yet", fill="#64748b")
            return

        epoch_key = "epoch" if "epoch" in rows[0] else next(iter(rows[0]))
        available = [(name, color) for name, color in series if any(name in row for row in rows)]
        if not available:
            canvas.create_text(width // 2, height // 2, text="No matching columns", fill="#64748b")
            return

        epochs = [row.get(epoch_key, index) for index, row in enumerate(rows)]
        values = [row[name] for name, _color in available for row in rows if name in row]
        min_x, max_x = min(epochs), max(epochs)
        min_y, max_y = min(values), max(values)
        if max_x == min_x:
            max_x = min_x + 1.0
        if max_y == min_y:
            max_y = min_y + 1.0

        canvas.create_line(pad_left, pad_top, pad_left, height - pad_bottom, fill="#94a3b8")
        canvas.create_line(pad_left, height - pad_bottom, width - pad_right, height - pad_bottom, fill="#94a3b8")
        canvas.create_text(pad_left - 8, pad_top, text=f"{max_y:.3g}", fill="#475569", anchor=tk.E)
        canvas.create_text(pad_left - 8, height - pad_bottom, text=f"{min_y:.3g}", fill="#475569", anchor=tk.E)
        canvas.create_text(pad_left, height - pad_bottom + 18, text=f"{min_x:.0f}", fill="#475569")
        canvas.create_text(width - pad_right, height - pad_bottom + 18, text=f"{max_x:.0f}", fill="#475569")

        legend_x = pad_left + 8
        legend_y = pad_top + 8
        for name, color in available:
            points = []
            for index, row in enumerate(rows):
                if name not in row:
                    continue
                x_value = row.get(epoch_key, index)
                y_value = row[name]
                x = pad_left + (x_value - min_x) / (max_x - min_x) * plot_w
                y = height - pad_bottom - (y_value - min_y) / (max_y - min_y) * plot_h
                points.extend([x, y])
            if len(points) >= 4:
                canvas.create_line(*points, fill=color, width=2)
            canvas.create_line(legend_x, legend_y, legend_x + 16, legend_y, fill=color, width=2)
            canvas.create_text(legend_x + 20, legend_y, text=name, fill="#334155", anchor=tk.W, font=("DejaVu Sans", 8))
            legend_y += 16

    def test_log(self, message: str) -> None:
        self.test_log_text.insert(tk.END, message + "\n")
        self.test_log_text.see(tk.END)

    def start_detection(self) -> None:
        if self.test_worker and self.test_worker.is_alive():
            return
        if self.preview_worker and self.preview_worker.is_alive():
            self.stop_preview()
            self.preview_worker.join(timeout=1.5)
        if self.capture_worker and self.capture_worker.is_alive():
            messagebox.showinfo("Camera busy", "Stop capture before starting detection.")
            return
        try:
            camera_type = self.test_camera_type_var.get()
            model_path = Path(self.test_model_var.get())
            width = int(self.test_width_var.get())
            height = int(self.test_height_var.get())
            fps = int(self.test_fps_var.get())
            conf = float(self.test_conf_var.get())
            if camera_type == "CSI":
                sensor_id = int(self.test_sensor_var.get())
                flip_method = int(self.test_flip_var.get())
                validate_mode(width, height, fps)
            else:
                depth_width = int(self.test_depth_width_var.get())
                depth_height = int(self.test_depth_height_var.get())
                depth_fps = int(self.test_depth_fps_var.get())
                flip_mode = self.test_flip_var.get()
                if flip_mode == "2":
                    flip_mode = "vertical"
        except Exception as exc:
            messagebox.showerror("Detection config error", str(exc))
            return

        self.test_stop.clear()
        if camera_type == "CSI":
            self.test_worker = DetectionWorker(
                model_path=model_path,
                sensor_id=sensor_id,
                width=width,
                height=height,
                fps=fps,
                flip_method=flip_method,
                conf=conf,
                frame_queue=self.test_frame_queue,
                log_queue=self.test_log_queue,
                stop_event=self.test_stop,
            )
        else:
            self.test_worker = RealSenseDetectionWorker(
                model_path=model_path,
                color_width=width,
                color_height=height,
                color_fps=fps,
                depth_width=depth_width,
                depth_height=depth_height,
                depth_fps=depth_fps,
                flip_mode=flip_mode,
                conf=conf,
                frame_queue=self.test_frame_queue,
                log_queue=self.test_log_queue,
                stop_event=self.test_stop,
            )
        self.test_worker.start()
        self.test_start_button.configure(state=tk.DISABLED)
        self.test_stop_button.configure(state=tk.NORMAL)
        self.status_var.set("Detecting")
        self.test_log("Detection requested")

    def stop_detection(self) -> None:
        self.test_stop.set()
        if self.test_worker and self.test_worker.is_alive():
            self.test_worker.join(timeout=2.0)
        self.test_start_button.configure(state=tk.NORMAL)
        self.test_stop_button.configure(state=tk.DISABLED)
        self.status_var.set("Idle")
        self.test_log("Detection stop requested")

    def show_test_frame(self, frame) -> None:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)
        width = max(self.test_preview_label.winfo_width(), 320)
        height = max(self.test_preview_label.winfo_height(), 240)
        image.thumbnail((width, height))
        self.test_photo = ImageTk.PhotoImage(image)
        self.test_preview_label.configure(image=self.test_photo)

    def run_command(self, command: list[str]) -> None:
        self.dataset_log("$ " + " ".join(command))

        def worker() -> None:
            process = subprocess.run(
                command,
                cwd=self.base_dir,
                text=True,
                capture_output=True,
                check=False,
            )
            if process.stdout:
                self.root.after(0, self.dataset_log, process.stdout.rstrip())
            if process.stderr:
                self.root.after(0, self.dataset_log, process.stderr.rstrip())
            self.root.after(0, self.dataset_log, f"exit code: {process.returncode}")

        threading.Thread(target=worker, daemon=True).start()

    def on_close(self) -> None:
        self.preview_stop.set()
        self.capture_stop.set()
        self.test_stop.set()
        if self.train_process and self.train_process.poll() is None:
            self.train_process.terminate()
        if self.export_process and self.export_process.poll() is None:
            self.export_process.terminate()
        self.root.after(200, self.root.destroy)


def main() -> None:
    root = tk.Tk()
    StatorDatasetGui(root)
    root.mainloop()


if __name__ == "__main__":
    main()

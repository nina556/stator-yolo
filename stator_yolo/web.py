from __future__ import annotations

import argparse
import base64
import json
import shutil
import subprocess
import sys
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

from .paths import ensure_project_dirs, repo_root


ROOT = repo_root()
WEB_ROOT = ROOT / "web"
SOURCE_DIR = ROOT / "data" / "frames" / "raw"
EXPORT_IMAGES = ROOT / "data" / "labeling" / "export" / "images"
EXPORT_LABELS = ROOT / "data" / "labeling" / "export" / "labels"
WEB_INFER_RESULT = ROOT / "runs" / "infer" / "web_preview.jpg"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
MIME_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".json": "application/json; charset=utf-8",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".bmp": "image/bmp",
    ".webp": "image/webp",
}
JOB_LOCK = threading.Lock()
JOB: dict[str, object] = {
    "process": None,
    "name": "",
    "status": "idle",
    "log": "",
    "started_at": None,
    "returncode": None,
}


def file_count(path: Path, suffixes: set[str]) -> int:
    if not path.exists():
        return 0
    return sum(1 for item in path.iterdir() if item.is_file() and item.suffix.lower() in suffixes)


def project_status() -> dict[str, object]:
    dataset = ROOT / "data" / "dataset"
    return {
        "export": {
            "images": file_count(EXPORT_IMAGES, IMAGE_SUFFIXES),
            "labels": file_count(EXPORT_LABELS, {".txt"}),
        },
        "dataset": {
            split: {
                "images": file_count(dataset / "images" / split, IMAGE_SUFFIXES),
                "labels": file_count(dataset / "labels" / split, {".txt"}),
            }
            for split in ("train", "val", "test")
        },
    }


def job_snapshot() -> dict[str, object]:
    with JOB_LOCK:
        return {key: value for key, value in JOB.items() if key != "process"}


def run_job(name: str, command: list[str]) -> None:
    with JOB_LOCK:
        process = JOB.get("process")
        if isinstance(process, subprocess.Popen) and process.poll() is None:
            raise ValueError(f"任务“{JOB['name']}”正在运行")
        JOB.update(name=name, status="running", log="", started_at=time.time(), returncode=None)
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        JOB["process"] = process

    def collect() -> None:
        assert process.stdout is not None
        for line in process.stdout:
            with JOB_LOCK:
                JOB["log"] = (str(JOB["log"]) + line)[-100_000:]
        returncode = process.wait()
        with JOB_LOCK:
            JOB["returncode"] = returncode
            JOB["status"] = "completed" if returncode == 0 else "failed"
            JOB["process"] = None

    threading.Thread(target=collect, daemon=True).start()


def safe_name(value: str) -> str:
    name = Path(value).name
    if not name or name != value or Path(name).suffix.lower() not in IMAGE_SUFFIXES:
        raise ValueError("Invalid image filename")
    return name


def list_images() -> list[dict[str, object]]:
    images = []
    for path in sorted(SOURCE_DIR.iterdir()):
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            label_path = EXPORT_LABELS / f"{path.stem}.txt"
            images.append({"name": path.name, "labeled": label_path.exists()})
    return images


class Handler(BaseHTTPRequestHandler):
    server_version = "StatorYoloWeb/1.0"

    def log_message(self, format: str, *args: object) -> None:
        print(f"[web] {self.address_string()} {format % args}")

    def send_bytes(self, data: bytes, content_type: str, status: int = 200) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def send_json(self, payload: object, status: int = 200) -> None:
        self.send_bytes(
            json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            "application/json; charset=utf-8",
            status,
        )

    def read_json(self) -> dict[str, object]:
        length = int(self.headers.get("Content-Length", "0"))
        if length > 50 * 1024 * 1024:
            raise ValueError("Request is too large")
        return json.loads(self.rfile.read(length) or b"{}")

    def do_GET(self) -> None:
        path = unquote(urlparse(self.path).path)
        try:
            if path == "/api/images":
                self.send_json({"images": list_images()})
                return
            if path == "/api/project/status":
                self.send_json(project_status())
                return
            if path == "/api/infer/result":
                if not WEB_INFER_RESULT.exists():
                    self.send_json({"error": "Inference result not found"}, HTTPStatus.NOT_FOUND)
                    return
                self.send_bytes(WEB_INFER_RESULT.read_bytes(), "image/jpeg")
                return
            if path == "/api/job":
                self.send_json(job_snapshot())
                return
            if path.startswith("/api/image/"):
                name = safe_name(path.removeprefix("/api/image/"))
                image = SOURCE_DIR / name
                if not image.exists():
                    self.send_json({"error": "Image not found"}, HTTPStatus.NOT_FOUND)
                    return
                self.send_bytes(image.read_bytes(), MIME_TYPES.get(image.suffix.lower(), "application/octet-stream"))
                return
            if path.startswith("/api/labels/"):
                name = safe_name(path.removeprefix("/api/labels/"))
                label = EXPORT_LABELS / f"{Path(name).stem}.txt"
                boxes = []
                if label.exists():
                    for line in label.read_text(encoding="utf-8").splitlines():
                        parts = line.split()
                        if len(parts) == 5 and parts[0] == "0":
                            boxes.append([float(value) for value in parts[1:]])
                self.send_json({"boxes": boxes})
                return

            static_path = "index.html" if path in {"/", ""} else path.lstrip("/")
            target = (WEB_ROOT / static_path).resolve()
            if WEB_ROOT.resolve() not in target.parents or not target.is_file():
                self.send_json({"error": "Not found"}, HTTPStatus.NOT_FOUND)
                return
            self.send_bytes(target.read_bytes(), MIME_TYPES.get(target.suffix.lower(), "application/octet-stream"))
        except (ValueError, OSError, json.JSONDecodeError) as exc:
            self.send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)

    def do_POST(self) -> None:
        path = unquote(urlparse(self.path).path)
        try:
            payload = self.read_json()
            if path == "/api/upload":
                name = safe_name(str(payload.get("name", "")))
                encoded = str(payload.get("data", ""))
                if "," in encoded:
                    encoded = encoded.split(",", 1)[1]
                data = base64.b64decode(encoded, validate=True)
                if not data:
                    raise ValueError("Uploaded image is empty")
                (SOURCE_DIR / name).write_bytes(data)
                self.send_json({"ok": True, "name": name})
                return
            if path == "/api/save":
                name = safe_name(str(payload.get("name", "")))
                source = SOURCE_DIR / name
                if not source.exists():
                    raise ValueError("Source image does not exist")
                raw_boxes = payload.get("boxes", [])
                if not isinstance(raw_boxes, list):
                    raise ValueError("Boxes must be a list")
                boxes: list[list[float]] = []
                for raw in raw_boxes:
                    if not isinstance(raw, list) or len(raw) != 4:
                        raise ValueError("Each box must contain four numbers")
                    box = [float(value) for value in raw]
                    if any(value < 0 or value > 1 for value in box):
                        raise ValueError("Box values must be between 0 and 1")
                    boxes.append(box)
                shutil.copy2(source, EXPORT_IMAGES / name)
                label = EXPORT_LABELS / f"{source.stem}.txt"
                label.write_text(
                    "".join(f"0 {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n" for x, y, w, h in boxes),
                    encoding="utf-8",
                )
                self.send_json({"ok": True, "label": str(label.relative_to(ROOT))})
                return
            if path == "/api/infer":
                name = safe_name(str(payload.get("name", "")))
                source = SOURCE_DIR / name
                if not source.exists():
                    raise ValueError("Source image does not exist")
                weights = list((ROOT / "runs").glob("**/weights/best.pt"))
                if not weights:
                    raise ValueError("未找到训练好的 best.pt，请先完成训练")
                model = max(weights, key=lambda item: item.stat().st_mtime)
                confidence = float(payload.get("confidence", 0.25))
                if not 0 <= confidence <= 1:
                    raise ValueError("Confidence must be between 0 and 1")
                WEB_INFER_RESULT.parent.mkdir(parents=True, exist_ok=True)
                completed = subprocess.run(
                    [
                        sys.executable,
                        "scripts/infer_image.py",
                        "--model",
                        str(model),
                        "--image",
                        str(source),
                        "--output",
                        str(WEB_INFER_RESULT),
                        "--conf",
                        str(confidence),
                    ],
                    cwd=ROOT,
                    capture_output=True,
                    text=True,
                    timeout=120,
                )
                if completed.returncode != 0:
                    raise ValueError((completed.stderr or completed.stdout or "检测失败")[-2000:])
                self.send_json({
                    "ok": True,
                    "model": str(model.relative_to(ROOT)),
                    "result": f"/api/infer/result?t={time.time_ns()}",
                })
                return
            if path == "/api/delete":
                name = safe_name(str(payload.get("name", "")))
                source = SOURCE_DIR / name
                if not source.exists():
                    raise ValueError("Source image does not exist")
                source.unlink()
                (EXPORT_IMAGES / name).unlink(missing_ok=True)
                (EXPORT_LABELS / f"{Path(name).stem}.txt").unlink(missing_ok=True)
                self.send_json({"ok": True, "name": name})
                return
            if path == "/api/dataset/validate":
                run_job(
                    "检查标签",
                    [
                        sys.executable,
                        "scripts/check_yolo_labels.py",
                        "--images-dir",
                        "data/labeling/export/images",
                        "--labels-dir",
                        "data/labeling/export/labels",
                    ],
                )
                self.send_json({"ok": True})
                return
            if path == "/api/dataset/split":
                run_job(
                    "划分数据集",
                    [
                        sys.executable,
                        "scripts/split_dataset.py",
                        "--images-dir",
                        "data/labeling/export/images",
                        "--labels-dir",
                        "data/labeling/export/labels",
                        "--output-dir",
                        "data/dataset",
                        "--train-ratio",
                        "0.8",
                        "--val-ratio",
                        "0.1",
                        "--test-ratio",
                        "0.1",
                    ],
                )
                self.send_json({"ok": True})
                return
            if path == "/api/dataset/augment":
                copies = int(payload.get("copies", 2))
                if not 1 <= copies <= 10:
                    raise ValueError("增强份数必须在 1 到 10 之间")
                run_job(
                    "增强训练集",
                    [
                        sys.executable,
                        "scripts/augment_dataset.py",
                        "--dataset-dir",
                        "data/dataset",
                        "--copies-per-image",
                        str(copies),
                    ],
                )
                self.send_json({"ok": True})
                return
            if path == "/api/train":
                epochs = int(payload.get("epochs", 100))
                batch = int(payload.get("batch", 8))
                image_size = int(payload.get("image_size", 640))
                if not 1 <= epochs <= 1000 or not 1 <= batch <= 128 or image_size not in {320, 416, 512, 640, 768, 960, 1280}:
                    raise ValueError("训练参数超出允许范围")
                yolo = Path(sys.executable).parent / "yolo"
                if not yolo.exists():
                    raise ValueError("未安装 Ultralytics YOLO")
                run_job(
                    "训练模型",
                    [
                        str(yolo),
                        "detect",
                        "train",
                        "model=yolov8n.pt",
                        "data=data/dataset.yaml",
                        f"project={ROOT / 'runs'}",
                        "name=stator_yolov8",
                        f"epochs={epochs}",
                        f"imgsz={image_size}",
                        f"batch={batch}",
                        "device=0",
                        "pretrained=True",
                        "workers=4",
                    ],
                )
                self.send_json({"ok": True})
                return
            if path == "/api/job/stop":
                with JOB_LOCK:
                    process = JOB.get("process")
                    if isinstance(process, subprocess.Popen) and process.poll() is None:
                        process.terminate()
                        JOB["status"] = "stopping"
                self.send_json({"ok": True})
                return
            self.send_json({"error": "Not found"}, HTTPStatus.NOT_FOUND)
        except (ValueError, OSError, json.JSONDecodeError, base64.binascii.Error) as exc:
            self.send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Stator YOLO browser labeling interface.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    ensure_project_dirs()
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    EXPORT_IMAGES.mkdir(parents=True, exist_ok=True)
    EXPORT_LABELS.mkdir(parents=True, exist_ok=True)
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"Stator YOLO Web: http://{args.host}:{args.port}", flush=True)
    print("Press Ctrl+C to stop the server.", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()

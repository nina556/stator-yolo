from __future__ import annotations

import argparse
import base64
import json
import shutil
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

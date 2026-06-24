from __future__ import annotations

from pathlib import Path


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def ensure_project_dirs(base_dir: Path | None = None) -> None:
    root = base_dir or repo_root()
    for relative in (
        "data/raw_videos",
        "data/frames/raw",
        "data/depth/raw",
        "data/labeling/export/images",
        "data/labeling/export/labels",
        "data/manifests",
        "runs",
    ):
        (root / relative).mkdir(parents=True, exist_ok=True)

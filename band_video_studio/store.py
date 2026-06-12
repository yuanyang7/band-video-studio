"""JSON-file metadata store for videos and their analysis artifacts.

Layout under DATA_DIR:
  videos.json                  index of all registered videos
  <video_id>/proxy.mp4         downsampled playback/analysis proxy
  <video_id>/analysis.json     audio + vision analysis results
  <video_id>/crops.json        per-player crop regions
  <video_id>/settings.json     last-used UI settings (render options, export range…)
  <video_id>/exports/          rendered edits
"""

from __future__ import annotations

import json
import threading
import uuid
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

_lock = threading.Lock()


def _index_path() -> Path:
    return DATA_DIR / "videos.json"


def _load_index() -> dict:
    path = _index_path()
    if path.exists():
        return json.loads(path.read_text())
    return {}


def _save_index(index: dict) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    tmp = _index_path().with_suffix(".tmp")
    tmp.write_text(json.dumps(index, indent=2))
    tmp.replace(_index_path())


def video_dir(video_id: str) -> Path:
    d = DATA_DIR / video_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def add_video(source_path: str, name: str, meta: dict) -> dict:
    video_id = uuid.uuid4().hex[:12]
    record = {
        "id": video_id,
        "name": name,
        "source_path": source_path,
        "meta": meta,
    }
    with _lock:
        index = _load_index()
        index[video_id] = record
        _save_index(index)
    video_dir(video_id)
    return record


def get_video(video_id: str) -> dict | None:
    with _lock:
        return _load_index().get(video_id)


def list_videos() -> list[dict]:
    with _lock:
        return list(_load_index().values())


def load_library() -> dict:
    """Library config: {"folders": [path, ...], "output_dir": path-or-empty}.

    output_dir is where finished exports are copied for easy access (defaults
    to ~/Downloads); empty string means keep them only inside the data dir.
    """
    path = DATA_DIR / "library.json"
    config = json.loads(path.read_text()) if path.exists() else {}
    config.setdefault("folders", [])
    config.setdefault("output_dir", "~/Downloads")
    return config


def save_library(config: dict) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = DATA_DIR / "library.json"
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(config, indent=2))
    tmp.replace(path)


def save_artifact(video_id: str, name: str, data: dict | list) -> None:
    path = video_dir(video_id) / f"{name}.json"
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2))
    tmp.replace(path)


def load_artifact(video_id: str, name: str) -> dict | list | None:
    path = video_dir(video_id) / f"{name}.json"
    if path.exists():
        return json.loads(path.read_text())
    return None

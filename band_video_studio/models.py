"""Download-and-cache for the free local model files (MediaPipe tasks)."""

from __future__ import annotations

import urllib.request
from pathlib import Path

from .store import DATA_DIR

_BASE = "https://storage.googleapis.com/mediapipe-models"

MODELS = {
    "yamnet.tflite": f"{_BASE}/audio_classifier/yamnet/float32/latest/yamnet.tflite",
    "face_landmarker.task": f"{_BASE}/face_landmarker/face_landmarker/float16/latest/face_landmarker.task",
    # small-face detector for wide room shots; the landmarker alone misses small faces
    "yunet.onnx": "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
}


def model_path(name: str) -> Path:
    """Return local path for a model, downloading it on first use."""
    dest = DATA_DIR / "models" / name
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(dest.suffix + ".tmp")
        urllib.request.urlretrieve(MODELS[name], tmp)
        tmp.replace(dest)
    return dest

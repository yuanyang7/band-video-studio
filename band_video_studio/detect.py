"""Fun-moment detection on free local models — no API calls.

Faces in a fixed wide room shot are small, so detection is two-stage:
YuNet (small-face detector) finds face boxes, each box is cropped, upscaled,
and run through the MediaPipe Face Landmarker for smile/jaw blendshapes.
Thresholds are calibrated against real rehearsal footage — blendshape smile
scores on small upscaled faces top out well below selfie-range values.

Strategy (cheap-first):
  1. Audio laugh candidates come from YAMNet (audio.analyze).
  2. Visually confirm/enrich them around each candidate.
  3. Optionally sweep the WHOLE video sparsely (including songs — funny
     reactions to mistakes happen mid-song) for multi-face smile moments.

The optional Claude deep pass (vision.py) only ever sees these pre-filtered
candidate windows, which keeps API cost near zero even for long videos.
"""

from __future__ import annotations

import numpy as np

from . import probe
from .models import model_path

# calibrated on real footage: a clearly laughing face measured smile≈0.45
SMILE_STRONG = 0.30
LAUGH_SMILE = 0.25
LAUGH_JAW = 0.15
FRAME_HEIGHT = probe.PROXY_HEIGHT


class FaceScanner:
    """YuNet face detection + per-face-crop blendshapes."""

    def __init__(self):
        import cv2
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision as mp_vision

        self._cv2 = cv2
        self._mp = mp
        self._detector = cv2.FaceDetectorYN.create(
            str(model_path("yunet.onnx")), "", (0, 0), score_threshold=0.6
        )
        self._landmarker = mp_vision.FaceLandmarker.create_from_options(
            mp_vision.FaceLandmarkerOptions(
                base_options=mp_python.BaseOptions(
                    model_asset_path=str(model_path("face_landmarker.task"))
                ),
                output_face_blendshapes=True,
                num_faces=1,  # one upscaled crop per call
                running_mode=mp_vision.RunningMode.IMAGE,
            )
        )

    def close(self):
        self._landmarker.close()

    def faces_for_frame(self, jpeg: bytes) -> list[dict]:
        """Per-face smile/laugh scores for one JPEG frame."""
        cv2, mp = self._cv2, self._mp
        img = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            return []
        h, w = img.shape[:2]
        self._detector.setInputSize((w, h))
        _, boxes = self._detector.detect(img)
        faces = []
        for box in boxes if boxes is not None else []:
            x, y, bw, bh = (int(v) for v in box[:4])
            margin = int(max(bw, bh) * 0.5)
            crop = img[max(0, y - margin): y + bh + margin, max(0, x - margin): x + bw + margin]
            if crop.size == 0:
                continue
            crop = cv2.resize(crop, (256, max(1, int(256 * crop.shape[0] / crop.shape[1]))))
            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            result = self._landmarker.detect(
                mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
            )
            for blendshapes in result.face_blendshapes:
                scores = {b.category_name: b.score for b in blendshapes}
                smile = (scores.get("mouthSmileLeft", 0) + scores.get("mouthSmileRight", 0)) / 2
                jaw = scores.get("jawOpen", 0)
                faces.append({
                    "smile": round(float(smile), 3),
                    "laughing": bool(smile > LAUGH_SMILE and jaw > LAUGH_JAW),
                })
        return faces


def scan_window(scanner: FaceScanner, source: str, start: float, end: float, interval: float = 1.5) -> dict:
    """Sample frames in a window; return best smile evidence found."""
    best = {"t": start, "smiling_faces": 0, "laughing_faces": 0, "max_smile": 0.0}
    for t, jpeg in probe.sample_frames(source, start, end, interval, height=FRAME_HEIGHT):
        faces = scanner.faces_for_frame(jpeg)
        smiling = sum(1 for f in faces if f["smile"] > SMILE_STRONG)
        laughing = sum(1 for f in faces if f["laughing"])
        max_smile = max((f["smile"] for f in faces), default=0.0)
        if (laughing, smiling, max_smile) > (best["laughing_faces"], best["smiling_faces"], best["max_smile"]):
            best = {"t": round(t, 2), "smiling_faces": smiling,
                    "laughing_faces": laughing, "max_smile": round(max_smile, 3)}
    return best


def view_activity(
    source: str,
    crops: dict[str, dict],
    start: float,
    end: float,
    fps: float = 2.0,
) -> dict[str, tuple[list[float], list[float]]]:
    """Per-view motion track over [start, end): {view: (times, motion)}.

    Measures frame-to-frame change inside each crop box, so the cut list can
    favour whoever is moving/playing now. One ffmpeg decode pass for the range.
    """
    if not crops or end <= start:
        return {}
    times: list[float] = []
    motion: dict[str, list[float]] = {name: [] for name in crops}
    prev = None
    prev_t = start
    for t, frame in probe.read_gray_frames(source, start, end, fps):
        h, w = frame.shape
        if prev is not None:
            diff = np.abs(frame.astype(np.int16) - prev)
            times.append(round((t + prev_t) / 2, 3))
            for name, box in crops.items():
                x0 = int(max(0, min(box["x"], 1)) * w)
                y0 = int(max(0, min(box["y"], 1)) * h)
                x1 = int(min(1.0, box["x"] + box["w"]) * w)
                y1 = int(min(1.0, box["y"] + box["h"]) * h)
                region = diff[y0:max(y0 + 1, y1), x0:max(x0 + 1, x1)]
                motion[name].append(float(region.mean()) if region.size else 0.0)
        prev, prev_t = frame.astype(np.int16), t
    return {name: (times, motion[name]) for name in crops}


def activity_scores(
    tracks: dict[str, tuple[list[float], list[float]]],
) -> dict[str, tuple[list[float], list[float]]]:
    """Z-score each view's own motion track so 'unusually active now' stands out.

    Per-view normalisation cancels a constantly-busy player (e.g. the drummer)
    so the score reflects *relative* activity rather than raw movement.
    """
    scored: dict[str, tuple[list[float], list[float]]] = {}
    for name, (times, motion) in tracks.items():
        arr = np.array(motion, dtype=float)
        if len(arr) == 0:
            scored[name] = (times, [])
            continue
        std = arr.std()
        z = (arr - arr.mean()) / std if std > 1e-6 else np.zeros_like(arr)
        scored[name] = (times, [round(float(v), 3) for v in z])
    return scored


def find_fun_moments(
    source: str,
    audio_result: dict,
    duration: float,
    sweep: bool = True,
    progress=None,
) -> list[dict]:
    """Fuse audio laugh candidates with visual smile confirmation."""
    moments: list[dict] = []
    scanner = FaceScanner()
    try:
        laughs = audio_result.get("laughs", [])
        for i, laugh in enumerate(laughs):
            if progress:
                progress(f"confirming laugh {i + 1}/{len(laughs)}")
            start = max(0.0, laugh["start"] - 2.0)
            end = min(duration, laugh["end"] + 3.0)
            visual = scan_window(scanner, source, start, end)
            score = laugh["score"] * 4 + 0.3 * visual["smiling_faces"] + 0.6 * visual["laughing_faces"]
            moments.append({
                "start": round(start, 2), "end": round(end, 2),
                "type": "laughter",
                "score": round(float(score), 2),
                "evidence": {"audio_laugh": laugh["score"], **visual},
            })

        if sweep:
            # whole video, songs included — mistake reactions happen mid-song
            step = 8.0
            total = int(duration / step) + 1
            t = 0.0
            while t < duration:
                if progress:
                    progress(f"smile sweep {int(t / step) + 1}/{total}")
                window_end = min(t + step, duration)
                covered = any(m["start"] <= t <= m["end"] for m in moments)
                if not covered:
                    visual = scan_window(scanner, source, t, window_end, interval=4.0)
                    if (visual["smiling_faces"] >= 2 or visual["laughing_faces"] >= 1
                            or visual["max_smile"] >= 0.75):
                        moments.append({
                            "start": round(max(0.0, visual["t"] - 3.0), 2),
                            "end": round(min(duration, visual["t"] + 4.0), 2),
                            "type": "smiles",
                            "score": round(0.3 * visual["smiling_faces"] + 0.6 * visual["laughing_faces"], 2),
                            "evidence": visual,
                        })
                t = window_end
    finally:
        scanner.close()

    # merge overlapping/adjacent moments, keeping the best-scoring evidence
    moments.sort(key=lambda m: m["start"])
    merged: list[dict] = []
    for m in moments:
        if merged and m["start"] <= merged[-1]["end"] + 2.0:
            prev = merged[-1]
            prev["end"] = max(prev["end"], m["end"])
            if m["score"] > prev["score"]:
                prev.update(score=m["score"], type=m["type"], evidence=m["evidence"])
        else:
            merged.append(m)
    return merged

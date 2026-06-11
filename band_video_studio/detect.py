"""Fun-moment detection on free local models — no API calls.

Strategy (cheap-first):
  1. Audio laugh candidates come from YAMNet (audio.analyze).
  2. Visually confirm/enrich them: sample frames around each candidate and run
     MediaPipe Face Landmarker blendshapes to count smiling/laughing faces.
  3. Optionally do a sparse smile sweep over non-music (chatting) stretches,
     where fun moments tend to live, without scanning the whole video.

The optional Claude deep pass (vision.py) only ever sees these pre-filtered
candidate windows, which keeps API cost near zero even for long videos.
"""

from __future__ import annotations

import numpy as np

from . import probe
from .models import model_path

SMILE_STRONG = 0.55   # blendshape score that counts as a clear smile
LAUGH_JAW = 0.25      # open jaw + smile reads as laughing


def _landmarker():
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision

    options = mp_vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(model_path("face_landmarker.task"))),
        output_face_blendshapes=True,
        num_faces=8,
        running_mode=mp_vision.RunningMode.IMAGE,
    )
    return mp_vision.FaceLandmarker.create_from_options(options)


def smile_scores_for_frame(landmarker, jpeg: bytes) -> list[dict]:
    """Per-face smile/laugh scores for one JPEG frame."""
    import cv2
    import mediapipe as mp

    arr = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
    if arr is None:
        return []
    rgb = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
    result = landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
    faces = []
    for blendshapes in result.face_blendshapes:
        scores = {b.category_name: b.score for b in blendshapes}
        smile = (scores.get("mouthSmileLeft", 0) + scores.get("mouthSmileRight", 0)) / 2
        jaw = scores.get("jawOpen", 0)
        faces.append({
            "smile": round(float(smile), 3),
            "laughing": bool(smile > SMILE_STRONG * 0.8 and jaw > LAUGH_JAW),
        })
    return faces


def scan_window(landmarker, source: str, start: float, end: float, interval: float = 1.5) -> dict:
    """Sample frames in a window; return best smile evidence found."""
    best = {"t": start, "smiling_faces": 0, "laughing_faces": 0, "max_smile": 0.0}
    for t, jpeg in probe.sample_frames(source, start, end, interval):
        faces = smile_scores_for_frame(landmarker, jpeg)
        smiling = sum(1 for f in faces if f["smile"] > SMILE_STRONG)
        laughing = sum(1 for f in faces if f["laughing"])
        max_smile = max((f["smile"] for f in faces), default=0.0)
        if (laughing, smiling, max_smile) > (best["laughing_faces"], best["smiling_faces"], best["max_smile"]):
            best = {"t": round(t, 2), "smiling_faces": smiling,
                    "laughing_faces": laughing, "max_smile": round(max_smile, 3)}
    return best


def _non_music_stretches(songs: list[dict], duration: float, min_len: float = 20.0) -> list[tuple[float, float]]:
    stretches, cursor = [], 0.0
    for song in songs:
        if song["start"] - cursor >= min_len:
            stretches.append((cursor, song["start"]))
        cursor = song["end"]
    if duration - cursor >= min_len:
        stretches.append((cursor, duration))
    return stretches


def find_fun_moments(
    source: str,
    audio_result: dict,
    duration: float,
    sweep_chat: bool = True,
    progress=None,
) -> list[dict]:
    """Fuse audio laugh candidates with visual smile confirmation."""
    moments: list[dict] = []
    landmarker = _landmarker()
    try:
        laughs = audio_result.get("laughs", [])
        for i, laugh in enumerate(laughs):
            if progress:
                progress(f"confirming laugh {i + 1}/{len(laughs)}")
            start = max(0.0, laugh["start"] - 2.0)
            end = min(duration, laugh["end"] + 3.0)
            visual = scan_window(landmarker, source, start, end)
            score = laugh["score"] + 0.3 * visual["smiling_faces"] + 0.6 * visual["laughing_faces"]
            moments.append({
                "start": round(start, 2), "end": round(end, 2),
                "type": "laughter",
                "score": round(float(score), 2),
                "evidence": {"audio_laugh": laugh["score"], **visual},
            })

        if sweep_chat:
            stretches = _non_music_stretches(audio_result.get("songs", []), duration)
            for j, (s_start, s_end) in enumerate(stretches):
                if progress:
                    progress(f"sweeping chat stretch {j + 1}/{len(stretches)}")
                # sparse: one probe every 8s; promote windows where ≥2 faces smile
                t = s_start
                while t < s_end:
                    window_end = min(t + 8.0, s_end)
                    visual = scan_window(landmarker, source, t, window_end, interval=4.0)
                    if visual["smiling_faces"] >= 2 or visual["laughing_faces"] >= 1:
                        covered = any(m["start"] <= visual["t"] <= m["end"] for m in moments)
                        if not covered:
                            moments.append({
                                "start": round(max(0.0, visual["t"] - 3.0), 2),
                                "end": round(min(duration, visual["t"] + 4.0), 2),
                                "type": "smiles",
                                "score": round(0.3 * visual["smiling_faces"] + 0.6 * visual["laughing_faces"], 2),
                                "evidence": visual,
                            })
                    t = window_end
    finally:
        landmarker.close()

    moments.sort(key=lambda m: m["start"])
    return moments

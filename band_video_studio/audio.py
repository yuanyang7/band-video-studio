"""Audio analysis on free local models — no API calls.

YAMNet (via MediaPipe Audio Classifier) gives per-window scores for music,
speech, and laughter. From those we derive:
  - songs:      sustained music segments (start/end of each rehearsal take)
  - laughs:     laughter events (audio candidates for fun moments)
  - highlights: energy peaks inside songs (solos, big moments)
"""

from __future__ import annotations

import numpy as np

from . import probe
from .models import model_path

WINDOW_S = 0.975  # YAMNet's native window hop
CHUNK_S = 600     # classify in 10-minute chunks to bound memory

_LAUGH_CLASSES = {"Laughter", "Giggle", "Snicker", "Chuckle, chortle", "Belly laugh"}


# ---------------------------------------------------------------- inference

def classify_audio(samples: np.ndarray, sample_rate: int) -> list[dict]:
    """Run YAMNet over the full clip. Returns [{t, music, speech, laugh}] per ~1s window."""
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import audio as mp_audio
    from mediapipe.tasks.python.components import containers

    options = mp_audio.AudioClassifierOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(model_path("yamnet.tflite"))),
        max_results=15,
    )
    windows: list[dict] = []
    chunk_len = int(CHUNK_S * sample_rate)
    with mp_audio.AudioClassifier.create_from_options(options) as classifier:
        for offset in range(0, len(samples), chunk_len):
            chunk = samples[offset:offset + chunk_len]
            if len(chunk) < sample_rate:  # skip sub-second tail
                break
            data = containers.AudioData.create_from_array(chunk.astype(float), sample_rate)
            for result in classifier.classify(data):
                scores = {c.category_name: c.score for c in result.classifications[0].categories}
                windows.append({
                    "t": round(offset / sample_rate + result.timestamp_ms / 1000.0, 3),
                    "music": round(max(scores.get("Music", 0.0),
                                       scores.get("Musical instrument", 0.0)), 4),
                    "speech": round(scores.get("Speech", 0.0), 4),
                    "laugh": round(max((scores.get(c, 0.0) for c in _LAUGH_CLASSES), default=0.0), 4),
                })
    return windows


def rms_energy(samples: np.ndarray, sample_rate: int, hop_s: float = WINDOW_S) -> np.ndarray:
    """RMS (dBFS) per hop, aligned with classify_audio windows."""
    hop = int(hop_s * sample_rate)
    n = len(samples) // hop
    if n == 0:
        return np.array([])
    frames = samples[: n * hop].reshape(n, hop)
    rms = np.sqrt(np.mean(frames**2, axis=1))
    return 20 * np.log10(np.maximum(rms, 1e-8))


# ------------------------------------------------------- pure segmentation
# (numpy-only, separated for unit testing)

def smooth(values: np.ndarray, k: int = 5) -> np.ndarray:
    if len(values) < k or k <= 1:
        return values
    kernel = np.ones(k) / k
    return np.convolve(values, kernel, mode="same")


def segments_from_scores(
    times: np.ndarray,
    scores: np.ndarray,
    on_threshold: float,
    off_threshold: float,
    min_len_s: float,
    merge_gap_s: float,
) -> list[tuple[float, float]]:
    """Hysteresis thresholding -> merged (start, end) segments."""
    segments: list[list[float]] = []
    active = False
    start = 0.0
    for t, s in zip(times, scores):
        if not active and s >= on_threshold:
            active, start = True, float(t)
        elif active and s < off_threshold:
            active = False
            segments.append([start, float(t)])
    if active:
        segments.append([start, float(times[-1]) + WINDOW_S])

    merged: list[list[float]] = []
    for seg in segments:
        if merged and seg[0] - merged[-1][1] <= merge_gap_s:
            merged[-1][1] = seg[1]
        else:
            merged.append(seg)
    return [(s, e) for s, e in merged if e - s >= min_len_s]


def find_highlights(
    times: np.ndarray,
    energy_db: np.ndarray,
    songs: list[tuple[float, float]],
    z_threshold: float = 1.2,
    min_len_s: float = 3.0,
    merge_gap_s: float = 4.0,
) -> list[dict]:
    """Energy peaks relative to each song's own level -> highlight segments."""
    highlights = []
    for start, end in songs:
        mask = (times >= start) & (times < end)
        if mask.sum() < 8:
            continue
        seg_energy = smooth(energy_db[mask], 5)
        mean, std = seg_energy.mean(), seg_energy.std()
        if std < 0.5:
            continue
        z = (seg_energy - mean) / std
        for h_start, h_end in segments_from_scores(
            times[mask], z, z_threshold, z_threshold * 0.6, min_len_s, merge_gap_s
        ):
            peak = float(z[(times[mask] >= h_start) & (times[mask] < h_end)].max())
            highlights.append({
                "start": round(h_start, 2), "end": round(h_end, 2),
                "score": round(peak, 2), "song": [round(start, 2), round(end, 2)],
            })
    return highlights


def find_laughs(
    times: np.ndarray,
    laugh_scores: np.ndarray,
    threshold: float = 0.12,
    merge_gap_s: float = 3.0,
) -> list[dict]:
    events = []
    for start, end in segments_from_scores(
        times, laugh_scores, threshold, threshold * 0.5, 0.5, merge_gap_s
    ):
        mask = (times >= start) & (times < end)
        events.append({
            "start": round(start, 2), "end": round(end, 2),
            "score": round(float(laugh_scores[mask].max()), 3),
        })
    return events


# ---------------------------------------------------------------- pipeline

def analyze(source: str, progress=None) -> dict:
    """Full audio pass: songs, highlights, laugh candidates, raw windows."""
    if progress:
        progress("decoding audio")
    samples = probe.extract_audio(source)
    if len(samples) == 0:
        return {"songs": [], "highlights": [], "laughs": [], "windows": []}

    if progress:
        progress("classifying audio (YAMNet)")
    windows = classify_audio(samples, probe.AUDIO_SR)
    if not windows:
        return {"songs": [], "highlights": [], "laughs": [], "windows": []}

    times = np.array([w["t"] for w in windows])
    music = smooth(np.array([w["music"] for w in windows]), 7)
    laugh = np.array([w["laugh"] for w in windows])
    energy = rms_energy(samples, probe.AUDIO_SR)
    n = min(len(energy), len(times))
    times, music, laugh, energy = times[:n], music[:n], laugh[:n], energy[:n]

    if progress:
        progress("segmenting songs")
    songs = segments_from_scores(
        times, music, on_threshold=0.35, off_threshold=0.2,
        min_len_s=30.0, merge_gap_s=12.0,
    )
    return {
        "songs": [{"start": round(s, 2), "end": round(e, 2)} for s, e in songs],
        "highlights": find_highlights(times, energy, songs),
        "laughs": find_laughs(times, laugh),
        "windows": windows,
    }

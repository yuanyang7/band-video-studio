"""Audio analysis on free local models — no API calls.

YAMNet (via MediaPipe Audio Classifier) gives per-window scores for music,
speech, and laughter. From those we derive:
  - songs:      sustained music segments (start/end of each rehearsal take)
  - laughs:     laughter events (audio candidates for fun moments)
  - highlights: energy peaks inside songs (solos, big moments)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from . import probe
from .models import model_path

WINDOW_S = 0.975  # YAMNet's native window hop
CHUNK_S = 600     # classify in 10-minute chunks to bound memory

_LAUGH_CLASSES = {"Laughter", "Giggle", "Snicker", "Chuckle, chortle", "Belly laugh"}
_VOCAL_CLASSES = {
    "Singing", "Male singing", "Female singing", "Child singing",
    "Choir", "Synthetic singing", "Rapping", "Humming", "Yodeling", "Chant",
    "Vocal music", "A capella", "Lullaby", "Mantra",
}


# ---------------------------------------------------------------- inference

def classify_audio(samples: np.ndarray, sample_rate: int) -> list[dict]:
    """Run YAMNet over the full clip. Returns [{t, music, speech, laugh}] per ~1s window."""
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import audio as mp_audio
    from mediapipe.tasks.python.components import containers

    options = mp_audio.AudioClassifierOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(model_path("yamnet.tflite"))),
        max_results=30,
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
                    "vocals": round(max((scores.get(c, 0.0) for c in _VOCAL_CLASSES), default=0.0), 4),
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
                "kind": "peak",
            })
    return highlights


def find_instrumental_sections(
    times: np.ndarray,
    vocals: np.ndarray,
    music: np.ndarray,
    songs: list[tuple[float, float]],
    vocals_off: float = 0.12,
    vocals_on: float = 0.22,
    music_floor: float = 0.15,
    min_len_s: float = 8.0,
    merge_gap_s: float = 4.0,
    max_song_fraction: float = 0.85,
) -> list[dict]:
    """Sustained stretches inside a song where music continues but vocals drop out."""
    sections: list[dict] = []
    for start, end in songs:
        mask = (times >= start) & (times < end)
        if mask.sum() < 8:
            continue
        seg_times = times[mask]
        seg_vocals = smooth(vocals[mask], 5)
        seg_music = smooth(music[mask], 5)

        raw: list[list[float]] = []
        active = False
        seg_start = 0.0
        for t, v, m in zip(seg_times, seg_vocals, seg_music):
            music_ok = m >= music_floor
            if not active and music_ok and v < vocals_off:
                active, seg_start = True, float(t)
            elif active and (v > vocals_on or not music_ok):
                active = False
                raw.append([seg_start, float(t)])
        if active:
            raw.append([seg_start, float(seg_times[-1]) + WINDOW_S])

        merged: list[list[float]] = []
        for seg in raw:
            if merged and seg[0] - merged[-1][1] <= merge_gap_s:
                merged[-1][1] = seg[1]
            else:
                merged.append(seg)

        song_len = max(1e-3, end - start)
        for h_start, h_end in merged:
            length = h_end - h_start
            if length < min_len_s:
                continue
            if length / song_len > max_song_fraction:
                continue
            seg_mask = (seg_times >= h_start) & (seg_times < h_end)
            sections.append({
                "start": round(h_start, 2), "end": round(h_end, 2),
                "score": round(float(seg_vocals[seg_mask].mean()), 3),
                "song": [round(start, 2), round(end, 2)],
                "kind": "instrumental",
            })
    return sections


# ------------------------------------------------------------ beat tracking
# Cuts feel musical when they land on the beat. We derive a beat grid from the
# audio with numpy only (no librosa): spectral-flux onset envelope -> tempo via
# autocorrelation -> phase-aligned beat times. Good enough to snap cut points to.

ONSET_HOP = 512        # samples per onset frame (~32 ms at 16 kHz)
ONSET_WIN = 1024       # analysis window for the onset STFT
TEMPO_MIN, TEMPO_MAX = 60.0, 180.0  # plausible rehearsal tempo range (BPM)


def onset_envelope(samples: np.ndarray, sample_rate: int) -> tuple[np.ndarray, np.ndarray]:
    """Spectral-flux onset strength per ~32 ms frame. Returns (times, envelope)."""
    if len(samples) < ONSET_WIN:
        return np.array([]), np.array([])
    n = 1 + (len(samples) - ONSET_WIN) // ONSET_HOP
    window = np.hanning(ONSET_WIN).astype(np.float32)
    frames = np.empty((n, ONSET_WIN), dtype=np.float32)
    for i in range(n):
        start = i * ONSET_HOP
        frames[i] = samples[start:start + ONSET_WIN] * window
    mag = np.abs(np.fft.rfft(frames, axis=1))
    flux = np.diff(mag, axis=0)
    flux[flux < 0] = 0.0  # half-wave rectify: onsets are energy *increases*
    env = flux.sum(axis=1)
    env = np.concatenate([[0.0], env])  # align length with frame count
    times = (np.arange(n) * ONSET_HOP + ONSET_WIN / 2) / sample_rate
    return times, env


def estimate_tempo(env: np.ndarray, hop_s: float) -> float:
    """Dominant tempo (BPM) of an onset envelope via autocorrelation, 0 if unclear."""
    if len(env) < 8:
        return 0.0
    e = env - env.mean()
    ac = np.correlate(e, e, mode="full")[len(e) - 1:]  # nonnegative lags
    lag_min = max(1, int(round(60.0 / TEMPO_MAX / hop_s)))
    lag_max = min(len(ac) - 1, int(round(60.0 / TEMPO_MIN / hop_s)))
    if lag_max <= lag_min:
        return 0.0
    band = ac[lag_min:lag_max + 1]
    if band.max() <= 0:
        return 0.0
    best_lag = lag_min + int(np.argmax(band))
    # octave correction: an integer 2-beat lag often aliases above the true beat
    # lag (which falls between bins). Prefer the faster tempo when its halved lag
    # is still a strong peak, so 60 -> 120 rather than reporting half-time.
    for _ in range(2):
        half = int(round(best_lag / 2))
        if half >= lag_min and ac[half] >= 0.5 * ac[best_lag]:
            best_lag = half
        else:
            break
    return 60.0 / (best_lag * hop_s)


def beat_grid(
    times: np.ndarray,
    env: np.ndarray,
    period_s: float,
    start: float,
    end: float,
) -> list[float]:
    """Phase-align a pulse train of spacing period_s to the envelope over [start, end)."""
    if period_s <= 0 or len(times) == 0 or end <= start:
        return []
    # try a handful of phase offsets within one beat, keep the one whose grid
    # points sample the most onset energy
    best_phase, best_score = 0.0, -1.0
    for frac in np.linspace(0, 1, 12, endpoint=False):
        phase = start + frac * period_s
        grid = np.arange(phase, end, period_s)
        if len(grid) == 0:
            continue
        score = float(np.interp(grid, times, env, left=0.0, right=0.0).sum())
        if score > best_score:
            best_phase, best_score = phase, score
    beats = np.arange(best_phase, end, period_s)
    return [round(float(b), 3) for b in beats if b >= start]


def detect_beats(
    samples: np.ndarray,
    sample_rate: int,
    songs: list[tuple[float, float]],
) -> tuple[list[float], float]:
    """Beat timestamps across all song spans plus the estimated tempo (BPM)."""
    times, env = onset_envelope(samples, sample_rate)
    if len(env) == 0:
        return [], 0.0
    hop_s = ONSET_HOP / sample_rate
    tempo = estimate_tempo(env, hop_s)
    if tempo <= 0:
        return [], 0.0
    period = 60.0 / tempo
    beats: list[float] = []
    spans = songs or [(float(times[0]), float(times[-1]) + hop_s)]
    for start, end in spans:
        beats.extend(beat_grid(times, env, period, start, end))
    return beats, round(tempo, 1)


def find_vocal_segments(
    times: np.ndarray,
    vocals: np.ndarray,
    on_threshold: float = 0.04,
    off_threshold: float = 0.02,
    min_len_s: float = 2.0,
    merge_gap_s: float = 3.0,
) -> list[dict]:
    """Stretches where someone is singing. Segment starts double as 'the singer
    just came in' cues for the auto edit to cut to the singer view."""
    segs = segments_from_scores(
        times, smooth(vocals, 5), on_threshold, off_threshold, min_len_s, merge_gap_s
    )
    return [{"start": round(s, 2), "end": round(e, 2)} for s, e in segs]


def find_stem_segments(
    times: np.ndarray,
    energies: np.ndarray,
    on_threshold: float = 0.08,
    off_threshold: float = 0.03,
    min_len_s: float = 2.0,
    merge_gap_s: float = 3.0,
) -> list[dict]:
    """Active stretches for an instrument stem (drums, bass, etc.)."""
    segs = segments_from_scores(
        times, smooth(energies, 5), on_threshold, off_threshold, min_len_s, merge_gap_s
    )
    return [{"start": round(s, 2), "end": round(e, 2)} for s, e in segs]


_TABS_GEN_DIR = Path(__file__).resolve().parent.parent.parent / "tabs-gen"
_DEMUCS_EXE = _TABS_GEN_DIR / ".venv" / "bin" / "demucs"


DEMUCS_STEMS = ("vocals", "drums", "bass", "other")


def _rms_track(
    samples: np.ndarray, sr: int, hop_s: float, time_offset: float,
) -> tuple[list[float], list[float]]:
    """RMS energy track (0..1 scale) aligned to ~1s windows."""
    hop = int(hop_s * sr)
    n = len(samples) // hop
    if n == 0:
        return [], []
    frames = samples[: n * hop].reshape(n, hop)
    rms = np.sqrt(np.mean(frames**2, axis=1))
    peak = rms.max()
    if peak > 0:
        rms = rms / peak
    times = [round(time_offset + i * hop_s, 3) for i in range(n)]
    energies = [round(float(v), 4) for v in rms]
    return times, energies


def demucs_all_stems(
    audio_path: str,
    time_offset: float = 0.0,
    hop_s: float = WINDOW_S,
) -> dict[str, tuple[list[float], list[float]]]:
    """Full 4-stem Demucs separation returning energy tracks per stem.

    Returns {stem_name: (times, energies)} for vocals, drums, bass, other.
    """
    import subprocess
    import tempfile

    if not _DEMUCS_EXE.exists():
        raise RuntimeError(f"demucs not found at {_DEMUCS_EXE}")

    audio_path = str(audio_path)
    with tempfile.TemporaryDirectory() as tmp:
        cmd = [
            str(_DEMUCS_EXE),
            "-n", "htdemucs",
            "-d", "mps",
            "--out", tmp,
            audio_path,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"demucs failed: {result.stderr[-500:]}")

        model_dir = Path(tmp) / "htdemucs"
        stem_dirs = list(model_dir.iterdir()) if model_dir.exists() else []
        if not stem_dirs:
            return {}
        stem_dir = stem_dirs[0]

        tracks: dict[str, tuple[list[float], list[float]]] = {}
        for stem in DEMUCS_STEMS:
            wav = stem_dir / f"{stem}.wav"
            if not wav.exists():
                continue
            samples = probe.extract_audio(str(wav))
            tracks[stem] = _rms_track(samples, probe.AUDIO_SR, hop_s, time_offset)
    return tracks


def demucs_vocal_track(
    audio_path: str,
    time_offset: float = 0.0,
    hop_s: float = WINDOW_S,
) -> tuple[list[float], list[float]]:
    """Separate vocals with Demucs and return an energy-based vocal track.

    Convenience wrapper around demucs_all_stems for backwards compat.
    """
    tracks = demucs_all_stems(audio_path, time_offset, hop_s)
    return tracks.get("vocals", ([], []))


def find_laughs(
    times: np.ndarray,
    laugh_scores: np.ndarray,
    threshold: float = 0.04,  # calibrated: distant-mic laughter under chatter peaks ~0.06
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
        progress("decoding audio", pct=0)
    samples = probe.extract_audio(source)
    if len(samples) == 0:
        return {"songs": [], "highlights": [], "laughs": [], "vocal_segments": [],
                "beats": [], "tempo": 0.0, "windows": []}

    if progress:
        progress("classifying audio (YAMNet)", pct=20)
    windows = classify_audio(samples, probe.AUDIO_SR)
    if not windows:
        return {"songs": [], "highlights": [], "laughs": [], "vocal_segments": [],
                "beats": [], "tempo": 0.0, "windows": []}

    times = np.array([w["t"] for w in windows])
    music = smooth(np.array([w["music"] for w in windows]), 7)
    laugh = np.array([w["laugh"] for w in windows])
    vocals = np.array([w.get("vocals", 0.0) for w in windows])
    energy = rms_energy(samples, probe.AUDIO_SR)
    n = min(len(energy), len(times))
    times, music, laugh, vocals, energy = times[:n], music[:n], laugh[:n], vocals[:n], energy[:n]

    if progress:
        progress("segmenting songs", pct=60)
    songs = segments_from_scores(
        times, music, on_threshold=0.35, off_threshold=0.2,
        min_len_s=30.0, merge_gap_s=12.0,
    )
    if progress:
        progress("tracking beats", pct=80)
    beats, tempo = detect_beats(samples, probe.AUDIO_SR, songs)
    return {
        "songs": [{"start": round(s, 2), "end": round(e, 2)} for s, e in songs],
        "highlights": (
            find_highlights(times, energy, songs)
            + find_instrumental_sections(times, vocals, music, songs)
        ),
        "laughs": find_laughs(times, laugh),
        "vocal_segments": find_vocal_segments(times, vocals),
        "beats": beats,
        "tempo": tempo,
        "windows": windows,
    }

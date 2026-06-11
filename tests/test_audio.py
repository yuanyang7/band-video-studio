import numpy as np

from band_video_studio.audio import (
    ONSET_HOP,
    beat_grid,
    estimate_tempo,
    find_highlights,
    find_laughs,
    onset_envelope,
    segments_from_scores,
)


def _click_track(bpm: float, sr: int = 16000, duration: float = 20.0) -> np.ndarray:
    """Silence with a short noise burst every beat at the given tempo."""
    n = int(sr * duration)
    sig = np.zeros(n, dtype=np.float32)
    period = int(sr * 60.0 / bpm)
    burst = int(sr * 0.02)
    rng = np.random.default_rng(0)
    for start in range(0, n - burst, period):
        sig[start:start + burst] = rng.standard_normal(burst).astype(np.float32)
    return sig


def test_estimate_tempo_recovers_bpm():
    sr = 16000
    sig = _click_track(120.0, sr=sr)
    times, env = onset_envelope(sig, sr)
    bpm = estimate_tempo(env, ONSET_HOP / sr)
    assert abs(bpm - 120.0) < 6.0  # within a few BPM


def test_beat_grid_lands_on_clicks():
    sr = 16000
    bpm = 100.0
    sig = _click_track(bpm, sr=sr, duration=10.0)
    times, env = onset_envelope(sig, sr)
    beats = beat_grid(times, env, 60.0 / bpm, 0.0, 10.0)
    assert len(beats) >= 8
    # consecutive beats are one period apart
    diffs = np.diff(beats)
    assert np.allclose(diffs, 60.0 / bpm, atol=0.05)


def test_segments_basic_hysteresis():
    times = np.arange(0, 100, 1.0)
    scores = np.zeros(100)
    scores[10:60] = 0.8  # one 50s "song"
    segs = segments_from_scores(times, scores, 0.35, 0.2, min_len_s=30, merge_gap_s=12)
    assert len(segs) == 1
    start, end = segs[0]
    assert 9 <= start <= 11 and 59 <= end <= 61


def test_segments_merge_short_gap_and_drop_short():
    times = np.arange(0, 200, 1.0)
    scores = np.zeros(200)
    scores[10:40] = 0.9
    scores[45:80] = 0.9   # 5s gap -> merged
    scores[150:160] = 0.9  # 10s blip -> dropped (min_len 30)
    segs = segments_from_scores(times, scores, 0.35, 0.2, min_len_s=30, merge_gap_s=12)
    assert len(segs) == 1
    assert segs[0][0] < 11 and segs[0][1] > 79


def test_find_laughs():
    times = np.arange(0, 60, 1.0)
    laugh = np.zeros(60)
    laugh[20:23] = 0.4
    events = find_laughs(times, laugh)
    assert len(events) == 1
    assert events[0]["score"] == 0.4
    assert 19 <= events[0]["start"] <= 21


def test_find_highlights_peak_inside_song():
    times = np.arange(0, 120, 1.0)
    energy = np.full(120, -30.0) + np.random.default_rng(0).normal(0, 0.5, 120)
    energy[50:58] = -18.0  # loud solo
    songs = [(10.0, 110.0)]
    highlights = find_highlights(times, energy, songs)
    assert len(highlights) >= 1
    h = highlights[0]
    assert 44 <= h["start"] <= 52 and h["end"] >= 55

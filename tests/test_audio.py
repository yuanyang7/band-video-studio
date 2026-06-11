import numpy as np

from band_video_studio.audio import find_highlights, find_laughs, segments_from_scores


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

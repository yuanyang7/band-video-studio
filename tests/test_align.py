import numpy as np

from band_video_studio.align import cross_correlate_offset


def test_recovers_known_offset():
    rng = np.random.default_rng(0)
    short = rng.standard_normal(200)
    long = rng.standard_normal(2000)
    offset = 750
    long[offset:offset + 200] = short  # plant the snippet at a known lag
    hop_s = 0.032
    found, conf = cross_correlate_offset(long, short, hop_s)
    assert abs(found - offset * hop_s) < hop_s  # within one hop
    assert conf > 0.9                            # near-perfect match


def test_offset_zero_at_start():
    rng = np.random.default_rng(1)
    short = rng.standard_normal(300)
    long = np.concatenate([short, rng.standard_normal(1700)])
    found, conf = cross_correlate_offset(long, short, 0.032)
    assert found < 0.05
    assert conf > 0.9


def test_weak_match_low_confidence():
    rng = np.random.default_rng(2)
    short = rng.standard_normal(200)
    long = rng.standard_normal(2000)  # unrelated noise — no real placement
    _, conf = cross_correlate_offset(long, short, 0.032)
    assert conf < 0.5


def test_short_longer_than_long_is_safe():
    found, conf = cross_correlate_offset(np.zeros(50), np.zeros(100), 0.032)
    assert found == 0.0 and conf == 0.0

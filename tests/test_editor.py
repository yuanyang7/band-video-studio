from band_video_studio.editor import (
    build_cutlist,
    build_smart_cutlist,
    drift_filter,
    fit_crop,
    is_pano,
    pano_sweep,
    slide_filter,
    transition_windows,
)


def test_cutlist_contiguous_and_no_repeat():
    cuts = build_cutlist(["drums", "bass", "keys"], 10.0, 50.0, switch_s=4.0, seed=7)
    assert cuts[0]["start"] == 10.0
    assert cuts[-1]["end"] == 50.0
    for a, b in zip(cuts, cuts[1:]):
        assert a["end"] == b["start"]
        assert a["view"] != b["view"]


def test_cutlist_single_view_allows_repeat():
    cuts = build_cutlist(["wide"], 0.0, 10.0, switch_s=4.0, seed=1)
    assert all(c["view"] == "wide" for c in cuts)
    assert cuts[-1]["end"] == 10.0


def test_fit_crop_expands_to_aspect_and_clamps():
    # tall narrow box on a 4K frame, horizontal target
    x, y, w, h = fit_crop({"x": 0.1, "y": 0.1, "w": 0.1, "h": 0.5}, 3840, 2160, 16 / 9)
    assert abs(w / h - 16 / 9) < 0.02
    assert x >= 0 and y >= 0 and x + w <= 3840 and y + h <= 2160
    assert w % 2 == 0 and h % 2 == 0


def test_fit_crop_box_near_edge_stays_inside():
    x, y, w, h = fit_crop({"x": 0.85, "y": 0.8, "w": 0.15, "h": 0.2}, 1920, 1080, 9 / 16)
    assert x + w <= 1920 and y + h <= 1080
    assert abs(w / h - 9 / 16) < 0.02


# --------------------------------------------------------- smart cut list

def test_smart_cutlist_contiguous_no_repeat():
    cuts = build_smart_cutlist(["drums", "bass", "keys"], 10.0, 50.0, switch_s=4.0, seed=3)
    assert cuts[0]["start"] == 10.0
    assert cuts[-1]["end"] == 50.0
    for a, b in zip(cuts, cuts[1:]):
        assert a["end"] == b["start"]
        assert a["view"] != b["view"]


def test_smart_cutlist_snaps_boundaries_to_beats():
    # beats every 0.5s; with switch_s=4 boundaries should land exactly on a beat
    beats = [round(0.5 * i, 3) for i in range(120)]
    cuts = build_smart_cutlist(["a", "b"], 0.0, 40.0, switch_s=4.0, beats=beats, seed=1)
    for c in cuts[:-1]:  # last end is the hard range end
        assert any(abs(c["end"] - b) < 1e-6 for b in beats)
    # segments stay within the even [0.6x, 1.8x] switch band
    for c in cuts:
        assert 0.6 * 4.0 - 1e-6 <= c["end"] - c["start"] <= 1.8 * 4.0 + 1e-6 or c["end"] == 40.0


def test_smart_cutlist_routes_fun_to_pano():
    events = [{"start": 0.0, "end": 60.0, "type": "fun"}]
    cuts = build_smart_cutlist(
        ["drums", "bass", "wide"], 0.0, 60.0, switch_s=4.0,
        events=events, pano_views={"wide"}, w_audio=10.0, seed=2,
    )
    # with a strong fun boost the pano view should dominate screen time
    pano_share = sum(c["end"] - c["start"] for c in cuts if c["view"] == "wide")
    assert pano_share > 0.5 * (cuts[-1]["end"] - cuts[0]["start"])


def test_smart_cutlist_spreads_views_evenly():
    cuts = build_smart_cutlist(["a", "b", "c", "d"], 0.0, 120.0, switch_s=4.0, seed=5)
    share = {}
    for c in cuts:
        share[c["view"]] = share.get(c["view"], 0.0) + c["end"] - c["start"]
    assert set(share) == {"a", "b", "c", "d"}  # every view used
    assert max(share.values()) < 2.0 * min(share.values())  # roughly balanced


def test_smart_cutlist_prefers_active_view_on_peak():
    # view "solo" is far more active during a peak; it should win that segment
    activity = {
        "solo": ([5.0], [3.0]),
        "rhythm": ([5.0], [-1.0]),
    }
    events = [{"start": 0.0, "end": 10.0, "type": "peak"}]
    cuts = build_smart_cutlist(
        ["solo", "rhythm"], 0.0, 8.0, switch_s=4.0,
        activity=activity, events=events, w_audio=5.0, seed=1,
    )
    assert cuts[0]["view"] == "solo"


# ------------------------------------------------------- camera movement

def test_is_pano_detects_wide_box():
    assert is_pano({"x": 0, "y": 0, "w": 0.9, "h": 0.2}, 16 / 9)  # very wide
    assert not is_pano({"x": 0, "y": 0, "w": 0.2, "h": 0.4}, 16 / 9)


def test_transition_windows_same_size_endpoints():
    prev = (1200, 200, 400, 712)
    crop = (100, 150, 480, 854)
    start, end = transition_windows(prev, crop, 1920, 1080)
    assert start[2:] == end[2:] == (crop[2], crop[3])  # constant window size
    assert end == crop
    for w in (start, end):  # inside the frame, even-valued
        assert 0 <= w[0] and w[0] + w[2] <= 1920
        assert 0 <= w[1] and w[1] + w[3] <= 1080
        assert w[0] % 2 == 0 and w[1] % 2 == 0


def test_pano_sweep_travels_horizontally():
    ends = pano_sweep((0, 200, 1900, 400), 16 / 9, 1920, 1080, seed=1)
    assert ends is not None
    (sx, _, sw, _), (ex, _, ew, _) = ends
    assert sw == ew  # constant window
    assert sx != ex  # actually moves


def test_pano_sweep_none_when_too_tight():
    # window wider than the pano box -> no room to travel
    assert pano_sweep((0, 0, 200, 1000), 16 / 9, 1920, 1080) is None


def test_slide_filter_uses_time_and_endpoints():
    f = slide_filter((100, 50, 480, 854), (600, 50, 480, 854), 1080, 1920, 1.5, 30.0)
    assert "crop=480:854" in f and "scale=1080:1920" in f
    assert "min(1,t/1.5000)" in f  # time-driven interpolation
    assert "(100+(500)" in f       # x animates 100 -> 600


def test_drift_filter_pushes_in_via_zoompan():
    f = drift_filter((0, 0, 960, 540), n_frames=90, out_w=1920, out_h=1080, fps=30.0, amount=0.06)
    assert "zoompan" in f and "s=1920x1080" in f and "fps=30.0" in f
    assert "1+0.0600" in f  # zoom grows from 1.0
    # zoompan emits frames on the input timebase; PTS must be re-stamped onto a
    # clean 1/fps grid or the muxed duration explodes ~500-1000x (frozen output)
    assert f.rstrip().endswith("setpts=N/30.0/TB")

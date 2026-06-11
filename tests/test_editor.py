from band_video_studio.editor import build_cutlist, fit_crop


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

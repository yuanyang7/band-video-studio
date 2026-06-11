from band_video_studio.detect import activity_scores


def test_activity_scores_zscore_peaks():
    # "solo" spikes at one moment, "steady" is constant
    tracks = {
        "solo": ([1.0, 2.0, 3.0, 4.0], [1.0, 1.0, 9.0, 1.0]),
        "steady": ([1.0, 2.0, 3.0, 4.0], [5.0, 5.0, 5.0, 5.0]),
    }
    scored = activity_scores(tracks)
    # the spike frame is well above the view's own mean
    assert scored["solo"][1][2] > 1.0
    assert scored["solo"][1][0] < 0.0
    # a flat track yields all-zero z-scores (no false "activity")
    assert all(abs(v) < 1e-6 for v in scored["steady"][1])


def test_activity_scores_handles_empty():
    assert activity_scores({"v": ([], [])})["v"] == ([], [])

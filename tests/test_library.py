from pathlib import Path

from band_video_studio.library import (
    find_new,
    scan_folders,
    top_expressions,
    top_fun_moments,
)


def _touch(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")
    return path


def test_scan_folders_finds_videos_recursively(tmp_path):
    a = _touch(tmp_path / "a.MP4")
    b = _touch(tmp_path / "sub" / "b.mov")
    _touch(tmp_path / "notes.txt")
    _touch(tmp_path / ".hidden" / "c.mp4")
    found = scan_folders([str(tmp_path)])
    assert set(found) == {a, b}


def test_scan_folders_skips_missing_folder(tmp_path):
    assert scan_folders([str(tmp_path / "nope")]) == []


def test_find_new_dedupes_by_resolved_path(tmp_path):
    a = _touch(tmp_path / "a.mp4")
    b = _touch(tmp_path / "b.mp4")
    new = find_new([a, b], {str(a)})
    assert new == [b]


def _items():
    v1 = {"id": "v1", "name": "monday.mp4"}
    v2 = {"id": "v2", "name": "friday.mp4"}
    a1 = {"fun_moments": [
        {"start": 10, "end": 15, "score": 2.0, "type": "laughter",
         "evidence": {"max_smile": 0.5}},
        {"start": 40, "end": 44, "score": 0.5, "type": "smiles",
         "evidence": {"max_smile": 0.9}},
    ]}
    a2 = {"fun_moments": [
        {"start": 5, "end": 9, "score": 3.0, "type": "laughter",
         "evidence": {"max_smile": 0.7}},
    ]}
    return [(v1, a1), (v2, a2)]


def test_top_fun_moments_ranks_across_videos():
    top = top_fun_moments(_items(), limit=2)
    assert [(m["video_id"], m["score"]) for m in top] == [("v2", 3.0), ("v1", 2.0)]
    assert top[0]["video_name"] == "friday.mp4"


def test_top_expressions_ranks_by_max_smile():
    top = top_expressions(_items(), limit=10)
    assert [m["max_smile"] for m in top] == [0.9, 0.7, 0.5]
    # a moment without smile evidence is skipped
    items = [({"id": "v", "name": "n"}, {"fun_moments": [{"start": 0, "end": 1, "score": 1}]})]
    assert top_expressions(items) == []

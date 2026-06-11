from band_video_studio.lyrics import align_lyrics


def _words(text: str, t0: float = 0.0, step: float = 0.4):
    return [{"t": round(t0 + i * step, 2), "word": w} for i, w in enumerate(text.split())]


def test_align_exact_lines_in_order():
    transcript = _words("hello darkness my old friend I've come to talk with you again")
    lyrics = "Hello darkness my old friend\nI've come to talk with you again"
    lines = align_lyrics(lyrics, transcript)
    assert lines[0]["start"] == 0.0
    assert lines[1]["start"] is not None
    assert lines[1]["start"] > lines[0]["start"]
    assert all(line["match"] > 0.8 for line in lines)


def test_align_unmatched_line_gets_none():
    transcript = _words("totally unrelated chatter about pizza and cables")
    lines = align_lyrics("bohemian rhapsody galileo figaro", transcript)
    assert lines[0]["start"] is None

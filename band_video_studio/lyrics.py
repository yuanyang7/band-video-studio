"""Optional lyrics matching: local whisper transcription + fuzzy line alignment.

Requires the `lyrics` extra (faster-whisper). Runs fully locally — no API.
"""

from __future__ import annotations

import re
from difflib import SequenceMatcher


def available() -> bool:
    try:
        import faster_whisper  # noqa: F401
        return True
    except ImportError:
        return False


def transcribe_words(source: str, start: float, end: float, progress=None) -> list[dict]:
    """Word-level transcript [{t, word}] for a time range, via faster-whisper."""
    from faster_whisper import WhisperModel

    from . import probe

    if progress:
        progress("loading whisper model", pct=10)
    model = WhisperModel("small", device="cpu", compute_type="int8")
    samples = probe.extract_audio(source, start=start, duration=end - start)

    if progress:
        progress("transcribing", pct=40)
    segments, _ = model.transcribe(samples, word_timestamps=True, vad_filter=True)
    words = []
    for segment in segments:
        for word in segment.words or []:
            words.append({"t": round(start + word.start, 2), "word": word.word.strip()})
    return words


def _norm(text: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", text.lower())


def align_lyrics(lyric_text: str, words: list[dict]) -> list[dict]:
    """Greedy in-order alignment of lyric lines to transcript word windows.

    Returns [{line, start, end, match}] — match is 0..1 similarity; lines that
    don't match anywhere get start/end None so the UI can show gaps.
    """
    lines = [line.strip() for line in lyric_text.splitlines() if line.strip()]
    tokens = [_norm(w["word"]) for w in words]
    flat = [(tok, w["t"]) for toks, w in zip(tokens, words) for tok in toks]

    results = []
    cursor = 0
    for line in lines:
        line_tokens = _norm(line)
        if not line_tokens or not flat:
            results.append({"line": line, "start": None, "end": None, "match": 0.0})
            continue
        n = len(line_tokens)
        best_score, best_i = 0.0, None
        horizon = min(len(flat), cursor + n * 12 + 60)  # don't scan the whole file
        for i in range(cursor, max(cursor, horizon - n + 1)):
            window = [t for t, _ in flat[i:i + n]]
            score = SequenceMatcher(None, " ".join(line_tokens), " ".join(window)).ratio()
            if score > best_score:
                best_score, best_i = score, i
        if best_i is not None and best_score >= 0.5:
            results.append({
                "line": line,
                "start": flat[best_i][1],
                "end": flat[min(best_i + n - 1, len(flat) - 1)][1],
                "match": round(best_score, 2),
            })
            cursor = best_i + n
        else:
            results.append({"line": line, "start": None, "end": None, "match": round(best_score, 2)})
    return results


def match(source: str, start: float, end: float, lyric_text: str, progress=None) -> dict:
    words = transcribe_words(source, start, end, progress=progress)
    return {
        "lines": align_lyrics(lyric_text, words),
        "transcript": " ".join(w["word"] for w in words),
    }

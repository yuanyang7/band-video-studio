"""Optional Claude-vision deep pass for nuanced fun-moment description.

Off by default; requires ANTHROPIC_API_KEY and the `claude` extra. To keep
cost near zero it only looks at the candidate windows already found by the
free local detectors (detect.py), never the whole video. For each candidate
it sends one small frame strip and asks for a judgment + caption.
"""

from __future__ import annotations

import base64
import json
import os

from . import probe

MODEL = "claude-opus-4-8"

_SCHEMA = {
    "type": "object",
    "properties": {
        "fun": {"type": "boolean"},
        "kind": {
            "type": "string",
            "enum": ["laughter", "funny_reaction", "mistake_reaction", "banter", "other", "not_fun"],
        },
        "caption": {"type": "string"},
        "confidence": {"type": "number"},
    },
    "required": ["fun", "kind", "caption", "confidence"],
    "additionalProperties": False,
}

_PROMPT = (
    "These are sequential frames (left to right, ~{interval}s apart) from a band "
    "rehearsal video, starting at {start:.0f}s. Local detectors flagged this window as a "
    "possible fun moment ({hint}). Judge whether something genuinely fun is happening — "
    "people laughing, a funny reaction to a flubbed note, joking around, playful banter. "
    "Neutral concentration or ordinary playing is not fun. Give a short caption a video "
    "editor could use as a clip title."
)


def available() -> bool:
    try:
        import anthropic  # noqa: F401
    except ImportError:
        return False
    return bool(os.environ.get("ANTHROPIC_API_KEY"))


def _frame_block(jpeg: bytes) -> dict:
    return {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": "image/jpeg",
            "data": base64.standard_b64encode(jpeg).decode(),
        },
    }


def judge_moment(client, source: str, start: float, end: float, hint: str) -> dict:
    """One API call: a strip of small frames from the window -> structured verdict."""
    interval = max(1.0, (end - start) / 5)  # at most ~6 frames per candidate
    content = [_frame_block(jpeg) for _, jpeg in probe.sample_frames(source, start, end, interval, height=300)]
    content.append({"type": "text", "text": _PROMPT.format(interval=interval, start=start, hint=hint)})

    response = client.messages.create(
        model=MODEL,
        max_tokens=300,
        output_config={"format": {"type": "json_schema", "schema": _SCHEMA}},
        messages=[{"role": "user", "content": content}],
    )
    text = next(b.text for b in response.content if b.type == "text")
    return json.loads(text)


def enrich_fun_moments(source: str, moments: list[dict], progress=None) -> list[dict]:
    """Annotate candidate moments in place with Claude verdicts; returns moments."""
    import anthropic

    client = anthropic.Anthropic()
    for i, moment in enumerate(moments):
        if progress:
            progress(f"Claude judging moment {i + 1}/{len(moments)}")
        hint = f"{moment['type']}, local score {moment['score']}"
        try:
            verdict = judge_moment(client, source, moment["start"], moment["end"], hint)
        except Exception as e:
            moment["claude"] = {"error": str(e)}
            continue
        moment["claude"] = verdict
        if verdict["fun"]:
            moment["score"] = round(moment["score"] + verdict["confidence"], 2)
            moment["caption"] = verdict["caption"]
            moment["type"] = verdict["kind"]
        else:
            moment["score"] = round(moment["score"] * 0.3, 2)
    return moments

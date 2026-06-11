"""Multicam-style auto edit from a single fixed wide shot.

The user draws one crop box per player (normalized 0..1 coords on the proxy
frame). For a chosen segment (typically one song) we build a cut list that
switches between those virtual "cameras" and render it with one ffmpeg
filtergraph from the ORIGINAL file, so 4K detail survives the crop.
"""

from __future__ import annotations

import random
import subprocess
from pathlib import Path

PRESETS = {
    "horizontal": (1920, 1080),
    "vertical": (1080, 1920),
}


def fit_crop(box: dict, src_w: int, src_h: int, target_aspect: float) -> tuple[int, int, int, int]:
    """Adjust a normalized box to the target aspect, centered and clamped.

    Returns integer (x, y, w, h) in source pixels, even-valued for yuv420.
    """
    x = box["x"] * src_w
    y = box["y"] * src_h
    w = max(box["w"] * src_w, 16)
    h = max(box["h"] * src_h, 16)
    cx, cy = x + w / 2, y + h / 2

    if w / h > target_aspect:
        h = w / target_aspect
    else:
        w = h * target_aspect
    # clamp inside the frame, preserving aspect by shrinking if needed
    scale = min(1.0, src_w / w, src_h / h)
    w, h = w * scale, h * scale
    x = min(max(cx - w / 2, 0), src_w - w)
    y = min(max(cy - h / 2, 0), src_h - h)

    even = lambda v: int(v) // 2 * 2
    return even(x), even(y), even(w), even(h)


def build_cutlist(
    views: list[str],
    start: float,
    end: float,
    switch_s: float = 4.0,
    seed: int | None = None,
) -> list[dict]:
    """Contiguous cuts over [start, end), random view order without repeats."""
    if not views:
        raise ValueError("need at least one view")
    rng = random.Random(seed)
    cuts, t, prev = [], start, None
    while t < end:
        choices = [v for v in views if v != prev] or views
        view = rng.choice(choices)
        cut_end = min(t + switch_s, end)
        # avoid a stub cut shorter than half a switch at the very end
        if end - cut_end < switch_s / 2:
            cut_end = end
        cuts.append({"start": round(t, 3), "end": round(cut_end, 3), "view": view})
        prev, t = view, cut_end
    return cuts


def render(
    source: str,
    src_w: int,
    src_h: int,
    crops: dict[str, dict],
    cuts: list[dict],
    orientation: str,
    out_path: Path,
    progress=None,
) -> Path:
    """Render the cut list to out_path with a single ffmpeg pass."""
    out_w, out_h = PRESETS[orientation]
    target_aspect = out_w / out_h

    px_crops = {name: fit_crop(box, src_w, src_h, target_aspect) for name, box in crops.items()}

    filters, labels = [], []
    for i, cut in enumerate(cuts):
        x, y, w, h = px_crops[cut["view"]]
        filters.append(
            f"[0:v]trim=start={cut['start']}:end={cut['end']},setpts=PTS-STARTPTS,"
            f"crop={w}:{h}:{x}:{y},scale={out_w}:{out_h},setsar=1[v{i}]"
        )
        labels.append(f"[v{i}]")
    seg_start, seg_end = cuts[0]["start"], cuts[-1]["end"]
    filters.append(f"{''.join(labels)}concat=n={len(cuts)}:v=1:a=0[vout]")
    filters.append(f"[0:a]atrim=start={seg_start}:end={seg_end},asetpts=PTS-STARTPTS[aout]")

    if progress:
        progress(f"rendering {len(cuts)} cuts ({orientation})")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-i", source,
        "-filter_complex", ";".join(filters),
        "-map", "[vout]", "-map", "[aout]",
        "-c:v", "libx264", "-preset", "medium", "-crf", "18",
        "-c:a", "aac", "-b:a", "192k",
        "-movflags", "+faststart",
        str(out_path),
    ]
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg render failed: {result.stderr.decode(errors='replace')[-2000:]}")
    return out_path

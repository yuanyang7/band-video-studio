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

import numpy as np

PRESETS = {
    "horizontal": (1920, 1080),
    "vertical": (1080, 1920),
}


def is_pano(box: dict, target_aspect: float) -> bool:
    """A view much wider than the output aspect — a group/room/pano framing."""
    h = max(box["h"], 1e-6)
    return (box["w"] / h) >= 1.3 * target_aspect


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


def _snap_boundary(target: float, lo: float, hi: float, beats: list[float]) -> float:
    """Pick the beat in [lo, hi] nearest `target`; fall back to target clamped."""
    candidates = [b for b in beats if lo <= b <= hi]
    if candidates:
        return min(candidates, key=lambda b: abs(b - target))
    return min(max(target, lo), hi)


def _activity_at(activity: dict, view: str, t: float) -> float:
    """Interpolated motion z-score for a view at time t (0 if unknown)."""
    track = activity.get(view) if activity else None
    if not track or not track[0]:
        return 0.0
    times, z = track
    return float(np.interp(t, times, z, left=z[0], right=z[-1]))


def _track_at(track, t: float) -> float:
    """Interpolated value of a (times, values) track at time t (0 if unknown)."""
    if not track or not track[0]:
        return 0.0
    times, values = track
    return float(np.interp(t, times, values, left=values[0], right=values[-1]))


def build_smart_cutlist(
    views: list[str],
    start: float,
    end: float,
    *,
    switch_s: float = 4.0,
    beats: list[float] | None = None,
    events: list[dict] | None = None,
    activity: dict | None = None,
    pano_views: set[str] | None = None,
    singer_views: set[str] | None = None,
    vocal_track: tuple[list[float], list[float]] | None = None,
    centers: dict[str, float] | None = None,
    seed: int | None = None,
    w_motion: float = 1.0,
    w_audio: float = 1.5,
    w_vocal: float = 1.2,
    w_center: float = 0.35,
    w_wide: float = 1.2,
    even_weight: float = 1.0,
    wide_refresh: float = 3.0,
) -> list[dict]:
    """Content-aware, beat-aligned cut list that still spreads views evenly.

    Boundaries step ~switch_s but snap to the nearest beat (kept within
    [0.6x, 1.8x] switch_s so it stays musical *and* even). Each segment goes to
    the view scoring highest on a fusion of:
      - per-view motion (activity z-score),
      - audio-event routing (peaks -> soloist, fun -> pano/room,
        vocal entrances and sustained singing -> singer views),
      - a small bonus for views framed near the centre of the stage,
      - a "wide refresh" bonus so a pano/group view returns every
        ~wide_refresh * switch_s seconds (keeps the whole band in the story),
      - an evenness penalty for views that already had more than their fair
        share of screen time.
    """
    if not views:
        raise ValueError("need at least one view")
    rng = random.Random(seed)
    beats = sorted(beats or [])
    events = events or []
    activity = activity or {}
    pano_views = pano_views or set()
    singer_views = singer_views or set()
    centers = centers or {}
    min_len, max_len = 0.6 * switch_s, 1.8 * switch_s

    cuts: list[dict] = []
    screen_time = {v: 0.0 for v in views}
    t, prev = start, None
    last_wide = start
    while t < end - 1e-3:
        lo, hi = min(t + min_len, end), min(t + max_len, end)
        cut_end = _snap_boundary(t + switch_s, lo, hi, beats)
        if end - cut_end < min_len:  # avoid a tiny stub at the very end
            cut_end = end
        mid = (t + cut_end) / 2

        overlapping = [e for e in events if e["start"] < cut_end and e["end"] > t]
        elapsed = max(1e-6, t - start)
        fair = elapsed / len(views)
        vocal_level = _track_at(vocal_track, mid)
        wide_due = pano_views and (t - last_wide) >= wide_refresh * switch_s

        choices = [v for v in views if v != prev] or views
        best_view, best_score, best_reason = choices[0], -1e9, "motion"
        # for peak/instrumental routing: which candidate is most active right now
        most_active = max(choices, key=lambda v: _activity_at(activity, v, mid))
        for v in choices:
            score = w_motion * _activity_at(activity, v, mid)
            reason = "motion"
            if v in singer_views and vocal_level > 0.12:
                # someone is singing right now — favour the singer, more so the
                # stronger (more emotive) the vocal is
                score += w_vocal * min(1.0, vocal_level / 0.4)
                reason = "vocals->singer"
            for e in overlapping:
                if e["type"] == "vocal_start" and v in singer_views:
                    score += w_audio  # the singer just came in — cut to them
                    reason = "vocal-start->singer"
                elif e["type"] == "fun" and v in pano_views:
                    score += w_audio
                    reason = "fun->pano"
                elif e["type"] in ("peak", "instrumental") and v == most_active:
                    score += w_audio
                    reason = f"{e['type']}->soloist"
            if v in centers:
                score += w_center * (1.0 - 2.0 * abs(centers[v] - 0.5))
            if wide_due and v in pano_views:
                score += w_wide
                reason = "wide-refresh"
            score -= even_weight * max(0.0, screen_time[v] - fair) / switch_s
            score += rng.uniform(0, 1e-3)  # deterministic tie-break
            if score > best_score:
                best_view, best_score, best_reason = v, score, reason

        cuts.append({"start": round(t, 3), "end": round(cut_end, 3),
                     "view": best_view, "reason": best_reason})
        screen_time[best_view] += cut_end - t
        if best_view in pano_views:
            last_wide = cut_end
        prev, t = best_view, cut_end
    return cuts


# ----------------------------------------------------------- camera movement
# Optional "shot by a human" feel: a subtle handheld drift / slow push-in within
# each shot, a pan across pano views, and glides that connect one framing to the
# next. Two ffmpeg techniques, picked so neither ever distorts the picture:
#   - push-in "drift" -> zoompan, which animates zoom inside one output-aspect
#     crop (its viewport aspect is locked to the input, so we feed it an
#     output-aspect region and only change zoom/offset).
#   - pano pans and view-to-view transitions -> a constant-size, output-aspect
#     window slid in x/y via crop(eval=frame). A fixed window size means a wide
#     horizontal sweep works even for a tall vertical output (the case zoompan
#     can't do without stretching).
# The geometry below is pure so it can be unit-tested without ffmpeg.


def _even(v) -> int:
    return int(v) // 2 * 2


def _window(cx: float, cy: float, w: int, h: int, src_w: int, src_h: int):
    """An (x, y, w, h) window of size w*h centred near (cx, cy), clamped to frame."""
    x = min(max(cx - w / 2, 0), src_w - w)
    y = min(max(cy - h / 2, 0), src_h - h)
    return (_even(x), _even(y), _even(w), _even(h))


def drift_filter(crop_px, n_frames: int, out_w: int, out_h: int, fps: float,
                 *, amount: float = 0.06, seed: int | None = None) -> str:
    """Subtle handheld push-in within one shot via zoompan (no aspect change)."""
    rng = random.Random(seed)
    x, y, w, h = crop_px
    # zoompan operates on the cropped region scaled up by `amount` of headroom so
    # it can push in without ever upscaling past the source detail.
    sw, sh = _even(w), _even(h)
    denom = max(1, n_frames - 1)
    p = f"min(1,on/{denom})"  # commas are literal inside the quoted expression
    z = f"(1+{amount:.4f}*{p})"  # 1.0 -> 1+amount, a slow push-in
    # drift the centre a touch in a seeded direction for a human, non-static feel
    dx, dy = rng.uniform(-1, 1) * amount * 0.5, rng.uniform(-1, 1) * amount * 0.5
    xe = f"((iw-iw/zoom)*(0.5+{dx:.4f}*{p}))"
    ye = f"((ih-ih/zoom)*(0.5+{dy:.4f}*{p}))"
    # zoompan emits frames on the *input* timebase, not 1/fps, so a source with a
    # fine timebase (e.g. 1/15360) explodes the duration ~500-1000x downstream.
    # Re-stamp PTS onto a clean 1/fps grid so concat/fps see the real duration.
    return (
        f"crop={sw}:{sh}:{x}:{y},"
        f"zoompan=z='{z}':x='{xe}':y='{ye}':d=1:s={out_w}x{out_h}:fps={fps},"
        f"setpts=N/{fps}/TB"
    )


def slide_filter(start_win, end_win, out_w: int, out_h: int, duration: float, fps: float) -> str:
    """Glide a constant-size output-aspect window start_win -> end_win.

    Used for pano sweeps and view-to-view transitions; pans/cranes without any
    zoom, so a wide horizontal move is undistorted even in a tall 9:16 frame.
    """
    x0, y0, w, h = start_win
    x1, y1, _, _ = end_win
    d = max(1e-3, duration)
    p = f"min(1,t/{d:.4f})"  # crop x/y expressions are evaluated per frame (t = timestamp)
    xe = f"({x0}+({x1 - x0})*{p})"
    ye = f"({y0}+({y1 - y0})*{p})"
    return f"crop={w}:{h}:x='{xe}':y='{ye}',scale={out_w}:{out_h}"


def pano_sweep(pano_box, target_aspect: float, src_w: int, src_h: int, seed: int | None = None):
    """Two end windows sweeping horizontally across a pano box, or None if too tight."""
    px, py, pw, ph = pano_box
    win_w, win_h = _even(ph * target_aspect), _even(ph)
    if win_w >= pw - 2 or win_w < 2:  # no room to travel (or degenerate)
        return None
    cy = py + ph / 2
    left = _window(px + win_w / 2, cy, win_w, win_h, src_w, src_h)
    right = _window(px + pw - win_w / 2, cy, win_w, win_h, src_w, src_h)
    return (left, right) if random.Random(seed).random() < 0.5 else (right, left)


def transition_windows(prev_crop, crop_px, src_w: int, src_h: int):
    """Glide from the previous view's framing to this one, at this view's zoom."""
    x, y, w, h = crop_px
    pcx, pcy = prev_crop[0] + prev_crop[2] / 2, prev_crop[1] + prev_crop[3] / 2
    start = _window(pcx, pcy, w, h, src_w, src_h)
    return start, (x, y, w, h)


def render(
    source: str,
    src_w: int,
    src_h: int,
    crops: dict[str, dict],
    cuts: list[dict],
    orientation: str,
    out_path: Path,
    camera_motion: bool = False,
    transitions: bool = False,
    fps: float = 30.0,
    seed: int | None = None,
    audio_source: str | None = None,
    audio_offset: float = 0.0,
    progress=None,
) -> Path:
    """Render the cut list to out_path with a single ffmpeg pass.

    With camera_motion/transitions each shot gets a subtle move (push-in drift,
    pano sweep, or a glide between views); otherwise the fast static crop chain
    is used. In motion mode every segment is normalised to `fps` so the concat
    stays uniform.

    If `audio_source` is given (a separately-recorded clean take aligned to the
    video), the original soundtrack is replaced by it: the reference begins at
    video time `audio_offset`, so the exported span [seg_start, seg_end] maps to
    reference time [seg_start - audio_offset, seg_end - audio_offset].
    """
    out_w, out_h = PRESETS[orientation]
    target_aspect = out_w / out_h

    px_crops = {name: fit_crop(box, src_w, src_h, target_aspect) for name, box in crops.items()}
    pano_px = {
        name: (_even(box["x"] * src_w), _even(box["y"] * src_h),
               _even(box["w"] * src_w), _even(box["h"] * src_h))
        for name, box in crops.items()
        if is_pano(box, target_aspect)
    }
    motion_mode = camera_motion or transitions

    # Each cut gets its own `-ss <seek> -i source` so ffmpeg keyframe-seeks
    # close to the cut start instead of decoding from 0 every time.  The trim
    # still handles frame-accurate boundaries (relative to the seeked input,
    # which starts at the seek point).  One extra un-seeked input supplies
    # audio for the full span (or the replacement audio_source does).
    SEEK_MARGIN = 5.0  # seconds before the cut to seek to (cover one GOP)
    inputs: list[str] = []
    filters, labels = [], []
    for i, cut in enumerate(cuts):
        seek = max(0.0, cut["start"] - SEEK_MARGIN)
        inputs += ["-ss", str(seek), "-i", source]
        # trim times are relative to the seeked input
        trim_start = cut["start"] - seek
        trim_end = cut["end"] - seek
        view = cut["view"]
        x, y, w, h = px_crops[view]
        pre = f"[{i}:v]trim=start={trim_start}:end={trim_end},setpts=PTS-STARTPTS,"
        dur = cut["end"] - cut["start"]
        if motion_mode:
            prev_view = cuts[i - 1]["view"] if i > 0 else None
            chain = None
            if transitions and prev_view and prev_view != view:
                s_win, e_win = transition_windows(px_crops[prev_view], (x, y, w, h), src_w, src_h)
                chain = slide_filter(s_win, e_win, out_w, out_h, dur, fps)
            elif camera_motion and view in pano_px:
                ends = pano_sweep(pano_px[view], target_aspect, src_w, src_h, seed=(seed or 0) + i)
                if ends:
                    chain = slide_filter(ends[0], ends[1], out_w, out_h, dur, fps)
            if chain is None and camera_motion:
                n = max(1, round(dur * fps))
                chain = drift_filter((x, y, w, h), n, out_w, out_h, fps, seed=(seed or 0) + i)
            if chain is None:  # static shot inside motion mode — keep fps uniform
                chain = f"crop={w}:{h}:{x}:{y},scale={out_w}:{out_h}"
            filters.append(f"{pre}{chain},fps={fps},setsar=1[v{i}]")
        else:
            filters.append(f"{pre}crop={w}:{h}:{x}:{y},scale={out_w}:{out_h},setsar=1[v{i}]")
        labels.append(f"[v{i}]")
    seg_start, seg_end = cuts[0]["start"], cuts[-1]["end"]
    filters.append(f"{''.join(labels)}concat=n={len(cuts)}:v=1:a=0[vout]")

    # audio: one dedicated input (un-seeked for atrim, or the replacement file)
    audio_idx = len(cuts)
    if audio_source:
        inputs += ["-i", audio_source]
        a_start = max(0.0, seg_start - audio_offset)
        a_end = max(a_start, seg_end - audio_offset)
        filters.append(f"[{audio_idx}:a]atrim=start={a_start}:end={a_end},asetpts=PTS-STARTPTS[aout]")
    else:
        inputs += ["-i", source]
        filters.append(f"[{audio_idx}:a]atrim=start={seg_start}:end={seg_end},asetpts=PTS-STARTPTS[aout]")

    if progress:
        progress(f"rendering {len(cuts)} cuts ({orientation})")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y", *inputs,
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

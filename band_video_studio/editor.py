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
    "horizontal_2k": (2560, 1440),
    "vertical_2k": (1440, 2560),
    "horizontal_4k": (3840, 2160),
    "vertical_4k": (2160, 3840),
}


def is_pano(box: dict, target_aspect: float) -> bool:
    """A view much wider than the output aspect — a group/room/pano framing."""
    h = max(box["h"], 1e-6)
    return (box["w"] / h) >= 1.3 * target_aspect


def _face_penalty(x: float, y: float, w: float, h: float,
                   faces: list[dict], src_w: int, src_h: int) -> float:
    """How much of the avoid-faces list falls inside the crop rectangle."""
    penalty = 0.0
    for f in faces:
        fx, fy = f["cx"] * src_w, f["cy"] * src_h
        if x <= fx <= x + w and y <= fy <= y + h:
            # face is inside crop — penalty proportional to how deep inside
            dx = min(fx - x, x + w - fx) / w
            dy = min(fy - y, y + h - fy) / h
            penalty += min(dx, dy) + 0.5  # 0.5 base + depth bonus
    return penalty


def fit_crop(
    box: dict, src_w: int, src_h: int, target_aspect: float,
    out_w: int = 0, out_h: int = 0,
    max_upscale: float = 2.0,
    avoid_faces: list[dict] | None = None,
) -> tuple[int, int, int, int]:
    """Adjust a normalized box to the target aspect, centered and clamped.

    When out_w/out_h are given, the crop is enlarged if needed so we never
    upscale beyond max_upscale (avoids blurry output from tiny boxes).

    When avoid_faces is given (list of {cx, cy} in normalized 0..1 coords),
    the crop is shifted away from those faces while keeping the target
    player centered.

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
    # enforce minimum crop size: at most max_upscale to the output
    if out_w and out_h:
        min_w = out_w / max_upscale
        min_h = out_h / max_upscale
        if w < min_w or h < min_h:
            grow = max(min_w / w, min_h / h)
            w, h = w * grow, h * grow
    # clamp inside the frame, preserving aspect by shrinking if needed
    scale = min(1.0, src_w / w, src_h / h)
    w, h = w * scale, h * scale
    x = min(max(cx - w / 2, 0), src_w - w)
    y = min(max(cy - h / 2, 0), src_h - h)

    # nudge away from other players' faces
    if avoid_faces:
        best_x, best_penalty = x, _face_penalty(x, y, w, h, avoid_faces, src_w, src_h)
        # try shifting left and right in small steps
        max_shift = w * 0.3
        for sign in (-1, 1):
            for step in (0.05, 0.1, 0.15, 0.2, 0.25, 0.3):
                nx = cx + sign * max_shift * (step / 0.3) - w / 2
                nx = min(max(nx, 0), src_w - w)
                pen = _face_penalty(nx, y, w, h, avoid_faces, src_w, src_h)
                if pen < best_penalty:
                    best_x, best_penalty = nx, pen
        x = best_x

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


ROLE_TO_STEM = {
    "singer": "vocals",
    "drums": "drums",
    "bass": "bass",
    "keys": "other",
}


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
    stem_tracks: dict[str, tuple[list[float], list[float]]] | None = None,
    role_map: dict[str, str] | None = None,
    centers: dict[str, float] | None = None,
    seed: int | None = None,
    w_motion: float = 1.0,
    w_audio: float = 1.5,
    w_vocal: float = 1.2,
    w_stem: float = 1.0,
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
      - per-instrument stem energy (drums/bass/keys boost matching views),
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
    stem_tracks = stem_tracks or {}
    role_map = role_map or {}
    centers = centers or {}
    min_len, max_len = 0.6 * switch_s, 1.8 * switch_s

    cuts: list[dict] = []
    screen_time = {v: 0.0 for v in views}
    last_seen = {v: start for v in views}
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
        most_active = max(choices, key=lambda v: _activity_at(activity, v, mid))
        for v in choices:
            score = w_motion * _activity_at(activity, v, mid)
            reason = "motion"
            # stem-aware boost: each role gets a bonus from its instrument track
            v_role = role_map.get(v, "")
            v_stem_name = ROLE_TO_STEM.get(v_role, "")
            if v_stem_name and v_stem_name in stem_tracks:
                stem_level = _track_at(stem_tracks[v_stem_name], mid)
                if stem_level > 0.05:
                    score += w_stem * min(1.0, stem_level / 0.15)
                    reason = f"{v_stem_name}->{v_role}"
            if v in singer_views and vocal_level > 0.02:
                score += w_vocal * min(1.0, vocal_level / 0.08)
                reason = "vocals->singer"
            for e in overlapping:
                if e["type"] == "vocal_start" and v in singer_views:
                    score += w_audio
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
            # evenness: penalize overrepresented, boost starved
            score -= even_weight * max(0.0, screen_time[v] - fair) / switch_s
            score += even_weight * max(0.0, fair - screen_time[v]) / switch_s * 0.5
            # recency: boost views not seen for a while
            gap = t - last_seen[v]
            if gap > switch_s * len(views):
                score += even_weight * min(gap / (switch_s * len(views)), 2.0)
                if reason == "motion":
                    reason = "recency"
            score += rng.uniform(0, 1e-3)
            if score > best_score:
                best_view, best_score, best_reason = v, score, reason

        cuts.append({"start": round(t, 3), "end": round(cut_end, 3),
                     "view": best_view, "reason": best_reason})
        screen_time[best_view] += cut_end - t
        last_seen[best_view] = cut_end
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
                 *, amount: float = 0.03, seed: int | None = None) -> str:
    """Subtle handheld camera drift via zoompan.

    Three things keep the motion fluid instead of pixel-steppy:
    - the crop is supersampled to 2x the output before zoompan, so zoompan's
      integer x/y rounding lands at half an output pixel;
    - the zoom has a constant base of (1+amount), so the drift has headroom
      from the very first frame (a pure push-in starts with zero headroom);
    - the sine amplitudes are sizable fractions of that headroom, keeping the
      wander well above the rounding threshold.
    """
    rng = random.Random(seed)
    x, y, w, h = crop_px
    sw, sh = _even(w), _even(h)
    denom = max(1, n_frames - 1)
    ss_w, ss_h = out_w * 2, out_h * 2

    # constant base zoom for headroom, plus a gentle extra push-in
    z = f"({1 + amount:.4f}+{amount * 0.5:.4f}*min(1,on/{denom}))"

    # XY drift: 3 slow overlapping sines, amplitudes as fractions of the
    # headroom around centre (sum stays under 0.5 so we never hit the edge)
    periods = [fps * rng.uniform(4, 6),
               fps * rng.uniform(8, 12),
               fps * rng.uniform(15, 22)]
    phases = [rng.uniform(0, 6.283) for _ in range(6)]
    ax = [rng.uniform(0.18, 0.26),
          rng.uniform(0.08, 0.14),
          rng.uniform(0.03, 0.06)]
    ay = [rng.uniform(0.18, 0.26),
          rng.uniform(0.08, 0.14),
          rng.uniform(0.03, 0.06)]

    x_terms = "+".join(
        f"{a:.4f}*sin(on/{p:.1f}*2*PI+{ph:.4f})"
        for a, p, ph in zip(ax, periods, phases[:3])
    )
    y_terms = "+".join(
        f"{a:.4f}*sin(on/{p:.1f}*2*PI+{ph:.4f})"
        for a, p, ph in zip(ay, periods, phases[3:])
    )
    xe = f"((iw-iw/zoom)*(0.5+{x_terms}))"
    ye = f"((ih-ih/zoom)*(0.5+{y_terms}))"

    return (
        f"crop={sw}:{sh}:{x}:{y},"
        f"scale={ss_w}:{ss_h}:flags=lanczos,"
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


def _merge_same_view(cuts: list[dict]) -> list[dict]:
    """Merge consecutive cuts with the same view into one longer shot."""
    if not cuts:
        return cuts
    merged = [dict(cuts[0])]
    for cut in cuts[1:]:
        if cut["view"] == merged[-1]["view"]:
            merged[-1]["end"] = cut["end"]
            merged[-1]["_merged"] = True
        else:
            merged.append(dict(cut))
    return merged


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
    sharpen: bool = False,
    denoise: bool = False,
    max_upscale: float = 2.0,
    face_map: dict[str, list[dict]] | None = None,
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
    cuts = _merge_same_view(cuts)
    out_w, out_h = PRESETS[orientation]
    target_aspect = out_w / out_h

    denoi = ",hqdn3d=4:3:6:4" if denoise else ""
    sharp = ",unsharp=5:5:0.8:3:3:0.4" if sharpen else ""
    # faces belonging to OTHER views that this view should try to exclude
    all_faces = face_map or {}
    all_other = {name: [f for other, fl in all_faces.items() if other != name for f in fl]
                 for name in crops}
    px_crops = {
        name: fit_crop(box, src_w, src_h, target_aspect, out_w, out_h,
                       max_upscale=max_upscale, avoid_faces=all_other.get(name))
        for name, box in crops.items()
    }
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
            if chain is None and camera_motion and not cut.get("_merged"):
                n = max(1, round(dur * fps))
                chain = drift_filter((x, y, w, h), n, out_w, out_h, fps, seed=(seed or 0) + i)
            if chain is None:  # static shot inside motion mode — keep fps uniform
                chain = f"crop={w}:{h}:{x}:{y},scale={out_w}:{out_h}"
            filters.append(f"{pre}{chain}{denoi}{sharp},fps={fps},setsar=1[v{i}]")
        else:
            filters.append(f"{pre}crop={w}:{h}:{x}:{y},scale={out_w}:{out_h}{denoi}{sharp},setsar=1[v{i}]")
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
        progress(f"rendering {len(cuts)} cuts ({orientation})", pct=80)
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

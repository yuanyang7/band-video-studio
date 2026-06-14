"""FastAPI app: REST API + static single-page frontend."""

from __future__ import annotations

import re
import shutil
from pathlib import Path

import numpy as np

from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import align, audio, detect, editor, jobs, library, lyrics, probe, store, vision

app = FastAPI(title="Band Video Studio")

WEB_DIR = Path(__file__).resolve().parent.parent / "web"


# ------------------------------------------------------------------ videos

class RegisterBody(BaseModel):
    path: str
    name: str | None = None


def _video_or_404(video_id: str) -> dict:
    video = store.get_video(video_id)
    if not video:
        raise HTTPException(404, "video not found")
    return video


def _prepare(video: dict, progress=None) -> dict:
    """Ensure the playback proxy exists (analysis & UI run off it)."""
    proxy = store.video_dir(video["id"]) / "proxy.mp4"
    probe.make_proxy(video["source_path"], proxy, progress=progress)
    return video


@app.get("/api/videos")
def list_videos():
    return store.list_videos()


def _import_job(video: dict):
    """Proxy + full default analysis, chained so a fresh import is ready to use.

    Detection runs automatically with the free local defaults; the user can
    still re-detect from the UI (e.g. to add the paid Claude pass).
    """
    def run(progress=None):
        _prepare(video, progress)
        return _run_analysis(video, AnalyzeBody(), progress)

    return jobs.submit("import", run)


@app.post("/api/videos/register")
def register_video(body: RegisterBody):
    src = Path(body.path).expanduser()
    if not src.exists():
        raise HTTPException(400, f"no file at {src}")
    meta = probe.probe(str(src))
    video = store.add_video(str(src), body.name or src.name, meta)
    job = _import_job(video)
    return {"video": video, "job": job["id"]}


@app.post("/api/videos/upload")
async def upload_video(file: UploadFile):
    uploads = store.DATA_DIR / "uploads"
    uploads.mkdir(parents=True, exist_ok=True)
    dest = uploads / file.filename
    with dest.open("wb") as f:
        shutil.copyfileobj(file.file, f)
    meta = probe.probe(str(dest))
    video = store.add_video(str(dest), file.filename, meta)
    job = _import_job(video)
    return {"video": video, "job": job["id"]}


@app.get("/api/videos/{video_id}")
def get_video(video_id: str):
    video = _video_or_404(video_id)
    stems_data = store.load_artifact(video_id, "stems")
    return {
        **video,
        "has_proxy": (store.video_dir(video_id) / "proxy.mp4").exists(),
        "analysis": store.load_artifact(video_id, "analysis"),
        "crops": store.load_artifact(video_id, "crops") or {},
        "lyrics": store.load_artifact(video_id, "lyrics"),
        "sync": store.load_artifact(video_id, "sync"),
        "settings": store.load_artifact(video_id, "settings") or {},
        "stems": list(stems_data.keys()) if stems_data else [],
        "capabilities": {"claude": vision.available(), "lyrics": lyrics.available()},
    }


@app.get("/api/videos/{video_id}/stream")
def stream_video(video_id: str):
    _video_or_404(video_id)
    proxy = store.video_dir(video_id) / "proxy.mp4"
    if not proxy.exists():
        raise HTTPException(409, "proxy not ready yet")
    return FileResponse(proxy, media_type="video/mp4")


@app.get("/api/videos/{video_id}/frame")
def frame(video_id: str, t: float = 0.0):
    video = _video_or_404(video_id)
    proxy = store.video_dir(video_id) / "proxy.mp4"
    source = str(proxy) if proxy.exists() else video["source_path"]
    return Response(probe.extract_frame_jpeg(source, t), media_type="image/jpeg")


# ---------------------------------------------------------------- analysis

class AnalyzeBody(BaseModel):
    fun_detection: bool = True      # local smile/laugh fusion (free)
    sweep: bool = True              # sparse smile sweep over the whole video
    claude_pass: bool = False       # optional paid deep pass on candidates


def _run_analysis(video: dict, body: AnalyzeBody, progress=None) -> dict:
    """Songs/highlights/fun-moment detection; shared by import and re-detect."""
    video_id = video["id"]
    proxy = str(store.video_dir(video_id) / "proxy.mp4")
    result = audio.analyze(video["source_path"], progress=progress)
    duration = video["meta"]["duration"]
    if body.fun_detection:
        result["fun_moments"] = detect.find_fun_moments(
            proxy, result, duration, sweep=body.sweep, progress=progress
        )
        if body.claude_pass and vision.available():
            result["fun_moments"] = vision.enrich_fun_moments(
                proxy, result["fun_moments"], progress=progress
            )
    else:
        result["fun_moments"] = []
    store.save_artifact(video_id, "analysis", result)

    # if a synced recording exists, run Demucs stem separation
    stems_result = _run_stem_separation(video_id, progress)

    summary = {"songs": len(result["songs"]), "fun_moments": len(result["fun_moments"])}
    if stems_result:
        summary["stems"] = list(stems_result.keys())
    return summary


def _run_stem_separation(video_id: str, progress=None) -> dict | None:
    """Run Demucs on the synced recording if available. Saves stems artifact."""
    sync = store.load_artifact(video_id, "sync")
    if not sync or not Path(sync.get("ref_path", "")).exists():
        return None
    try:
        if progress:
            progress("separating stems with Demucs", pct=85)
        stem_tracks = audio.demucs_all_stems(
            sync["ref_path"], time_offset=sync["offset"],
        )
        if stem_tracks:
            store.save_artifact(video_id, "stems", stem_tracks)
            if progress:
                progress(f"stems separated: {', '.join(stem_tracks.keys())}", pct=90)
            return stem_tracks
    except Exception as exc:
        if progress:
            progress(f"Demucs unavailable ({exc}), skipping stems", pct=90)
    return None


@app.post("/api/videos/{video_id}/analyze")
def analyze(video_id: str, body: AnalyzeBody):
    video = _video_or_404(video_id)

    def run(progress=None):
        _prepare(video, progress)
        return _run_analysis(video, body, progress)

    return jobs.submit("analyze", run)


# ------------------------------------------------------------------ lyrics

class LyricsBody(BaseModel):
    start: float
    end: float
    song_name: str = ""
    lyrics: str


@app.post("/api/videos/{video_id}/lyrics-match")
def lyrics_match(video_id: str, body: LyricsBody):
    video = _video_or_404(video_id)
    if not lyrics.available():
        raise HTTPException(400, "install the 'lyrics' extra: uv sync --extra lyrics")

    def run(progress=None):
        result = lyrics.match(video["source_path"], body.start, body.end, body.lyrics, progress=progress)
        result["song_name"] = body.song_name
        result["range"] = [body.start, body.end]
        store.save_artifact(video_id, "lyrics", result)
        return {"matched": sum(1 for line in result["lines"] if line["start"] is not None)}

    return jobs.submit("lyrics", run)


# --------------------------------------------------- sync to a recording

@app.post("/api/videos/{video_id}/sync-audio")
async def sync_audio(video_id: str, file: UploadFile):
    """Upload a clean recording of a song played in the video and align it.

    Only computes the alignment (offset/duration/confidence) for the UI to
    review — the actual export is done by Render edit, which picks up this
    alignment to crop the matching span and use the recording as its audio.
    """
    video = _video_or_404(video_id)
    sync_dir = store.video_dir(video_id) / "sync"
    sync_dir.mkdir(parents=True, exist_ok=True)
    ref_path = sync_dir / Path(file.filename).name
    with ref_path.open("wb") as f:
        shutil.copyfileobj(file.file, f)

    def run(progress=None):
        _prepare(video, progress)
        # align on the proxy soundtrack (decodes far faster than the 4K source)
        proxy = str(store.video_dir(video_id) / "proxy.mp4")
        alignment_result = align.align(proxy, str(ref_path), progress=progress)
        if alignment_result.duration <= 0:
            raise RuntimeError("no overlap found — is this a recording of a song in this video?")
        result = alignment_result.to_dict()
        result["file"] = file.filename
        result["ref_path"] = str(ref_path)
        store.save_artifact(video_id, "sync", result)
        # separate stems right after alignment so they're ready for export
        stems = _run_stem_separation(video_id, progress)
        if stems:
            result["stems"] = list(stems.keys())
        return result

    return jobs.submit("sync", run)


@app.get("/api/videos/{video_id}/sync-audio/file")
def sync_audio_file(video_id: str):
    """Stream the aligned clean recording so the UI can audition the alignment."""
    _video_or_404(video_id)
    sync = store.load_artifact(video_id, "sync")
    path = Path(sync.get("ref_path", "")) if sync else None
    if not path or not path.exists():
        raise HTTPException(404, "no aligned recording")
    return FileResponse(path)


@app.delete("/api/videos/{video_id}/sync-audio")
def clear_sync(video_id: str):
    """Forget the alignment so Render edit goes back to the original soundtrack."""
    _video_or_404(video_id)
    path = store.video_dir(video_id) / "sync.json"
    if path.exists():
        path.unlink()
    return {"ok": True}


# ----------------------------------------------------------------- editing

class CropsBody(BaseModel):
    crops: dict[str, dict]  # name -> {x, y, w, h} normalized 0..1


@app.put("/api/videos/{video_id}/crops")
def save_crops(video_id: str, body: CropsBody):
    _video_or_404(video_id)
    store.save_artifact(video_id, "crops", body.crops)
    return {"ok": True}


class SettingsBody(BaseModel):
    settings: dict


@app.put("/api/videos/{video_id}/settings")
def save_settings(video_id: str, body: SettingsBody):
    """Persist the UI's current settings (render options, export range, …).

    The frontend auto-saves these on every change so reopening a video
    restores exactly where the user left off.
    """
    _video_or_404(video_id)
    store.save_artifact(video_id, "settings", body.settings)
    return {"ok": True}


class EditBody(BaseModel):
    start: float
    end: float
    orientation: str = "horizontal"  # or "vertical"
    switch_s: float = 4.0            # target cadence (smart mode snaps it to beats)
    views: list[str] | None = None   # subset of crop names; default all
    smart: bool = True               # content-aware, beat-aligned switching
    camera_motion: bool = False      # subtle handheld drift / push-in / pano pan
    transitions: bool = True         # glide between views (connects the scene)
    sharpen: bool = False            # unsharp-mask to recover detail after crop upscale
    denoise: bool = False            # temporal denoise for low-light footage
    max_upscale: float = 2.0         # max upscale factor for crop boxes
    name: str | None = None          # optional output filename (sans extension)
    seed: int | None = None


def _vocal_segments(analysis: dict) -> list[dict]:
    """Return vocal_segments, deriving from windows if the key is missing (old analyses)."""
    segs = analysis.get("vocal_segments")
    if segs:
        return segs
    windows = analysis.get("windows") or []
    if not windows:
        return []
    times = np.array([w["t"] for w in windows])
    vocals = np.array([w.get("vocals", 0.0) for w in windows])
    return audio.find_vocal_segments(times, vocals)


def _events_from_analysis(analysis: dict | None, start: float, end: float) -> list[dict]:
    """Highlights/instrumentals/fun moments overlapping [start, end] as routing events."""
    if not analysis:
        return []
    events: list[dict] = []
    for h in analysis.get("highlights", []):
        events.append({"start": h["start"], "end": h["end"],
                       "type": "instrumental" if h.get("kind") == "instrumental" else "peak"})
    for m in analysis.get("fun_moments", []):
        events.append({"start": m["start"], "end": m["end"], "type": "fun"})
    return [e for e in events if e["start"] < end and e["end"] > start]


_SINGER_NAME_HINTS = ("sing", "vocal", "vox", "唱", "主唱")
_NAME_HINTS: dict[str, tuple[str, ...]] = {
    "singer": ("sing", "vocal", "vox", "唱", "主唱"),
    "drums": ("drum", "鼓"),
    "bass": ("bass", "贝斯", "貝斯"),
    "keys": ("key", "piano", "synth", "organ", "键盘", "鍵盤"),
}


def _singer_views(crops: dict[str, dict], views: list[str]) -> set[str]:
    """Views marked role=singer, or (with no explicit role) named like a singer."""
    out = set()
    for v in views:
        role = (crops[v].get("role") or "").lower()
        if role == "singer" or (not role and any(h in v.lower() for h in _SINGER_NAME_HINTS)):
            out.add(v)
    return out


def _role_map(crops: dict[str, dict], views: list[str]) -> dict[str, str]:
    """Map view name -> instrument role, from explicit role or name hints."""
    out: dict[str, str] = {}
    for v in views:
        role = (crops[v].get("role") or "").lower()
        if role in ("singer", "drums", "bass", "keys"):
            out[v] = role
            continue
        for r, hints in _NAME_HINTS.items():
            if any(h in v.lower() for h in hints):
                out[v] = r
                break
    return out




def _export_filename(body: EditBody, start: float, end: float, synced: bool) -> str:
    """User-chosen name (sanitized) or the descriptive default."""
    base = re.sub(r"[^\w\- .()\[\]]", "", (body.name or "").strip(), flags=re.UNICODE)
    base = base.strip(". ").removesuffix(".mp4")
    if not base:
        suffix = "_synced" if synced else ""
        base = f"edit_{int(start)}-{int(end)}_{body.orientation}{suffix}"
    return base + ".mp4"


def _resolve_sync(video_id: str, start: float, end: float):
    """Resolve sync artifact into audio source, offset, and clamped range."""
    sync = store.load_artifact(video_id, "sync")
    if sync and Path(sync.get("ref_path", "")).exists():
        audio_source, audio_offset = sync["ref_path"], sync["offset"]
        cover_start, cover_end = sync["offset"], sync["offset"] + sync["duration"]
        start, end = max(start, cover_start), min(end, cover_end)
        if end - start < 1.0:
            raise RuntimeError(
                "export range falls outside the aligned recording "
                f"({cover_start:.1f}s–{cover_end:.1f}s)"
            )
        return audio_source, audio_offset, start, end
    return None, 0.0, start, end


def _generate_cuts(video_id: str, crops: dict, views: list[str],
                   body: EditBody, start: float, end: float,
                   audio_source: str | None, audio_offset: float,
                   progress=None) -> list[dict]:
    """Generate the cut list (smart or simple) without rendering."""
    if not body.smart:
        return editor.build_cutlist(views, start, end, body.switch_s, body.seed)

    analysis = store.load_artifact(video_id, "analysis")
    out_w, out_h = editor.PRESETS[body.orientation]
    proxy = str(store.video_dir(video_id) / "proxy.mp4")
    view_crops = {v: crops[v] for v in views}
    if progress:
        progress("measuring per-view motion", pct=10)
    tracks = detect.view_activity(proxy, view_crops, start, end)
    activity = detect.activity_scores(tracks)
    pano = {v for v in views
            if (crops[v].get("role") or "").lower() == "wide"
            or editor.is_pano(crops[v], out_w / out_h)}

    vocal_track = None
    vocal_segs: list[dict] = []
    vocal_method = "none"
    saved_stems = store.load_artifact(video_id, "stems")
    stem_tracks: dict[str, tuple[list[float], list[float]]] = {}
    if saved_stems:
        stem_tracks = {k: (v[0], v[1]) for k, v in saved_stems.items() if v and v[0]}
        if "vocals" in stem_tracks:
            times, energies = stem_tracks["vocals"]
            if times:
                vocal_track = (times, energies)
                vocal_segs = audio.find_vocal_segments(
                    np.array(times), np.array(energies),
                    on_threshold=0.15, off_threshold=0.05,
                )
                vocal_method = "demucs"
    if not vocal_track and audio_source:
        ref_windows = audio.classify_audio(
            probe.extract_audio(audio_source), probe.AUDIO_SR,
        )
        if ref_windows:
            ref_vocals = np.array([w.get("vocals", 0.0) for w in ref_windows])
            ref_times = np.array([w["t"] + audio_offset for w in ref_windows])
            vocal_track = (ref_times.tolist(), ref_vocals.tolist())
            vocal_segs = audio.find_vocal_segments(ref_times, ref_vocals)
            vocal_method = "yamnet-clean"
    if not vocal_track:
        windows = (analysis or {}).get("windows") or []
        if windows:
            vocal_track = (
                [w["t"] for w in windows],
                [w.get("vocals", 0.0) for w in windows],
            )
            vocal_segs = _vocal_segments(analysis)
            vocal_method = "yamnet-camera"

    singers = _singer_views(crops, views)
    roles = _role_map(crops, views)
    events = _events_from_analysis(analysis, start, end)
    for seg in vocal_segs:
        events.append({"start": seg["start"],
                       "end": min(seg["end"], seg["start"] + 5.0),
                       "type": "vocal_start"})
    events = [e for e in events if e["start"] < end and e["end"] > start]
    stem_names = sorted(stem_tracks.keys()) if stem_tracks else []
    if progress:
        progress(
            f"smart cut: {len(views)} views, "
            f"singers={singers or 'none'}, "
            f"pano={pano or 'none'}, "
            f"vocal_track={vocal_method}, "
            f"stems={stem_names or 'none'}, "
            f"{len(vocal_segs)} vocal segs, "
            f"{len(events)} events",
            pct=60,
        )
    return editor.build_smart_cutlist(
        views, start, end,
        switch_s=body.switch_s,
        beats=(analysis or {}).get("beats"),
        events=events,
        activity=activity, pano_views=pano,
        singer_views=singers,
        vocal_track=vocal_track,
        stem_tracks=stem_tracks,
        role_map=roles,
        centers={v: crops[v]["x"] + crops[v]["w"] / 2 for v in views},
        seed=body.seed,
    )


class SimulateBody(BaseModel):
    start: float
    end: float
    orientation: str = "horizontal"
    switch_s: float = 4.0
    views: list[str] | None = None
    smart: bool = True
    seed: int | None = None


@app.post("/api/videos/{video_id}/simulate")
def simulate(video_id: str, body: SimulateBody):
    """Generate a cut list without rendering — for preview and manual editing."""
    video = _video_or_404(video_id)
    crops = store.load_artifact(video_id, "crops") or {}
    views = body.views or list(crops.keys())
    if not views or any(v not in crops for v in views):
        raise HTTPException(400, "define crops first (or unknown view name)")

    edit_body = EditBody(
        start=body.start, end=body.end, orientation=body.orientation,
        switch_s=body.switch_s, views=body.views, smart=body.smart, seed=body.seed,
    )

    def run(progress=None):
        audio_source, audio_offset, start, end = _resolve_sync(
            video_id, body.start, body.end)
        cuts = _generate_cuts(
            video_id, crops, views, edit_body, start, end,
            audio_source, audio_offset, progress)
        return {"cuts": cuts, "views": list(crops.keys()),
                "start": start, "end": end}

    return jobs.submit("simulate", run)


class RenderBody(BaseModel):
    start: float
    end: float
    cuts: list[dict]
    orientation: str = "horizontal"
    camera_motion: bool = False
    transitions: bool = True
    sharpen: bool = False
    denoise: bool = False
    max_upscale: float = 2.0
    name: str | None = None
    seed: int | None = None


@app.post("/api/videos/{video_id}/edit")
def make_edit(video_id: str, body: RenderBody):
    """Render a cut list (from simulate or manual edits) into a video file."""
    video = _video_or_404(video_id)
    crops = store.load_artifact(video_id, "crops") or {}
    used_views = {c["view"] for c in body.cuts}
    if any(v not in crops for v in used_views):
        raise HTTPException(400, "cut list references unknown view name")
    if body.orientation not in editor.PRESETS:
        raise HTTPException(400, f"orientation must be one of {list(editor.PRESETS)}")

    def run(progress=None):
        audio_source, audio_offset, start, end = _resolve_sync(
            video_id, body.start, body.end)
        cuts = body.cuts

        if progress:
            progress("detecting faces for crop avoidance")
        proxy = str(store.video_dir(video_id) / "proxy.mp4")
        mid_t = (start + end) / 2
        faces = detect.detect_face_boxes(proxy, mid_t)
        face_map = detect.assign_faces_to_views(faces, crops) if faces else {}

        synced = bool(audio_source)
        suffix = "_synced" if synced else ""
        base = re.sub(r"[^\w\- .()\[\]]", "", (body.name or "").strip(), flags=re.UNICODE)
        base = base.strip(". ").removesuffix(".mp4")
        if not base:
            base = f"edit_{int(start)}-{int(end)}_{body.orientation}{suffix}"
        out = store.video_dir(video_id) / "exports" / (base + ".mp4")

        editor.render(
            video["source_path"], video["meta"]["width"], video["meta"]["height"],
            crops, cuts, body.orientation, out,
            camera_motion=body.camera_motion, transitions=body.transitions,
            sharpen=body.sharpen, denoise=body.denoise,
            max_upscale=body.max_upscale, face_map=face_map,
            fps=video["meta"].get("fps") or 30.0, seed=body.seed,
            audio_source=audio_source, audio_offset=audio_offset, progress=progress,
        )
        result = {
            "file": out.name, "cuts": len(cuts), "synced": synced,
            "cut_details": cuts,
        }
        # copy the finished export to the user's output folder (e.g. ~/Downloads)
        output_dir = (store.load_library().get("output_dir") or "").strip()
        if output_dir:
            try:
                dest_dir = Path(output_dir).expanduser()
                dest_dir.mkdir(parents=True, exist_ok=True)
                dest = dest_dir / out.name
                shutil.copy2(out, dest)
                result["saved_to"] = str(dest)
            except OSError as exc:
                result["save_error"] = f"could not copy to {output_dir}: {exc}"
        return result

    return jobs.submit("edit", run)


@app.get("/api/videos/{video_id}/exports")
def list_exports(video_id: str):
    _video_or_404(video_id)
    exports = store.video_dir(video_id) / "exports"
    if not exports.exists():
        return []
    return sorted(f.name for f in exports.glob("*.mp4"))


@app.get("/api/videos/{video_id}/exports/{name}")
def download_export(video_id: str, name: str):
    _video_or_404(video_id)
    path = store.video_dir(video_id) / "exports" / Path(name).name
    if not path.exists():
        raise HTTPException(404, "export not found")
    return FileResponse(path, media_type="video/mp4", filename=path.name)


# ----------------------------------------------------------------- library

class FolderBody(BaseModel):
    path: str


class ScanBody(BaseModel):
    limit: int | None = None  # import at most N new videos this run


@app.get("/api/library")
def get_library():
    config = store.load_library()
    return {
        "folders": config.get("folders", []),
        "output_dir": config.get("output_dir", ""),
        "videos": len(store.list_videos()),
    }


@app.put("/api/library/output-dir")
def set_output_dir(body: FolderBody):
    """Set where finished exports are copied; empty disables the copy."""
    path = body.path.strip()
    if path and not Path(path).expanduser().is_dir():
        raise HTTPException(400, f"not a folder: {Path(path).expanduser()}")
    config = store.load_library()
    config["output_dir"] = path
    store.save_library(config)
    return get_library()


@app.post("/api/library/folders")
def add_library_folder(body: FolderBody):
    folder = Path(body.path).expanduser()
    if not folder.is_dir():
        raise HTTPException(400, f"not a folder: {folder}")
    config = store.load_library()
    if str(folder) not in config["folders"]:
        config["folders"].append(str(folder))
        store.save_library(config)
    return get_library()


@app.post("/api/library/folders/remove")
def remove_library_folder(body: FolderBody):
    config = store.load_library()
    config["folders"] = [f for f in config["folders"] if f != body.path]
    store.save_library(config)
    return get_library()


@app.post("/api/library/scan")
def scan_library(body: ScanBody | None = None):
    """Find new videos in the library folders and import them one at a time.

    A single sequential job: each new file is registered, proxied and analyzed
    before the next starts, so a big NAS folder doesn't fan out into dozens of
    concurrent ffmpeg/model runs. A file that fails (corrupt, still uploading,
    NAS hiccup) is skipped and reported — it never aborts the whole run. With
    `limit` set, at most that many videos are imported per run, so a huge
    backlog can be worked through in batches.
    """
    folders = store.load_library().get("folders", [])
    if not folders:
        raise HTTPException(400, "add a library folder first")
    limit = body.limit if body else None

    def run(progress=None):
        if progress:
            progress("scanning folders", pct=0)
        existing = {v["source_path"] for v in store.list_videos()}
        new_files = library.find_new(library.scan_folders(folders), existing)
        batch = new_files[:limit] if limit else new_files
        imported, failed = [], []
        for i, path in enumerate(batch):
            label = f"{path.name} ({i + 1}/{len(batch)})"
            if progress:
                progress(f"importing {label}", pct=int(((i) / max(len(batch), 1)) * 100))
            try:
                meta = probe.probe(str(path))
                video = store.add_video(str(path), path.name, meta)
                _prepare(video, progress=lambda msg, **kw: progress and progress(f"{label}: {msg}", **kw))
                _run_analysis(video, AnalyzeBody(),
                              progress=lambda msg, **kw: progress and progress(f"{label}: {msg}", **kw))
                imported.append(video["name"])
            except Exception as e:
                failed.append({"file": str(path), "error": str(e).split("\n")[0]})
        return {"scanned": len(folders), "new": len(new_files),
                "imported": imported, "failed": failed,
                "remaining": len(new_files) - len(batch)}

    return jobs.submit("library scan", run)


@app.get("/api/library/best")
def library_best(limit: int = 20):
    """Cross-video best-of: funniest moments and most exaggerated expressions."""
    items = [(v, store.load_artifact(v["id"], "analysis")) for v in store.list_videos()]
    items = [(v, a) for v, a in items if a]
    return {
        "analyzed_videos": len(items),
        "fun": library.top_fun_moments(items, limit),
        "expressions": library.top_expressions(items, limit),
    }


# -------------------------------------------------------------------- jobs

@app.get("/api/jobs/{job_id}")
def get_job(job_id: str):
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(404, "job not found")
    return job


app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web")

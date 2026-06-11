"""FastAPI app: REST API + static single-page frontend."""

from __future__ import annotations

import shutil
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import align, audio, detect, editor, jobs, lyrics, probe, store, vision

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


@app.post("/api/videos/register")
def register_video(body: RegisterBody):
    src = Path(body.path).expanduser()
    if not src.exists():
        raise HTTPException(400, f"no file at {src}")
    meta = probe.probe(str(src))
    video = store.add_video(str(src), body.name or src.name, meta)
    job = jobs.submit("proxy", lambda progress=None: _prepare(video, progress) and None)
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
    job = jobs.submit("proxy", lambda progress=None: _prepare(video, progress) and None)
    return {"video": video, "job": job["id"]}


@app.get("/api/videos/{video_id}")
def get_video(video_id: str):
    video = _video_or_404(video_id)
    return {
        **video,
        "has_proxy": (store.video_dir(video_id) / "proxy.mp4").exists(),
        "analysis": store.load_artifact(video_id, "analysis"),
        "crops": store.load_artifact(video_id, "crops") or {},
        "lyrics": store.load_artifact(video_id, "lyrics"),
        "sync": store.load_artifact(video_id, "sync"),
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


@app.post("/api/videos/{video_id}/analyze")
def analyze(video_id: str, body: AnalyzeBody):
    video = _video_or_404(video_id)

    def run(progress=None):
        _prepare(video, progress)
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
        return {"songs": len(result["songs"]), "fun_moments": len(result["fun_moments"])}

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
        result = align.align(proxy, str(ref_path), progress=progress)
        if result["duration"] <= 0:
            raise RuntimeError("no overlap found — is this a recording of a song in this video?")
        result["file"] = file.filename
        result["ref_path"] = str(ref_path)
        store.save_artifact(video_id, "sync", result)
        return result

    return jobs.submit("sync", run)


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


class EditBody(BaseModel):
    start: float
    end: float
    orientation: str = "horizontal"  # or "vertical"
    switch_s: float = 4.0            # target cadence (smart mode snaps it to beats)
    views: list[str] | None = None   # subset of crop names; default all
    smart: bool = True               # content-aware, beat-aligned switching
    camera_motion: bool = False      # subtle handheld drift / push-in / pano pan
    transitions: bool = False        # glide between views (connects the scene)
    seed: int | None = None


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


@app.post("/api/videos/{video_id}/edit")
def make_edit(video_id: str, body: EditBody):
    video = _video_or_404(video_id)
    crops = store.load_artifact(video_id, "crops") or {}
    views = body.views or list(crops.keys())
    if not views or any(v not in crops for v in views):
        raise HTTPException(400, "define crops first (or unknown view name)")
    if body.orientation not in editor.PRESETS:
        raise HTTPException(400, "orientation must be horizontal or vertical")

    def run(progress=None):
        analysis = store.load_artifact(video_id, "analysis")
        out_w, out_h = editor.PRESETS[body.orientation]

        # if a clean recording has been aligned to this video, use it as the
        # soundtrack (muting the original) instead of the camera audio, and clamp
        # the export to the span it actually covers so the audio never runs out
        sync = store.load_artifact(video_id, "sync")
        audio_source, audio_offset = None, 0.0
        start, end = body.start, body.end
        if sync and Path(sync.get("ref_path", "")).exists():
            audio_source, audio_offset = sync["ref_path"], sync["offset"]
            cover_start, cover_end = sync["offset"], sync["offset"] + sync["duration"]
            start, end = max(start, cover_start), min(end, cover_end)
            if end - start < 1.0:
                raise RuntimeError(
                    "export range falls outside the aligned recording "
                    f"({cover_start:.1f}s–{cover_end:.1f}s)"
                )

        if body.smart:
            proxy = str(store.video_dir(video_id) / "proxy.mp4")
            view_crops = {v: crops[v] for v in views}
            if progress:
                progress("measuring per-view motion")
            tracks = detect.view_activity(proxy, view_crops, start, end)
            activity = detect.activity_scores(tracks)
            pano = {v for v in views if editor.is_pano(crops[v], out_w / out_h)}
            cuts = editor.build_smart_cutlist(
                views, start, end,
                switch_s=body.switch_s,
                beats=(analysis or {}).get("beats"),
                events=_events_from_analysis(analysis, start, end),
                activity=activity, pano_views=pano, seed=body.seed,
            )
        else:
            cuts = editor.build_cutlist(views, start, end, body.switch_s, body.seed)
        suffix = "_synced" if audio_source else ""
        out = store.video_dir(video_id) / "exports" / (
            f"edit_{int(start)}-{int(end)}_{body.orientation}{suffix}.mp4"
        )
        editor.render(
            video["source_path"], video["meta"]["width"], video["meta"]["height"],
            crops, cuts, body.orientation, out,
            camera_motion=body.camera_motion, transitions=body.transitions,
            fps=video["meta"].get("fps") or 30.0, seed=body.seed,
            audio_source=audio_source, audio_offset=audio_offset, progress=progress,
        )
        return {"file": out.name, "cuts": len(cuts), "synced": bool(audio_source)}

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


# -------------------------------------------------------------------- jobs

@app.get("/api/jobs/{job_id}")
def get_job(job_id: str):
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(404, "job not found")
    return job


app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web")

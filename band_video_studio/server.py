"""FastAPI app: REST API + static single-page frontend."""

from __future__ import annotations

import shutil
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import audio, detect, editor, jobs, lyrics, probe, store, vision

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
    sweep_chat: bool = True         # sparse smile sweep over non-music stretches
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
                proxy, result, duration, sweep_chat=body.sweep_chat, progress=progress
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
    switch_s: float = 4.0
    views: list[str] | None = None   # subset of crop names; default all
    seed: int | None = None


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
        cuts = editor.build_cutlist(views, body.start, body.end, body.switch_s, body.seed)
        out = store.video_dir(video_id) / "exports" / (
            f"edit_{int(body.start)}-{int(body.end)}_{body.orientation}.mp4"
        )
        editor.render(
            video["source_path"], video["meta"]["width"], video["meta"]["height"],
            crops, cuts, body.orientation, out, progress=progress,
        )
        return {"file": out.name, "cuts": len(cuts)}

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

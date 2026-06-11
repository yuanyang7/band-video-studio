"""ffprobe metadata, proxy generation, audio extraction, frame sampling.

Everything is an ffmpeg subprocess; analysis works on downsampled data so 4K
sources stay cheap. Only the final export touches the original file.
"""

from __future__ import annotations

import json
import subprocess
import threading
from collections import defaultdict
from pathlib import Path

import numpy as np

AUDIO_SR = 16000  # mono 16 kHz — what YAMNet expects


def _run(cmd: list[str]) -> bytes:
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(f"{cmd[0]} failed: {result.stderr.decode(errors='replace')[-2000:]}")
    return result.stdout


def probe(path: str) -> dict:
    """Return duration, resolution, fps, codec info for a media file."""
    out = _run([
        "ffprobe", "-v", "error", "-print_format", "json",
        "-show_format", "-show_streams", path,
    ])
    info = json.loads(out)
    video = next((s for s in info["streams"] if s["codec_type"] == "video"), None)
    audio = next((s for s in info["streams"] if s["codec_type"] == "audio"), None)
    fps = 0.0
    if video and video.get("avg_frame_rate", "0/0") != "0/0":
        num, den = video["avg_frame_rate"].split("/")
        fps = float(num) / float(den) if float(den) else 0.0
    return {
        "duration": float(info["format"].get("duration", 0)),
        "width": video["width"] if video else 0,
        "height": video["height"] if video else 0,
        "fps": round(fps, 3),
        "video_codec": video["codec_name"] if video else None,
        "audio_codec": audio["codec_name"] if audio else None,
    }


# one lock per proxy path: a second caller (e.g. analyze clicked while the
# register-time transcode is still running) waits instead of reading a
# half-written file
_proxy_locks: defaultdict[str, threading.Lock] = defaultdict(threading.Lock)


def _is_valid_media(path: Path) -> bool:
    try:
        probe(str(path))
        return True
    except RuntimeError:
        return False


def make_proxy(src: str, dest: Path, height: int = 540, progress=None) -> Path:
    """Downscaled H.264 proxy for browser playback and frame sampling.

    Safe under concurrency: transcodes to a temp file and renames atomically,
    so dest only ever exists complete. A pre-existing broken dest (from an
    interrupted run) is detected and re-transcoded.
    """
    with _proxy_locks[str(dest)]:
        if dest.exists():
            if _is_valid_media(dest):
                return dest
            dest.unlink()  # leftover partial file from a crashed/killed run
        if progress:
            progress("transcoding proxy")
        tmp = dest.with_name(dest.name + ".part.mp4")
        _run([
            "ffmpeg", "-y", "-i", src,
            "-vf", f"scale=-2:{height}",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "26",
            "-c:a", "aac", "-b:a", "96k",
            "-movflags", "+faststart",
            str(tmp),
        ])
        tmp.replace(dest)
    return dest


def extract_audio(src: str, start: float = 0.0, duration: float | None = None) -> np.ndarray:
    """Decode audio to float32 mono PCM at AUDIO_SR."""
    cmd = ["ffmpeg", "-v", "error"]
    if start:
        cmd += ["-ss", str(start)]
    cmd += ["-i", src]
    if duration is not None:
        cmd += ["-t", str(duration)]
    cmd += ["-vn", "-ac", "1", "-ar", str(AUDIO_SR), "-f", "f32le", "-"]
    raw = _run(cmd)
    return np.frombuffer(raw, dtype=np.float32)


def extract_frame_jpeg(src: str, t: float, height: int = 540) -> bytes:
    """Single frame at time t as JPEG bytes (for the crop editor and vision)."""
    return _run([
        "ffmpeg", "-v", "error", "-ss", str(t), "-i", src,
        "-frames:v", "1", "-vf", f"scale=-2:{height}",
        "-f", "image2", "-c:v", "mjpeg", "-q:v", "4", "-",
    ])


def sample_frames(src: str, start: float, end: float, interval: float, height: int = 360):
    """Yield (timestamp, jpeg_bytes) every `interval` seconds in [start, end)."""
    t = start
    while t < end:
        yield t, extract_frame_jpeg(src, t, height=height)
        t += interval

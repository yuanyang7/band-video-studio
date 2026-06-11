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


PROXY_HEIGHT = 1080  # high enough for small-face detection in wide shots


def make_proxy(src: str, dest: Path, height: int = PROXY_HEIGHT, progress=None) -> Path:
    """Downscaled H.264 proxy for browser playback and frame sampling.

    Safe under concurrency: transcodes to a temp file and renames atomically,
    so dest only ever exists complete. A pre-existing broken or outdated
    (wrong height) dest is detected and re-transcoded. Uses the macOS
    hardware encoder when available, falling back to libx264.
    """
    with _proxy_locks[str(dest)]:
        if dest.exists():
            if _is_valid_media(dest) and probe(str(dest))["height"] == height:
                return dest
            dest.unlink()  # partial file, or proxy from an older lower-res version
        if progress:
            progress("transcoding proxy", pct=5)
        tmp = dest.with_name(dest.name + ".part.mp4")
        base = ["ffmpeg", "-y", "-i", src, "-vf", f"scale=-2:{height}"]
        tail = ["-c:a", "aac", "-b:a", "96k", "-movflags", "+faststart", str(tmp)]
        try:
            _run(base + ["-c:v", "h264_videotoolbox", "-b:v", "6M"] + tail)
        except RuntimeError:
            _run(base + ["-c:v", "libx264", "-preset", "veryfast", "-crf", "26"] + tail)
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


def read_gray_frames(src: str, start: float, end: float, fps: float, height: int = 180):
    """Yield (timestamp, grayscale ndarray) over [start, end) in ONE ffmpeg pass.

    Decodes downscaled single-channel rawvideo so per-region motion can be
    measured cheaply — far faster than per-frame seeks via extract_frame_jpeg.
    """
    if end <= start or fps <= 0:
        return
    # probe the scaled width so we can reshape the raw byte stream
    width = max(2, round(probe(src)["width"] * height / max(1, probe(src)["height"])))
    width -= width % 2
    cmd = [
        "ffmpeg", "-v", "error", "-ss", str(start), "-i", src, "-t", str(end - start),
        "-vf", f"fps={fps},scale={width}:{height}",
        "-pix_fmt", "gray", "-f", "rawvideo", "-",
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    frame_bytes = width * height
    i = 0
    try:
        while True:
            buf = proc.stdout.read(frame_bytes)
            if len(buf) < frame_bytes:
                break
            frame = np.frombuffer(buf, dtype=np.uint8).reshape(height, width)
            yield start + i / fps, frame
            i += 1
    finally:
        proc.stdout.close()
        proc.wait()

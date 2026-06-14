"""I/O for alignment: ffmpeg-backed audio decode and aligned-audio muxing.

Everything here shells out to ffmpeg; the rest of the package is pure numpy.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import TYPE_CHECKING, Union

import numpy as np

if TYPE_CHECKING:
    from .core import AlignmentResult

AUDIO_SR = 16000  # mono 16 kHz — matches the rate the onset envelope expects


class AlignmentError(RuntimeError):
    """Raised by the alignment package for decode, sanity, or ffmpeg failures."""


def _run(cmd: list[str]) -> bytes:
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        raise AlignmentError(
            f"{cmd[0]} failed: {result.stderr.decode(errors='replace')[-2000:]}"
        )
    return result.stdout


def extract_audio(src: str, start: float = 0.0, duration: float | None = None) -> np.ndarray:
    """Decode audio to float32 mono PCM at AUDIO_SR.

    Call directly when you want raw samples for your own analysis without going
    through `align()`. Requires ffmpeg on PATH.
    """
    cmd = ["ffmpeg", "-v", "error"]
    if start:
        cmd += ["-ss", str(start)]
    cmd += ["-i", src]
    if duration is not None:
        cmd += ["-t", str(duration)]
    cmd += ["-vn", "-ac", "1", "-ar", str(AUDIO_SR), "-f", "f32le", "-"]
    raw = _run(cmd)
    return np.frombuffer(raw, dtype=np.float32)


def mux_aligned_audio(
    video: str,
    audio: str,
    alignment: Union["AlignmentResult", float],
    output: str | Path,
    *,
    start: float | None = None,
    end: float | None = None,
    video_codec: str = "copy",
    audio_codec: str = "aac",
    audio_bitrate: str = "192k",
) -> Path:
    """Produce a new file with `audio` muxed onto `video` at the aligned offset.

    `alignment` is either an `AlignmentResult` or a bare `float` offset
    (seconds into the video where the reference audio begins).

    `start`/`end` are times in the VIDEO timeline. If omitted and `alignment`
    is an `AlignmentResult`, they default to its `covered_range()` so the
    output never references audio past the end of the reference recording.
    Explicit values outside that range raise `AlignmentError`.

    Defaults stream-copy the video (no re-encode) and re-encode audio to AAC.
    Requires ffmpeg on PATH.
    """
    # late import: AlignmentResult lives in core, which imports from this module
    from .core import AlignmentResult

    if isinstance(alignment, AlignmentResult):
        offset = alignment.offset
        cov_start, cov_end = alignment.covered_range()
    else:
        offset = float(alignment)
        cov_start, cov_end = None, None

    if start is None:
        start = cov_start if cov_start is not None else offset
    if end is None:
        if cov_end is None:
            raise AlignmentError(
                "`end` is required when `alignment` is a bare offset; pass an "
                "AlignmentResult or specify start/end explicitly."
            )
        end = cov_end

    if end <= start:
        raise AlignmentError(f"end ({end}) must be greater than start ({start})")
    if cov_start is not None and (start < cov_start - 1e-3 or end > cov_end + 1e-3):
        raise AlignmentError(
            f"requested span [{start:.3f}, {end:.3f}] falls outside the aligned "
            f"recording's covered range [{cov_start:.3f}, {cov_end:.3f}]"
        )

    a_start = max(0.0, start - offset)
    a_end = max(a_start, end - offset)

    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg", "-y",
        "-ss", f"{start:.3f}", "-to", f"{end:.3f}", "-i", str(video),
        "-ss", f"{a_start:.3f}", "-to", f"{a_end:.3f}", "-i", str(audio),
        "-map", "0:v:0", "-map", "1:a:0",
        "-c:v", video_codec,
        "-c:a", audio_codec, "-b:a", audio_bitrate,
        "-movflags", "+faststart",
        str(out_path),
    ]
    _run(cmd)
    return out_path

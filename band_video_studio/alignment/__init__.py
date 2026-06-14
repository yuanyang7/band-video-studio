"""Audio-to-video alignment as a reusable library.

Two-device band-rehearsal setup: a phone films the take while a separate
recorder captures the same performance in high quality. This package finds
*where* in the video that performance sits and *how far* it is offset, then
optionally muxes the clean audio onto the matching video span.

Algorithm: decode both inputs to mono 16 kHz PCM, build spectral-flux onset
envelopes (timbre/level invariant — only the rhythm of attacks matters), and
locate the reference inside the video via FFT-based normalized cross-correlation.
A single global offset describes the match; the peak correlation (0..1) is the
confidence. No drift correction.

Dependencies: numpy + ffmpeg on PATH. Nothing else from band_video_studio is
required — this sub-package is safe to import in isolation.

Quickstart::

    from band_video_studio.alignment import align, mux_aligned_audio

    result = align("take.mp4", "clean.wav")
    print(result.offset, result.confidence)         # e.g. 12.347 0.872

    if result.is_confident():
        mux_aligned_audio("take.mp4", "clean.wav", result, "out.mp4")

See `band_video_studio/alignment/README.md` for the full API reference,
recipes, and assumptions/limits.
"""

from .core import AlignmentResult, align
from .io import AUDIO_SR, AlignmentError, extract_audio, mux_aligned_audio
from .signal import ONSET_HOP, ONSET_WIN, cross_correlate_offset, onset_envelope

__all__ = [
    "AUDIO_SR",
    "ONSET_HOP",
    "ONSET_WIN",
    "AlignmentError",
    "AlignmentResult",
    "align",
    "cross_correlate_offset",
    "extract_audio",
    "mux_aligned_audio",
    "onset_envelope",
]

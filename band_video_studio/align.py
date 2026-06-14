"""Backwards-compatible shim: alignment now lives in `band_video_studio.alignment`.

Kept so internal code and external callers using `from band_video_studio import align`
or `from band_video_studio.align import cross_correlate_offset` keep working.
New code should import from `band_video_studio.alignment` directly.
"""

from .alignment import (  # noqa: F401
    AUDIO_SR,
    ONSET_HOP,
    ONSET_WIN,
    AlignmentError,
    AlignmentResult,
    align,
    cross_correlate_offset,
    extract_audio,
    mux_aligned_audio,
    onset_envelope,
)

"""Top-level `align()` entry point and the `AlignmentResult` value type."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from .io import AUDIO_SR, AlignmentError, extract_audio
from .signal import ONSET_HOP, cross_correlate_offset, onset_envelope


@dataclass(frozen=True)
class AlignmentResult:
    """Result of aligning a reference recording against a video's soundtrack.

    All times are in seconds.

    - `offset`         — where the reference begins in the video timeline.
    - `duration`       — how much of the reference fits before the video ends
                         (`min(ref_duration, video_duration - offset)`).
    - `confidence`     — 0..1, peak of normalized cross-correlation. As a rule
                         of thumb: > 0.3 is usually a real match for noisy
                         phone-vs-recorder captures; matched sources push much
                         higher. Below ~0.15 is almost always noise.
    - `ref_duration`   — full length of the reference recording.
    - `video_duration` — full length of the video's audio track.
    """

    offset: float
    duration: float
    confidence: float
    ref_duration: float
    video_duration: float

    def is_confident(self, threshold: float = 0.3) -> bool:
        return self.confidence >= threshold

    def covered_range(self) -> tuple[float, float]:
        """Video-timeline span that the reference recording covers."""
        return (self.offset, self.offset + self.duration)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "AlignmentResult":
        return cls(
            offset=float(d["offset"]),
            duration=float(d["duration"]),
            confidence=float(d["confidence"]),
            ref_duration=float(d["ref_duration"]),
            video_duration=float(d["video_duration"]),
        )


def align(video: str, audio: str, *, progress=None) -> AlignmentResult:
    """Locate `audio` (clean reference recording) inside `video`'s soundtrack.

    Decodes both to mono 16 kHz PCM, builds spectral-flux onset envelopes, and
    finds the best lag via FFT-based normalized cross-correlation. Returns an
    `AlignmentResult`. FFT-backed, so a multi-minute reference against a
    multi-hour video stays cheap.

    Raises `AlignmentError` if either input fails to decode or if the reference
    is longer than the video. Requires ffmpeg on PATH.

    >>> r = align("take.mp4", "clean.wav")
    >>> print(r.offset, r.confidence)
    """
    if progress:
        progress("decoding audio", pct=10)
    video_samples = extract_audio(video)
    ref_samples = extract_audio(audio)
    if len(video_samples) == 0 or len(ref_samples) == 0:
        raise AlignmentError("could not decode audio from the video or the reference file")

    if progress:
        progress("matching onset envelopes", pct=50)
    _, video_env = onset_envelope(video_samples, AUDIO_SR)
    _, ref_env = onset_envelope(ref_samples, AUDIO_SR)
    if len(ref_env) > len(video_env):
        raise AlignmentError("the reference recording is longer than the video")

    hop_s = ONSET_HOP / AUDIO_SR
    offset, confidence = cross_correlate_offset(video_env, ref_env, hop_s)

    video_duration = len(video_samples) / AUDIO_SR
    ref_duration = len(ref_samples) / AUDIO_SR
    duration = min(ref_duration, video_duration - offset)
    return AlignmentResult(
        offset=round(offset, 3),
        duration=round(max(0.0, duration), 3),
        confidence=round(confidence, 3),
        ref_duration=round(ref_duration, 3),
        video_duration=round(video_duration, 3),
    )

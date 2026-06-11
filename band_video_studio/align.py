"""Align a clean reference recording to the take captured in a video.

The band films a rehearsal on a phone (distant, noisy mic) while a separate
recorder captures the same performance in high quality. This finds *where* in
the video that performance sits and *how far* it is offset, so we can cut the
matching video span and drop the clean audio onto it.

Both signals are the same take, so a single global offset describes the match.
We compare spectral-flux onset envelopes (timbre/level invariant — only the
rhythm of energy changes matters) via normalized cross-correlation. The peak
gives the lag; its height (0..1) is the confidence.
"""

from __future__ import annotations

import numpy as np

from . import probe
from .audio import ONSET_HOP, onset_envelope


def cross_correlate_offset(
    long_env: np.ndarray,
    short_env: np.ndarray,
    hop_s: float,
) -> tuple[float, float]:
    """Best lag (seconds) of `short_env` inside `long_env` and its 0..1 confidence.

    Normalized cross-correlation over every valid placement of the short
    envelope within the long one (FFT-based, so a multi-minute clip against a
    multi-hour video stays cheap). The returned offset is the time into the long
    signal where the short one begins; confidence is the peak correlation, sub-hop
    refined by parabolic interpolation around it.
    """
    n, m = len(long_env), len(short_env)
    if m == 0 or n < m:
        return 0.0, 0.0

    a = long_env - long_env.mean()
    b = short_env - short_env.mean()
    b_norm = float(np.sqrt(np.sum(b * b)))
    if b_norm < 1e-9:
        return 0.0, 0.0

    # correlation via convolution with the reversed kernel; lag L -> index m-1+L
    size = 1 << (n + m - 1).bit_length()
    corr = np.fft.irfft(np.fft.rfft(a, size) * np.fft.rfft(b[::-1], size), size)
    corr = corr[m - 1 : n]  # valid lags L = 0 .. n-m

    # per-lag norm of the long window via prefix sums of a^2
    energy = np.concatenate([[0.0], np.cumsum(a * a)])
    win_norm = np.sqrt(np.maximum(energy[m : n + 1] - energy[: n - m + 1], 1e-12))
    ncc = corr / (win_norm * b_norm)

    k = int(np.argmax(ncc))
    peak = float(ncc[k])
    # parabolic sub-hop refinement of the lag using the two neighbouring bins
    lag = float(k)
    if 0 < k < len(ncc) - 1:
        y0, y1, y2 = ncc[k - 1], ncc[k], ncc[k + 1]
        denom = y0 - 2 * y1 + y2
        if abs(denom) > 1e-9:
            lag += 0.5 * (y0 - y2) / denom
    return max(0.0, lag * hop_s), max(0.0, peak)


def align(video_source: str, ref_audio: str, progress=None) -> dict:
    """Locate `ref_audio` within `video_source`'s soundtrack.

    Returns {offset, duration, confidence, video_duration} in seconds — `offset`
    is where the reference begins in the video; `duration` is how much of the
    reference fits before the video ends.
    """
    if progress:
        progress("decoding audio")
    video_samples = probe.extract_audio(video_source)
    ref_samples = probe.extract_audio(ref_audio)
    if len(video_samples) == 0 or len(ref_samples) == 0:
        raise RuntimeError("could not decode audio from the video or the reference file")

    if progress:
        progress("matching onset envelopes")
    _, video_env = onset_envelope(video_samples, probe.AUDIO_SR)
    _, ref_env = onset_envelope(ref_samples, probe.AUDIO_SR)
    if len(ref_env) > len(video_env):
        raise RuntimeError("the reference recording is longer than the video")

    hop_s = ONSET_HOP / probe.AUDIO_SR
    offset, confidence = cross_correlate_offset(video_env, ref_env, hop_s)

    video_duration = len(video_samples) / probe.AUDIO_SR
    ref_duration = len(ref_samples) / probe.AUDIO_SR
    duration = min(ref_duration, video_duration - offset)
    return {
        "offset": round(offset, 3),
        "duration": round(max(0.0, duration), 3),
        "ref_duration": round(ref_duration, 3),
        "confidence": round(confidence, 3),
        "video_duration": round(video_duration, 3),
    }

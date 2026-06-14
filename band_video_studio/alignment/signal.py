"""Signal-processing primitives for audio alignment.

Pure numpy. No ffmpeg, no I/O. Useful directly when a caller already has PCM
in memory and wants to skip the wrapper or run cross-correlation against a
custom-built envelope.
"""

from __future__ import annotations

import numpy as np

ONSET_HOP = 512   # samples per onset frame (~32 ms at 16 kHz)
ONSET_WIN = 1024  # analysis window for the onset STFT


def onset_envelope(samples: np.ndarray, sample_rate: int) -> tuple[np.ndarray, np.ndarray]:
    """Spectral-flux onset strength per ~32 ms frame. Returns (times, envelope).

    Energy *increases* only (half-wave rectified): captures the rhythm of attacks
    independent of timbre or absolute level. That invariance is why a noisy phone
    capture aligns against a clean recorder take of the same performance.

    Call directly if you already have PCM samples and want to skip the ffmpeg
    decode in `alignment.io.extract_audio`.
    """
    if len(samples) < ONSET_WIN:
        return np.array([]), np.array([])
    n = 1 + (len(samples) - ONSET_WIN) // ONSET_HOP
    window = np.hanning(ONSET_WIN).astype(np.float32)
    frames = np.empty((n, ONSET_WIN), dtype=np.float32)
    for i in range(n):
        start = i * ONSET_HOP
        frames[i] = samples[start:start + ONSET_WIN] * window
    mag = np.abs(np.fft.rfft(frames, axis=1))
    flux = np.diff(mag, axis=0)
    flux[flux < 0] = 0.0
    env = flux.sum(axis=1)
    env = np.concatenate([[0.0], env])
    times = (np.arange(n) * ONSET_HOP + ONSET_WIN / 2) / sample_rate
    return times, env


def cross_correlate_offset(
    long_env: np.ndarray,
    short_env: np.ndarray,
    hop_s: float,
) -> tuple[float, float]:
    """Best lag (seconds) of `short_env` inside `long_env` and its 0..1 confidence.

    Normalized cross-correlation over every valid placement of the short
    envelope within the long one (FFT-based, so a multi-minute clip against a
    multi-hour video stays cheap). The returned offset is the time into the long
    signal where the short one begins; confidence is the peak correlation,
    sub-hop refined by parabolic interpolation around it.

    Call directly when you want to align against custom envelopes (e.g. RMS
    energy, chroma) instead of the default spectral-flux one.
    """
    n, m = len(long_env), len(short_env)
    if m == 0 or n < m:
        return 0.0, 0.0

    a = long_env - long_env.mean()
    b = short_env - short_env.mean()
    b_norm = float(np.sqrt(np.sum(b * b)))
    if b_norm < 1e-9:
        return 0.0, 0.0

    size = 1 << (n + m - 1).bit_length()
    corr = np.fft.irfft(np.fft.rfft(a, size) * np.fft.rfft(b[::-1], size), size)
    corr = corr[m - 1 : n]

    energy = np.concatenate([[0.0], np.cumsum(a * a)])
    win_norm = np.sqrt(np.maximum(energy[m : n + 1] - energy[: n - m + 1], 1e-12))
    ncc = corr / (win_norm * b_norm)

    k = int(np.argmax(ncc))
    peak = float(ncc[k])
    lag = float(k)
    if 0 < k < len(ncc) - 1:
        y0, y1, y2 = ncc[k - 1], ncc[k], ncc[k + 1]
        denom = y0 - 2 * y1 + y2
        if abs(denom) > 1e-9:
            lag += 0.5 * (y0 - y2) / denom
    return max(0.0, lag * hop_s), max(0.0, peak)

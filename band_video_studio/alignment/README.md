# `band_video_studio.alignment`

Find where a clean reference recording sits inside a video's noisy soundtrack,
and optionally swap the clean audio onto the video.

## What it does

Decodes both inputs to mono 16 kHz PCM, builds a spectral-flux **onset
envelope** for each (energy *increases* per ~32 ms frame — timbre- and
level-invariant, so phone mic vs. recorder of the same performance still
match on rhythm), and locates the reference inside the video via FFT-based
**normalized cross-correlation**. A single global offset describes the match;
the peak correlation (0..1) is the confidence.

A bare offset is all the editor needs to map any span of the video timeline
onto the matching span of the clean recording.

## Install

```bash
pip install band-video-studio
```

System requirement: `ffmpeg` and `ffprobe` on `PATH`. No other runtime
dependencies (the sub-package is numpy + stdlib + ffmpeg subprocess only —
nothing else from the parent project is imported).

## Quickstart

```python
from band_video_studio.alignment import align, mux_aligned_audio

result = align("take.mp4", "clean.wav")
print(result.offset, result.confidence)
# 12.347 0.872

if result.is_confident():
    mux_aligned_audio("take.mp4", "clean.wav", result, "out.mp4")
```

`align()` does pure compute and returns an `AlignmentResult`.
`mux_aligned_audio()` produces a new file that stream-copies the video and
muxes in the reference audio shifted by `result.offset`.

## API reference

| Symbol | Description |
| --- | --- |
| [`align(video, audio, *, progress=None)`](core.py) | Compute `AlignmentResult` for the reference inside the video. |
| [`AlignmentResult`](core.py) | Frozen dataclass: `offset`, `duration`, `confidence`, `ref_duration`, `video_duration`. Helpers `is_confident()`, `covered_range()`, `to_dict()`, `from_dict()`. |
| [`mux_aligned_audio(video, audio, alignment, output, *, start=None, end=None, ...)`](io.py) | Write a new file with the reference audio muxed at the aligned offset. Accepts `AlignmentResult` or a bare float offset. |
| [`AlignmentError`](io.py) | Single exception type for all package failures (decode, ffmpeg, bad spans). |
| [`extract_audio(src, start=0, duration=None)`](io.py) | Lower-level: decode any file to float32 mono PCM at `AUDIO_SR`. |
| [`onset_envelope(samples, sample_rate)`](signal.py) | Lower-level: spectral-flux onset envelope of PCM samples. |
| [`cross_correlate_offset(long_env, short_env, hop_s)`](signal.py) | Lower-level: best lag (seconds) and confidence of a short envelope inside a long one. |
| `AUDIO_SR`, `ONSET_HOP`, `ONSET_WIN` | Constants used internally; exported for callers building custom pipelines. |

## Assumptions & limits

- **Single global offset.** Both signals are assumed to be the same take, with
  no drift between recorders. There is no resampling / DTW pass — if your
  clocks drift over an hour-long take you'll see the alignment work near the
  start and slip toward the end. (For rehearsal-length takes this is
  effectively never a problem.)
- **Reference ≤ video.** If the reference recording is longer than the
  video's soundtrack, `align()` raises `AlignmentError`.
- **Confidence interpretation.** `confidence` is the peak normalized
  cross-correlation, so it depends on how similar the two captures are. A
  matched-source test signal can score `> 0.95`. A real phone-vs-recorder
  pairing typically lands `0.5..0.85`. Below ~`0.2` is almost always noise.
  Use `is_confident(threshold)` rather than a hard equality check — the
  default `0.3` is a sensible floor for the noisy case.
- **No drift correction, no per-region adjustment.** If you need either,
  build it yourself on top of `cross_correlate_offset` over windows.

## Recipes

### Batch-align a folder of takes against a reference

```python
from pathlib import Path
from band_video_studio.alignment import align

ref = "clean.wav"
for video in Path("takes").glob("*.mp4"):
    r = align(str(video), ref)
    print(f"{video.name}: offset={r.offset:.3f} conf={r.confidence:.2f}")
```

### Align against a custom envelope (e.g. RMS energy)

```python
import numpy as np
from band_video_studio.alignment import extract_audio, cross_correlate_offset, AUDIO_SR

def rms_env(samples, hop=512, win=1024):
    n = 1 + (len(samples) - win) // hop
    frames = np.stack([samples[i*hop:i*hop+win] for i in range(n)])
    return np.sqrt((frames ** 2).mean(axis=1))

video = extract_audio("take.mp4")
ref = extract_audio("clean.wav")
hop_s = 512 / AUDIO_SR
offset, conf = cross_correlate_offset(rms_env(video), rms_env(ref), hop_s)
```

### Mux only a sub-span of the aligned recording

```python
result = align("take.mp4", "clean.wav")
# pick a 30s slice from inside the covered range, in VIDEO timeline coords
cov_start, cov_end = result.covered_range()
mux_aligned_audio(
    "take.mp4", "clean.wav", result, "highlight.mp4",
    start=cov_start + 10, end=cov_start + 40,
)
```

## How it compares

A bare waveform cross-correlation needs the two signals to look alike sample
for sample — fine for two recordings off the same mixer split, useless for a
phone capture vs. a clean recorder. By cross-correlating *onset envelopes*
instead, we throw away timbre and absolute level and keep only the rhythm of
attacks, which is the same in both captures of the same performance. The cost
is one extra FFT per signal; the win is "it just works" across very different
recording chains.

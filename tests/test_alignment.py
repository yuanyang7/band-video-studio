import json
import subprocess

import numpy as np
import pytest

from band_video_studio.alignment import (
    AlignmentError,
    AlignmentResult,
    align,
    cross_correlate_offset,
    mux_aligned_audio,
)


# ----------------------------------------------------------- cross_correlate_offset


def test_recovers_known_offset():
    rng = np.random.default_rng(0)
    short = rng.standard_normal(200)
    long = rng.standard_normal(2000)
    offset = 750
    long[offset:offset + 200] = short
    hop_s = 0.032
    found, conf = cross_correlate_offset(long, short, hop_s)
    assert abs(found - offset * hop_s) < hop_s
    assert conf > 0.9


def test_offset_zero_at_start():
    rng = np.random.default_rng(1)
    short = rng.standard_normal(300)
    long = np.concatenate([short, rng.standard_normal(1700)])
    found, conf = cross_correlate_offset(long, short, 0.032)
    assert found < 0.05
    assert conf > 0.9


def test_weak_match_low_confidence():
    rng = np.random.default_rng(2)
    short = rng.standard_normal(200)
    long = rng.standard_normal(2000)
    _, conf = cross_correlate_offset(long, short, 0.032)
    assert conf < 0.5


def test_short_longer_than_long_is_safe():
    found, conf = cross_correlate_offset(np.zeros(50), np.zeros(100), 0.032)
    assert found == 0.0 and conf == 0.0


# --------------------------------------------------------------- AlignmentResult


def test_alignment_result_roundtrip_through_json():
    r = AlignmentResult(
        offset=1.234, duration=10.0, confidence=0.87,
        ref_duration=10.5, video_duration=30.0,
    )
    d = r.to_dict()
    # exact key set the server's artifact JSON has relied on
    assert set(d) == {"offset", "duration", "confidence", "ref_duration", "video_duration"}
    via_json = AlignmentResult.from_dict(json.loads(json.dumps(d)))
    assert via_json == r


def test_alignment_result_confidence_helpers():
    r = AlignmentResult(offset=0, duration=5, confidence=0.4,
                        ref_duration=5, video_duration=10)
    assert r.is_confident()
    assert r.is_confident(0.3)
    assert not r.is_confident(0.5)
    assert r.covered_range() == (0, 5)


# ----------------------------------------------------- align() + mux end-to-end


@pytest.fixture(scope="module")
def synthetic_take(tmp_path_factory):
    """A 10s video whose soundtrack contains a 4s reference clip starting at
    t=3s, plus the standalone reference file. The reference is a non-periodic
    rhythm of noise bursts so the cross-correlation has a single sharp peak
    (not a periodic ambiguity ladder). align() should recover offset≈3."""
    d = tmp_path_factory.mktemp("align_media")
    video = d / "take.mp4"
    ref = d / "ref.wav"

    # Bursts of pink noise at irregular times — unique signature, strong onsets.
    bursts = (
        "anoisesrc=d=0.15:c=pink:a=0.8:r=16000[b1];"
        "aevalsrc=0:d=0.35:s=16000[g1];"
        "anoisesrc=d=0.10:c=pink:a=0.8:r=16000[b2];"
        "aevalsrc=0:d=0.70:s=16000[g2];"
        "anoisesrc=d=0.20:c=pink:a=0.8:r=16000[b3];"
        "aevalsrc=0:d=0.30:s=16000[g3];"
        "anoisesrc=d=0.15:c=pink:a=0.8:r=16000[b4];"
        "aevalsrc=0:d=0.55:s=16000[g4];"
        "anoisesrc=d=0.10:c=pink:a=0.8:r=16000[b5];"
        "aevalsrc=0:d=0.40:s=16000[g5];"
        "anoisesrc=d=0.30:c=pink:a=0.8:r=16000[b6];"
        "aevalsrc=0:d=0.70:s=16000[g6];"
        "[b1][g1][b2][g2][b3][g3][b4][g4][b5][g5][b6][g6]"
        "concat=n=12:v=0:a=1[ref]"
    )
    # Standalone reference.
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y",
         "-filter_complex", bursts + ";[ref]aformat=channel_layouts=mono[mono]",
         "-map", "[mono]", str(ref)],
        check=True,
    )
    # Video soundtrack: 3s silence + the same reference (read from file) + 3s silence.
    track = (
        "[1:a]aresample=16000[r];"
        "aevalsrc=0:d=3:s=16000[s1];"
        "aevalsrc=0:d=3:s=16000[s2];"
        "[s1][r][s2]concat=n=3:v=0:a=1[aout]"
    )
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y",
         "-f", "lavfi", "-i", "testsrc2=size=320x180:rate=10:duration=10",
         "-i", str(ref),
         "-filter_complex", track, "-map", "0:v", "-map", "[aout]",
         "-c:v", "libx264", "-preset", "ultrafast", "-c:a", "aac",
         "-shortest", str(video)],
        check=True,
    )
    return video, ref


def test_align_recovers_offset_end_to_end(synthetic_take):
    video, ref = synthetic_take
    result = align(str(video), str(ref))
    assert isinstance(result, AlignmentResult)
    assert abs(result.offset - 3.0) < 0.1  # within ~100ms
    assert result.confidence > 0.5
    assert 3.5 < result.ref_duration < 4.5
    assert 9.5 < result.video_duration < 10.5


def test_align_raises_when_reference_longer_than_video(tmp_path):
    short_vid = tmp_path / "short.mp4"
    long_ref = tmp_path / "long.wav"
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y",
         "-f", "lavfi", "-i", "testsrc2=size=160x90:rate=10:duration=1",
         "-f", "lavfi", "-i", "sine=frequency=440:duration=1:sample_rate=16000",
         "-c:v", "libx264", "-preset", "ultrafast", "-c:a", "aac",
         "-shortest", str(short_vid)],
        check=True,
    )
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y",
         "-f", "lavfi", "-i", "sine=frequency=440:duration=5:sample_rate=16000",
         "-ac", "1", str(long_ref)],
        check=True,
    )
    with pytest.raises(AlignmentError):
        align(str(short_vid), str(long_ref))


def test_mux_aligned_audio_roundtrip(synthetic_take, tmp_path):
    video, ref = synthetic_take
    result = align(str(video), str(ref))
    out = tmp_path / "out.mp4"
    mux_aligned_audio(str(video), str(ref), result, out)
    assert out.exists() and out.stat().st_size > 0

    info = json.loads(subprocess.check_output([
        "ffprobe", "-v", "error", "-print_format", "json",
        "-show_format", "-show_streams", str(out),
    ]))
    codecs = {s["codec_type"] for s in info["streams"]}
    assert {"video", "audio"} <= codecs
    duration = float(info["format"]["duration"])
    # output covers result.duration (the overlap of ref inside video)
    assert abs(duration - result.duration) < 0.3


def test_mux_with_bare_offset_requires_explicit_span(synthetic_take, tmp_path):
    video, ref = synthetic_take
    out = tmp_path / "out.mp4"
    # bare float offset, no covered_range to fall back on
    with pytest.raises(AlignmentError):
        mux_aligned_audio(str(video), str(ref), 3.0, out)
    # explicit span works
    mux_aligned_audio(str(video), str(ref), 3.0, out, start=3.0, end=7.0)
    assert out.exists()


def test_mux_rejects_span_outside_covered_range(synthetic_take, tmp_path):
    video, ref = synthetic_take
    result = align(str(video), str(ref))
    out = tmp_path / "out.mp4"
    with pytest.raises(AlignmentError):
        # ask for video time well past offset+duration
        mux_aligned_audio(
            str(video), str(ref), result, out,
            start=result.offset, end=result.offset + result.duration + 5,
        )

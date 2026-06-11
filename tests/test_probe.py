import subprocess
import threading

import pytest

from band_video_studio import probe


@pytest.fixture(scope="module")
def tiny_video(tmp_path_factory):
    path = tmp_path_factory.mktemp("media") / "tiny.mp4"
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y",
         "-f", "lavfi", "-i", "testsrc2=size=320x180:rate=10:duration=2",
         "-f", "lavfi", "-i", "sine=frequency=440:duration=2",
         "-c:v", "libx264", "-preset", "ultrafast", "-c:a", "aac", "-shortest", str(path)],
        check=True,
    )
    return path


def test_make_proxy_heals_partial_file(tiny_video, tmp_path):
    dest = tmp_path / "proxy.mp4"
    dest.write_bytes(b"\x00" * 4096)  # simulates an interrupted transcode
    probe.make_proxy(str(tiny_video), dest, height=120)
    assert probe.probe(str(dest))["duration"] > 1.5  # valid, re-transcoded


def test_make_proxy_concurrent_calls_both_get_valid_file(tiny_video, tmp_path):
    dest = tmp_path / "proxy.mp4"
    errors = []

    def worker():
        try:
            probe.make_proxy(str(tiny_video), dest, height=120)
            probe.probe(str(dest))  # must always be readable when returned
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=worker) for _ in range(3)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors
    assert not list(dest.parent.glob("*.part.mp4"))  # no temp leftovers

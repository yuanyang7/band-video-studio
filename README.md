# Band Video Studio

Interactive app for analyzing and auto-editing band rehearsal videos (fixed-camera, long-form, possibly 4K).

## What it does

**Analysis**
- **Song detection** — finds start/end of each song rehearsal by audio energy (music vs. chatting).
- **Highlights** — energy peaks within songs (solos, big moments).
- **Fun moments** — samples video frames and uses Claude vision to spot laughter, funny reactions to mistakes, and other amusing moments. Results land on an interactive timeline; click a marker to jump there.

**Editing**
- **Lyrics matching** *(optional, needs `faster-whisper`)* — give it a song name + lyrics, it transcribes and aligns lyric lines to timestamps.
- **Multicam-style auto edit** — define a crop region per player on the fixed wide shot, and it cuts a song into a view-switching edit (horizontal 16:9 or vertical 9:16).

## Setup

Requires `ffmpeg`/`ffprobe` on PATH and [uv](https://docs.astral.sh/uv/).

```sh
uv sync                     # core
uv sync --extra lyrics      # + whisper lyrics alignment (large download)
```

Fun-moment detection calls the Claude API — set `ANTHROPIC_API_KEY` (audio-based analysis works without it).

## Run

```sh
uv run uvicorn band_video_studio.server:app --reload
```

Open http://127.0.0.1:8000.

Workflow in the UI:
1. **Add video** — upload, or register a local file path (recommended for big 4K files; nothing is copied).
2. **Analyze** — runs audio analysis (songs + highlights), optionally fun-moment detection on the chosen time range.
3. Browse the **timeline** — songs (green), highlights (amber), fun moments (red). Click to seek.
4. **Crops** — draw a box per player on a frame, then **Export edit** for a song segment.

## Architecture

```
band_video_studio/
  probe.py    ffprobe metadata + frame/proxy extraction
  audio.py    PCM energy analysis: song segmentation + highlights
  vision.py   frame-grid sampling + Claude vision fun-moment detection
  lyrics.py   optional whisper transcription + lyric line alignment
  editor.py   ffmpeg filtergraph multicam-style crop/cut renderer
  store.py    JSON metadata store under data/
  jobs.py     background job manager
  server.py   FastAPI app + REST API
web/          single-page frontend (player, timeline, crop editor)
```

All heavy lifting is ffmpeg subprocesses; analysis runs on a downsampled proxy/audio so 4K sources stay cheap until final export, which renders from the original.

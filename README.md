# Band Video Studio

Interactive app for analyzing and auto-editing band rehearsal videos (fixed-camera, long-form, possibly 4K).

## What it does

**Analysis — free, local models by default (no API calls)**
- **Song detection** — YAMNet audio classification finds sustained music segments → start/end of each song rehearsal.
- **Highlights** — energy peaks within songs (solos, big moments).
- **Fun moments** — YAMNet laughter detection fused with MediaPipe face-blendshape smile detection on sampled frames, plus a sparse smile sweep over chat sections. Results land on an interactive timeline; click a marker to jump there.
- **Claude deep pass** *(optional, opt-in, costs API tokens)* — sends only the pre-filtered candidate windows (a handful of small frames each) to Claude vision for a verdict + clip caption, e.g. "funny reaction to a flubbed note". Never scans the whole video.

**Editing**
- **Lyrics matching** *(optional, local whisper)* — give it a song name + lyrics, it transcribes and aligns lyric lines to timestamps.
- **Multicam-style auto edit** — define a crop region per player on the fixed wide shot, and it cuts a song into a view-switching edit (horizontal 16:9 or vertical 9:16), rendered from the original file so 4K detail survives the crop.

## Setup

Requires `ffmpeg`/`ffprobe` on PATH and [uv](https://docs.astral.sh/uv/).

```sh
uv sync                       # core (all local analysis)
uv sync --extra lyrics        # + whisper lyrics alignment (large download)
uv sync --extra claude        # + optional Claude deep pass (needs ANTHROPIC_API_KEY)
```

Local model files (YAMNet ~4 MB, Face Landmarker ~3 MB) download automatically on first analysis into `data/models/`.

## Run

```sh
uv run uvicorn band_video_studio.server:app --reload
```

Open http://127.0.0.1:8000.

Workflow in the UI:
1. **Add video** — upload, or register a local file path (recommended for big 4K files; nothing is copied).
2. **Analyze** — songs + highlights + fun moments; optionally tick the Claude deep pass.
3. Browse the **timeline** — songs (green), highlights (amber), fun moments (red). Click to seek; click list items to jump; double-click a song to fill the export range.
4. **Crops** — grab a frame, drag a box per player, save, then **Export edit**.

## Architecture

```
band_video_studio/
  probe.py    ffprobe metadata + proxy/audio/frame extraction (ffmpeg)
  models.py   download-and-cache for local MediaPipe model files
  audio.py    YAMNet music/laugh classification + energy → songs, highlights, laugh candidates
  detect.py   face-blendshape smile detection + audio/visual fusion → fun moments
  vision.py   optional Claude deep pass on candidate windows only
  lyrics.py   optional faster-whisper transcription + lyric line alignment
  editor.py   ffmpeg filtergraph multicam-style crop/cut renderer
  store.py    JSON metadata store under data/
  jobs.py     background job manager
  server.py   FastAPI app + REST API
web/          single-page frontend (player, timeline, crop editor)
tests/        unit tests for segmentation, cutlist, crop fitting, lyric alignment
```

Analysis runs on a 540p proxy and 16 kHz mono audio, so 4K sources stay cheap; only the final export renders from the original.

## Cost model

Everything runs locally for free. The only thing that can spend money is the explicit **Claude deep pass** checkbox, and it only judges the candidate windows the free detectors already found — typically a few API calls per rehearsal video, each with ~6 small (300 px) frames.

# Band Video Studio

Interactive app for analyzing and auto-editing band rehearsal videos (fixed-camera, long-form, possibly 4K).

## What it does

**Analysis — free, local models by default (no API calls). Runs automatically on import; re-detect any time from the UI.**
- **Song detection** — YAMNet audio classification finds sustained music segments → start/end of each song rehearsal.
- **Highlights** — energy peaks within songs (solos, big moments).
- **Singing detection** — YAMNet vocal classes mark sung stretches; entrances drive the auto edit's cut to the singer.
- **Fun moments** — YAMNet laughter detection fused with MediaPipe face-blendshape smile detection on sampled frames, plus a sparse smile sweep over chat sections. Results land on an interactive timeline; click a marker to jump there.
- **Claude deep pass** *(optional, opt-in, costs API tokens)* — sends only the pre-filtered candidate windows (a handful of small frames each) to Claude vision for a verdict + clip caption, e.g. "funny reaction to a flubbed note". Never scans the whole video.

**Editing**
- **Sync to recording** — upload a clean, separately-recorded take of a song that plays in the video; onset-envelope cross-correlation locates *where* in the video that take sits. Audition the result with a two-track mixer (camera audio vs. aligned recording, toggle either) before exporting. When an alignment exists, **Export** crops the aligned span and replaces the camera audio with your recording (original muted). Fully local, no API.
- **Shooting simulation** — define a virtual camera (crop region) per player on the fixed wide shot, optionally tagging one as *singer* and one as *wide*. The cut list switches between those cameras like a multicam shoot: to the singer when the vocal comes in (and while singing is strong), to the most active player on energy peaks/instrumentals, back to the wide view at a regular cadence, with a mild bias toward centre-stage framing and an evenness guarantee so every player gets screen time. Optional glide transitions connect the views instead of hard cuts.
- **Export** — independent of the panels above: pick a range on the timeline, orientation (16:9 or 9:16), and an optional file name; rendered from the original file so 4K detail survives the crop.
- **Lyrics matching** *(optional, local whisper, experimental)* — give it a song name + lyrics, it transcribes and aligns lyric lines to timestamps.

**Library**
- **Watched folders** — point the library at folders (e.g. a NAS mount); **Scan** registers any new video files (no copying) and imports them one at a time — proxy + detection sequentially, so a big folder never fans out into dozens of concurrent ffmpeg/model runs. Deep trees are walked cheaply (hidden dirs pruned), bad files are skipped and reported, and an optional per-scan limit lets you work through a huge backlog in batches.
- **Cross-video best-of** — once videos are analyzed, **Best moments** aggregates the cached results across the whole library: globally funniest moments (fused laugh/smile score) and most exaggerated expressions (peak smile blendshape). Click any entry to open that video at that moment. Smile scores are only roughly comparable across videos.

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
1. **Add video** — upload, or register a local file path (recommended for big 4K files; nothing is copied). Detection runs automatically; **Re-detect** any time (e.g. to add the Claude deep pass).
2. Browse the **timeline** — songs (green), highlights (amber), singing (teal), fun moments (red). Click to seek; click list items to jump; double-click a song to fill the export range.
3. **Sync to recording** *(optional)* — upload a clean recording of a song in the video; it aligns, fills the export range, and shows a two-track mixer to audition camera audio vs. the recording.
4. **Shooting simulation** — grab a frame, drag a virtual camera per player (tag singer/wide roles), save, tune the switching options.
5. **Export** — pick range, orientation, and an optional file name. With an alignment present the recording becomes the soundtrack (camera muted).

## Architecture

```
band_video_studio/
  probe.py    ffprobe metadata + proxy/audio/frame extraction (ffmpeg)
  models.py   download-and-cache for local MediaPipe model files
  audio.py    YAMNet music/laugh/vocal classification + energy → songs, highlights, singing, laugh candidates
  detect.py   face-blendshape smile detection + audio/visual fusion → fun moments
  vision.py   optional Claude deep pass on candidate windows only
  lyrics.py   optional faster-whisper transcription + lyric line alignment
  align.py    onset-envelope cross-correlation: locate a clean recording within a video
  library.py  media library: watched-folder scanning + cross-video best-of aggregation
  editor.py   ffmpeg filtergraph multicam-style crop/cut renderer + muted-sync render
  store.py    JSON metadata store under data/
  jobs.py     background job manager
  server.py   FastAPI app + REST API
web/          single-page frontend (player, timeline, crop editor)
tests/        unit tests for segmentation, cutlist, crop fitting, lyric alignment
```

Analysis runs on a 540p proxy and 16 kHz mono audio, so 4K sources stay cheap; only the final export renders from the original.

## Cost model

Everything runs locally for free. The only thing that can spend money is the explicit **Claude deep pass** checkbox, and it only judges the candidate windows the free detectors already found — typically a few API calls per rehearsal video, each with ~6 small (300 px) frames.

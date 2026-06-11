"""Media library: watched folders (e.g. a NAS mount) + cross-video queries.

A library is a list of folders. Scanning walks them for video files, registers
anything new, and runs the standard import (proxy + detection) sequentially so
a big folder doesn't fan out into dozens of concurrent ffmpeg/YAMNet jobs.

Once per-video analysis exists, cross-video questions are cheap aggregations
over the cached analysis.json files: globally funniest moments, most
exaggerated expressions, and so on. Note smile scores are only roughly
comparable across videos (different rooms, faces and distances).
"""

from __future__ import annotations

import os
from pathlib import Path

VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".avi", ".m4v", ".mts", ".webm"}


def scan_folders(folders: list[str]) -> list[Path]:
    """All video files under the given folders, recursively.

    Built on os.walk so huge, deeply nested trees stay cheap: hidden
    directories are pruned before descent (never walked at all), only video
    files are kept, and nothing but the matches is held in memory.
    """
    found: list[Path] = []
    for folder in folders:
        root = Path(folder).expanduser()
        if not root.is_dir():
            continue
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = sorted(d for d in dirnames if not d.startswith("."))
            for name in sorted(filenames):
                if not name.startswith(".") and Path(name).suffix.lower() in VIDEO_EXTS:
                    found.append(Path(dirpath) / name)
    return found


def find_new(files: list[Path], existing_paths: set[str]) -> list[Path]:
    """Files not yet registered, comparing resolved paths."""
    known = {str(Path(p).expanduser().resolve()) for p in existing_paths}
    return [f for f in files if str(f.resolve()) not in known]


# ------------------------------------------------- cross-video aggregation
# items: [(video_record, analysis_dict), ...] — pure, unit-testable.

def top_fun_moments(items: list[tuple[dict, dict]], limit: int = 20) -> list[dict]:
    """Globally funniest moments across all analyzed videos, by fused score."""
    out = []
    for video, analysis in items:
        for m in (analysis or {}).get("fun_moments", []):
            out.append({
                "video_id": video["id"], "video_name": video["name"],
                "start": m["start"], "end": m["end"],
                "score": m.get("score", 0.0), "type": m.get("type", ""),
                "caption": m.get("caption", ""),
            })
    out.sort(key=lambda m: m["score"], reverse=True)
    return out[:limit]


def top_expressions(items: list[tuple[dict, dict]], limit: int = 20) -> list[dict]:
    """Most exaggerated expressions: fun moments ranked by their peak smile."""
    out = []
    for video, analysis in items:
        for m in (analysis or {}).get("fun_moments", []):
            smile = (m.get("evidence") or {}).get("max_smile")
            if smile is None:
                continue
            out.append({
                "video_id": video["id"], "video_name": video["name"],
                "start": m["start"], "end": m["end"],
                "max_smile": smile, "type": m.get("type", ""),
                "caption": m.get("caption", ""),
            })
    out.sort(key=lambda m: m["max_smile"], reverse=True)
    return out[:limit]

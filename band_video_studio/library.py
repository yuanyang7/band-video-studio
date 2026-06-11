"""素材库: watched media folders (e.g. a NAS mount) + cross-video queries.

A library is a list of folders. Scanning walks them for video files, registers
anything new, and runs the standard import (proxy + detection) sequentially so
a big folder doesn't fan out into dozens of concurrent ffmpeg/YAMNet jobs.

Once per-video analysis exists, cross-video questions are cheap aggregations
over the cached analysis.json files: globally funniest moments, most
exaggerated expressions, and so on. Note smile scores are only roughly
comparable across videos (different rooms, faces and distances).
"""

from __future__ import annotations

from pathlib import Path

VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".avi", ".m4v", ".mts", ".webm"}


def scan_folders(folders: list[str]) -> list[Path]:
    """All video files under the given folders (recursive, hidden dirs skipped)."""
    found: list[Path] = []
    for folder in folders:
        root = Path(folder).expanduser()
        if not root.is_dir():
            continue
        for p in sorted(root.rglob("*")):
            if p.suffix.lower() not in VIDEO_EXTS or not p.is_file():
                continue
            if any(part.startswith(".") for part in p.relative_to(root).parts):
                continue
            found.append(p)
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

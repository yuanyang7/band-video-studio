"""Tiny in-process background job manager (thread per job)."""

from __future__ import annotations

import threading
import traceback
import uuid
from typing import Any, Callable

_jobs: dict[str, dict] = {}
_lock = threading.Lock()


def submit(kind: str, fn: Callable[..., Any], *args, **kwargs) -> dict:
    job_id = uuid.uuid4().hex[:12]
    job = {"id": job_id, "kind": kind, "status": "running", "progress": "", "result": None, "error": None}
    with _lock:
        _jobs[job_id] = job

    def run():
        try:
            job["result"] = fn(*args, progress=lambda msg: job.update(progress=msg), **kwargs)
            job["status"] = "done"
        except Exception as e:  # surface the full traceback to the API
            job["status"] = "error"
            job["error"] = f"{e}\n{traceback.format_exc()}"

    threading.Thread(target=run, daemon=True).start()
    return job


def get(job_id: str) -> dict | None:
    with _lock:
        return _jobs.get(job_id)

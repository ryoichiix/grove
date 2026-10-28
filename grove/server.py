"""FastAPI server: serves the UI and streams agent events over Server-Sent Events."""
from __future__ import annotations

import json
import queue
import threading
import time
from pathlib import Path
from typing import Optional

from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse

from .agent import Agent
from .factory import ROOT, make_client, resolve_mode
from .skills import SkillStore
from .tasks import FAMILIES, all_tasks, get_task, heldout_tasks, train_tasks

app = FastAPI(title="Grove")
store = SkillStore(ROOT / "skills")
RESULTS = ROOT / "results" / "eval_latest.json"
RUN_LOCK = threading.Lock()


def _skill_dict(s):
    return {"name": s.name, "description": s.description, "version": s.version,
            "family": s.family, "text": s.text}


@app.get("/")
def index():
    return FileResponse(ROOT / "web" / "index.html")


@app.get("/api/state")
def state():
    client = make_client()
    last = None
    if RESULTS.exists():
        try:
            last = json.loads(RESULTS.read_text(encoding="utf-8"))
        except Exception:
            last = None
    return {
        "mode": resolve_mode(),
        "model": getattr(client, "label", "?"),
        "families": FAMILIES,
        "tasks": [{"id": t.id, "family": t.family, "split": t.split, "title": t.title} for t in all_tasks()],
        "skills": [_skill_dict(s) for s in store.list()],
        "eval": last,
    }


@app.get("/api/run")
def run(kind: str = "learn", task: Optional[str] = None):
    if not RUN_LOCK.acquire(blocking=False):
        return JSONResponse({"error": "a run is already in progress"}, status_code=409)
    q: queue.Queue = queue.Queue()

    def emit(type_: str, **data):
        q.put({"type": type_, "t": time.time(), **data})

    def worker():
        try:
            if kind == "reset":
                store.clear()
                RESULTS.unlink(missing_ok=True)
                emit("reset")
                return
            client = make_client(on_call=lambda **d: emit("llm_call", **d))
            agent = Agent(client, store, emit)
            emit("run_start", kind=kind, model=getattr(client, "label", "?"), mode=resolve_mode())
            if kind in ("learn", "all"):
                tasks = [get_task(task)] if task else train_tasks()
                agent.learn_all(tasks)
            if kind in ("eval", "all"):
                agent.run_eval(heldout_tasks(), RESULTS)
        except Exception as e:  # surface every failure in the UI instead of dying silently
            emit("error", message=f"{type(e).__name__}: {e}")
        finally:
            q.put(None)
            RUN_LOCK.release()

    threading.Thread(target=worker, daemon=True).start()

    def gen():
        while True:
            try:
                item = q.get(timeout=15)
            except queue.Empty:
                yield ": keepalive\n\n"
                continue
            if item is None:
                yield 'data: {"type":"end"}\n\n'
                break
            yield f"data: {json.dumps(item, default=str)}\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

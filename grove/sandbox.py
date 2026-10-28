"""Execute model-written code in a subprocess with time and memory limits.

NOTE: this is process isolation, not a security boundary. Only run code from models you chose.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

RUNNER = str(Path(__file__).with_name("runner.py"))
R = "\x00GROVE_RESULT\x00"
E = "\x00GROVE_ERROR\x00"


def _limits(timeout: int):
    try:
        import resource
    except ImportError:  # Windows
        return None

    def apply():
        resource.setrlimit(resource.RLIMIT_CPU, (timeout + 2, timeout + 2))
        resource.setrlimit(resource.RLIMIT_AS, (1 << 30, 1 << 30))

    return apply


def run_code(code: str, inputs: dict, timeout: int = 10) -> dict:
    """Return {'ok': True, 'result': ...} or {'ok': False, 'error': '...'}."""
    payload = json.dumps({"code": code, "inputs": inputs})
    try:
        p = subprocess.run(
            [sys.executable, "-I", RUNNER],
            input=payload, capture_output=True, text=True, timeout=timeout,
            preexec_fn=_limits(timeout),
        )
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": f"TimeoutError: solve() ran longer than {timeout}s"}

    out = p.stdout
    if R in out:
        raw = out.rsplit(R, 1)[1].splitlines()[0]
        try:
            return {"ok": True, "result": json.loads(raw)}
        except json.JSONDecodeError:
            return {"ok": False, "error": "solve() returned something that is not JSON-serialisable"}
    if E in out:
        return {"ok": False, "error": out.rsplit(E, 1)[1].strip()}
    tail = (p.stderr or out or "process exited without a result").strip().splitlines()[-4:]
    return {"ok": False, "error": "\n".join(tail)}

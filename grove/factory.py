"""Pick the model backend and provider routing.

Providers (all serve Gemma 4):
  gemini        Gemma 4 via the Gemini API
  digitalocean  Gemma 4 via DigitalOcean serverless inference
  hybrid        solving/selecting/eval on DigitalOcean (high rate limit), skill authoring on the Gemini API

Modes: auto (cache then live) | live | replay (cache only) | mock (scripted stand-in, never Gemma).
"""
from __future__ import annotations

import os
from pathlib import Path

from .llm import DigitalOceanClient, GemmaClient, RoutedClient
from .mock import MockClient

ROOT = Path(__file__).resolve().parent.parent


def load_dotenv(path: Path = ROOT / ".env"):
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip("\"'"))


def gemini_key():
    return os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")


def do_key():
    return os.getenv("DIGITALOCEAN_MODEL_ACCESS_KEY") or os.getenv("MODEL_ACCESS_KEY")


def resolve_mode() -> str:
    load_dotenv()
    mode = os.getenv("GROVE_MODE", "").lower()
    if mode in ("mock", "live", "replay", "auto"):
        return mode
    return "auto" if (gemini_key() or do_key()) else "mock"


def resolve_provider() -> str:
    load_dotenv()
    p = os.getenv("GROVE_PROVIDER", "").lower()
    if p in ("gemini", "digitalocean", "hybrid"):
        return p
    g, d = bool(gemini_key()), bool(do_key())
    if g and d:
        return "hybrid"
    if d:
        return "digitalocean"
    return "gemini"


def eval_workers() -> int:
    """Parallel eval calls: worth it on DigitalOcean (120 req/min), pointless on the Gemini free tier."""
    default = "4" if (resolve_mode() != "mock" and resolve_provider() in ("digitalocean", "hybrid")) else "1"
    try:
        return max(1, int(os.getenv("GROVE_EVAL_WORKERS", default)))
    except ValueError:
        return 1


def make_client(on_call=None):
    mode = resolve_mode()
    if mode == "mock":
        return MockClient(on_call=on_call)
    kw = dict(mode=mode, cache_dir=str(ROOT / "cache"), on_call=on_call)
    prov = resolve_provider()
    if prov == "gemini":
        return RoutedClient(GemmaClient(**kw))
    if prov == "digitalocean":
        return RoutedClient(DigitalOceanClient(**kw))
    return RoutedClient(DigitalOceanClient(**kw), GemmaClient(**kw))  # hybrid

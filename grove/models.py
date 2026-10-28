"""List the model ids a provider actually exposes to your key (so you never guess an id)."""
from __future__ import annotations

import os

import requests

from . import llm
from .factory import do_key, gemini_key, load_dotenv


def list_model_ids(provider: str) -> list[str]:
    load_dotenv()
    if provider == "digitalocean":
        key = do_key()
        if not key:
            raise llm.LLMError("DIGITALOCEAN_MODEL_ACCESS_KEY is not set")
        r = requests.get(f"{llm.DO_ROOT}/models", headers={"Authorization": f"Bearer {key}"}, timeout=30)
        if r.status_code != 200:
            raise llm.LLMError(f"HTTP {r.status_code}: {r.text[:300]}")
        return sorted(m["id"] for m in r.json().get("data", []))
    if provider == "gemini":
        key = gemini_key()
        if not key:
            raise llm.LLMError("GEMINI_API_KEY is not set")
        ids, token = [], None
        for _ in range(10):
            params = {"pageSize": 200, **({"pageToken": token} if token else {})}
            r = requests.get(f"{llm.API_ROOT}/models", headers={"x-goog-api-key": key}, params=params, timeout=30)
            if r.status_code != 200:
                raise llm.LLMError(f"HTTP {r.status_code}: {r.text[:300]}")
            data = r.json()
            ids += [m["name"].split("/", 1)[-1] for m in data.get("models", [])]
            token = data.get("nextPageToken")
            if not token:
                break
        return sorted(ids)
    raise ValueError("provider must be 'digitalocean' or 'gemini'")

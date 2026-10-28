"""Gemma 4 through two providers, plus the plumbing a live demo needs.

  * GemmaClient         Gemma 4 via the Gemini API  (POST .../v1beta/models/<id>:generateContent)
  * DigitalOceanClient  Gemma 4 via DigitalOcean serverless inference (OpenAI-compatible
                        POST https://inference.do-ai.run/v1/chat/completions)
  * RoutedClient        sends skill-authoring calls to one provider and everything else to another

Every model call in Grove goes through BaseClient.generate(), which:
  * spaces live calls out (Gemini free tier ~15 req/min; DigitalOcean tier 1 up to 120 req/min)
  * retries 429/5xx with backoff, and can fall back to a second model after repeated 429s
  * caches every response on disk by (provider, model, prompt) hash, so a run can be replayed exactly
    (GROVE_MODE=replay) if the venue wifi dies on stage.
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from pathlib import Path
from typing import Callable, Optional

import requests

API_ROOT = "https://generativelanguage.googleapis.com/v1beta"
DO_ROOT = "https://inference.do-ai.run/v1"


class LLMError(RuntimeError):
    pass


class BaseClient:
    provider = "base"

    def __init__(self, api_key, model, fallback_model, mode, min_gap, cache_dir, on_call):
        self.api_key = api_key
        self.model = model
        self.fallback_model = fallback_model
        self.mode = mode  # auto (cache then live) | live (never read cache) | replay (cache only)
        self.min_gap = min_gap
        self.cache_dir = Path(cache_dir or os.getenv("GROVE_CACHE", "cache"))
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.on_call: Optional[Callable[..., None]] = on_call
        self._lock = threading.Lock()
        self._last_live = 0.0
        self._use_fallback = False
        self.calls = 0

    # ------------------------------------------------------------------ public
    @property
    def mode_tag(self) -> str:
        return {"replay": "replay", "live": "live", "auto": "live+cache"}.get(self.mode, self.mode)

    @property
    def label(self) -> str:
        return f"{self.model} via {self.provider} ({self.mode_tag})"

    def clients(self):
        return [self]

    def generate(self, prompt: str, *, system: Optional[str] = None,
                 temperature: float = 0.2, tag: str = "") -> str:
        key = self._key(prompt, system, temperature)
        path = self.cache_dir / f"{key}.json"
        t0 = time.time()

        if self.mode in ("auto", "replay") and path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            self._emit(tag, data.get("model", self.model), True, 0.0, len(prompt), len(data["text"]))
            return data["text"]
        if self.mode == "replay":
            raise LLMError(f"replay mode: no cached response for this prompt (provider={self.provider}, tag={tag})")
        if not self.api_key:
            raise LLMError(self._missing_key_message())

        text, used = self._live(prompt, system, temperature)
        path.write_text(
            json.dumps({"provider": self.provider, "model": used, "tag": tag, "prompt": prompt, "text": text},
                       indent=2),
            encoding="utf-8",
        )
        self._emit(tag, used, False, time.time() - t0, len(prompt), len(text))
        return text

    # ----------------------------------------------------------- provider hooks
    def _missing_key_message(self) -> str:
        return "API key is not set"

    def _request(self, model: str, full_prompt: str, temperature: float) -> requests.Response:
        raise NotImplementedError

    @staticmethod
    def _parse(data: dict) -> str:
        raise NotImplementedError

    # ----------------------------------------------------------------- private
    def _key(self, prompt, system, temperature) -> str:
        blob = json.dumps(
            {"provider": self.provider, "model": self.model, "system": system, "prompt": prompt, "t": temperature},
            sort_keys=True,
        )
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:32]

    def _emit(self, tag, model, cached, seconds, n_in, n_out):
        self.calls += 1
        if self.on_call:
            self.on_call(tag=tag, model=model, cached=cached, seconds=round(seconds, 2),
                         chars_in=n_in, chars_out=n_out, provider=self.provider)

    def _throttle(self):
        with self._lock:
            wait = self.min_gap - (time.time() - self._last_live)
            if wait > 0:
                time.sleep(wait)
            self._last_live = time.time()

    def _live(self, prompt, system, temperature):
        full = f"SYSTEM INSTRUCTIONS:\n{system}\n\n{prompt}" if system else prompt
        rate_limited = 0
        last_err = "unknown error"
        for attempt in range(6):
            model = self.fallback_model if (self._use_fallback and self.fallback_model) else self.model
            self._throttle()
            try:
                r = self._request(model, full, temperature)
            except requests.RequestException as e:  # network blip
                last_err = f"network error: {e}"
                time.sleep(min(2 ** attempt, 15))
                continue

            if r.status_code == 200:
                return self._parse(r.json()), model

            last_err = f"HTTP {r.status_code}: {r.text[:300]}"
            if r.status_code == 429:
                rate_limited += 1
                if rate_limited >= 2 and self.fallback_model and not self._use_fallback:
                    self._use_fallback = True  # switch to the second model
                    continue
                retry_after = r.headers.get("Retry-After")
                time.sleep(float(retry_after) if retry_after else min(3 * 2 ** attempt, 30))
            elif r.status_code >= 500:  # includes 503 "capacity constrained"
                time.sleep(min(2 ** attempt, 15))
            else:  # 400/401/403/404: our fault, do not hammer
                raise LLMError(f"{self.provider}: {last_err}")
        raise LLMError(f"{self.provider} failed after retries ({last_err})")


class GemmaClient(BaseClient):
    """Gemma 4 through the Gemini API."""

    provider = "Gemini API"

    def __init__(self, api_key=None, model=None, fallback_model=None, mode="auto",
                 min_gap=None, cache_dir=None, on_call=None):
        super().__init__(
            api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"),
            model or os.getenv("GROVE_MODEL", "gemma-4-31b-it"),
            fallback_model or os.getenv("GROVE_FALLBACK_MODEL", "gemma-4-26b-a4b-it"),
            mode,
            float(os.getenv("GROVE_MIN_GAP", "4.5")) if min_gap is None else min_gap,
            cache_dir, on_call,
        )

    def _missing_key_message(self):
        return "GEMINI_API_KEY is not set (get one at https://aistudio.google.com/apikey)"

    def _request(self, model, full_prompt, temperature):
        body = {
            "contents": [{"role": "user", "parts": [{"text": full_prompt}]}],
            "generationConfig": {"temperature": temperature, "maxOutputTokens": int(os.getenv("GROVE_MAX_TOKENS", "8192"))},
        }
        return requests.post(
            f"{API_ROOT}/models/{model}:generateContent",
            headers={"x-goog-api-key": self.api_key, "Content-Type": "application/json"},
            json=body, timeout=120,
        )

    @staticmethod
    def _parse(data: dict) -> str:
        cands = data.get("candidates") or []
        if not cands:
            raise LLMError(f"empty response from model: {json.dumps(data.get('promptFeedback', data))[:300]}")
        parts = (cands[0].get("content") or {}).get("parts") or []
        text = "".join(p.get("text", "") for p in parts if not p.get("thought"))
        if not text.strip():
            raise LLMError(f"model returned no text (finishReason={cands[0].get('finishReason')})")
        return text


class DigitalOceanClient(BaseClient):
    """Gemma 4 through DigitalOcean serverless inference (OpenAI-compatible chat completions)."""

    provider = "DigitalOcean"

    def __init__(self, api_key=None, model=None, fallback_model=None, mode="auto",
                 min_gap=None, cache_dir=None, on_call=None):
        super().__init__(
            api_key or os.getenv("DIGITALOCEAN_MODEL_ACCESS_KEY") or os.getenv("MODEL_ACCESS_KEY"),
            model or os.getenv("GROVE_DO_MODEL", "gemma-4-31B-it"),
            fallback_model or os.getenv("GROVE_DO_FALLBACK_MODEL") or None,
            mode,
            float(os.getenv("GROVE_DO_MIN_GAP", "0.6")) if min_gap is None else min_gap,
            cache_dir, on_call,
        )

    def _missing_key_message(self):
        return ("DIGITALOCEAN_MODEL_ACCESS_KEY is not set (create a model access key under "
                "Serverless inference > Model Access Keys in the DigitalOcean control panel)")

    def _request(self, model, full_prompt, temperature):
        body = {
            "model": model,
            "messages": [{"role": "user", "content": full_prompt}],
            "temperature": temperature,
            "max_completion_tokens": int(os.getenv("GROVE_MAX_TOKENS", "8192")),
        }
        return requests.post(
            f"{DO_ROOT}/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            json=body, timeout=120,
        )

    @staticmethod
    def _parse(data: dict) -> str:
        choices = data.get("choices") or []
        if not choices:
            raise LLMError(f"empty response from model: {json.dumps(data)[:300]}")
        msg = choices[0].get("message") or {}
        content = msg.get("content")
        if isinstance(content, list):  # some servers return content parts
            content = "".join(p.get("text", "") for p in content if isinstance(p, dict))
        if not content or not str(content).strip():
            raise LLMError(f"model returned no text (finish_reason={choices[0].get('finish_reason')})")
        return str(content)


class RoutedClient:
    """Skill authoring (reflect/repair) can run on a different provider from solving/selecting."""

    mode = "routed"

    def __init__(self, work: BaseClient, reflect: Optional[BaseClient] = None):
        self.work = work
        self.reflect = reflect or work

    def clients(self):
        return [self.work] if self.reflect is self.work else [self.work, self.reflect]

    @property
    def label(self) -> str:
        if self.reflect is self.work:
            return self.work.label
        return (f"Gemma 4 · solving on {self.work.provider}, skill-writing on {self.reflect.provider} "
                f"({self.work.mode_tag})")

    def generate(self, prompt, *, system=None, temperature=0.2, tag=""):
        c = self.reflect if tag.startswith(("reflect", "repair")) else self.work
        return c.generate(prompt, system=system, temperature=temperature, tag=tag)

"""The live Gemini path, verified against a local stub server (no network, no key needed)."""
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

import grove.llm as llm
from grove.llm import GemmaClient, LLMError


class Stub:
    def __init__(self, responses):
        self.responses = list(responses)  # (status, body-dict) in order
        self.requests = []
        stub = self

        class H(BaseHTTPRequestHandler):
            def do_POST(self):
                n = int(self.headers.get("Content-Length", 0))
                stub.requests.append({"path": self.path, "key": self.headers.get("x-goog-api-key"),
                                      "body": json.loads(self.rfile.read(n))})
                status, body = stub.responses.pop(0)
                data = json.dumps(body).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, *a):
                pass

        self.server = HTTPServer(("127.0.0.1", 0), H)
        self.url = f"http://127.0.0.1:{self.server.server_port}/v1beta"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self):
        self.server.shutdown()


def ok(text, with_thought=False):
    parts = [{"text": "hidden reasoning", "thought": True}] if with_thought else []
    return 200, {"candidates": [{"content": {"parts": parts + [{"text": text}]}, "finishReason": "STOP"}]}


@pytest.fixture
def make(tmp_path, monkeypatch):
    stubs = []
    monkeypatch.setattr(llm.time, "sleep", lambda s: None)

    def _make(responses, **kw):
        s = Stub(responses)
        stubs.append(s)
        monkeypatch.setattr(llm, "API_ROOT", s.url)
        c = GemmaClient(api_key="test-key", model="gemma-4-31b-it", fallback_model="gemma-4-26b-a4b-it",
                        min_gap=0, cache_dir=str(tmp_path / "cache"), **kw)
        return c, s

    yield _make
    for s in stubs:
        s.close()


def test_request_shape_thought_filtering_and_cache(make):
    calls = []
    c, stub = make([ok("hello", with_thought=True)], on_call=lambda **d: calls.append(d))
    assert c.generate("say hello", tag="t1") == "hello"  # thought part is dropped
    req = stub.requests[0]
    assert req["path"] == "/v1beta/models/gemma-4-31b-it:generateContent"
    assert req["key"] == "test-key"
    assert req["body"]["contents"][0]["parts"][0]["text"] == "say hello"
    assert c.generate("say hello", tag="t1") == "hello"  # second call: served from cache, no HTTP
    assert len(stub.requests) == 1
    assert [d["cached"] for d in calls] == [False, True]


def test_two_429s_switch_to_fallback_model_and_still_cache_under_primary(make):
    c, stub = make([(429, {"error": "quota"}), (429, {"error": "quota"}), ok("from fallback")])
    assert c.generate("p", tag="x") == "from fallback"
    paths = [r["path"] for r in stub.requests]
    assert paths[0].endswith("gemma-4-31b-it:generateContent")
    assert paths[1].endswith("gemma-4-31b-it:generateContent")
    assert paths[2].endswith("gemma-4-26b-a4b-it:generateContent")
    assert c.generate("p", tag="x") == "from fallback" and len(stub.requests) == 3


def test_client_errors_fail_fast_without_retrying(make):
    c, stub = make([(404, {"error": "model not found"})])
    with pytest.raises(LLMError, match="404"):
        c.generate("p")
    assert len(stub.requests) == 1


def test_empty_candidates_raise(make):
    c, _ = make([(200, {"candidates": [], "promptFeedback": {"blockReason": "SAFETY"}})])
    with pytest.raises(LLMError, match="empty response"):
        c.generate("p")


def test_replay_mode_never_touches_the_network(make):
    c, stub = make([ok("cached answer")])
    c.generate("p")
    r = GemmaClient(api_key=None, model="gemma-4-31b-it", mode="replay", min_gap=0, cache_dir=str(c.cache_dir))
    assert r.generate("p") == "cached answer"
    with pytest.raises(LLMError, match="replay mode"):
        r.generate("something never seen")
    assert len(stub.requests) == 1


# ---------------------------------------------------------------- DigitalOcean path
from grove.llm import DigitalOceanClient, RoutedClient


def do_ok(text):
    return 200, {"choices": [{"message": {"role": "assistant", "content": text}, "finish_reason": "stop"}]}


@pytest.fixture
def make_do(tmp_path, monkeypatch):
    stubs = []
    monkeypatch.setattr(llm.time, "sleep", lambda s: None)

    def _make(responses, **kw):
        s = Stub(responses)
        stubs.append(s)
        monkeypatch.setattr(llm, "DO_ROOT", s.url.replace("/v1beta", "/v1"))
        c = DigitalOceanClient(api_key="do-key", model="gemma-4-31B-it", min_gap=0,
                               cache_dir=str(tmp_path / "cache"), **kw)
        return c, s

    yield _make
    for s in stubs:
        s.close()


def test_do_request_shape_auth_and_cache(make_do):
    c, stub = make_do([do_ok("hi from do")])
    assert c.generate("say hi", system="be brief", tag="t") == "hi from do"
    req = stub.requests[0]
    assert req["path"] == "/v1/chat/completions"
    assert req["body"]["model"] == "gemma-4-31B-it"
    assert "say hi" in req["body"]["messages"][0]["content"] and "be brief" in req["body"]["messages"][0]["content"]
    assert c.generate("say hi", system="be brief", tag="t") == "hi from do"
    assert len(stub.requests) == 1


def test_do_retries_503_then_succeeds_and_401_fails_fast(make_do):
    c, stub = make_do([(503, {"error": "busy"}), do_ok("ok")])
    assert c.generate("p") == "ok" and len(stub.requests) == 2
    c2, stub2 = make_do([(401, {"error": "bad key"})])
    with pytest.raises(LLMError, match="401"):
        c2.generate("other prompt")
    assert len(stub2.requests) == 1


def test_cache_keys_differ_per_provider(tmp_path):
    a = GemmaClient(api_key="x", cache_dir=str(tmp_path))
    b = DigitalOceanClient(api_key="x", cache_dir=str(tmp_path))
    assert a._key("p", None, 0.2) != b._key("p", None, 0.2)


def test_routed_client_sends_reflection_to_second_provider(tmp_path):
    class Fake:
        def __init__(self, n): self.n, self.seen = n, []
        def generate(self, prompt, *, system=None, temperature=0.2, tag=""):
            self.seen.append(tag); return self.n
    w, r = Fake("work"), Fake("reflect")
    rc = RoutedClient(w, r)
    assert rc.generate("p", tag="solve:a:r1") == "work"
    assert rc.generate("p", tag="select:a") == "work"
    assert rc.generate("p", tag="reflect:a") == "reflect"
    assert rc.generate("p", tag="repair:a") == "reflect"

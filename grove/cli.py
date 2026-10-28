"""python -m grove.cli <serve|learn|eval|all|smoke|reset>"""
from __future__ import annotations

import argparse
import sys

from .agent import Agent
from .factory import ROOT, make_client, resolve_mode
from .skills import SkillStore
from .tasks import heldout_tasks, train_tasks


def _printer(type_, **d):
    if type_ == "llm_call":
        print(f"   [call] {d['tag']:<30} {d.get('provider','')[:6]:<7}{d['model']:<20} {'cache' if d['cached'] else 'live '} {d['seconds']}s")
    elif type_ == "attempt":
        print(f" - {d['task']} round {d['round']}: {'PASS' if d['passed'] else 'FAIL'}"
              f"   skills={d['skills'] or '-'}")
    elif type_ == "reflection":
        print(f"   [reflection] {d['text']}")
    elif type_ == "skill_written":
        print(f"   [skill] wrote {d['name']} v{d['version']} -> {d['path']}")
    elif type_ == "eval_task":
        print(f" - eval {d['mode']:<6} {d['task']:<20} {'PASS' if d['passed'] else 'FAIL'}")
    elif type_ == "eval_done":
        print(f"\nHELD-OUT RESULT  raw {d['raw_pass']}/{d['n']}  ->  with skills {d['skills_pass']}/{d['n']}  ({d['model']})")
    elif type_ == "error":
        print("ERROR:", d["message"])


def main(argv=None):
    ap = argparse.ArgumentParser(prog="grove")
    ap.add_argument("cmd", choices=["serve", "learn", "eval", "all", "smoke", "reset", "models"])
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--provider", choices=["gemini", "digitalocean"], default="digitalocean",
                    help="for 'models': which provider's model ids to list")
    args = ap.parse_args(argv)

    store = SkillStore(ROOT / "skills")
    if args.cmd == "serve":
        import uvicorn
        print(f"Grove on http://127.0.0.1:{args.port}   mode={resolve_mode()}")
        uvicorn.run("grove.server:app", host="127.0.0.1", port=args.port)
        return 0
    if args.cmd == "reset":
        store.clear()
        print("skills cleared")
        return 0

    if args.cmd == "models":
        from .models import list_model_ids
        from .llm import LLMError
        try:
            ids = list_model_ids(args.provider)
        except LLMError as e:
            print("ERROR:", e)
            return 1
        gem = [i for i in ids if "gemma" in i.lower()]
        print(f"{len(ids)} models on {args.provider}. Gemma ones:")
        for i in gem or ["(none found)"]:
            print("  ", i)
        return 0

    client = make_client(on_call=lambda **d: _printer("llm_call", **d))
    print(f"model: {getattr(client, 'label', '?')}   mode: {resolve_mode()}")
    if args.cmd == "smoke":
        out = client.generate("Reply with exactly the single word: OK", tag="smoke")
        print("model replied:", out.strip()[:80])
        return 0
    agent = Agent(client, store, _printer)
    if args.cmd in ("learn", "all"):
        agent.learn_all(train_tasks())
    if args.cmd in ("eval", "all"):
        agent.run_eval(heldout_tasks(), ROOT / "results" / "eval_latest.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())

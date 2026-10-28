"""The Grove loop: attempt -> fail -> reflect -> write an Agent Skill -> retry.

Design rules that keep the result honest:
  * The retry prompt contains the new SKILL.md but NOT the failure log, so a pass can only come
    from knowledge that travelled through the skill.
  * Evaluation compares pass@1 with and without skills on tasks the agent never learned from.
  * Everything the model says is passed through to the UI/event stream; nothing is scripted here.
"""
from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Callable, Optional

from .skills import SkillStore, parse_skill, strip_fences, validate_skill
from .tasks import Task, check_task

SOLVE_RULES = (
    "Write Python (standard library only) that defines `solve(inputs)`. `inputs` is a dict as described "
    "above. Return the answer as plain JSON-compatible Python values. You cannot see the data in "
    "advance, so write robust code. Reply with exactly one ```python code block and nothing else."
)

REFLECT_FORMAT = """Write your answer in exactly this shape:

REFLECTION: <2-4 sentences: what went wrong, and the underlying convention or rule that explains it>

```markdown
---
name: <lowercase-kebab-case, max 64 chars>
description: <ONE line, max 300 chars, starting with "Use when", describing the general situation (not this one task)>
---
# <Title>

## When to use
<the kind of input/situation that triggers this skill>

## Procedure
<numbered, concrete steps, including a short correct code snippet>

## Pitfalls
<what to watch for>

## Verification
<how to sanity-check the result before answering>
```

Rules: the skill must be general (it will be reused on different tasks of the same kind), must not contain
this task's answer or numbers, and must follow the format above exactly."""


def extract_code(text: str) -> str:
    m = re.search(r"```(?:python|py)?\s*\n(.*?)```", text, re.S)
    return (m.group(1) if m else text).strip()


def parse_reflection(text: str) -> tuple[str, str]:
    """Split a model answer into (reflection, skill_markdown)."""
    fence = re.search(r"```(?:markdown|md)?\s*\n(.*?)\n```", text, re.S)
    skill = fence.group(1).strip() if fence else ""
    if not skill:
        i = text.find("\n---")
        skill = text[i + 1:].strip() if i != -1 else text.strip()
    refl = text.split("```", 1)[0]
    refl = re.sub(r"^\s*REFLECTION:\s*", "", refl).strip()
    return refl, strip_fences(skill)


class Agent:
    def __init__(self, client, store: SkillStore, emit: Optional[Callable[..., None]] = None,
                 max_rounds: int = 3, llm_select: Optional[bool] = None):
        self.client = client
        self.store = store
        self.emit = emit or (lambda *a, **k: None)
        self.max_rounds = max_rounds
        self.llm_select = (os.getenv("GROVE_SELECTOR", "auto") != "lexical") if llm_select is None else llm_select

    def _select(self, task: Task):
        """Progressive disclosure: the agent sees only names + descriptions and decides what to load.
        A cheap lexical shortlist goes first; if it finds nothing, the model chooses (one small call)."""
        picked = self.store.select(task.prompt)
        library = self.store.list()
        if picked or not library or not self.llm_select:
            return picked
        listing = "\n".join(f"- {s.name}: {s.description}" for s in library)
        prompt = (f"[TASK:{task.id}]\nYou are an agent deciding which saved skills to load before starting "
                  f"a task. Skills are listed as name: description.\n\n{listing}\n\nTASK:\n{task.prompt}\n\n"
                  "Which skills clearly apply to this task? Reply with a comma-separated list of exact skill "
                  "names, or the single word NONE if none apply.")
        raw = self.client.generate(prompt, tag=f"select:{task.id}", temperature=0.0)
        return [s for s in library if s.name in raw][:2]

    # ------------------------------------------------------------- prompts
    @staticmethod
    def _skill_block(skills) -> str:
        if not skills:
            return ""
        parts = ["SKILLS FROM PAST EXPERIENCE (procedural knowledge; follow it when it applies):"]
        for s in skills:
            parts.append(f"--- skill: {s.name} ---\n{s.text.strip()}\n--- end skill ---")
        return "\n".join(parts) + "\n\n"

    def _solve(self, task: Task, skills, tag: str) -> str:
        prompt = (f"You are a careful Python engineer.\n[TASK:{task.id}]\n\n"
                  f"{self._skill_block(skills)}TASK:\n{task.prompt}\n\n{SOLVE_RULES}")
        return extract_code(self.client.generate(prompt, tag=tag))

    def _reflect(self, task: Task, code: str, log: str, active) -> Optional[dict]:
        self.emit("reflect_start", task=task.id, family=task.family)
        related = ""
        if active:
            related = ("Existing related skills (if one already covers this topic, output an improved "
                       "version under the SAME name instead of a duplicate):\n"
                       + "\n".join(f"--- {s.name} ---\n{s.text.strip()}" for s in active) + "\n\n")
        prompt = (
            "You are the learning module of a coding agent. The agent just FAILED a task. Turn the failure "
            "into a reusable Agent Skill (agentskills.io format) so this agent, and any other agent, does "
            f"not make the same mistake on similar tasks.\n[TASK:{task.id}]\n\n"
            f"Task given to the agent:\n{task.prompt}\n\n"
            f"The agent's code:\n```python\n{code}\n```\n\n"
            f"What the checker reported:\n{log}\n\n{related}{REFLECT_FORMAT}"
        )
        raw = self.client.generate(prompt, tag=f"reflect:{task.id}", temperature=0.3)
        reflection, skill_md = parse_reflection(raw)
        errors = validate_skill(skill_md)
        if errors:
            self.emit("skill_invalid", task=task.id, errors=errors)
            repair = (f"{prompt}\n\nYour previous SKILL.md was invalid: {'; '.join(errors)}.\n"
                      f"Previous attempt:\n{raw}\n\nReturn the corrected answer in the same shape.")
            raw = self.client.generate(repair, tag=f"repair:{task.id}", temperature=0.1)
            reflection2, skill_md = parse_reflection(raw)
            reflection = reflection2 or reflection
            errors = validate_skill(skill_md)
            if errors:
                self.emit("skill_invalid", task=task.id, errors=errors, final=True)
                return None
        self.emit("reflection", task=task.id, family=task.family, text=reflection)
        info = self.store.save(skill_md, origin={
            "task": task.id, "family": task.family, "log": log, "reflection": reflection})
        self.emit("skill_written", task=task.id, family=task.family, **info)
        return info

    # --------------------------------------------------------------- learn
    def learn(self, task: Task) -> dict:
        self.emit("task_start", task=task.id, family=task.family, title=task.title, phase="learn")
        active = self._select(task)
        learned = None
        for rnd in range(1, self.max_rounds + 1):
            code = self._solve(task, active, tag=f"solve:{task.id}:r{rnd}")
            res = check_task(task, code)
            self.emit("attempt", task=task.id, family=task.family, round=rnd,
                      skills=[s.name for s in active], code=code, passed=res.passed, log=res.log)
            if res.passed:
                self.emit("task_done", task=task.id, family=task.family, passed=True,
                          rounds=rnd, skill=learned)
                return {"task": task.id, "passed": True, "rounds": rnd, "skill": learned}
            if rnd == self.max_rounds:
                break
            info = self._reflect(task, code, res.log, active)
            if not info:
                break
            learned = info["name"]
            fresh = self.store.get(learned)
            active = [s for s in active if s.name != learned] + ([fresh] if fresh else [])
        self.emit("task_done", task=task.id, family=task.family, passed=False,
                  rounds=self.max_rounds, skill=learned)
        return {"task": task.id, "passed": False, "rounds": self.max_rounds, "skill": learned}

    def learn_all(self, tasks) -> list[dict]:
        import os
        workers = int(os.getenv("GROVE_LEARN_WORKERS", "1") or 1)
        if workers > 1 and len(tasks) > 1:  # tasks are different families, so they can learn in parallel
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(min(workers, len(tasks))) as ex:
                out = list(ex.map(self.learn, tasks))
        else:
            out = [self.learn(t) for t in tasks]
        self.emit("learn_done", results=out, skills=len(self.store.list()))
        return out

    # ---------------------------------------------------------------- eval
    def eval_task(self, task: Task, use_skills: bool) -> dict:
        mode = "skills" if use_skills else "raw"
        active = self._select(task) if use_skills else []
        code = self._solve(task, active, tag=f"eval-{mode}:{task.id}")
        res = check_task(task, code)
        out = {"task": task.id, "family": task.family, "mode": mode, "passed": res.passed,
               "skills": [s.name for s in active]}
        self.emit("eval_task", log=None if res.passed else res.log, **out)
        return out

    def run_eval(self, tasks, results_path: Optional[Path] = None) -> dict:
        summary = {"n": len(tasks), "model": getattr(self.client, "label", "?"),
                   "ts": time.time(), "raw": [], "skills": []}
        self.emit("eval_start", n=len(tasks))
        from .factory import eval_workers
        workers = min(eval_workers(), len(tasks)) if tasks else 1
        for mode in ("raw", "skills"):
            if workers > 1:  # DigitalOcean allows ~120 req/min; order of results is preserved
                from concurrent.futures import ThreadPoolExecutor
                with ThreadPoolExecutor(workers) as ex:
                    summary[mode] = list(ex.map(lambda t: self.eval_task(t, mode == "skills"), tasks))
            else:
                for t in tasks:
                    summary[mode].append(self.eval_task(t, mode == "skills"))
        summary["raw_pass"] = sum(r["passed"] for r in summary["raw"])
        summary["skills_pass"] = sum(r["passed"] for r in summary["skills"])
        self.emit("eval_done", n=summary["n"], raw_pass=summary["raw_pass"],
                  skills_pass=summary["skills_pass"], model=summary["model"])
        if results_path:
            Path(results_path).parent.mkdir(parents=True, exist_ok=True)
            Path(results_path).write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return summary

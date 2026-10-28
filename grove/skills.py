"""Skill store: Agent Skills (agentskills.io) on disk, with validation, versioning and selection.

A skill is a folder:  skills/<name>/SKILL.md  (+ references/ for provenance).
SKILL.md = YAML-style front matter (name, description) + markdown instructions.
Agents use progressive disclosure: discovery (name+description) -> activation (full SKILL.md).
"""
from __future__ import annotations

import difflib
import json
import re
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

NAME_RE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
MAX_NAME = 64
MAX_DESC = 1024

STOP = set(
    """the and for with you are given return use used using when this that from each its
    list dict over value values data input inputs into per not all any one has have text
    plain line lines file raw will your can then than them they which what how also only
    such where while there their about after before between should would could does done
    make makes being been were was but out get got set sets new old whose via without
    task tasks write code python function
    round rounded decimal decimals float string integer ints iso""".split()  # last row: answer-format words
)


@dataclass
class Skill:
    name: str
    description: str
    body: str
    text: str
    version: int
    path: Path
    family: str = ""


def strip_fences(text: str) -> str:
    t = text.strip()
    m = re.match(r"^```[a-zA-Z]*\n(.*?)\n```\s*$", t, re.S)
    return m.group(1).strip() if m else t


def parse_skill(text: str) -> tuple[dict, str]:
    """Return (front_matter, body). Raises ValueError if there is no front matter."""
    t = strip_fences(text).lstrip("﻿")
    if not t.startswith("---"):
        raise ValueError("SKILL.md must start with '---' front matter")
    end = t.find("\n---", 3)
    if end == -1:
        raise ValueError("front matter is not closed with '---'")
    fm: dict = {}
    for line in t[3:end].strip().splitlines():
        if ":" not in line:
            continue
        k, v = line.split(":", 1)
        v = v.strip()
        if len(v) >= 2 and v[0] == v[-1] and v[0] in "\"'":
            v = v[1:-1]
        fm[k.strip()] = v
    body = t[end + 4:].strip()
    return fm, body


def validate_skill(text: str) -> list[str]:
    """Check a SKILL.md against the Agent Skills format. Empty list = valid."""
    try:
        fm, body = parse_skill(text)
    except ValueError as e:
        return [str(e)]
    errors = []
    name = fm.get("name", "")
    if not name:
        errors.append("front matter is missing 'name'")
    elif len(name) > MAX_NAME:
        errors.append(f"name is {len(name)} chars; max is {MAX_NAME}")
    elif not NAME_RE.match(name):
        errors.append("name must be lowercase letters/digits separated by single hyphens")
    desc = fm.get("description", "")
    if not desc.strip():
        errors.append("front matter is missing 'description'")
    elif len(desc) > MAX_DESC:
        errors.append(f"description is {len(desc)} chars; max is {MAX_DESC}")
    if len(body) < 60:
        errors.append("body is too short to be useful instructions")
    return errors


def _tokens(text: str) -> set[str]:
    out = set()
    for t in re.findall(r"[a-z][a-z0-9]{2,}", text.lower()):
        if t not in STOP:
            out.add(t[:6])  # crude stemmer: export/exports/exported -> export
    return out


import threading as _threading
_SAVE_LOCK = _threading.RLock()


class SkillStore:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.meta_path = self.root.parent / ".grove" / "meta.json"
        self.hist_root = self.root.parent / ".grove" / "history"

    # ---------------------------------------------------------------- meta
    def _meta(self) -> dict:
        try:
            return json.loads(self.meta_path.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def _write_meta(self, meta: dict):
        self.meta_path.parent.mkdir(parents=True, exist_ok=True)
        self.meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    # ---------------------------------------------------------------- read
    def list(self) -> list[Skill]:
        skills = []
        meta = self._meta()
        for d in sorted(self.root.iterdir()):
            f = d / "SKILL.md"
            if d.name.startswith(".") or not f.is_file():
                continue
            text = f.read_text(encoding="utf-8")
            try:
                fm, body = parse_skill(text)
            except ValueError:
                continue
            hist = self.hist_root / d.name
            version = len(list(hist.glob("v*.md"))) if hist.exists() else 1
            skills.append(Skill(d.name, fm.get("description", ""), body, text,
                                version, f, meta.get(d.name, {}).get("family", "")))
        return skills

    def get(self, name: str) -> Optional[Skill]:
        return next((s for s in self.list() if s.name == name), None)

    # --------------------------------------------------------------- write
    def save(self, text: str, origin: Optional[dict] = None) -> dict:
        with _SAVE_LOCK:
            return self._save(text, origin)

    def _save(self, text: str, origin: Optional[dict] = None) -> dict:
        """Validate and write a skill. Returns {name, version, path, diff, created, description}."""
        text = strip_fences(text).strip() + "\n"
        errors = validate_skill(text)
        if errors:
            raise ValueError("; ".join(errors))
        fm, _ = parse_skill(text)
        name = fm["name"]
        folder = self.root / name
        f = folder / "SKILL.md"
        old = f.read_text(encoding="utf-8") if f.exists() else ""
        folder.mkdir(parents=True, exist_ok=True)
        f.write_text(text, encoding="utf-8")

        hist = self.hist_root / name
        hist.mkdir(parents=True, exist_ok=True)
        version = len(list(hist.glob("v*.md"))) + 1
        (hist / f"v{version}.md").write_text(text, encoding="utf-8")

        diff = [
            {"t": ("+" if ln.startswith("+") else "-" if ln.startswith("-") else " "), "s": ln[1:]}
            for ln in difflib.unified_diff(old.splitlines(), text.splitlines(), lineterm="", n=200)
            if not ln.startswith(("+++", "---", "@@"))
        ] if old else [{"t": "+", "s": ln} for ln in text.splitlines()]

        if origin:
            ref = folder / "references"
            ref.mkdir(exist_ok=True)
            (ref / f"origin-v{version}.md").write_text(
                f"# Where skill `{name}` v{version} came from\n\n"
                f"- written: {datetime.now(timezone.utc).isoformat(timespec='seconds')}\n"
                f"- failing task: `{origin.get('task', '?')}` (family `{origin.get('family', '?')}`)\n\n"
                f"## What the agent saw\n\n```\n{origin.get('log', '').strip()}\n```\n\n"
                f"## The agent's own reflection\n\n{origin.get('reflection', '').strip()}\n",
                encoding="utf-8",
            )
            meta = self._meta()
            meta[name] = {"family": origin.get("family", ""), "task": origin.get("task", "")}
            self._write_meta(meta)

        return {"name": name, "version": version, "path": str(f),
                "description": fm.get("description", ""), "diff": diff, "created": not old}

    def clear(self):
        for d in self.root.iterdir():
            if d.is_dir() and not d.name.startswith("."):
                shutil.rmtree(d)
        shutil.rmtree(self.root.parent / ".grove", ignore_errors=True)

    # ------------------------------------------------------------- selection
    def _scores(self, prompt: str, skills: list[Skill]) -> list[tuple[float, int, Skill]]:
        """Lexical relevance with IDF-style weights: words shared by many skills count for less.
        Returns (score, number of distinct description words matched, skill), best first."""
        p = _tokens(prompt)
        desc = {s.name: _tokens(s.name.replace("-", " ") + " " + s.description) for s in skills}
        body = {s.name: _tokens(s.body) - desc[s.name] for s in skills}
        df: dict[str, int] = {}
        for s in skills:
            for t in desc[s.name] | body[s.name]:
                df[t] = df.get(t, 0) + 1
        out = []
        for s in skills:
            hit = p & desc[s.name]
            sc = sum(3 / df[t] for t in hit) + sum(1 / df[t] for t in p & body[s.name])
            out.append((sc, len(hit), s))
        return sorted(out, key=lambda x: -x[0])

    def score(self, prompt: str, skill: Skill) -> float:
        return next((sc for sc, _, s in self._scores(prompt, self.list()) if s.name == skill.name), 0.0)

    def select(self, prompt: str, limit: int = 2) -> list[Skill]:
        """Lexical shortlist, tuned for PRECISION: a skill needs >= 3 distinct description-word
        matches, so generic words ("records", "export") cannot activate an unrelated skill.
        Recall comes from the model-assisted fallback in Agent._select."""
        scored = self._scores(prompt, self.list())
        if not scored or scored[0][0] < 2.0:
            return []
        top = scored[0][0]
        return [s for sc, n, s in scored if n >= 3 and sc >= 2.0 and sc >= 0.5 * top][:limit]

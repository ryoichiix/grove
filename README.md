# Grove: agents that grow their own skills

> Built at **Hacktoberfest Hack Day Hyderabad x NIAT-AuroraU** (28 Oct 2026) by Team Rockerz.
> Categories: **Best Open-Source AI Project** and **Best Use of Gemma 4**.

An agent that **fails in front of you, explains why in its own words, writes a reusable
[Agent Skill](https://agentskills.io) (`SKILL.md`), and never makes that mistake again**, including on tasks it
has never seen. The learned skills are plain folders anyone can `git clone` and load into any
Agent-Skills-compatible agent.

Agents forget. Grove turns each failure into durable, human-readable, version-controlled knowledge.

## Model (for the Gemma 4 prize)

| | |
|---|---|
| Model | **Gemma 4 31B IT** (`gemma-4-31b-it` on the Gemini API, `gemma-4-31B-it` on DigitalOcean) |
| Access 1 | **Gemini API**, REST `models/<id>:generateContent` (`GemmaClient`, [`grove/llm.py`](grove/llm.py)) |
| Access 2 | **DigitalOcean Gradient serverless inference**, OpenAI-compatible `https://inference.do-ai.run/v1/chat/completions` (`DigitalOceanClient`) |
| License | Gemma 4 is open-weight under Apache 2.0, see the [Gemma docs](https://ai.google.dev/gemma/docs/core) |
| Providers | `GROVE_PROVIDER=gemini | digitalocean | hybrid` (hybrid = solving/eval on DigitalOcean for its higher rate limit, skill-writing on the Gemini API) |

Every model call goes through `generate()` in `grove/llm.py`; a `RoutedClient` picks the provider per call.
The UI's **Model calls** tab lists each call live (purpose, provider, model id, live vs cache, seconds).

## Where the AI is (and what the team wrote)

Gemma does three jobs; everything around them is our own code (about 1,500 lines, written today):

1. **Solve**: writes Python for a task whose input hides a convention (semicolon CSVs with decimal commas,
   German month names, overlapping API pages, `1.2k` / `(3M)` / `n/a` numbers).
2. **Reflect**: reads its own failing code and the checker log and writes an explanation plus a complete
   `SKILL.md` (front matter + procedure + pitfalls + verification).
3. **Choose**: when the cheap lexical shortlist finds nothing, picks which saved skills to load from
   names + descriptions only (progressive disclosure).

Our harness (`grove/agent.py`): the attempt, reflect, validate, retry loop; a spec validator and versioned
skill store (`grove/skills.py`); a sandboxed runner with time and memory limits (`grove/sandbox.py`); the task
suite and checkers (`grove/tasks.py`); a throttled, cached, replayable Gemini client; the SSE server and the
growing-tree UI.

## How it proves the learning is real

* **The retry prompt contains the new skill but not the failure log**, so a pass can only come from knowledge
  that travelled through the `SKILL.md` (asserted in `tests/test_pipeline.py`).
* **Held-out evaluation**: skills are learned from 4 train tasks; we then score pass@1 on **10 tasks never
  learned from**, once with plain Gemma and once with Gemma + its learned skills. Two of the ten are
  **controls with no quirk**, to show skills do no harm.
* **Provenance**: every skill folder has `references/origin-vN.md` holding the failing task, the checker log
  and the model's own reflection. Every node in the tree is a real file in `skills/`.
* **Skills are versioned**: if a later failure touches the same topic the model patches the skill and the
  UI shows the diff.

### Results (fill in from your live run)

| | Gemma alone | Gemma + learned skills |
|---|---|---|
| Held-out pass@1 | _ / 10 | _ / 10 |

Produced by `python -m grove eval` (also saved to `results/eval_latest.json`).

## Run it

```bash
pip install -r requirements.txt
cp .env.example .env            # DIGITALOCEAN_MODEL_ACCESS_KEY and/or GEMINI_API_KEY (both = hybrid)

python -m grove models          # lists the Gemma ids your key can use (default provider: digitalocean)

python -m grove smoke           # one tiny call: proves the model id and key work
python -m grove serve           # UI at http://127.0.0.1:8000
# or headless:
python -m grove all             # learn from the 4 train tasks, then evaluate on the 10 held-out ones
python -m grove reset           # wipe learned skills
python -m pytest -q             # 22 tests (pipeline verified with a scripted mock model)
```

**Modes** (`GROVE_MODE`): `auto` (default with a key: reads the on-disk cache, then calls Gemma),
`live` (always calls Gemma), `replay` (cache only, for when wifi dies on stage), `mock`
(scripted stand-in used for tests and UI work, clearly badged **MOCK** in the UI; it is not Gemma).
Every live response is cached in `cache/` by prompt hash, so a full run can be replayed exactly.

The Gemini free tier is about 15 requests/minute (DigitalOcean tier 1 allows up to 120), so calls are spaced `GROVE_MIN_GAP` seconds apart
(default 4.5) and a 429 triggers backoff and a switch to the fallback model.

## Agent Skills compliance

Each skill is `skills/<name>/SKILL.md` with YAML front matter (`name`: lowercase letters, digits, single
hyphens, 64 chars or fewer, matching the folder; `description`: 1024 chars or fewer) and a markdown body,
plus an optional `references/` folder. `validate_skill()` enforces this before anything is saved; an invalid
answer gets one automatic repair round. Discovery uses name + description; activation loads the full file.

## Honest limitations

* "Learning" here is **text memory in `SKILL.md` files, not weight updates**. That is the point: it is
  auditable, portable and reviewable by a human.
* The task suite is **synthetic by design**: each family hides a convention the prompt never mentions, and the
  baseline writes code blind (like coding against an export you have not opened). The held-out numbers measure
  how well a learned convention transfers to sibling tasks, not general coding skill.
* Skill selection is lexical-then-model; with unusual skill wording the model fallback does the work.
* The sandbox is process isolation with limits, not a security boundary. Only run models you chose.
* Small free-tier quotas mean a full run takes a few minutes; use the cache.

## Layout

```
grove/            agent loop, Gemini client, skills store, sandbox, tasks, server, CLI
web/index.html    the live skill-tree UI (no build step)
skills/           the exported, installable skill library (grows as the agent learns)
cache/            cached model responses (replayable runs)
tests/            22 tests: tasks, validator, loop, selection, eval
```

License: MIT (see `LICENSE`). Gemma 4 weights are used under their own Apache 2.0 terms.

import pytest

from grove.agent import Agent, extract_code, parse_reflection
from grove.mock import SKILLS, MockClient
from grove.skills import SkillStore, validate_skill
from grove.tasks import heldout_tasks, train_tasks


@pytest.fixture
def store(tmp_path):
    return SkillStore(tmp_path / "skills")


def make_agent(store, events=None):
    ev = events if events is not None else []
    return Agent(MockClient(delay=0), store, lambda t, **d: ev.append((t, d))), ev


def test_validator_accepts_mock_skills_and_rejects_bad():
    for _, text in SKILLS.values():
        assert validate_skill(text) == []
    assert validate_skill("no front matter here")
    assert validate_skill("---\nname: Bad Name\ndescription: x\n---\n" + "body " * 30)
    assert validate_skill("---\nname: ok-name\n---\n" + "body " * 30)  # missing description


def test_extractors():
    assert extract_code("hi\n```python\nx = 1\n```\nbye") == "x = 1"
    refl, skill = parse_reflection("REFLECTION: it broke.\n\n```markdown\n---\nname: a\n---\nbody\n```")
    assert refl == "it broke." and skill.startswith("---")


def test_learn_loop_writes_valid_skills_and_passes_on_retry(store):
    agent, events = make_agent(store)
    results = agent.learn_all(train_tasks())
    assert all(r["passed"] for r in results)
    assert all(r["rounds"] == 2 and r["skill"] for r in results)
    assert len(store.list()) == 4
    for s in store.list():
        assert validate_skill(s.text) == []
        assert (s.path.parent / "references").exists()  # provenance written
    kinds = [t for t, _ in events]
    assert kinds.count("skill_written") == 4 and kinds.count("reflect_start") == 4


def test_retry_prompt_has_no_failure_log(store):
    seen = []
    agent, _ = make_agent(store)
    real = agent.client.generate

    def spy(prompt, **kw):
        seen.append((kw.get("tag", ""), prompt))
        return real(prompt, **kw)

    agent.client.generate = spy
    agent.learn(train_tasks()[0])
    solves = [p for tag, p in seen if tag.startswith("solve")]
    assert len(solves) == 2
    assert "CHECK FAILED" not in solves[1] and "SAMPLE OF THE RAW INPUT" not in solves[1]
    assert "--- skill:" in solves[1]


def test_lexical_selector_is_precise_and_full_selection_is_exact(store):
    agent, _ = make_agent(store)
    agent.learn_all(train_tasks())
    for t in heldout_tasks() + train_tasks():
        if t.family == "control":
            continue
        own = SKILLS[t.family][0]
        lexical = [s.name for s in store.select(t.prompt)]
        assert set(lexical) <= {own}, f"{t.id}: lexical shortlist activated a wrong skill: {lexical}"
        full = [s.name for s in agent._select(t)]  # lexical, then model-assisted fallback
        assert full == [own], f"{t.id}: expected [{own}], got {full}"


def test_eval_shows_lift_on_heldout(store, tmp_path):
    agent, _ = make_agent(store)
    agent.learn_all(train_tasks())
    summary = agent.run_eval(heldout_tasks(), tmp_path / "eval.json")
    assert summary["n"] == 10
    assert summary["raw_pass"] == 2  # only the two controls pass without skills
    assert summary["skills_pass"] == 10
    assert (tmp_path / "eval.json").exists()


def test_skill_patch_creates_new_version_and_diff(store):
    _, text = SKILLS["eu_csv"]
    first = store.save(text)
    assert first["created"] and first["version"] == 1
    second = store.save(text.replace("Verification", "Verification (patched)"))
    assert not second["created"] and second["version"] == 2
    assert any(d["t"] == "+" for d in second["diff"]) and any(d["t"] == "-" for d in second["diff"])


def test_first_attempt_on_a_new_family_never_has_skills(store):
    agent, events = make_agent(store)
    agent.learn_all(train_tasks())
    firsts = [d for t, d in events if t == "attempt" and d["round"] == 1]
    assert len(firsts) == 4
    assert all(d["skills"] == [] for d in firsts), [d["skills"] for d in firsts]


def test_model_assisted_selection_rescues_a_lexical_miss(store):
    agent, _ = make_agent(store)
    agent.learn_all(train_tasks())
    task = heldout_tasks()[0]
    store.select = lambda prompt, limit=2: []  # force the lexical shortlist to miss
    picked = agent._select(task)
    assert [s.name for s in picked] == [SKILLS[task.family][0]]
    agent.llm_select = False
    assert agent._select(task) == []

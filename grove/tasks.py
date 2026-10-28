"""Quirk tasks: small coding tasks whose input hides a convention the prompt never mentions.

Four quirk families (a skill learned on one train task should transfer to its held-out siblings)
plus a control family with no quirk, to show the skills do no harm.

The agent writes `solve(inputs)`. It never sees the raw data up front, exactly like writing code
against a real export you have not opened. A failed attempt gets an error / mismatch log plus a
small sample of the raw input, which is what a human debugger would look at next.
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from datetime import date, timedelta

from .sandbox import run_code

FAMILIES = {
    "eu_csv": "European CSV exports",
    "localized_dates": "Localized dates",
    "paged_api": "Paginated API",
    "human_units": "Human-readable units",
    "control": "Control (no quirk)",
}


@dataclass
class Task:
    id: str
    family: str
    split: str  # "train" | "heldout"
    title: str
    prompt: str
    inputs: dict
    expected: object
    sample: str
    naive_code: str
    reference_code: str


@dataclass
class CheckResult:
    passed: bool
    got: object
    log: str
    ran: bool = True


# ------------------------------------------------------------------ comparison
def same(exp, got) -> bool:
    if isinstance(exp, bool) or isinstance(got, bool):
        return exp == got
    if isinstance(exp, (int, float)) and isinstance(got, (int, float)):
        return abs(exp - got) <= 0.01
    if isinstance(exp, dict):
        return isinstance(got, dict) and set(exp) == set(got) and all(same(exp[k], got[k]) for k in exp)
    if isinstance(exp, list):
        return isinstance(got, list) and len(exp) == len(got) and all(same(a, b) for a, b in zip(exp, got))
    return exp == got


def _short(x, n=220) -> str:
    s = json.dumps(x, default=str)
    return s if len(s) <= n else s[:n] + " ..."


def check_task(task: Task, code: str) -> CheckResult:
    run = run_code(code, json.loads(json.dumps(task.inputs)))
    if not run["ok"]:
        log = f"CHECK FAILED: your code crashed.\n{run['error']}"
        if task.sample:
            log += f"\n\nSAMPLE OF THE RAW INPUT:\n{task.sample}"
        return CheckResult(False, None, log, ran=False)
    got = run["result"]
    if same(task.expected, got):
        return CheckResult(True, got, "CHECK PASSED", True)
    log = (f"CHECK FAILED: solve() returned a wrong answer.\n"
           f"returned: {_short(got)}\nexpected: {_short(task.expected)}")
    if task.sample:
        log += f"\n\nSAMPLE OF THE RAW INPUT:\n{task.sample}"
    return CheckResult(False, got, log, True)


# ================================================================== family: eu_csv
_REGIONS = ["North", "South", "East", "West"]


def _eu_rows(seed, n=40):
    rng = random.Random(seed)
    rows = [
        {"id": f"ORD-{1000 + i}",
         "date": f"2025-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}",
         "region": _REGIONS[i % 4] if i < 8 else rng.choice(_REGIONS),
         "amount": round(rng.uniform(40, 9000), 2)}
        for i in range(n)
    ]
    return rows


def _eu_num(x: float) -> str:
    return f"{x:,.2f}".replace(",", "§").replace(".", ",").replace("§", ".")


def _eu_text(rows) -> str:
    lines = ["\ufeffid;date;region;amount"]
    for r in rows:
        lines.append(f'{r["id"]};{r["date"]};{r["region"]};"{_eu_num(r["amount"])}"')
    return "\n".join(lines) + "\n"


_EU_NAIVE = ("import csv, io\n"
             "def solve(inputs):\n"
             "    rows = list(csv.DictReader(io.StringIO(inputs['csv_text'])))\n")
_EU_REF = ("import csv, io\n"
           "def num(s):\n"
           "    return float(s.replace('.', '').replace(',', '.'))\n"
           "def solve(inputs):\n"
           "    text = inputs['csv_text'].lstrip('\\ufeff')\n"
           "    rows = list(csv.DictReader(io.StringIO(text), delimiter=';'))\n")
_EU_TAILS = {
    "sum": "    return round(sum(NUM(r['amount']) for r in rows), 2)\n",
    "avg": ("    groups = {}\n"
            "    for r in rows:\n"
            "        groups.setdefault(r['region'], []).append(NUM(r['amount']))\n"
            "    return {k: round(sum(v) / len(v), 2) for k, v in groups.items()}\n"),
    "max": "    return max(rows, key=lambda r: NUM(r['amount']))['id']\n",
}
_EU_INTRO = ("You are given `inputs['csv_text']`, the raw text of a CSV file exported from the "
             "company's sales system. Its columns are: id, date, region, amount. ")


def _eu_tasks():
    spec = [
        ("eu_csv_1", "train", "Total revenue", 11, "sum",
         "Return the total revenue, i.e. the sum of `amount` over all rows, as a float rounded to 2 decimals."),
        ("eu_csv_2", "heldout", "Average per region", 12, "avg",
         "Return a dict mapping each region to the average `amount` of its rows, each rounded to 2 decimals."),
        ("eu_csv_3", "heldout", "Biggest order", 13, "max",
         "Return the `id` of the row with the largest `amount` (a string)."),
    ]
    out = []
    for tid, split, title, seed, kind, ask in spec:
        rows = _eu_rows(seed)
        text = _eu_text(rows)
        if kind == "sum":
            exp = round(sum(r["amount"] for r in rows), 2)
        elif kind == "avg":
            exp = {}
            for reg in _REGIONS:
                v = [r["amount"] for r in rows if r["region"] == reg]
                exp[reg] = round(sum(v) / len(v), 2)
        else:
            exp = max(rows, key=lambda r: r["amount"])["id"]
        out.append(Task(
            tid, "eu_csv", split, title, _EU_INTRO + ask, {"csv_text": text}, exp,
            "\n".join(text.splitlines()[:4]),
            _EU_NAIVE + _EU_TAILS[kind].replace("NUM(", "float("),
            _EU_REF + _EU_TAILS[kind].replace("NUM(", "num("),
        ))
    return out


# ========================================================= family: localized_dates
_DE = ["Jan", "Feb", "Mär", "Apr", "Mai", "Jun", "Jul", "Aug", "Sep", "Okt", "Nov", "Dez"]
_LABELS = ["Server migration", "Backup verified", "Patch rollout", "Incident review",
           "Load test", "Certificate renewal", "Access audit", "Failover drill"]


def _events(seed, n=45):
    rng = random.Random(seed)
    start = date(2024, 10, 1)
    evs = [(start + timedelta(days=rng.randint(0, 364)), rng.choice(_LABELS)) for _ in range(n)]
    for k in range(6):  # make sure March 2025 is populated
        evs.append((date(2025, 3, 3 + 4 * k), rng.choice(_LABELS)))
    rng.shuffle(evs)
    return evs


def _ev_text(evs) -> str:
    return "\n".join(f"{d.day} {_DE[d.month - 1]} {d.year} - {lab}" for d, lab in evs) + "\n"


_DT_NAIVE = ("from datetime import datetime\n"
             "def parse(s):\n"
             "    return datetime.strptime(s.strip(), '%d %b %Y').date()\n")
_DT_REF = ("from datetime import date\n"
           "MON = {'Jan': 1, 'Feb': 2, 'Mär': 3, 'Apr': 4, 'Mai': 5, 'Jun': 6, 'Jul': 7, "
           "'Aug': 8, 'Sep': 9, 'Okt': 10, 'Nov': 11, 'Dez': 12}\n"
           "def parse(s):\n"
           "    d, m, y = s.strip().split()\n"
           "    return date(int(y), MON[m], int(d))\n")
_DT_COMMON = ("def solve(inputs):\n"
              "    dates = [parse(l.split(' - ')[0]) for l in inputs['log_text'].splitlines() if l.strip()]\n")
_DT_TAILS = {
    "march": "    return sum(1 for d in dates if d.year == 2025 and d.month == 3)\n",
    "first": "    return min(dates).isoformat()\n",
    "bymonth": ("    out = {}\n"
                "    for d in dates:\n"
                "        k = f'{d.year}-{d.month:02d}'\n"
                "        out[k] = out.get(k, 0) + 1\n"
                "    return out\n"),
}
_DT_INTRO = ("You are given `inputs['log_text']`, a plain-text operations log with one event per line. "
             "Each line starts with the date of the event, followed by ' - ' and a description. ")


def _date_tasks():
    spec = [
        ("localized_dates_1", "train", "Events in March 2025", 21, "march",
         "Return how many events happened in March 2025 (an int)."),
        ("localized_dates_2", "heldout", "Earliest event", 22, "first",
         "Return the date of the earliest event as an ISO string 'YYYY-MM-DD'."),
        ("localized_dates_3", "heldout", "Events per month", 23, "bymonth",
         "Return a dict mapping 'YYYY-MM' to the number of events in that month (only months that have events)."),
    ]
    out = []
    for tid, split, title, seed, kind, ask in spec:
        evs = _events(seed)
        text = _ev_text(evs)
        ds = [d for d, _ in evs]
        if kind == "march":
            exp = sum(1 for d in ds if d.year == 2025 and d.month == 3)
        elif kind == "first":
            exp = min(ds).isoformat()
        else:
            exp = {}
            for d in ds:
                k = f"{d.year}-{d.month:02d}"
                exp[k] = exp.get(k, 0) + 1
        out.append(Task(
            tid, "localized_dates", split, title, _DT_INTRO + ask, {"log_text": text}, exp,
            "\n".join(text.splitlines()[:5]),
            _DT_NAIVE + _DT_COMMON + _DT_TAILS[kind],
            _DT_REF + _DT_COMMON + _DT_TAILS[kind],
        ))
    return out


# ============================================================== family: paged_api
def _pages(seed, n=60, size=10, stride=8):
    rng = random.Random(seed)
    values = rng.sample(range(10, 900), n)
    recs = [{"id": i + 1, "value": values[i]} for i in range(n)]
    recs[8]["value"], recs[16]["value"], recs[29]["value"] = 999, 990, 985  # ids 9, 17, 30
    starts = list(range(0, n, stride))
    toks = ["start"] + [f"p{k}" for k in range(1, len(starts))]
    pages = {}
    for k, s in enumerate(starts):
        pages[toks[k]] = {"items": recs[s:s + size], "next": toks[k + 1] if k + 1 < len(starts) else ""}
    return recs, pages


_PG_NAIVE = ("def fetch_all(inputs):\n"
             "    items, token = [], None\n"
             "    while True:\n"
             "        page = inputs['fetch_page'](token)\n"
             "        items.extend(page['items'])\n"
             "        token = page.get('next')\n"
             "        if not token:\n"
             "            break\n"
             "    return items\n")
_PG_REF = ("def fetch_all(inputs):\n"
           "    seen, token = {}, None\n"
           "    while True:\n"
           "        page = inputs['fetch_page'](token)\n"
           "        for it in page['items']:\n"
           "            seen[it['id']] = it\n"
           "        token = page.get('next')\n"
           "        if not token:\n"
           "            break\n"
           "    return list(seen.values())\n")
_PG_TAILS = {
    "count": "def solve(inputs):\n    return len(fetch_all(inputs))\n",
    "sum": "def solve(inputs):\n    return sum(i['value'] for i in fetch_all(inputs))\n",
    "top3": ("def solve(inputs):\n"
             "    items = sorted(fetch_all(inputs), key=lambda i: -i['value'])\n"
             "    return [i['id'] for i in items[:3]]\n"),
}
_PG_INTRO = ("`inputs['fetch_page']` is a function `fetch_page(token=None)` that returns one page of a "
             "paginated records API as a dict with keys `items` (list of records, each a dict with integer "
             "`id` and integer `value`) and `next` (token for the following page). Call it with no token "
             "to get the first page. ")


def _paged_tasks():
    spec = [
        ("paged_api_1", "train", "Count all records", 31, "count",
         "Fetch every page and return the total number of records (an int)."),
        ("paged_api_2", "heldout", "Sum of values", 32, "sum",
         "Fetch every page and return the sum of `value` over all records (an int)."),
        ("paged_api_3", "heldout", "Top 3 records", 33, "top3",
         "Fetch every page and return the list of the `id`s of the 3 records with the highest `value`, highest first."),
    ]
    out = []
    for tid, split, title, seed, kind, ask in spec:
        recs, pages = _pages(seed)
        exp = {"count": len(recs), "sum": sum(r["value"] for r in recs), "top3": [9, 17, 30]}[kind]
        sample = (f"page 'start' item ids: {[i['id'] for i in pages['start']['items']]}, next={pages['start']['next']!r}\n"
                  f"page 'p1' item ids:    {[i['id'] for i in pages['p1']['items']]}, next={pages['p1']['next']!r}\n"
                  f"last page next token: {pages[sorted(pages)[-1]]['next']!r}")
        out.append(Task(
            tid, "paged_api", split, title, _PG_INTRO + ask, {"_pages": pages}, exp, sample,
            _PG_NAIVE + _PG_TAILS[kind], _PG_REF + _PG_TAILS[kind],
        ))
    return out


# ============================================================ family: human_units
_MULT = {"": 1, "k": 1e3, "M": 1e6, "B": 1e9}


def _metrics(seed, n=30):
    rng = random.Random(seed)
    rows = []
    for i in range(n):
        name = f"svc-{i + 1:02d}"
        r = rng.random()
        if r < 0.12:
            rows.append((name, "n/a", None))
            continue
        suf = rng.choice(["", "k", "k", "M", "B"])
        if suf == "":
            mant = rng.randint(5, 900)
            raw, val = f"{mant}", float(mant)
        else:
            mant = round(rng.uniform(0.1, 99.9), 1)
            raw, val = f"{mant:.1f}{suf}", round(mant * _MULT[suf], 2)
        if rng.random() < 0.15:
            raw, val = f"({raw})", -val
        rows.append((name, raw, val))
    return rows


_HU_NAIVE = ("def parse_rows(inputs):\n"
             "    rows = []\n"
             "    for line in inputs['metrics_text'].splitlines():\n"
             "        if not line.strip():\n"
             "            continue\n"
             "        name, raw = line.split(':', 1)\n"
             "        rows.append((name.strip(), float(raw.strip())))\n"
             "    return rows\n")
_HU_REF = ("MULT = {'k': 1e3, 'M': 1e6, 'B': 1e9}\n"
           "def parse_value(raw):\n"
           "    raw = raw.strip()\n"
           "    if raw.lower() in ('n/a', 'na', '-', ''):\n"
           "        return None\n"
           "    neg = raw.startswith('(') and raw.endswith(')')\n"
           "    if neg:\n"
           "        raw = raw[1:-1]\n"
           "    mult = 1.0\n"
           "    if raw and raw[-1] in MULT:\n"
           "        mult = MULT[raw[-1]]\n"
           "        raw = raw[:-1]\n"
           "    v = float(raw) * mult\n"
           "    return -v if neg else v\n"
           "def parse_rows(inputs):\n"
           "    rows = []\n"
           "    for line in inputs['metrics_text'].splitlines():\n"
           "        if not line.strip():\n"
           "            continue\n"
           "        name, raw = line.split(':', 1)\n"
           "        v = parse_value(raw)\n"
           "        if v is not None:\n"
           "            rows.append((name.strip(), v))\n"
           "    return rows\n")
_HU_HEAD = "def solve(inputs):\n    rows = parse_rows(inputs)\n"
_HU_TAILS = {
    "sum": "    return round(sum(v for _, v in rows), 2)\n",
    "max": "    return max(rows, key=lambda r: r[1])[0]\n",
    "avg": "    return round(sum(v for _, v in rows) / len(rows), 2)\n",
}
_HU_INTRO = ("You are given `inputs['metrics_text']`, a plain-text metrics export with one service per "
             "line in the form `service-name: value`. ")


def _unit_tasks():
    spec = [
        ("human_units_1", "train", "Total of all services", 41, "sum",
         "Return the sum of all service values as a float rounded to 2 decimals. "
         "Services without a reading are ignored."),
        ("human_units_2", "heldout", "Busiest service", 42, "max",
         "Return the name of the service with the largest value."),
        ("human_units_3", "heldout", "Average reading", 43, "avg",
         "Return the average value over the services that have a reading, rounded to 2 decimals."),
    ]
    out = []
    for tid, split, title, seed, kind, ask in spec:
        rows = _metrics(seed)
        text = "\n".join(f"{n}: {raw}" for n, raw, _ in rows) + "\n"
        vals = [(n, v) for n, _, v in rows if v is not None]
        if kind == "sum":
            exp = round(sum(v for _, v in vals), 2)
        elif kind == "max":
            exp = max(vals, key=lambda r: r[1])[0]
        else:
            exp = round(sum(v for _, v in vals) / len(vals), 2)
        out.append(Task(
            tid, "human_units", split, title, _HU_INTRO + ask, {"metrics_text": text}, exp,
            "\n".join(text.splitlines()[:6]),
            _HU_NAIVE + _HU_HEAD + _HU_TAILS[kind],
            _HU_REF + _HU_HEAD + _HU_TAILS[kind],
        ))
    return out


# ================================================================ family: control
_WORDS = "alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima mike".split()


def _control_tasks():
    out = []
    for tid, title, seed, kind, ask in [
        ("control_1", "Count the words", 51, "count", "Return the number of words in it (an int)."),
        ("control_2", "Reverse the words", 52, "rev", "Return the list of its words in reverse order."),
    ]:
        rng = random.Random(seed)
        words = [rng.choice(_WORDS) for _ in range(rng.randint(12, 20))]
        text = "".join(w + rng.choice([" ", "  ", "\n", " "]) for w in words)
        exp = len(words) if kind == "count" else list(reversed(words))
        code = ("def solve(inputs):\n    words = inputs['text'].split()\n"
                + ("    return len(words)\n" if kind == "count" else "    return list(reversed(words))\n"))
        out.append(Task(tid, "control", "heldout", title,
                        "`inputs['text']` is a block of free text. " + ask,
                        {"text": text}, exp, "", code, code))
    return out


# ===================================================================== registry
def all_tasks() -> list[Task]:
    return _eu_tasks() + _date_tasks() + _paged_tasks() + _unit_tasks() + _control_tasks()


def get_task(task_id: str) -> Task:
    return next(t for t in all_tasks() if t.id == task_id)


def train_tasks() -> list[Task]:
    return [t for t in all_tasks() if t.split == "train"]


def heldout_tasks() -> list[Task]:
    return [t for t in all_tasks() if t.split == "heldout"]

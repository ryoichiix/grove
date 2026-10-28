"""MOCK model for UI development and offline tests. NOT Gemma.

It returns scripted answers: the naive solution until a skill for the right family is in the prompt,
then the reference solution; and a canned SKILL.md when asked to reflect. It exists so the whole
pipeline (sandbox, checkers, validator, store, selection, events, UI) can be exercised without an API
key. Anything shown to judges must come from the live model (GROVE_MODE=live or replay of a live cache).
"""
from __future__ import annotations

import re
import time

from .tasks import get_task

SKILLS = {
    "eu_csv": ("parse-european-csv-exports", """---
name: parse-european-csv-exports
description: Use when reading a CSV export of sales or other business records where numbers may use decimal commas and the delimiter may not be a comma.
---
# Parse European CSV exports

## When to use
A CSV export from a business system whose amounts look like `1.234,56`, or where the header parses as a single column.

## Procedure
1. Strip a leading BOM (`\\ufeff`) from the raw text.
2. Parse with `csv.DictReader(io.StringIO(text), delimiter=';')`; if the header collapses into one column, sniff the delimiter.
3. Convert numbers with `float(s.replace('.', '').replace(',', '.'))`: dots are thousands separators, the comma is the decimal mark.
4. Aggregate only after converting.

## Pitfalls
Quoted amounts contain the delimiter-like comma. `float('1.234,56')` raises or misreads.

## Verification
Check the header names exist and that parsed amounts are plausible magnitudes.
"""),
    "localized_dates": ("parse-localized-month-names", """---
name: parse-localized-month-names
description: Use when an operations log or event list has dates written with non-English month abbreviations such as Mär, Mai, Okt or Dez.
---
# Parse localized month names

## When to use
Event lines that start with a date like `7 Mär 2025`, where `strptime('%b')` raises ValueError.

## Procedure
1. Do not rely on `strptime` with `%b`; it only knows the current locale's English names.
2. Build an explicit month map (`Mär`->3, `Mai`->5, `Okt`->10, `Dez`->12, ...).
3. Split the date text on whitespace and build `date(year, month, day)`.
4. Group or filter on the resulting `date` objects.

## Pitfalls
Only some months differ from English, so a parser can seem to work on a few rows and crash later.

## Verification
Count parsed lines against non-empty input lines.
"""),
    "paged_api": ("fetch-all-pages-and-dedupe", """---
name: fetch-all-pages-and-dedupe
description: Use when collecting records from a paginated API function that returns items plus a next token, where pages may overlap.
---
# Fetch all pages and dedupe

## When to use
A `fetch_page(token)` style API with `items` and `next`, where the last page returns an empty-string token.

## Procedure
1. Start with `token=None` and loop until `next` is falsy (empty string counts).
2. Store records in a dict keyed by `id` so repeated records collapse.
3. Aggregate over `list(seen.values())`, never over the raw concatenation.

## Pitfalls
Adjacent pages repeat boundary records, which inflates counts, sums and top-N lists.

## Verification
Compare the number of unique ids with the number of items fetched; they should differ when pages overlap.
"""),
    "human_units": ("parse-human-readable-numbers", """---
name: parse-human-readable-numbers
description: Use when a plain-text metrics export lists values written as 1.2k, 3.4M, 2B, (2.5k) for negatives, or n/a for missing readings.
---
# Parse human-readable numbers

## When to use
Lines like `svc-01: 12.5k` or `svc-02: (3.1M)` or `svc-03: n/a`.

## Procedure
1. Treat `n/a`, `-` and empty as missing and skip them.
2. Parentheses mean negative: strip them and negate.
3. A trailing `k`, `M` or `B` multiplies by 1e3, 1e6 or 1e9.
4. Only then aggregate.

## Pitfalls
`float('12.5k')` raises ValueError. Missing readings must not count in averages.

## Verification
Count parsed rows versus lines and confirm the missing ones were excluded.
"""),
}

REFLECTIONS = {
    "eu_csv": "The file is a European-style export: it is semicolon-delimited and amounts use a decimal comma with dot thousands separators, so the default comma parser saw one big column.",
    "localized_dates": "Month names are German abbreviations, so English-only strptime fails on Mär, Mai, Okt and Dez; the dates need an explicit month map.",
    "paged_api": "Consecutive pages overlap, so the same records come back twice. Aggregating over the raw concatenation double-counts; records must be deduplicated by id.",
    "human_units": "Values carry magnitude suffixes (k, M, B), parentheses for negatives and n/a for missing readings, so a bare float() cannot parse them.",
}


class MockClient:
    mode = "mock"
    label = "MOCK scripted model (not Gemma)"

    def __init__(self, on_call=None, delay: float = 0.35):
        self.on_call = on_call
        self.delay = delay
        self.calls = 0

    def generate(self, prompt: str, *, system=None, temperature=0.2, tag: str = "") -> str:
        time.sleep(self.delay)
        m = re.search(r"\[TASK:([a-z_0-9]+)\]", prompt)
        task = get_task(m.group(1))
        fam = task.family
        if tag.startswith("select"):
            text = SKILLS[fam][0] if (fam != "control" and f"- {SKILLS[fam][0]}:" in prompt) else "NONE"
        elif tag.startswith(("reflect", "repair")):
            text = f"REFLECTION: {REFLECTIONS[fam]}\n\n```markdown\n{SKILLS[fam][1].strip()}\n```"
        else:
            right = fam == "control" or f"--- skill: {SKILLS[fam][0]} ---" in prompt
            code = task.reference_code if right else task.naive_code
            text = f"```python\n{code}\n```"
        self.calls += 1
        if self.on_call:
            self.on_call(tag=tag, model="mock-scripted", cached=False,
                         seconds=self.delay, chars_in=len(prompt), chars_out=len(text))
        return text

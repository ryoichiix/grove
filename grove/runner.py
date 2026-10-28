"""Runs untrusted solution code in a child process. Reads JSON {code, inputs} on stdin."""
import copy
import json
import sys
import traceback

R = "\x00GROVE_RESULT\x00"
E = "\x00GROVE_ERROR\x00"

payload = json.load(sys.stdin)
code, inputs = payload["code"], payload["inputs"]

pages = inputs.pop("_pages", None)
if pages is not None:
    def fetch_page(token=None):
        key = token or "start"
        if key not in pages:
            raise KeyError(f"unknown page token {token!r}")
        return copy.deepcopy(pages[key])

    inputs["fetch_page"] = fetch_page

ns = {"__name__": "grove_solution"}
try:
    exec(compile(code, "<solution>", "exec"), ns)
    if "solve" not in ns:
        raise NameError("your code must define solve(inputs)")
    result = ns["solve"](inputs)
    sys.stdout.write("\n" + R + json.dumps(result, default=str) + "\n")
except BaseException:
    tb = traceback.format_exc().strip().splitlines()
    sys.stdout.write("\n" + E + "\n".join(tb[-6:]) + "\n")

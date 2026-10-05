"""Which answers' encoders search? Census over every logged loop answer.

    python search_census.py                 # all runs with a responses.jsonl
    python search_census.py --show 2        # also print two examples per group

An encoder "searches" if, anywhere inside `encode` (including functions it
defines), it calls a *renderer*: the candidate's own `decode`, or any other
function in the answer, module-level or nested in encode, that draws an ellipse (its source both
loops over the 16x16 grid and uses cos/sin). That is what analysis by
synthesis needs: draw a guess, compare, adjust. A purely measuring encoder
(moments, thresholds, peak pixels) never calls one.

Static, from the source of each answer (`responses.jsonl` text); answers
with no code block or a syntax error are counted separately. Groups are the
run-id prefix: a = shipped, b = prompt hint, r = Haiku + reasoning low,
l = GPT-6 Luna + reasoning low.
"""

from __future__ import annotations

import argparse
import ast
import json
from collections import defaultdict
from pathlib import Path

from evo.loop import extract_code

HERE = Path(__file__).parent


def classify(code: str | None) -> str:
    if code is None:
        return "no_code"
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return "syntax_error"
    top = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    enc = top.get("encode")
    if enc is None:
        return "no_encode"

    def draws(fn: ast.FunctionDef) -> bool:
        src = ast.get_source_segment(code, fn) or ""
        return "range(16)" in src and ("cos(" in src or "sin(" in src)

    # renderers anywhere: module level or nested inside encode (a first
    # version looked only at module level and missed nested ones)
    every = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]
    renderers = {fn.name for fn in every
                 if fn.name != "encode" and (fn.name == "decode" or draws(fn))}
    called = {c.func.id for c in ast.walk(enc)
              if isinstance(c, ast.Call) and isinstance(c.func, ast.Name)}
    return "search" if called & renderers else "measure"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--show", type=int, default=0)
    args = ap.parse_args()
    groups = defaultdict(lambda: defaultdict(int))
    per_run = {}
    champs = {}
    examples = defaultdict(list)
    for f in sorted((HERE / "runs").glob("*/responses.jsonl")):
        run = f.parent.name
        if run.startswith(("probe", "mock", "s0")):
            continue
        g = run[0]
        c = defaultdict(int)
        for line in f.read_text().splitlines():
            r = json.loads(line)
            if r.get("status") == "api_error":
                continue
            k = classify(extract_code(r.get("text", "")))
            c[k] += 1
            groups[g][k] += 1
            if k == "search" and len(examples[g]) < args.show:
                examples[g].append((run, r.get("generation"), r.get("score")))
        per_run[run] = c
        best = f.parent / "best.py"
        if best.exists():
            champs[run] = classify(best.read_text())

    print(f"{'run':<5} {'answers':>7} {'search':>7} {'measure':>8} {'other':>6}  champion")
    for run, c in per_run.items():
        n = sum(c.values())
        other = n - c["search"] - c["measure"]
        print(f"{run:<5} {n:7d} {c['search']:7d} {c['measure']:8d} {other:6d}  {champs.get(run, '-')}")
    print()
    names = {"a": "A shipped", "b": "B hint", "r": "C1 Haiku+reasoning",
             "l": "C2 Luna+reasoning"}
    for g, c in groups.items():
        n = sum(c.values())
        coded = c["search"] + c["measure"]
        print(f"{names.get(g, g):<20} {c['search']:4d}/{coded:<4d} parseable encoders search "
              f"({100 * c['search'] / max(coded, 1):4.1f}%); champions searching: "
              f"{sum(1 for r, k in champs.items() if r[0] == g and k == 'search')}/"
              f"{sum(1 for r in champs if r[0] == g)}; unparseable/no code: {n - coded}")
    for g, ex in examples.items():
        print(f"  {g} examples: {ex}")


if __name__ == "__main__":
    main()

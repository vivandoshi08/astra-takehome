"""Score PREDICTIONS_smallnet.md.

    python experiments/score_smallnet.py

Writes runs/smallnet/smallnet_scored.md (all cells, then with the three
cells run during development excluded).
"""

from __future__ import annotations

import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
RUNS = HERE / "runs"
DATASETS = ("xor", "spiral", "spiral3")
SPIRALS = ("spiral", "spiral3")
SEEN = {("lam_std", "spiral", 0), ("lam_std", "spiral3", 4), ("bald_sine", "spiral", 5)}
UNCON = {"lam": "A_lamarckian", "bald": "B_baldwinian"}


def load(grid, exclude=frozenset()):
    acc, conns = defaultdict(list), defaultdict(list)
    for r in csv.DictReader(open(RUNS / grid / "results.csv")):
        if r["outcome"] != "ok" or (r["arm"], r["dataset"], int(r["seed"])) in exclude:
            continue
        acc[(r["arm"], r["dataset"])].append(float(r["heldout_acc"]))
        conns[(r["arm"], r["dataset"])].append(float(r["conns"]))
    return acc, conns


def diff(a, x, b, y, d):
    (mx, sx), (my, sy) = [(statistics.mean(v), statistics.stdev(v)) for v in (a[(x, d)], b[(y, d)])]
    return mx - my, math.sqrt((sx ** 2 + sy ** 2) / 2)


def score(exclude, label, out):
    sn, sc = load("smallnet", exclude)
    al, ac = load("allocation")
    out += [f"## {label}", "",
            "| arm | xor | spiral | spiral3 |", "|---|--:|--:|--:|"]
    for arm in ("lam_std", "bald_std", "lam_sine", "bald_sine"):
        cells = [f"{statistics.mean(sn[(arm, d)]):.3f} ± {statistics.stdev(sn[(arm, d)]):.3f} "
                 f"({statistics.mean(sc[(arm, d)]):.1f} conns, n={len(sn[(arm, d)])})" for d in DATASETS]
        out.append(f"| {arm} (<= 6) | " + " | ".join(cells) + " |")
    for s, a in UNCON.items():
        cells = [f"{statistics.mean(al[(a, d)]):.3f} ± {statistics.stdev(al[(a, d)]):.3f} "
                 f"({statistics.mean(ac[(a, d)]):.1f} conns)" for d in DATASETS]
        out.append(f"| {a} (unconstrained) | " + " | ".join(cells) + " |")
    out += ["", "```"]
    for s, a in UNCON.items():
        dd, sp = diff(sn, f"{s}_std", al, a, "xor")
        ok = abs(dd) <= max(sp, 0.02)
        out.append(f"S1a {s:4} xor   <=6 - unconstrained {dd:+.3f} (spread {sp:.3f}): "
                   f"{'supported' if ok else 'contradicted'}")
        for d in SPIRALS:
            dd, sp = diff(sn, f"{s}_std", al, a, d)
            v = "supported" if dd < -sp else "contradicted" if dd > sp else "inconclusive"
            out.append(f"S1b {s:4} {d:8} <=6 - unconstrained {dd:+.3f} (spread {sp:.3f}): {v}")
    for s in UNCON:
        for d in DATASETS:
            dd, sp = diff(sn, f"{s}_sine", sn, f"{s}_std", d)
            if d == "xor":
                v = "supported" if abs(dd) <= sp else "contradicted"
                out.append(f"S3  {s:4} xor      sine - std {dd:+.3f} (spread {sp:.3f}): {v}")
            else:
                v = "supported" if dd > sp else "contradicted" if dd < -sp else "inconclusive"
                out.append(f"S2  {s:4} {d:8} sine - std {dd:+.3f} (spread {sp:.3f}): {v}")
    for d in SPIRALS:
        dd, sp = diff(sn, "lam_std", sn, "bald_std", d)
        out.append(f"S4  {d:8} lam_std - bald_std {dd:+.3f} (spread {sp:.3f}): "
                   f"{'holds' if dd > 0 else 'does not hold'}")
    out += ["```", ""]


def main() -> int:
    out = ["# Small-network track — scored", ""]
    rows = list(csv.DictReader(open(RUNS / "smallnet" / "results.csv")))
    bad = [r["cell"] for r in rows if r["outcome"] != "ok"]
    over = [r["cell"] for r in rows if r["outcome"] == "ok" and float(r["conns"]) > 6]
    out += [f"Cells not ok: {bad or 'none'}; champions over 6 connections: {over or 'none'}", ""]
    score(frozenset(), "All cells (seeds 0–9)", out)
    score(SEEN, "Excluding the three cells run during development", out)
    text = "\n".join(out) + "\n"
    print(text)
    (RUNS / "smallnet" / "smallnet_scored.md").write_text(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

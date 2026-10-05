"""Score PREDICTIONS_mechanism2.md (Test A from existing logs, Test B from new runs).

    python experiments/score_mechanism2.py [--only-a]

Writes runs/mechanism2/mechanism2_scored.md (or prints Test A alone with
--only-a, before the Test B runs exist).
"""

from __future__ import annotations

import csv
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
RUNS = HERE / "runs"
SPIRALS = ("spiral", "spiral3")


def children(path):
    """[(loss, kind, same_shape)] for non-elite children of one run's log."""
    by_gen = defaultdict(dict)
    for line in open(path):
        r = json.loads(line)
        by_gen[r["gen"]][r["key"]] = r
    out = []
    for g in sorted(by_gen):
        if g == 0:
            continue
        prev = by_gen[g - 1]
        for key, r in by_gen[g].items():
            if key in prev or len(r["parents"]) != 2:
                continue
            p1, p2 = r["parents"]
            if p1 not in prev or p2 not in prev:
                continue
            loss = r["pre_acc"] - (prev[p1]["acc"] + prev[p2]["acc"]) / 2
            kind = "clone" if p1 == p2 else "cross"
            same = (kind == "clone" and r["nodes"] == prev[p1]["nodes"]
                    and r["conns"] == prev[p1]["conns"])
            out.append((loss, kind, same))
    return out


def loss_summary(arm_dir):
    allc, per_run = [], []
    for f in sorted(arm_dir.glob("*/genomes.jsonl")):
        c = children(f)
        allc += c
        per_run.append((f.parent.name, statistics.median(x[0] for x in c)))
    return allc, per_run


def med(xs):
    xs = list(xs)
    return (statistics.median(xs), len(xs)) if xs else (float("nan"), 0)


def test_a(out):
    allc, per_run = loss_summary(RUNS / "mechanism_logs" / "lamarckian_log")
    mc, nc = med(x[0] for x in allc if x[1] == "clone")
    mx, nx = med(x[0] for x in allc if x[1] == "cross")
    ms, ns = med(x[0] for x in allc if x[2])
    d = mc - mx
    if d >= 0.05:
        v = "H4 supported (clones keep clearly more)"
    elif abs(d) < 0.03:
        v = "H3 supported (clones lose about as much)" if mc <= -0.08 else \
            "|d| < 0.03 but clones lose < 0.08 (neither H3 nor H4 fits)"
    else:
        v = "in between: both contribute"
    out += ["Test A (Lamarckian, default config, runs/mechanism_logs):",
            f"  clones  median loss {mc:+.3f} (n={nc})",
            f"  crosses median loss {mx:+.3f} (n={nx})",
            f"  clones with unchanged node/connection counts {ms:+.3f} (n={ns})",
            f"  d = clones - crosses = {d:+.3f} -> {v}",
            "  per-run median loss (all children): " +
            ", ".join(f"{k} {m:+.3f}" for k, m in per_run)]
    _, per_b = loss_summary(RUNS / "mechanism_logs" / "baldwinian_log")
    allb, _ = loss_summary(RUNS / "mechanism_logs" / "baldwinian_log")
    out.append(f"  (Baldwinian reference: clones {med(x[0] for x in allb if x[1] == 'clone')[0]:+.3f}, "
               f"crosses {med(x[0] for x in allb if x[1] == 'cross')[0]:+.3f})")


def acc(grid):
    a = defaultdict(list)
    for r in csv.DictReader(open(RUNS / grid / "results.csv")):
        if r["outcome"] == "ok":
            a[(r["arm"], r["dataset"])].append(float(r["heldout_acc"]))
    return a


def test_b(out):
    new, old = acc("mechanism2"), acc("mechanism")
    out += ["", "Test B (low parameter mutation):",
            "| arm | xor | spiral | spiral3 |", "|---|--:|--:|--:|"]
    table = {("lam", "low"): new, ("bald", "low"): new, ("lam", "def"): old, ("bald", "def"): old}
    names = {("lam", "low"): "lam_lowmut", ("bald", "low"): "bald_lowmut",
             ("lam", "def"): "lam_d10", ("bald", "def"): "bald_d10"}
    for k, name in names.items():
        cells = []
        for d in ("xor", *SPIRALS):
            xs = table[k][(name, d)]
            cells.append(f"{statistics.mean(xs):.3f} ± {statistics.stdev(xs):.3f} (n={len(xs)})")
        out.append(f"| {name} | " + " | ".join(cells) + " |")
    out.append("")
    for s, target, ok_fn in (("lamarckian", "> -0.08", lambda m: m > -0.08),
                             ("baldwinian", "< -0.10", lambda m: m < -0.10)):
        allc, per_run = loss_summary(RUNS / "mechanism2_logs" / f"{s}_lowmut_log")
        m, n = med(x[0] for x in allc)
        tag = "B1" if s == "lamarckian" else "B3"
        out.append(f"{tag} {s} low-mutation children median loss {m:+.3f} (n={n}; predicted "
                   f"{target}): {'supported' if ok_fn(m) else 'contradicted'}")
    for d in SPIRALS:
        lm, ld = new[("lam_lowmut", d)], old[("lam_d10", d)]
        bm, bd = new[("bald_lowmut", d)], old[("bald_d10", d)]
        dl = statistics.mean(lm) - statistics.mean(ld)
        db = statistics.mean(bm) - statistics.mean(bd)
        inter = dl - db
        spread = math.sqrt(statistics.mean([statistics.variance(x) for x in (lm, ld, bm, bd)]))
        v = ("supported" if inter > spread else "contradicted" if inter < -spread
             else "inconclusive")
        out.append(f"B2 {d:8} lam change {dl:+.3f}, bald change {db:+.3f}, "
                   f"interaction {inter:+.3f} (spread {spread:.3f}): {v}")


def main() -> int:
    out = ["# Mechanism round 2 — scored", "", "```"]
    test_a(out)
    if "--only-a" not in sys.argv:
        test_b(out)
    out.append("```")
    text = "\n".join(out) + "\n"
    print(text)
    if "--only-a" not in sys.argv:
        (RUNS / "mechanism2" / "mechanism2_scored.md").write_text(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

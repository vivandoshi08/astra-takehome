"""Score PREDICTIONS_mechanism.md.

    python experiments/score_mechanism.py

Reads runs/mechanism/results.csv (E1, E2), each mechanism run's
fitness_history.csv (species counts), runs/mechanism_logs/*/genomes.jsonl
(E3), and runs/allocation/results.csv (consistency check). Writes
runs/mechanism/mechanism_scored.md.
"""

from __future__ import annotations

import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
RUNS = HERE / "runs"
SPIRALS = ("spiral", "spiral3")
DEPTHS = ("bald_d10", "bald_d30", "bald_d100", "bald_d300")


def load(grid):
    acc, rows = defaultdict(list), list(csv.DictReader(open(RUNS / grid / "results.csv")))
    for r in rows:
        if r["outcome"] == "ok":
            acc[(r["arm"], r["dataset"])].append(float(r["heldout_acc"]))
    return acc, rows


def ms(xs):
    return statistics.mean(xs), statistics.stdev(xs)


def cmp(acc, x, y, d):
    (mx, sx), (my, sy) = ms(acc[(x, d)]), ms(acc[(y, d)])
    return mx - my, math.sqrt((sx ** 2 + sy ** 2) / 2)


def verdict(diff, spread, sign=+1):
    if abs(diff) <= spread:
        return "inconclusive"
    return "supported" if diff * sign > 0 else "contradicted"


def max_species(grid, arm, d):
    out = []
    for f in sorted((RUNS / grid / arm).glob(f"{d}_s*/fitness_history.csv")):
        out.append(max(int(r["n_species"]) for r in csv.DictReader(open(f))))
    return out


def head_start(path):
    """(child pre - mean parent post) for non-elite children; gains late."""
    by_gen = defaultdict(dict)
    for line in open(path):
        r = json.loads(line)
        by_gen[r["gen"]][r["key"]] = r
    deltas, gains = [], []
    for g in sorted(by_gen):
        if g >= 20:
            gains += [r["acc"] - r["pre_acc"] for r in by_gen[g].values() if r["steps"] > 0]
        if g == 0:
            continue
        prev = by_gen[g - 1]
        for key, r in by_gen[g].items():
            if key in prev:      # elite carried over, not a child
                continue
            parents = [prev[p] for p in r["parents"] if p in prev]
            if parents:
                deltas.append(r["pre_acc"] - statistics.mean(p["acc"] for p in parents))
    return deltas, gains


def main() -> int:
    acc, rows = load("mechanism")
    bad = [r["cell"] for r in rows if r["outcome"] != "ok"]
    out = ["# Mechanism experiments — scored", "", f"Cells not ok: {bad or 'none'}", "",
           "| arm | xor | spiral | spiral3 | max species (median per run: spiral / spiral3) |",
           "|---|--:|--:|--:|--:|"]
    for arm in (*DEPTHS, "lam_d10", "lam_topo", "bald_topo"):
        cells = []
        for d in ("xor", *SPIRALS):
            xs = acc.get((arm, d), [])
            cells.append(f"{statistics.mean(xs):.3f} ± {statistics.stdev(xs):.3f} (n={len(xs)})"
                         if len(xs) > 1 else "—")
        sp = [statistics.median(max_species("mechanism", arm, d) or [float('nan')]) for d in SPIRALS]
        out.append(f"| {arm} | " + " | ".join(cells) + f" | {sp[0]:g} / {sp[1]:g} |")
    out += ["", "```"]

    for d in SPIRALS:
        diff, spread = cmp(acc, "bald_d300", "bald_d10", d)
        out.append(f"M1 {d:8} bald_d300 - bald_d10 {diff:+.3f} (spread {spread:.3f}): "
                   f"{verdict(diff, spread)}")
        rev = []
        for a, b in zip(DEPTHS, DEPTHS[1:]):
            dd, sp = cmp(acc, b, a, d)
            if dd < -sp:
                rev.append(f"{a}->{b} {dd:+.3f}")
        out.append(f"   {d:8} reversals beyond spread along depth: {rev or 'none'}")
        gaps = {a: cmp(acc, "lam_d10", a, d)[0] for a in DEPTHS}
        g10, gmin = gaps["bald_d10"], min(gaps[a] for a in DEPTHS[1:])
        v = ("supported" if gmin <= 0.5 * g10 else
             "contradicted" if all(gaps[a] >= g10 for a in DEPTHS[1:]) else "partial")
        out.append(f"M2 {d:8} gap to lam_d10 by depth: " +
                   ", ".join(f"{a[5:]} {gaps[a]:+.3f}" for a in DEPTHS) + f" -> {v}")

    for d in SPIRALS:
        lam = max_species("mechanism", "lam_topo", d)
        crashed = [r["cell"] for r in rows if r["arm"] == "lam_topo" and r["dataset"] == d
                   and r["outcome"] != "ok"]
        med = statistics.median(lam) if lam else float("nan")
        v = "supported" if not crashed and med <= 5 else "contradicted"
        out.append(f"M3 {d:8} lam_topo max species per run {lam}; crashed {len(crashed)}; "
                   f"median {med:g} -> {v}")
        diff, spread = cmp(acc, "lam_topo", "bald_topo", d)
        orig = cmp(acc, "lam_d10", "bald_d10", d)[0]
        out.append(f"M4 {d:8} lam_topo - bald_topo {diff:+.3f} (spread {spread:.3f}): "
                   f"{verdict(diff, spread)}; ratio to weight-aware gap {orig:+.3f}: "
                   f"{diff / orig:.2f}x")

    for strat in ("lamarckian", "baldwinian"):
        deltas, gains = [], []
        for f in sorted((RUNS / "mechanism_logs" / f"{strat}_log").glob("*/genomes.jsonl")):
            dl, gn = head_start(f)
            deltas += dl
            gains += gn
        md, mg = statistics.median(deltas), statistics.median(gains)
        target = "> -0.05" if strat == "lamarckian" else "< -0.10"
        ok = md > -0.05 if strat == "lamarckian" else md < -0.10
        out.append(f"M5 {strat:11} child pre - parent post: median {md:+.3f} over "
                   f"{len(deltas)} children (predicted {target}): "
                   f"{'supported' if ok else 'contradicted'}; median gain gens 20-29 {mg:+.3f}")

    alloc, _ = load("allocation")
    pairs = {"bald_d10": "A_baldwinian", "bald_d100": "B_baldwinian", "lam_d10": "A_lamarckian"}
    mism = []
    for m, a in pairs.items():
        for d in ("xor", *SPIRALS):
            if sorted(acc[(m, d)]) != sorted(alloc[(a, d)]):
                mism.append(f"{m} vs {a} on {d}")
    out.append(f"consistency with the allocation grid: {'identical' if not mism else mism}")
    out.append("```")
    text = "\n".join(out) + "\n"
    print(text)
    (RUNS / "mechanism" / "mechanism_scored.md").write_text(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

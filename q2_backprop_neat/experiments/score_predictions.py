"""Score the allocation predictions in PREDICTIONS_allocation.md.

    python experiments/score_predictions.py [runs/allocation/results.csv]

Reads the grid's results.csv (ok cells only), prints a held-out table per
dataset x strategy x arm, then each prediction's verdict by the rule fixed
in PREDICTIONS_allocation.md. Writes runs/allocation/predictions_scored.md.
"""

from __future__ import annotations

import csv
import math
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
ARMS = "ABCDE"
STRATS = ("lamarckian", "baldwinian")
SPIRALS = ("spiral", "spiral3")


def load(path):
    cells = {}
    for r in csv.DictReader(open(path)):
        if r["outcome"] != "ok":
            continue
        arm, strat = r["arm"].split("_", 1)
        cells.setdefault((r["dataset"], strat, arm), []).append(float(r["heldout_acc"]))
    return cells


def stats(xs):
    return statistics.mean(xs), (statistics.stdev(xs) if len(xs) > 1 else float("nan"))


def compare(cells, d, s, x, sign):
    """Verdict for 'arm x vs A has sign `sign`' (-1: x < A)."""
    mx, sx = stats(cells[(d, s, x)])
    ma, sa = stats(cells[(d, s, "A")])
    diff, spread = mx - ma, math.sqrt((sx ** 2 + sa ** 2) / 2)
    if abs(diff) > spread:
        verdict = "supported" if diff * sign > 0 else "contradicted"
    else:
        verdict = "inconclusive"
    return diff, spread, verdict


def main() -> int:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "runs/allocation/results.csv"
    cells = load(path)
    out = ["# Allocation predictions — scored", "",
           f"From `{path.relative_to(HERE)}`; held-out accuracy, mean ± sd over seeds.", ""]

    out += ["| dataset | strategy | " + " | ".join(ARMS) + " |",
            "|---|---|" + "--:|" * len(ARMS)]
    for d in ("xor", *SPIRALS):
        for s in STRATS:
            row = []
            for a in ARMS:
                xs = cells.get((d, s, a), [])
                m, sd = stats(xs) if xs else (float("nan"), float("nan"))
                row.append(f"{m:.3f} ± {sd:.3f} (n={len(xs)})")
            out.append(f"| {d} | {s} | " + " | ".join(row) + " |")
    out.append("")

    lines = []
    for d in SPIRALS:
        diff, spread, v = compare(cells, d, "baldwinian", "B", -1)
        lines.append(f"P1  {d:8} baldwinian  B-A {diff:+.3f} (spread {spread:.3f}): {v}")
        dl, sl, _ = compare(cells, d, "lamarckian", "B", -1)
        gap, big = dl - diff, max(sl, spread)
        v2 = ("supported" if gap > big else "contradicted" if gap < -big else "inconclusive")
        lines.append(f"P1b {d:8} (B-A) lamarckian {dl:+.3f} vs baldwinian {diff:+.3f}: "
                     f"gap {gap:+.3f} (spread {big:.3f}): {v2}")
        for s in STRATS:
            diff, spread, v = compare(cells, d, s, "C", -1)
            lines.append(f"P2  {d:8} {s:11} C-A {diff:+.3f} (spread {spread:.3f}): {v}")
        diff, spread, v = compare(cells, d, "baldwinian", "D", -1)
        lines.append(f"P3  {d:8} baldwinian  D-A {diff:+.3f} (spread {spread:.3f}): {v}")
    for s in STRATS:
        for a in "BCDE":
            diff, spread, _ = compare(cells, "xor", s, a, -1)
            beyond = "beyond spread" if abs(diff) > spread else "within spread"
            lines.append(f"xor {s:11} {a}-A {diff:+.3f} (spread {spread:.3f}): {beyond}")
    for s in STRATS:
        for d in SPIRALS:
            diff, spread, _ = compare(cells, d, s, "E", +1)
            lines.append(f"E   {d:8} {s:11} E-A {diff:+.3f} (spread {spread:.3f}) [no prediction]")

    out += ["```", *lines, "```"]
    text = "\n".join(out) + "\n"
    print(text)
    (path.parent / "predictions_scored.md").write_text(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

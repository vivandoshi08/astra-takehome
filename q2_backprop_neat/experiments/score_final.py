"""Score PREDICTIONS_mechanism_final.md.

    python experiments/score_final.py

Writes runs/mechanism_final/final_scored.md.
"""

from __future__ import annotations

import csv
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE / "experiments"))
from score_mechanism2 import loss_summary  # noqa: E402

RUNS = HERE / "runs"
SPIRALS = ("spiral", "spiral3")


def main() -> int:
    acc = defaultdict(list)
    rows = list(csv.DictReader(open(RUNS / "mechanism_final" / "results.csv")))
    for r in rows:
        if r["outcome"] == "ok":
            acc[(r["arm"], r["dataset"])].append(float(r["heldout_acc"]))
    out = ["# Mechanism final test (seeds 20–29) — scored", "",
           f"Cells not ok: {[r['cell'] for r in rows if r['outcome'] != 'ok'] or 'none'}", "",
           "| arm | xor | spiral | spiral3 |", "|---|--:|--:|--:|"]
    for arm in ("lam_default", "bald_default", "lam_norep", "bald_norep"):
        cells = [f"{statistics.mean(acc[(arm, d)]):.3f} ± {statistics.stdev(acc[(arm, d)]):.3f} "
                 f"(n={len(acc[(arm, d)])})" for d in ("xor", *SPIRALS)]
        out.append(f"| {arm} | " + " | ".join(cells) + " |")
    out += ["", "```"]
    med = {}
    for tag in ("default", "norep"):
        allc, per_run = loss_summary(RUNS / "mechanism_final_logs" / f"lam_{tag}_log")
        med[tag] = statistics.median(x[0] for x in allc)
        out.append(f"lam_{tag}: median child loss {med[tag]:+.3f} (n={len(allc)}); per run " +
                   ", ".join(f"{k} {m:+.3f}" for k, m in per_run))
    d1 = med["norep"] - med["default"]
    out.append(f"F1 change in median child loss {d1:+.3f} (predicted >= +0.03): "
               f"{'supported' if d1 >= 0.03 else 'contradicted'}")
    for d in SPIRALS:
        ld, ln = acc[("lam_default", d)], acc[("lam_norep", d)]
        bd, bn = acc[("bald_default", d)], acc[("bald_norep", d)]
        dl = statistics.mean(ln) - statistics.mean(ld)
        db = statistics.mean(bn) - statistics.mean(bd)
        spread = math.sqrt(statistics.mean([statistics.variance(x) for x in (ld, ln, bd, bn)]))
        inter = dl - db
        v = "supported" if inter > spread else "contradicted" if inter < -spread else "inconclusive"
        out.append(f"F2 {d:8} lam change {dl:+.3f}, bald change {db:+.3f}, interaction "
                   f"{inter:+.3f} (spread {spread:.3f}): {v}")
        gap = statistics.mean(ld) - statistics.mean(bd)
        sp = math.sqrt((statistics.variance(ld) + statistics.variance(bd)) / 2)
        v3 = "supported" if gap > sp else "contradicted" if gap < -sp else "inconclusive"
        out.append(f"F3 {d:8} lam_default - bald_default {gap:+.3f} (spread {sp:.3f}): {v3}")
    out.append("```")
    text = "\n".join(out) + "\n"
    print(text)
    (RUNS / "mechanism_final" / "final_scored.md").write_text(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

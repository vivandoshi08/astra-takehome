"""Score PREDICTIONS_confirm.md from runs/confirm/results.csv.

    python experiments/score_confirm.py
"""

from __future__ import annotations

import csv
import statistics
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
SEEDS_0_9 = {  # (claim, dataset): seeds 0-9 difference, from runs/allocation
    ("C1", "spiral"): 0.088, ("C1", "spiral3"): 0.148,
    ("C2", "spiral"): 0.053, ("C2", "spiral3"): 0.063,
    ("C3", "spiral"): -0.026, ("C3", "spiral3"): -0.017,
    ("C4", "spiral"): 0.034, ("C4", "spiral3"): 0.085,
}
CLAIMS = {  # claim: (arm x, arm y, expected sign of mean(x) - mean(y))
    "C1": ("A_lamarckian", "A_baldwinian", +1),
    "C2": ("B_baldwinian", "A_baldwinian", +1),
    "C3": ("B_lamarckian", "A_lamarckian", -1),
    "C4": ("A_lamarckian", "B_baldwinian", +1),
}


def main() -> int:
    rows = [r for r in csv.DictReader(open(HERE / "runs/confirm/results.csv"))]
    acc = {}
    for r in rows:
        if r["outcome"] == "ok":
            acc.setdefault((r["arm"], r["dataset"]), []).append(float(r["heldout_acc"]))
    bad = [r["cell"] for r in rows if r["outcome"] != "ok"]
    out = ["# Confirmation (seeds 10–12) — scored", "",
           f"Cells not ok: {bad or 'none'}", "",
           "| arm | xor | spiral | spiral3 |", "|---|--:|--:|--:|"]
    for arm in ("A_lamarckian", "A_baldwinian", "B_lamarckian", "B_baldwinian"):
        cells = []
        for d in ("xor", "spiral", "spiral3"):
            xs = acc.get((arm, d), [])
            cells.append(f"{statistics.mean(xs):.3f} ± {statistics.stdev(xs):.3f} (n={len(xs)})"
                         if len(xs) > 1 else "—")
        out.append(f"| {arm} | " + " | ".join(cells) + " |")
    out += ["", "```"]
    for c, (x, y, sign) in CLAIMS.items():
        for d in ("spiral", "spiral3"):
            diff = statistics.mean(acc[(x, d)]) - statistics.mean(acc[(y, d)])
            ok = "holds" if diff * sign > 0 else "does NOT hold"
            out.append(f"{c} {d:8} {x} - {y}: {diff:+.3f} (seeds 0-9: "
                       f"{SEEDS_0_9[(c, d)]:+.3f}) -> {ok}")
    out.append("```")
    text = "\n".join(out) + "\n"
    print(text)
    (HERE / "runs/confirm/confirm_scored.md").write_text(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

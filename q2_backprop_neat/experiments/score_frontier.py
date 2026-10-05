"""Score PREDICTIONS_frontier.md and draw the frontier.

    python experiments/score_frontier.py

Writes runs/frontier/frontier_scored.md and figures/frontier.png.
"""

from __future__ import annotations

import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent.parent
RUNS = HERE / "runs"
DATASETS = ("xor", "spiral", "spiral3")
LIN = ("lin_0", "lin_0.001", "lin_0.003", "lin_0.01", "lin_0.03")
MULT = ("lin_0", "mult_0.1", "mult_0.3", "mult_1")
STRATS = {"lam": ("Lamarckian, 10 x 100%", "#2a78d6"), "bald": ("Baldwinian, 100 x 10%", "#eb6834")}
FLOOR = {"xor": 0.513, "spiral": 0.520, "spiral3": 0.567}


def load():
    acc, conns = defaultdict(list), defaultdict(list)
    rows = list(csv.DictReader(open(RUNS / "frontier" / "results.csv")))
    for r in rows:
        if r["outcome"] == "ok":
            k = (r["arm"], r["dataset"])
            acc[k].append(float(r["heldout_acc"]))
            conns[k].append(float(r["conns"]))
    return acc, conns, rows


def ms(xs):
    return statistics.mean(xs), statistics.stdev(xs)


def diff(d, a, b, k):
    (ma, sa), (mb, sb) = ms(d[(a, k)]), ms(d[(b, k)])
    return ma - mb, math.sqrt((sa ** 2 + sb ** 2) / 2)


def lam_label(tag):
    return "λ=0" if tag == "lin_0" else ("λ=" + tag.split("_")[1])


def plot(acc, conns):
    fig, axes = plt.subplots(2, 3, figsize=(11, 6.6), sharey="col")
    for row, (tags, form) in enumerate(((LIN, "linear: acc − λ·C"),
                                        (MULT, "mult: 1 − (1 − acc)(1 + λ√C)"))):
        for col, d in enumerate(DATASETS):
            ax = axes[row, col]
            ax.axhline(FLOOR[d], color="#b4b2a9", lw=1, ls="--", zorder=0)
            for s, (label, colour) in STRATS.items():
                xs = [statistics.mean(conns[(f"{s}_{t}", d)]) for t in tags]
                ys = [statistics.mean(acc[(f"{s}_{t}", d)]) for t in tags]
                es = [statistics.stdev(acc[(f"{s}_{t}", d)]) for t in tags]
                ax.errorbar(xs, ys, yerr=es, color=colour, lw=2, marker="o", ms=6,
                            capsize=0, elinewidth=1, label=label,
                            mec="#fcfcfb", mew=1.5)
                for t, x, y in zip(tags, xs, ys):
                    if t in (tags[0], tags[-1]):
                        ax.annotate(lam_label(t), (x, y), textcoords="offset points",
                                    xytext=(5, 4), fontsize=8, color="#5f5e5a")
            ax.set_title(f"{d} — {form}", fontsize=10)
            ax.grid(color="#e8e6dc", lw=0.6)
            ax.set_axisbelow(True)
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
            if row == 1:
                ax.set_xlabel("champion enabled connections (mean of 10 seeds)")
            if col == 0:
                ax.set_ylabel("held-out accuracy (mean ± sd)")
    axes[0, 0].legend(frameon=False, fontsize=8, loc="lower right")
    fig.text(0.01, 0.005, "dashed line: majority-class floor on held-out", fontsize=8,
             color="#5f5e5a")
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    out = HERE / "figures" / "frontier.png"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


def main() -> int:
    acc, conns, rows = load()
    out = ["# Complexity frontier — scored", "",
           f"Cells not ok: {[r['cell'] for r in rows if r['outcome'] != 'ok'] or 'none'}", "",
           "| strategy | penalty | " + " | ".join(f"{d} held-out | {d} conns" for d in DATASETS) + " |",
           "|---|---|" + "--:|--:|" * len(DATASETS)]
    for s in STRATS:
        for t in LIN + MULT[1:]:
            cells = []
            for d in DATASETS:
                (ma, sa), (mc, sc) = ms(acc[(f"{s}_{t}", d)]), ms(conns[(f"{s}_{t}", d)])
                cells.append(f"{ma:.3f} ± {sa:.3f} | {mc:.1f} ± {sc:.1f}")
            form = "linear" if t.startswith("lin") else "mult"
            out.append(f"| {s} | {form} {lam_label(t)} | " + " | ".join(cells) + " |")
    out += ["", "```"]
    ok1 = []
    for s in STRATS:
        for d in DATASETS:
            dd, sp = diff(conns, f"{s}_lin_0.03", f"{s}_lin_0", d)
            ok1.append(dd < -sp)
            out.append(f"FP1 {s:4} {d:8} conns λ0.03 − λ0 {dd:+.1f} (spread {sp:.1f})")
    out.append(f"FP1 {'supported' if all(ok1) else 'contradicted'} ({sum(ok1)}/6 cells)")
    rev = []
    for s in STRATS:
        for d in DATASETS:
            for a, b in zip(LIN, LIN[1:]):
                dd, sp = diff(conns, f"{s}_{b}", f"{s}_{a}", d)
                if dd > sp:
                    rev.append(f"{s} {d} {a}->{b} {dd:+.1f}")
    out.append(f"FP2 increases in conns beyond spread along λ: {rev or 'none'} -> "
               f"{'supported' if not rev else 'contradicted'}")
    ok3 = []
    for s in STRATS:
        for d in DATASETS:
            dd, sp = diff(acc, f"{s}_lin_0.01", f"{s}_lin_0", d)
            want = abs(dd) <= sp if d == "xor" else dd < -sp
            ok3.append(want)
            out.append(f"FP3 {s:4} {d:8} held-out λ0.01 − λ0 {dd:+.3f} (spread {sp:.3f}): "
                       f"{'as predicted' if want else 'not as predicted'}")
    out.append(f"FP3 {'supported' if all(ok3) else 'partly supported' if any(ok3) else 'contradicted'} "
               f"({sum(ok3)}/6 cells as predicted)")
    better = []
    for s in STRATS:
        for d in DATASETS:
            for t in LIN[1:] + MULT[1:]:
                dd, sp = diff(acc, f"{s}_{t}", f"{s}_lin_0", d)
                if dd > sp:
                    better.append(f"{s} {d} {t} {dd:+.3f}")
    out.append(f"FP4 penalised arms beating λ=0 beyond spread: {better or 'none'} -> "
               f"{'supported' if not better else 'contradicted'}")
    ok5 = []
    for s in STRATS:
        rel = {d: statistics.mean(conns[(f"{s}_mult_1", d)]) / statistics.mean(conns[(f"{s}_lin_0", d)])
               for d in ("xor", "spiral3")}
        ok5.append(rel["xor"] > rel["spiral3"])
        out.append(f"FP5 {s:4} conns ratio mult λ1 / λ0: xor {rel['xor']:.2f}, spiral3 {rel['spiral3']:.2f}")
    out.append(f"FP5 {'supported' if all(ok5) else 'contradicted' if not any(ok5) else 'mixed'}")
    n6 = sum(statistics.mean(acc[(f"lam_{t}", d)]) >= statistics.mean(acc[(f"bald_{t}", d)])
             for t in LIN for d in ("spiral", "spiral3"))
    out.append(f"FP6 Lamarckian >= Baldwinian in {n6}/10 (spiral, spiral3) x linear λ cells -> "
               f"{'supported' if n6 == 10 else 'contradicted' if n6 <= 5 else 'mostly'}")
    alloc = defaultdict(list)
    for r in csv.DictReader(open(RUNS / "allocation" / "results.csv")):
        if r["outcome"] == "ok":
            alloc[(r["arm"], r["dataset"])].append(float(r["heldout_acc"]))
    same = all(sorted(acc[(f"{s}_lin_0", d)]) == sorted(alloc[(a, d)])
               for s, a in (("lam", "A_lamarckian"), ("bald", "B_baldwinian")) for d in DATASETS)
    out.append(f"λ=0 reproduces the allocation grid's cells: {same}")
    out.append("```")
    path = plot(acc, conns)
    out.append(f"\nFigure: `{path.relative_to(HERE)}`")
    text = "\n".join(out) + "\n"
    print(text)
    (RUNS / "frontier" / "frontier_scored.md").write_text(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

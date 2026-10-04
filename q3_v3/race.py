"""Score your model, the baselines, and any loop champions on the same
terms, and plot what each one does.

    python race.py                       # your model vs the baselines
    python race.py runs/r1/best.py ...   # include loop champions

Every entry goes through tasks/ellipses.py::evaluate — the same function
that scores loop candidates. The table is the race; the plots
(race_<name>.png) show each model's reconstructions next to its inputs.

Pass --images to score on a different image set (any (n, 256) array
exposed by data.py, e.g. HELD_OUT).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import data
import baselines
import student
import plots
from evo.loop import compile_candidate
from tasks import ellipses


def entries(champion_paths):
    yield "student (yours)", student.fns()
    yield "mean", baselines.mean_fns()
    yield "pca (k=6)", baselines.pca_fns()
    yield "vae (k=6)", baselines.vae_fns()
    for p in champion_paths:
        path = Path(p)
        fns = compile_candidate(path.read_text(), ellipses.ENTRY_POINT,
                                isolated=ellipses.ISOLATED_NAMESPACES)
        if fns is None:
            print(f"[skip] {p}: does not compile to (encode, decode)")
            continue
        yield f"champion ({path.parent.name})", fns


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("champions", nargs="*",
                    help="paths to best.py files from runs/")
    ap.add_argument("--images", default="PUBLIC",
                    help="name of an image array in data.py (default PUBLIC)")
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args()

    images = getattr(data, args.images)
    print(f"scoring on data.{args.images} "
          f"({len(images)} images; higher is better, 0.0 is perfect)\n")

    results = []
    for name, fns in entries(args.champions):
        score = ellipses.evaluate(fns, images=images)
        results.append((score, name, fns))
        print(f"  {name:24s} {score:+.4f}")

    if not args.no_plots:
        print()
        for score, name, fns in results:
            if score == float("-inf"):
                continue
            slug = "".join(c if c.isalnum() else "_" for c in name).strip("_")
            print(" ", plots.plot_candidate(fns, f"race_{slug}.png",
                                            images=images))


if __name__ == "__main__":
    main()

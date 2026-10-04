"""Plotting helpers: decision boundaries and fitness curves.

    python viz.py --dataset spiral --champion runs/spiral/champion.pkl

or from your own code:

    from viz import plot_boundary
    plot_boundary(net.predict, X, y, "spiral", out="boundary.png")
"""

from __future__ import annotations

import argparse
import csv
import pickle
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless-safe; drop this if you want interactive windows
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))


def plot_boundary(predict_fn, X, y, title: str = "", out: str = "boundary.png",
                  resolution: int = 300) -> str:
    """Contour the classifier's decision regions with the data on top."""
    X = np.asarray(X)
    pad = 0.25
    x_min, x_max = X[:, 0].min() - pad, X[:, 0].max() + pad
    y_min, y_max = X[:, 1].min() - pad, X[:, 1].max() + pad
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, resolution),
                         np.linspace(y_min, y_max, resolution))
    grid = np.c_[xx.ravel(), yy.ravel()]
    zz = np.asarray(predict_fn(grid)).reshape(xx.shape)

    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.contourf(xx, yy, zz, levels=[-0.5, 0.5, 1.5], alpha=0.25,
                colors=["#4C72B0", "#DD8452"])
    for label, colour in ((0, "#4C72B0"), (1, "#DD8452")):
        pts = X[np.asarray(y) == label]
        ax.scatter(pts[:, 0], pts[:, 1], s=9, c=colour, edgecolors="none",
                   label=f"class {label}")
    ax.set_title(title)
    ax.set_aspect("equal")
    ax.legend(loc="upper right", frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def plot_fitness(history_csv: str, out: str = "fitness.png") -> str:
    """Best and mean fitness per generation, from a run's fitness_history.csv."""
    gens, best, mean = [], [], []
    with open(history_csv) as fh:
        for row in csv.DictReader(fh):
            gens.append(int(row["generation"]))
            best.append(float(row["best_fitness"]))
            mean.append(float(row["mean_fitness"]))

    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot(gens, best, label="population best")
    ax.plot(gens, mean, label="population mean", alpha=0.75)
    ax.set_xlabel("generation")
    ax.set_ylabel("fitness")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def main() -> int:
    from datasets import DATASETS, get_dataset
    from network import GenomeNetwork, complexity
    import neat

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", choices=DATASETS, default="spiral")
    ap.add_argument("--champion", default=None)
    ap.add_argument("--config", default=str(HERE / "neat_config.ini"))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    champion = args.champion or f"runs/{args.dataset}/champion.pkl"
    out = args.out or f"boundary_{args.dataset}.png"

    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         args.config)
    with open(champion, "rb") as fh:
        genome = pickle.load(fh)
    net = GenomeNetwork(genome, config)

    X_train, y_train, X_test, y_test = get_dataset(args.dataset)
    nodes, conns = complexity(genome)
    title = (f"{args.dataset} — held-out {net.accuracy(X_test, y_test):.3f}, "
             f"{nodes} nodes / {conns} conns")
    path = plot_boundary(net.predict, X_test, y_test, title=title, out=out)
    print(f"wrote {path}")

    hist = Path(champion).parent / "fitness_history.csv"
    if hist.exists():
        print(f"wrote {plot_fitness(str(hist), out.replace('boundary', 'fitness'))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

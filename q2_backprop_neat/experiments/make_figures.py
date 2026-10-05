"""Regenerate every Q2 figure in figures/ from committed champions and logs.

    python experiments/make_figures.py

Trains nothing: it only loads champion.pkl / summary.json / fitness_history.csv
from runs/. Champions are always rebuilt with network.champion_net(), so a
Baldwinian champion is drawn with the refined parameters that earned its
fitness, not the unrefined weights its genome still carries.

Every plotted champion's held-out accuracy is recomputed here and compared
with its summary.json; a mismatch is printed and recorded in
figures/champions.csv (column acc_matches_summary).

Outputs (figures/):
    seeds_<dataset>.png          all 10 seeds, A_lamarckian and B_baldwinian
    seeds_plain.png              all 3 plain-NEAT seeds, each dataset
    representative_<dataset>.png median-seed champion of plain NEAT,
                                 A_lamarckian and B_baldwinian, side by side
    wide_view.png                the representative champions on [-3, 3]^2
    fitness_representative.png   per-generation curves of the representative runs
    inheritance_signature.png    mean fit accuracy before / after refinement,
                                 all 10 seeds, A_lamarckian / A_baldwinian /
                                 B_baldwinian
    radius_accuracy.png          spirals: held-out accuracy by radius tercile
    champions.csv                one row per plotted champion (accuracy check,
                                 size, per-tercile accuracy, label changes per ray)

Representative seed ("median"): sort an arm's seeds by (held-out accuracy,
seed) and take position (n - 1) // 2, i.e. the lower median; ties in accuracy
go to the lower seed. For 10 seeds that is the 5th of 10; for plain NEAT's 3
seeds it is the true median.
"""

from __future__ import annotations

import csv
import json
import pickle
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
from datasets import DATASETS, get_dataset  # noqa: E402
from network import champion_net, complexity, load_config  # noqa: E402

RUNS = HERE / "runs"
FIG = HERE / "figures"
CONFIG = HERE / "neat_config.ini"

# Class colours, identical to viz.plot_boundary.
C0, C1 = "#4C72B0", "#DD8452"
PAD = 0.25  # viz.plot_boundary's margin around the data

# Series colours for line / strip charts (validated with the dataviz
# skill's palette checker; colour is never the only cue - legends and
# x-positions carry identity too).
BEST, MEAN = "#2a78d6", "#eb6834"
ARM_COLOUR = {"plain": "#8c8c8c", "A_lamarckian": "#4a3aa7",
              "B_baldwinian": "#1baf7a"}

ARMS = {
    "plain": ("baseline_plain", range(3), "plain NEAT (no refinement)"),
    "A_lamarckian": ("allocation/A_lamarckian", range(10),
                     "A_lamarckian (10 steps x 100%)"),
    "B_baldwinian": ("allocation/B_baldwinian", range(10),
                     "B_baldwinian (100 steps x top 10%)"),
}
SPIRAL_TURNS = {"spiral": 2.6, "spiral3": 3.0}  # datasets.py, units of pi
R_MAX = 1.1                                      # noise-free outer radius

plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 9, "axes.edgecolor": "#888888",
    "axes.linewidth": 0.6, "xtick.color": "#555555", "ytick.color": "#555555",
    "axes.labelcolor": "#333333", "savefig.facecolor": "white",
    "savefig.bbox": "tight", "savefig.pad_inches": 0.12,
})


# ---- loading ---------------------------------------------------------------
class Champion:
    """One committed champion, rebuilt and re-scored on the held-out split."""

    def __init__(self, arm: str, dataset: str, seed: int, config):
        sub, _, _ = ARMS[arm]
        self.arm, self.dataset, self.seed = arm, dataset, seed
        self.dir = RUNS / sub / f"{dataset}_s{seed}"
        with (self.dir / "champion.pkl").open("rb") as fh:
            self.genome = pickle.load(fh)
        self.summary = json.loads((self.dir / "summary.json").read_text())
        self.net = champion_net(self.genome, config)
        self.nodes, self.conns = complexity(self.genome)
        _, _, X_test, y_test = get_dataset(dataset)
        self.acc = self.net.accuracy(X_test, y_test)
        self.summary_acc = float(self.summary["heldout_acc"])
        self.matches = abs(self.acc - self.summary_acc) < 1e-12
        self.genome_type = type(self.genome).__name__
        self.refined = getattr(self.genome, "refined_params", None) is not None

    @property
    def label(self) -> str:
        return f"{self.arm} s{self.seed}"

    def stats_line(self) -> str:
        return (f"held-out {self.acc:.3f} · {self.conns} enabled conns · "
                f"{self.nodes} nodes")


def representative(champs: list[Champion]) -> Champion:
    """Lower median by held-out accuracy; ties to the lower seed."""
    ranked = sorted(champs, key=lambda c: (c.acc, c.seed))
    return ranked[(len(ranked) - 1) // 2]


def read_history(path: Path) -> dict[str, np.ndarray]:
    with path.open() as fh:
        rows = list(csv.DictReader(fh))
    return {k: np.array([float(r[k]) if r[k] != "" else np.nan for r in rows])
            for k in rows[0]}


# ---- boundary drawing (same encoding as viz.plot_boundary) -----------------
def extent_of(X, pad=PAD):
    return (X[:, 0].min() - pad, X[:, 0].max() + pad,
            X[:, 1].min() - pad, X[:, 1].max() + pad)


def draw_boundary(ax, net, X, y, extent=None, resolution=300, s=9,
                  show_box=None):
    """viz.plot_boundary on a given axis: predicted regions, data on top.

    Adds a thin dark line at forward() = 0.5, the threshold predict() uses,
    so the boundary stays visible in small panels.
    """
    x0, x1, y0, y1 = extent or extent_of(X)
    xx, yy = np.meshgrid(np.linspace(x0, x1, resolution),
                         np.linspace(y0, y1, resolution))
    grid = np.c_[xx.ravel(), yy.ravel()]
    p = net.forward(grid).reshape(xx.shape)
    zz = (p > 0.5).astype(float)
    ax.contourf(xx, yy, zz, levels=[-0.5, 0.5, 1.5], alpha=0.25,
                colors=[C0, C1])
    if 0 < zz.mean() < 1:
        ax.contour(xx, yy, p, levels=[0.5], colors="#222222", linewidths=0.6)
    for label, colour in ((0, C0), (1, C1)):
        pts = X[np.asarray(y) == label]
        ax.scatter(pts[:, 0], pts[:, 1], s=s, c=colour, edgecolors="none",
                   label=f"class {label}")
    if show_box is not None:
        bx0, bx1, by0, by1 = show_box
        ax.add_patch(Rectangle((bx0, by0), bx1 - bx0, by1 - by0, fill=False,
                               ls="--", lw=0.7, ec="#444444"))
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_aspect("equal")


def bare(ax):
    ax.set_xticks([])
    ax.set_yticks([])


# ---- quantitative checks ---------------------------------------------------
def knn_predict(X_train, y_train, Q, k=5, chunk=4000):
    out = np.empty(len(Q))
    for i in range(0, len(Q), chunk):
        q = Q[i:i + chunk]
        d = ((q[:, None, :] - X_train[None, :, :]) ** 2).sum(-1)
        nn = np.argpartition(d, k, axis=1)[:, :k]
        out[i:i + chunk] = (y_train[nn].mean(1) > 0.5).astype(float)
    return out


class KNN:
    """5-NN on the training split: a reference for what the data supports
    (AUDIT.md reports k-NN held-out 0.987-1.000 on the spirals)."""

    def __init__(self, dataset):
        X_train, y_train, _, _ = get_dataset(dataset)
        self.X, self.y = X_train, y_train

    def predict(self, Q):
        return knn_predict(self.X, self.y, np.asarray(Q))


def tercile_edges(dataset):
    _, _, X_test, _ = get_dataset(dataset)
    r = np.linalg.norm(X_test, axis=1)
    return np.r_[0.0, np.quantile(r, [1 / 3, 2 / 3]), np.inf]


def tercile_accuracy(predict, dataset):
    """Held-out accuracy in each radius tercile (about 50 points each)."""
    _, _, X_test, y_test = get_dataset(dataset)
    r = np.linalg.norm(X_test, axis=1)
    edges = tercile_edges(dataset)
    pred = np.asarray(predict(X_test))
    accs, ns, floors = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (r >= lo) & (r < hi)
        accs.append(float((pred[m] == y_test[m]).mean()))
        ns.append(int(m.sum()))
        floors.append(float(max(y_test[m].mean(), 1 - y_test[m].mean())))
    return accs, ns, floors


def label_changes_per_ray(predict, n_rays=360, n_r=500):
    """Mean number of predicted-label changes along rays from the origin,
    r in [0, 1.1]. A straight cut gives at most 1 per ray; the generating
    spirals alternate label `turns` times (2.6 for spiral, 3.0 for spiral3)."""
    phi = np.linspace(0, 2 * np.pi, n_rays, endpoint=False)
    r = np.linspace(0, R_MAX, n_r)
    P = np.c_[(r[:, None] * np.cos(phi)).ravel(), (r[:, None] * np.sin(phi)).ravel()]
    L = np.asarray(predict(P)).reshape(n_r, n_rays)
    changes = np.abs(np.diff(L, axis=0)).sum(0)
    return float(changes.mean()), float((changes >= 2).mean())


# ---- figures ---------------------------------------------------------------
def fig_seeds(dataset, champs, reps, floor):
    _, _, X, y = get_dataset(dataset)
    fig, axes = plt.subplots(2, 10, figsize=(17, 5.0))
    for row, arm in enumerate(("A_lamarckian", "B_baldwinian")):
        for col, c in enumerate(sorted(champs[arm], key=lambda c: c.seed)):
            ax = axes[row, col]
            draw_boundary(ax, c.net, X, y, resolution=160, s=2.5)
            bare(ax)
            is_rep = c is reps[arm]
            ax.set_title(f"s{c.seed}  {c.acc:.3f}" + ("  (median)" if is_rep else "")
                         + f"\n{c.conns} conns · {c.nodes} nodes", fontsize=7.5,
                         fontweight="bold" if is_rep else "normal")
            if is_rep:
                for sp in ax.spines.values():
                    sp.set_linewidth(2.0)
                    sp.set_edgecolor("#222222")
        axes[row, 0].set_ylabel(ARMS[arm][2].replace(" (", "\n("), fontsize=8)
    fig.suptitle(f"{dataset}: champions of all 10 seeds, held-out split (150 points; "
                 f"majority-class floor {floor:.3f}). Seed order, not sorted; "
                 f"framed = representative (lower median) seed", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.95), h_pad=2.2)
    out = FIG / f"seeds_{dataset}.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def fig_plain_seeds(champs_by_ds):
    fig, axes = plt.subplots(3, 3, figsize=(10, 10.4))
    for i, ds in enumerate(DATASETS):
        _, _, X, y = get_dataset(ds)
        for j, c in enumerate(sorted(champs_by_ds[ds]["plain"], key=lambda c: c.seed)):
            ax = axes[i, j]
            draw_boundary(ax, c.net, X, y, resolution=200, s=4)
            bare(ax)
            ax.set_title(f"{ds} · plain NEAT s{c.seed}\n{c.stats_line()}", fontsize=8.5)
    fig.suptitle("Plain NEAT (no refinement): all 3 seeds' champions, held-out split",
                 fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out = FIG / "seeds_plain.png"
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return out


def fig_representative(dataset, reps, floor):
    _, _, X, y = get_dataset(dataset)
    fig, axes = plt.subplots(1, 3, figsize=(14, 5.0))
    for ax, arm in zip(axes, ("plain", "A_lamarckian", "B_baldwinian")):
        c = reps[arm]
        n = len(ARMS[arm][1])
        draw_boundary(ax, c.net, X, y)
        ax.set_title(f"{ARMS[arm][2]}\nseed {c.seed} (median of {n})\n{c.stats_line()}",
                     fontsize=9)
    # One legend below the panels, so it never covers a data point.
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=2,
               frameon=False, fontsize=9, bbox_to_anchor=(0.5, -0.08))
    fig.suptitle(f"{dataset}: representative champions on the held-out split "
                 f"(150 points; majority-class floor {floor:.3f})", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = FIG / f"representative_{dataset}.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def fig_wide(reps_by_ds):
    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    for i, ds in enumerate(DATASETS):
        _, _, X, y = get_dataset(ds)
        for j, arm in enumerate(("plain", "A_lamarckian", "B_baldwinian")):
            c = reps_by_ds[ds][arm]
            ax = axes[i, j]
            draw_boundary(ax, c.net, X, y, extent=(-3, 3, -3, 3), resolution=300,
                          s=3, show_box=extent_of(X))
            ax.set_title(f"{ds} · {c.label}\n{c.stats_line()}", fontsize=8.5)
            ax.tick_params(labelsize=7)
    fig.suptitle("Representative champions on [-3, 3]^2; dashed box = the region "
                 "the boundary plots show (data +/- 0.25)", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out = FIG / "wide_view.png"
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return out


def last_improvement(best: np.ndarray) -> int:
    """Last generation at which the population-best fitness rose."""
    up = np.flatnonzero(np.diff(best) > 1e-12)
    return int(up[-1] + 1) if len(up) else 0


def fig_fitness(reps_by_ds):
    facts = []
    fig, axes = plt.subplots(3, 2, figsize=(11, 9.5), sharex=True, sharey=True)
    for i, ds in enumerate(DATASETS):
        for j, arm in enumerate(("A_lamarckian", "B_baldwinian")):
            c = reps_by_ds[ds][arm]
            h = read_history(c.dir / "fitness_history.csv")
            # penalty is 0 in these runs, so fitness = fit-split accuracy
            # after refinement and mean_fitness is exactly mean_acc.
            assert c.summary["args"]["penalty"] == 0.0
            assert np.allclose(h["mean_fitness"], h["mean_acc"])
            g = h["generation"]
            facts.append((ds, c.label, h["best_fitness"][-1] - h["best_pre_acc"][-1],
                          h["mean_acc"][-1] - h["mean_pre_acc"][-1],
                          np.abs(h["mean_acc"] - h["mean_pre_acc"])[-10:].max(),
                          [int(k) for k in np.flatnonzero(h["best_fitness"] < h["best_pre_acc"] - 1e-12)],
                          last_improvement(h["best_fitness"])))
            ax = axes[i, j]
            ax.plot(g, h["best_fitness"], color=BEST, lw=2, label="best, after refinement (= fitness)")
            ax.plot(g, h["best_pre_acc"], color=BEST, lw=1.4, ls="--", label="best, before refinement")
            ax.plot(g, h["mean_fitness"], color=MEAN, lw=2, label="mean, after refinement (= fitness)")
            ax.plot(g, h["mean_pre_acc"], color=MEAN, lw=1.4, ls="--", label="mean, before refinement")
            ax.axvline(c.summary["champion_generation"], color="#999999", lw=0.8, ls=":")
            ax.set_title(f"{ds} · {c.label} ({ARMS[arm][2].split('(')[1][:-1]})\n"
                         f"champion from generation {c.summary['champion_generation']} "
                         f"(dotted line), held-out {c.acc:.3f}", fontsize=8.5)
            ax.grid(axis="y", color="#e6e6e6", lw=0.6)
            ax.set_ylim(0.4, 1.0)
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
            if i == 2:
                ax.set_xlabel("generation")
            if j == 0:
                ax.set_ylabel("fit-split accuracy")
    axes[0, 0].legend(loc="lower right", frameon=False, fontsize=7.5)
    fig.suptitle("Representative runs: population best and mean fit-split accuracy per "
                 "generation, before and after that generation's refinement", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out = FIG / "fitness_representative.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out, facts


def fig_signature():
    arms = [("A_lamarckian", "10 steps x 100%"), ("A_baldwinian", "10 steps x 100%"),
            ("B_baldwinian", "100 steps x top 10%")]
    fig, axes = plt.subplots(3, 3, figsize=(13, 9.5), sharex=True, sharey=True)
    rows = []
    for i, ds in enumerate(DATASETS):
        for j, (arm, alloc) in enumerate(arms):
            H = [read_history(RUNS / "allocation" / arm / f"{ds}_s{s}" / "fitness_history.csv")
                 for s in range(10)]
            g = H[0]["generation"]
            pre = np.array([h["mean_pre_acc"] for h in H])
            post = np.array([h["mean_acc"] for h in H])
            ax = axes[i, j]
            for k in range(10):
                ax.plot(g, post[k], color=MEAN, lw=0.6, alpha=0.3)
                ax.plot(g, pre[k], color=MEAN, lw=0.6, alpha=0.3, ls="--")
            ax.plot(g, np.median(post, 0), color=MEAN, lw=2.2,
                    label="mean after refinement (median of 10 seeds)")
            ax.plot(g, np.median(pre, 0), color=MEAN, lw=2.2, ls="--",
                    label="mean before refinement (median of 10 seeds)")
            ax.set_title(f"{ds} · {arm} ({alloc})", fontsize=9)
            ax.grid(axis="y", color="#e6e6e6", lw=0.6)
            ax.set_ylim(0.4, 0.95)
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
            if i == 2:
                ax.set_xlabel("generation")
            if j == 0:
                ax.set_ylabel("population-mean fit-split accuracy")
            gain = np.array([h["mean_gain_refined"] for h in H])
            best_gap = np.array([h["best_fitness"][-1] - h["best_pre_acc"][-1] for h in H])
            last_up = [last_improvement(h["best_fitness"]) for h in H]
            dip0 = sum(h["best_fitness"][0] < h["best_pre_acc"][0] - 1e-12 for h in H)
            rows.append((ds, arm, np.median(pre[:, -1]), np.median(post[:, -1]),
                         int((gain < 0).sum()), gain.size, np.median(best_gap),
                         np.median(last_up), dip0))
    axes[0, 0].legend(loc="upper left", frameon=False, fontsize=7.5)
    fig.suptitle("Population-mean fit-split accuracy before and after each generation's "
                 "refinement, all 10 seeds (thin lines = one seed, bold = median)",
                 fontsize=9.5)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = FIG / "inheritance_signature.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out, rows


def fig_radius(champs_by_ds, knn_terc):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True)
    arms = ("plain", "A_lamarckian", "B_baldwinian")
    offs = dict(zip(arms, (-0.25, 0.0, 0.25)))
    for ax, ds in zip(axes, SPIRAL_TURNS):
        edges = tercile_edges(ds)
        turns = SPIRAL_TURNS[ds]
        rev = lambda r: r / R_MAX * turns / 2  # noqa: E731  revolutions from the centre
        floors = None
        for arm in arms:
            T = np.array([c.terc for c in champs_by_ds[ds][arm]])
            floors = champs_by_ds[ds][arm][0].terc_floor
            for b in range(3):
                xs = b + offs[arm] + np.linspace(-0.06, 0.06, len(T))
                ax.scatter(xs, T[:, b], s=16, color=ARM_COLOUR[arm], alpha=0.85,
                           edgecolors="none", label=ARMS[arm][2] if b == 0 else None)
                ax.plot([b + offs[arm] - 0.1, b + offs[arm] + 0.1], [np.median(T[:, b])] * 2,
                        color="#222222", lw=1.6)
        for b in range(3):
            ax.plot([b - 0.38, b + 0.38], [floors[b]] * 2, color="#888888", lw=0.9, ls=":")
            ax.plot([b - 0.38, b + 0.38], [knn_terc[ds][b]] * 2, color="#222222", lw=0.9, ls="--")
        ns = champs_by_ds[ds]["plain"][0].terc_n
        labels = []
        for b, name in enumerate(("inner", "middle", "outer")):
            lo, hi = edges[b], edges[b + 1]
            hi_txt = f"{hi:.2f}" if np.isfinite(hi) else "max"
            rv_hi = f"{rev(hi):.2f}" if np.isfinite(hi) else f"{turns / 2:.2f}+"
            labels.append(f"{name} third (n={ns[b]})\nr {lo:.2f}-{hi_txt}\n"
                          f"revolutions {rev(lo):.2f}-{rv_hi}")
        ax.set_xticks(range(3))
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_title(f"{ds} ({turns / 2:.2f} revolutions)", fontsize=9.5)
        ax.set_ylim(0.2, 1.03)
        ax.grid(axis="y", color="#e6e6e6", lw=0.6)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axes[0].set_ylabel("held-out accuracy within the radius third")
    axes[0].plot([], [], color="#222222", lw=1.6, label="median of the arm's seeds")
    axes[0].plot([], [], color="#222222", lw=0.9, ls="--", label="5-NN on the training split")
    axes[0].plot([], [], color="#888888", lw=0.9, ls=":", label="majority class within the third")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=3,
               frameon=False, fontsize=8, bbox_to_anchor=(0.5, -0.1))
    fig.suptitle("Spirals: held-out accuracy by radius r = |x| (equal-count thirds of the "
                 "150 held-out points). One dot per seed: plain NEAT 3 seeds, the others 10",
                 fontsize=9.5)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out = FIG / "radius_accuracy.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


# ---- main ------------------------------------------------------------------
def main() -> int:
    FIG.mkdir(exist_ok=True)
    config = load_config(CONFIG)
    champs_by_ds, reps_by_ds, knn_terc, knn_ray = {}, {}, {}, {}
    mismatches = []
    for ds in DATASETS:
        champs = {arm: [Champion(arm, ds, s, config) for s in seeds]
                  for arm, (_, seeds, _) in ARMS.items()}
        for cs in champs.values():
            for c in cs:
                if not c.matches:
                    mismatches.append(c)
                    print(f"MISMATCH {ds} {c.label}: recomputed {c.acc:.6f}, "
                          f"summary.json {c.summary_acc:.6f}")
                if ds in SPIRAL_TURNS:
                    c.terc, c.terc_n, c.terc_floor = tercile_accuracy(c.net.predict, ds)
                    c.ray_changes, c.ray_frac2 = label_changes_per_ray(c.net.predict)
        if ds in SPIRAL_TURNS:
            knn = KNN(ds)
            knn_terc[ds] = tercile_accuracy(knn.predict, ds)[0]
            knn_ray[ds] = label_changes_per_ray(knn.predict)
        champs_by_ds[ds] = champs
        reps_by_ds[ds] = {arm: representative(cs) for arm, cs in champs.items()}
        floor = champs["A_lamarckian"][0].summary["heldout_majority_class"]
        print(f"wrote {fig_seeds(ds, champs, reps_by_ds[ds], floor)}")
        print(f"wrote {fig_representative(ds, reps_by_ds[ds], floor)}")
    print(f"wrote {fig_plain_seeds(champs_by_ds)}")
    print(f"wrote {fig_wide(reps_by_ds)}")
    out, fit_facts = fig_fitness(reps_by_ds)
    print(f"wrote {out}")
    out, sig_rows = fig_signature()
    print(f"wrote {out}")
    print(f"wrote {fig_radius(champs_by_ds, knn_terc)}")

    # Table behind every claim in figures/README.md.
    cols = ["dataset", "arm", "seed", "representative", "heldout_acc",
            "summary_heldout_acc", "acc_matches_summary", "enabled_conns", "nodes",
            "genome_type", "has_refined_params", "champion_generation",
            "acc_r_inner", "acc_r_middle", "acc_r_outer", "label_changes_per_ray",
            "frac_rays_ge2_changes", "run_dir"]
    with (FIG / "champions.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for ds in DATASETS:
            for arm, cs in champs_by_ds[ds].items():
                for c in cs:
                    sp = ds in SPIRAL_TURNS
                    w.writerow([ds, arm, c.seed, int(c is reps_by_ds[ds][arm]),
                                f"{c.acc:.4f}", f"{c.summary_acc:.4f}", int(c.matches),
                                c.conns, c.nodes, c.genome_type, int(c.refined),
                                c.summary.get("champion_generation", ""),
                                *([f"{a:.3f}" for a in c.terc] if sp else ["", "", ""]),
                                f"{c.ray_changes:.2f}" if sp else "",
                                f"{c.ray_frac2:.2f}" if sp else "",
                                str(c.dir.relative_to(HERE))])
            if ds in SPIRAL_TURNS:
                w.writerow([ds, "5-NN reference", "", 0, "", "", "", "", "", "", "", "",
                            *[f"{a:.3f}" for a in knn_terc[ds]],
                            f"{knn_ray[ds][0]:.2f}", f"{knn_ray[ds][1]:.2f}", ""])
    print(f"wrote {FIG / 'champions.csv'}")

    # Console summary.
    n = sum(len(cs) for d in champs_by_ds.values() for cs in d.values())
    print(f"\naccuracy check: {n - len(mismatches)}/{n} champions match summary.json")
    for ds in DATASETS:
        for arm, c in reps_by_ds[ds].items():
            print(f"representative {ds:8} {arm:13} seed {c.seed}  {c.stats_line()}")
    for ds in SPIRAL_TURNS:
        print(f"\n{ds}: label changes per ray (ideal ~{SPIRAL_TURNS[ds]}, "
              f"5-NN {knn_ray[ds][0]:.2f}); per-third accuracy inner/middle/outer "
              f"(5-NN {', '.join(f'{a:.2f}' for a in knn_terc[ds])})")
        for arm, cs in champs_by_ds[ds].items():
            T = np.array([c.terc for c in cs])
            rc = np.array([c.ray_changes for c in cs])
            print(f"  {arm:13} changes/ray median {np.median(rc):.2f} "
                  f"(min {rc.min():.2f}, max {rc.max():.2f}); per-third median "
                  f"{', '.join(f'{v:.2f}' for v in np.median(T, 0))}")
    print()
    for ds, label, bgap, mgap, mmax, below, last in fit_facts:
        print(f"fitness {ds:8} {label:17} last gen: best after-before {bgap:+.3f}, mean "
              f"after-before {mgap:+.3f}; max |mean gap| over last 10 gens {mmax:.3f}; "
              f"best after < best before in gens {below}; best last rose at gen {last}")
    print()
    for ds, arm, pre, post, neg, cells, gap, last_up, dip0 in sig_rows:
        print(f"signature {ds:8} {arm:13} last generation, median of 10 seeds: mean before "
              f"{pre:.3f}, after {post:.3f}; best after - best before {gap:+.3f}; "
              f"best fitness last rose at gen {last_up:.1f} (median); best after < best "
              f"before at gen 0 in {dip0}/10 seeds; refinement lowered "
              f"the refined genomes' mean fit accuracy in {neg}/{cells} (seed, generation) cells")
    return 1 if mismatches else 0


if __name__ == "__main__":
    raise SystemExit(main())

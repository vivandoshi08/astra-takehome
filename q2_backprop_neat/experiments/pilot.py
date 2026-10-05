"""Pilot for the allocation study: measure, then predict, before any grid runs.

    python experiments/pilot.py [--datasets xor spiral spiral3] [--seed 100]

For each dataset:
  1. Run main.py's Evaluator for 30 generations with the default allocation
     (10 steps x all genomes, Baldwinian, lr 0.3) and keep copies of the
     population at generations 0, 10, 20 and 29. Seed 100: no grid uses it.
  2. Sample 25 genomes from each snapshot (100 per dataset) and refine each
     from its own genome weights to 300 steps (one resumed Adam run),
     recording fit accuracy, fit loss and validation accuracy at
     CHECKPOINTS.
  3. From those curves:
       learning curve   median share of the 300-step gain reached by s steps
       rank reliability Spearman rho between the ranking after s steps and
                        after 300 (ranking key = selection's: fit accuracy,
                        then lower loss)
       top-10 overlap   how many of the 300-step top 10 are top 10 after s
       pre-screen       recall of the 300-step top 10 within the top k by
                        pre-refinement fitness, k = 10, 25, 50, 100
       late bloomers    how many of the 300-step top 6 successive halving
                        (100x2, 50x4, 25x8, 12x16, 6x32; 6 finalists) would
                        eliminate, and whether it keeps the 300-step best
  4. Apply the decision rule (PREDICTION_RULE) and print the predicted
     allocation per dataset.

Uses the fit split for refinement and ranking; validation accuracy is
recorded but not used by the rule; held-out is never loaded. Writes
runs/pilot/<dataset>/curves.csv and runs/pilot/pilot.json. Gradient steps
spent (snapshot run + curves) are counted and reported as pilot cost.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import sys
from pathlib import Path

import neat
import numpy as np

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

import main as driver  # noqa: E402
from budget import StepBudget  # noqa: E402
from datasets import get_dataset  # noqa: E402
from network import GenomeNetwork, load_config  # noqa: E402
from refine import refine  # noqa: E402

LR = 0.3
SNAP_GENS = (0, 10, 20, 29)
PER_SNAP = 25
# 2, 6, 14, 30, 62 are where successive halving ranks / ends, so its cuts
# are replayed exactly.
CHECKPOINTS = (0, 2, 5, 6, 10, 14, 20, 30, 50, 62, 100, 200, 300)
TOP = 10
STEP_GRID = (2, 5, 10, 20, 50, 100, 200)
FRACTIONS = (0.10, 0.25, 0.50, 1.00)
HALVING = driver.halving_schedule(100, 2, 2, 5)
PREDICTION_RULE = (
    "steps s* = smallest s in STEP_GRID with Spearman rho(s, 300) >= 0.9; "
    "fraction f* = smallest f in FRACTIONS whose top-f by pre-refinement "
    "fitness holds >= 9 of the 300-step top 10; generations = 30000 // "
    "(100 f* s*). Successive halving is predicted to lose to the fixed "
    "allocation if it eliminates the 300-step best genome or >= 2 of the "
    "300-step top 6 (late bloomers; halving keeps 6 finalists)."
)


def ranks(keys):
    """Rank positions (0 = best) of a list of sortable keys; ties averaged."""
    order = sorted(range(len(keys)), key=lambda i: keys[i])
    r = np.empty(len(keys))
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and keys[order[j + 1]] == keys[order[i]]:
            j += 1
        r[[order[k] for k in range(i, j + 1)]] = (i + j) / 2
        i = j + 1
    return r


def spearman(a, b):
    a, b = a - a.mean(), b - b.mean()
    return float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum()))


def snapshots(dataset, seed, X_fit, y_fit, X_val, y_val):
    config = load_config(HERE / "neat_config.ini")  # fresh: see DEVLOG #10
    args = driver.build_parser().parse_args(
        ["--steps", "10", "--strategy", "baldwinian", "--lr", str(LR)])
    budget = StepBudget()
    ev = driver.Evaluator(args, config, X_fit, y_fit, X_val, y_val, budget)
    pop = neat.Population(config, seed=seed)
    ev.pop = pop
    snaps = {}

    def fitness(genomes, cfg):
        ev(genomes, cfg)
        if ev.gen - 1 in SNAP_GENS:
            snaps[ev.gen - 1] = [copy.deepcopy(g) for _, g in genomes]

    pop.run(fitness, 30)
    return config, snaps, budget.spent


def curves(config, genomes, X_fit, y_fit, X_val, y_val):
    rows, spent = [], 0
    for gi, (gen, g) in enumerate(genomes):
        net = GenomeNetwork(g, config)  # the genome's own (Baldwinian) weights
        state, done = None, 0
        for cp in CHECKPOINTS:
            if cp > done:
                res = refine(net, X_fit, y_fit, cp - done, LR, state=state)
                state, done = res.state, cp
                spent += res.steps
            rows.append({"genome": gi, "snap_gen": gen, "key": g.key, "steps": cp,
                         "conns": sum(c.enabled for c in g.connections.values()),
                         "fit_acc": net.accuracy(X_fit, y_fit),
                         "fit_loss": net.loss(X_fit, y_fit),
                         "val_acc": net.accuracy(X_val, y_val)})
    return rows, spent


def analyse(rows):
    n = 1 + max(r["genome"] for r in rows)
    at = {s: sorted((r for r in rows if r["steps"] == s), key=lambda r: r["genome"])
          for s in CHECKPOINTS}
    key = {s: [(-r["fit_acc"], r["fit_loss"]) for r in at[s]] for s in CHECKPOINTS}
    rk = {s: ranks(key[s]) for s in CHECKPOINTS}
    top_final = set(np.argsort(rk[300], kind="stable")[:TOP])

    gain_final = np.array([b["fit_acc"] - a["fit_acc"] for a, b in zip(at[0], at[300])])
    out = {"n_genomes": n, "steps": {}}
    for s in CHECKPOINTS[1:]:
        gain = np.array([b["fit_acc"] - a["fit_acc"] for a, b in zip(at[0], at[s])])
        share = np.median(gain[gain_final > 0] / gain_final[gain_final > 0]) if (gain_final > 0).any() else float("nan")
        top_s = set(np.argsort(rk[s], kind="stable")[:TOP])
        out["steps"][s] = {
            "median_fit_acc": float(np.median([r["fit_acc"] for r in at[s]])),
            "median_val_acc": float(np.median([r["val_acc"] for r in at[s]])),
            "median_share_of_final_gain": float(share),
            "spearman_vs_300": spearman(rk[s], rk[300]),
            "top10_overlap": len(top_s & top_final),
        }
    pre = np.argsort(rk[0], kind="stable")
    out["prescreen_recall"] = {f: len(set(pre[:int(round(n * f))]) & top_final)
                               for f in FRACTIONS}
    out["spearman_pre_vs_300"] = spearman(rk[0], rk[300])

    # Successive halving replayed on these curves: round r keeps the best n_r
    # by the ranking after the steps given so far (0, 2, 6, 14, 30), as
    # main.py does. Finalists are the 6 alive in the last round.
    final = len(HALVING) and HALVING[-1][0]
    top_k = set(np.argsort(rk[300], kind="stable")[:final])
    best = int(np.argsort(rk[300], kind="stable")[0])
    survivors_by_round, alive, given = [], list(range(n)), 0
    for n_r, s_r in HALVING:
        alive = sorted(alive, key=lambda i: rk[given][i])[:min(n_r, len(alive))]
        survivors_by_round.append(set(alive))
        given += s_r
    out["halving_late_bloomers"] = len(top_k - survivors_by_round[-1])
    out["halving_keeps_best"] = best in survivors_by_round[-1]
    out["halving_top6_cut_by_round"] = [len(top_k - sv) for sv in survivors_by_round]
    return out


def predict(a):
    s_star = next((s for s in STEP_GRID if a["steps"][s]["spearman_vs_300"] >= 0.9), 300)
    f_star = next((f for f in FRACTIONS if a["prescreen_recall"][f] >= 9), 1.0)
    per_gen = max(1, int(round(100 * f_star))) * s_star
    return {"steps": s_star, "refine_fraction": f_star,
            "generations": 30_000 // per_gen,
            "halving_predicted_to_lose": (a["halving_late_bloomers"] >= 2
                                          or not a["halving_keeps_best"])}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", nargs="+", default=["xor", "spiral", "spiral3"])
    ap.add_argument("--seed", type=int, default=100)
    args = ap.parse_args()

    out_root = HERE / "runs" / "pilot"
    report = {"rule": PREDICTION_RULE, "seed": args.seed, "lr": LR, "datasets": {}}
    for d in args.datasets:
        X_train, y_train = get_dataset(d)[:2]
        X_fit, y_fit, X_val, y_val = driver.split_validation(X_train, y_train, 0.2)
        config, snaps, snap_cost = snapshots(d, args.seed, X_fit, y_fit, X_val, y_val)
        rng = np.random.default_rng(args.seed)
        picked = []
        for gen in SNAP_GENS:
            pool = snaps[gen]
            for i in sorted(rng.choice(len(pool), PER_SNAP, replace=False)):
                picked.append((gen, pool[i]))
        rows, curve_cost = curves(config, picked, X_fit, y_fit, X_val, y_val)

        (out_root / d).mkdir(parents=True, exist_ok=True)
        with (out_root / d / "curves.csv").open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        a = analyse(rows)
        a["prediction"] = predict(a)
        a["cost"] = {"snapshot_run_steps": snap_cost, "curve_steps": curve_cost}
        report["datasets"][d] = a

        print(f"\n== {d}  (pilot cost {snap_cost + curve_cost:,} steps: "
              f"snapshot run {snap_cost:,} + curves {curve_cost:,})")
        print(" steps | median fit acc | median val acc | share of 300-step gain | "
              "rho vs 300 | top-10 overlap")
        for s, v in a["steps"].items():
            print(f" {s:5} | {v['median_fit_acc']:14.3f} | {v['median_val_acc']:14.3f} | "
                  f"{v['median_share_of_final_gain']:22.2f} | {v['spearman_vs_300']:10.3f} | "
                  f"{v['top10_overlap']:>2}/10")
        print(f" pre-refinement ranking: rho vs 300 = {a['spearman_pre_vs_300']:.3f}; "
              "top-10 recall in top k: " +
              ", ".join(f"{int(f * 100)}% -> {r}/10" for f, r in a["prescreen_recall"].items()))
        print(f" successive halving would eliminate {a['halving_late_bloomers']}/6 of the "
              f"300-step top 6 (cumulative per round: {a['halving_top6_cut_by_round']}); "
              f"keeps the 300-step best: {a['halving_keeps_best']}")
        p = a["prediction"]
        print(f" PREDICTION: {p['steps']} steps x {p['refine_fraction']:.0%} refined -> "
              f"{p['generations']} generations; halving predicted to "
              f"{'LOSE to' if p['halving_predicted_to_lose'] else 'match or beat'} it")

    with (out_root / "pilot.json").open("w") as fh:
        json.dump(report, fh, indent=2, default=str)
    print(f"\nrule: {PREDICTION_RULE}\nwrote {out_root}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

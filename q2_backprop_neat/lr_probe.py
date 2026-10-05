"""Pick a default Adam learning rate for refine(), from training data only.

    python lr_probe.py

Samples about 50 genomes per dataset from plain-NEAT runs (fitness = training
accuracy) at mixed generations, refines each one at every learning rate in a
log grid, and reports, after 10, 50 and 200 steps:

    reduction  median relative training-loss reduction, (L0 - L) / L0
    incr       share of genomes whose training loss ended above where it started
    div        share that diverged: loss more than doubled, or not finite
    dacc       median change in training accuracy, percentage points
    const      share that ended at the constant predictor: loss within 1% of
               the entropy of the training class balance (p = mean(y) everywhere)

10, 50 and 200 steps are one resumed refinement (10, then 40 more, then 150
more), which test_gradients.py checks is bit-identical to separate runs.

Only the fit split is used: main.split_validation() of get_dataset(...)[:2],
the same 280 points main.py refines on. Neither the validation split (champion
selection) nor the held-out arrays are touched. Nothing is written to disk.
"""

from __future__ import annotations

import copy
import sys
import time
from pathlib import Path

import neat
import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

from datasets import DATASETS, get_dataset  # noqa: E402
from network import GenomeNetwork, load_config  # noqa: E402
from main import split_validation  # noqa: E402
from refine import refine  # noqa: E402

CONFIG = load_config(HERE / "neat_config.ini")
LRS = (0.003, 0.01, 0.03, 0.1, 0.2, 0.3, 0.5, 1.0, 3.0)
STEPS = (10, 50, 200)
SEEDS = (100, 101)
KEEP = (0, 10, 25, 50, 99)     # generations sampled from
PER_GEN = 5                    # genomes per kept generation and seed -> 50 per dataset
MAX_INCREASE = 0.05            # an lr that raises the loss more often than this is out


def sample_genomes(name):
    X, y = split_validation(*get_dataset(name)[:2], 0.2)[:2]
    found = []
    for seed in SEEDS:
        gen = 0

        def ev(genomes, cfg):
            nonlocal gen
            for _, g in genomes:
                g.fitness = GenomeNetwork(g, cfg).accuracy(X, y)
            if gen in KEEP:
                for _, g in genomes[::max(1, len(genomes) // PER_GEN)][:PER_GEN]:
                    found.append(copy.deepcopy(g))
            gen += 1

        neat.Population(CONFIG, seed=seed).run(ev, max(KEEP) + 1)
    return found


def probe(name, genomes):
    """results[lr][steps] = list of (L0, L, acc0, acc) over genomes."""
    X, y = split_validation(*get_dataset(name)[:2], 0.2)[:2]
    results = {lr: {s: [] for s in STEPS} for lr in LRS}
    for g in genomes:
        for lr in LRS:
            net = GenomeNetwork(g, CONFIG)
            acc0, state, done, L0 = net.accuracy(X, y), None, 0, None
            for s in STEPS:
                r = refine(net, X, y, s - done, lr, state=state)
                state, done = r.state, s
                L0 = r.loss_before if L0 is None else L0
                results[lr][s].append((L0, r.loss_after, acc0, net.accuracy(X, y),
                                       const_loss(y)))
    return results


def const_loss(y):
    """Training loss of the best constant prediction, p = mean(y)."""
    m = float(np.mean(y))
    return -(m * np.log(m) + (1 - m) * np.log(1 - m))


def stats(rows):
    a = np.array(rows)
    L0, L, acc0, acc, Lc = a.T
    finite = np.isfinite(L)
    return {
        "reduction": float(np.median(np.where(finite, (L0 - L) / L0, -np.inf))),
        "incr": float(np.mean(~finite | (L > L0))),
        "div": float(np.mean(~finite | (L > 2 * L0))),
        "dacc": float(np.median(acc - acc0)) * 100,
        "const": float(np.mean(np.abs(L - Lc) <= 0.01 * Lc)),
    }


def table(title, res):
    print(f"\n{title}")
    print("  lr     | " + " | ".join(f"{s:>3} steps: reduction incr  div  dacc const"
                                       for s in STEPS))
    for lr in LRS:
        cells = []
        for s in STEPS:
            t = stats(res[lr][s])
            cells.append(f"          {t['reduction']:+8.1%} {t['incr']:4.0%} "
                         f"{t['div']:4.0%} {t['dacc']:+5.1f} {t['const']:5.0%}")
        print(f"  {lr:<6} | " + " | ".join(cells))


def main() -> int:
    started = time.time()
    per_dataset = {}
    for name in DATASETS:
        genomes = sample_genomes(name)
        n_hidden = sum(any(k not in CONFIG.genome_config.output_keys for k in
                           GenomeNetwork(g, CONFIG).eval_order) for g in genomes)
        print(f"{name}: {len(genomes)} genomes from seeds {SEEDS}, generations {KEEP}; "
              f"{n_hidden} with live hidden nodes")
        per_dataset[name] = probe(name, genomes)

    pooled = {lr: {s: sum((per_dataset[d][lr][s] for d in DATASETS), [])
                   for s in STEPS} for lr in LRS}
    for name in DATASETS:
        table(f"{name}", per_dataset[name])
    table("all datasets pooled", pooled)

    # Recommendation: among lrs that raise the loss on at most MAX_INCREASE of
    # genomes at every step count and on every dataset, the one whose worst
    # (dataset, steps) cell is closest to the best lr for that cell.
    cells = [(d, s) for d in DATASETS for s in STEPS]
    best = {c: max(stats(per_dataset[c[0]][lr][c[1]])["reduction"] for lr in LRS)
            for c in cells}
    print("\nrecommendation (training loss only)")
    scores = {}
    for lr in LRS:
        st = {c: stats(per_dataset[c[0]][lr][c[1]]) for c in cells}
        worst_incr = max(t["incr"] for t in st.values())
        ratio = min(st[c]["reduction"] / best[c] for c in cells)
        safe = worst_incr <= MAX_INCREASE
        if safe:
            scores[lr] = ratio
        print(f"  lr {lr:<6}: worst share with loss up {worst_incr:4.0%}; worst cell "
              f"reaches {ratio:6.1%} of the best lr's reduction"
              f"{'' if safe else '  (excluded: raises the loss too often)'}")
    for s in STEPS:
        top = max(LRS, key=lambda lr: stats(pooled[lr][s])["reduction"])
        print(f"  best pooled median reduction at {s:>3} steps: lr {top}")
    pick = max(scores, key=scores.get)
    print(f"\nrecommended default lr = {pick}: it never raises the training loss on more "
          f"than {MAX_INCREASE:.0%} of genomes in any cell, and in its worst "
          f"(dataset, steps) cell it still gets {scores[pick]:.0%} of the reduction "
          "of the best lr for that cell.")
    print(f"({time.time() - started:.0f} s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

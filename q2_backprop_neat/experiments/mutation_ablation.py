"""Mechanism round 3: how much fit accuracy does each mutation operator erase?

    python experiments/mutation_ablation.py

Refined Lamarckian genomes (populations at generations 10, 20, 29 of a
Lamarckian 10 x 100% run, seed 100) are mutated K times by one operator in
isolation, and each mutant's fit accuracy is compared with its parent's.
Forward passes only; the snapshot run itself spends 30,000 steps per
dataset. See PREDICTIONS_mechanism3.md. Writes runs/mechanism3/.
"""

from __future__ import annotations

import copy
import json
import random
import statistics
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

SEED = 100
SNAP_GENS = (10, 20, 29)
K = 20
ZERO = dict(weight_mutate_rate=0.0, weight_replace_rate=0.0, bias_mutate_rate=0.0,
            bias_replace_rate=0.0, activation_mutate_rate=0.0, node_add_prob=0.0,
            node_delete_prob=0.0, conn_add_prob=0.0, conn_delete_prob=0.0,
            enabled_mutate_rate=0.0)
OPERATORS = {
    "perturb": dict(weight_mutate_rate=0.8, bias_mutate_rate=0.7),
    "replace": dict(weight_replace_rate=0.1, bias_replace_rate=0.1),
    "activation": dict(activation_mutate_rate=0.1),
    "structural": dict(node_add_prob=0.3, node_delete_prob=0.1, conn_add_prob=0.5,
                       conn_delete_prob=0.2, enabled_mutate_rate=0.01),
}


def snapshot(dataset, X_fit, y_fit, X_val, y_val):
    config = load_config(HERE / "neat_config.ini")
    args = driver.build_parser().parse_args(
        ["--steps", "10", "--strategy", "lamarckian", "--lr", "0.3"])
    ev = driver.Evaluator(args, config, X_fit, y_fit, X_val, y_val, StepBudget())
    pop = neat.Population(config, seed=SEED)
    ev.pop = pop
    snaps = []

    def fitness(genomes, cfg):
        ev(genomes, cfg)
        if ev.gen - 1 in SNAP_GENS:
            snaps.extend(copy.deepcopy(g) for _, g in genomes)

    pop.run(fitness, 30)
    return config, snaps


def operator_config(config, rates, base=ZERO):
    gc = copy.deepcopy(config.genome_config)
    for k, v in {**base, **rates}.items():
        setattr(gc, k, v)
    return gc


def main() -> int:
    out_dir = HERE / "runs" / "mechanism3"
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {}
    lowmut = load_config(HERE / "neat_config_lowmut.ini").genome_config
    for d in ("spiral", "spiral3"):
        X_fit, y_fit, X_val, y_val = driver.split_validation(*get_dataset(d)[:2], 0.2)
        config, genomes = snapshot(d, X_fit, y_fit, X_val, y_val)
        ops = {name: operator_config(config, r) for name, r in OPERATORS.items()}
        ops["full_default"] = copy.deepcopy(config.genome_config)
        ops["full_lowmut"] = copy.deepcopy(lowmut)
        random.seed(0)
        changes = {name: [] for name in ops}
        for g in genomes:
            base = GenomeNetwork(g, config).accuracy(X_fit, y_fit)
            for name, gc in ops.items():
                for _ in range(K):
                    m = copy.deepcopy(g)
                    m.mutate(gc)
                    changes[name].append(GenomeNetwork(m, config).accuracy(X_fit, y_fit) - base)
        full = statistics.mean(changes["full_default"])
        rows = {}
        for name, xs in changes.items():
            mean = statistics.mean(xs)
            rows[name] = {"mean": mean, "median": statistics.median(xs),
                          "damaged_frac": float(np.mean(np.array(xs) <= -0.05)),
                          "share": mean / full if full else float("nan")}
        report[d] = {"n_genomes": len(genomes), "mutants_per_genome": K, "ops": rows}
        print(f"\n== {d}: {len(genomes)} refined Lamarckian genomes x {K} mutants per operator")
        print(f"{'operator':13} {'mean change':>12} {'median':>8} {'share of full':>14} {'mutants losing >= 0.05':>23}")
        for name, r in rows.items():
            print(f"{name:13} {r['mean']:+12.3f} {r['median']:+8.3f} {r['share']:14.2f} "
                  f"{r['damaged_frac']:23.0%}")

    lines = []
    for d, rep in report.items():
        o = rep["ops"]
        lines.append(f"C1 {d}: full_default {o['full_default']['mean']:+.3f} vs full_lowmut "
                     f"{o['full_lowmut']['mean']:+.3f} -> "
                     f"{'supported' if abs(o['full_default']['mean'] - o['full_lowmut']['mean']) <= 0.02 else 'contradicted'}")
        lines.append(f"C2 {d}: perturb share {o['perturb']['share']:.2f} (<= 0.25) -> "
                     f"{'supported' if o['perturb']['share'] <= 0.25 else 'contradicted'}")
        lines.append(f"H5 {d}: replace share {o['replace']['share']:.2f} (>= 0.4) -> "
                     f"{'supported' if o['replace']['share'] >= 0.4 else 'not supported'}")
        lines.append(f"H6 {d}: activation share {o['activation']['share']:.2f} (>= 0.4) -> "
                     f"{'supported' if o['activation']['share'] >= 0.4 else 'not supported'}")
        lines.append(f"   {d}: structural share {o['structural']['share']:.2f} (reported)")
    print("\n" + "\n".join(lines))
    report["verdicts"] = lines
    (out_dir / "ablation.json").write_text(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

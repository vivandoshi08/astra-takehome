"""Evolve a classifier for one of the 2-D problems with NEAT.

    python main.py --dataset spiral3
    python main.py --dataset spiral3 --steps 40 --refine-fraction 0.25

As shipped this is plain NEAT: fitness is training accuracy, with no gradient
refinement and no complexity term. Adding those is the question.

You have a fixed budget of gradient steps per run (see budget.py). Generations
are DERIVED from your allocation so every run costs the same total compute:

    --steps            gradient steps per refined genome
    --refine-fraction  what share of the population you refine (1.0 = all)

Writes to runs/<run-id>/:
    fitness_history.csv   generation, best fitness, mean fitness
    champion.pkl          the best genome found
    summary.json          args, git commit, budget.summary(), champion stats

Refuses to overwrite an existing run unless --overwrite is given.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import pickle
import statistics
import subprocess
import sys
import time
from pathlib import Path

import neat
import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

from budget import DEFAULT_BUDGET, BudgetExhausted, StepBudget, plan  # noqa: E402
from datasets import DATASETS, get_dataset  # noqa: E402
from network import (GenomeNetwork, champion_net, complexity,  # noqa: E402
                     load_config)


def git_commit() -> str:
    """HEAD, suffixed -dirty if code in this directory differs from it.

    runs/ is excluded: run outputs are untracked until committed and would
    otherwise mark every run dirty.
    """
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=HERE,
                             capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--", ".",
                                ":(exclude)runs"], cwd=HERE,
                               capture_output=True, text=True).stdout.strip()
        return out + ("-dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def evaluate_genome(genome, config, X, y, budget=None, steps=0) -> float:
    """Fitness for one genome."""
    # Parameters this evaluation refined, if any; champion_net() loads them.
    # Cleared first so an elite that is not refined this generation cannot
    # carry a stale vector from the last one.
    genome.refined_params = None
    net = GenomeNetwork(genome, config)

    # ------------------------------------------------------------------
    # TODO: weight refinement goes here.
    #
    # net.get_params() / net.set_params() / net.forward() are your handles;
    # network.py documents them. Charge the budget for what you spend:
    #
    #     budget.charge(steps)
    #
    # After refining: genome.refined_params = net.get_params(), and project
    # onto [net.lower, net.upper] so write_back() stores what was evaluated.
    # ------------------------------------------------------------------

    return net.accuracy(X, y)


class Recorder(neat.reporting.BaseReporter):
    def __init__(self):
        self.history: list[tuple[int, float, float]] = []
        self.champion = None
        self.champion_fitness = float("-inf")
        self._gen = 0

    def post_evaluate(self, config, population, species, best_genome):
        fits = [g.fitness for g in population.values() if g.fitness is not None]
        if fits:
            self.history.append((self._gen, max(fits), statistics.mean(fits)))
        if best_genome is not None and best_genome.fitness is not None:
            if best_genome.fitness > self.champion_fitness:
                self.champion_fitness = best_genome.fitness
                self.champion = copy.deepcopy(best_genome)
        self._gen += 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", choices=DATASETS, default="spiral")
    ap.add_argument("--steps", type=int, default=10,
                    help="gradient steps per refined genome")
    ap.add_argument("--refine-fraction", type=float, default=1.0,
                    help="share of the population you refine (1.0 = all)")
    ap.add_argument("--budget", type=int, default=DEFAULT_BUDGET)
    ap.add_argument("--generations", type=int, default=None,
                    help="override the derived generation count")
    ap.add_argument("--config", default=str(HERE / "neat_config.ini"))
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--overwrite", action="store_true",
                    help="allow replacing an existing runs/<run-id>/")
    args = ap.parse_args()

    if args.steps == 0 and args.generations is None:
        ap.error("--steps 0 (no refinement) needs --generations: "
                 "plan() divides by steps")

    run_dir = HERE / "runs" / (args.run_id or args.dataset)
    if (run_dir / "fitness_history.csv").exists() and not args.overwrite:
        ap.error(f"{run_dir} already holds a run; pick another --run-id "
                 "or pass --overwrite")
    run_dir.mkdir(parents=True, exist_ok=True)

    X_train, y_train, X_test, y_test = get_dataset(args.dataset)

    config = load_config(args.config)
    pop_size = config.pop_size
    planned = (plan(pop_size, args.steps, args.refine_fraction, args.budget)
               if args.steps > 0 else None)
    generations = args.generations or planned
    budget = StepBudget(args.budget)

    print(f"dataset {args.dataset} | pop {pop_size} | {args.steps} steps/genome "
          f"| refine {args.refine_fraction:.0%} | -> {generations} generations "
          f"within {args.budget:,} steps")
    if planned is not None and generations > planned:
        print(f"WARNING: --generations {generations} > the {planned} this "
              "allocation affords; the run will stop when the budget is spent")

    pop = neat.Population(config, seed=args.seed)
    if not args.quiet:
        pop.add_reporter(neat.StdOutReporter(False))
    recorder = Recorder()
    pop.add_reporter(recorder)

    def evaluate(genomes, cfg):
        for _, genome in genomes:
            genome.fitness = evaluate_genome(
                genome, cfg, X_train, y_train, budget=budget, steps=args.steps)

    started = time.time()
    commit = git_commit()
    exhausted = None
    try:
        pop.run(evaluate, generations)
    except BudgetExhausted as e:
        # Keep the partial logs, but the run does not count.
        exhausted = str(e)
        print(f"\nBUDGET EXCEEDED — this run is invalid: {e}")

    with (run_dir / "fitness_history.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["generation", "best_fitness", "mean_fitness"])
        w.writerows(recorder.history)

    summary = {
        "args": vars(args),
        "git_commit": commit,
        "seconds": round(time.time() - started, 2),
        "generations_planned": generations,
        "generations_run": len(recorder.history),
        "budget": budget.summary(),
        "budget_spent": budget.spent,
        "budget_limit": budget.limit,
        "budget_refinements": budget.refinements,
        "within_budget": exhausted is None and budget.spent <= budget.limit,
        "budget_error": exhausted,
        "heldout_majority_class": float(max(y_test.mean(), 1 - y_test.mean())),
    }

    champ = recorder.champion
    if champ is not None:
        with (run_dir / "champion.pkl").open("wb") as fh:
            pickle.dump(champ, fh)
        net = champion_net(champ, config)
        nodes, conns = complexity(champ)
        summary.update({
            "champion_fitness": recorder.champion_fitness,
            "champion_refined": getattr(champ, "refined_params", None) is not None,
            "train_acc": net.accuracy(X_train, y_train),
            "heldout_acc": net.accuracy(X_test, y_test),
            "nodes": nodes,
            "conns": conns,
            "max_abs_param": float(np.abs(net.get_params()).max()),
        })
        print(f"\ndataset          : {args.dataset}")
        print(f"generations      : {len(recorder.history)}")
        print(f"budget           : {budget.summary()}")
        print(f"train accuracy   : {summary['train_acc']:.3f}")
        print(f"held-out accuracy: {summary['heldout_acc']:.3f} "
              f"(majority class {summary['heldout_majority_class']:.3f})")
        print(f"champion size    : {nodes} nodes, {conns} enabled connections")

    with (run_dir / "summary.json").open("w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"wrote {run_dir}/")
    return 1 if exhausted else 0


if __name__ == "__main__":
    raise SystemExit(main())

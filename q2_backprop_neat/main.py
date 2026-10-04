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
"""

from __future__ import annotations

import argparse
import copy
import csv
import pickle
import statistics
import sys
from pathlib import Path

import neat

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

from budget import DEFAULT_BUDGET, StepBudget, plan  # noqa: E402
from datasets import DATASETS, get_dataset  # noqa: E402
from network import GenomeNetwork, complexity  # noqa: E402


def evaluate_genome(genome, config, X, y, budget=None, steps=0) -> float:
    """Fitness for one genome."""
    net = GenomeNetwork(genome, config)

    # ------------------------------------------------------------------
    # TODO: weight refinement goes here.
    #
    # net.get_params() / net.set_params() / net.forward() are your handles;
    # network.py documents them. Charge the budget for what you spend:
    #
    #     budget.charge(steps)
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
    args = ap.parse_args()

    run_dir = HERE / "runs" / (args.run_id or args.dataset)
    run_dir.mkdir(parents=True, exist_ok=True)

    X_train, y_train, X_test, y_test = get_dataset(args.dataset)

    config = neat.Config(
        neat.DefaultGenome, neat.DefaultReproduction,
        neat.DefaultSpeciesSet, neat.DefaultStagnation, args.config,
    )
    pop_size = config.pop_size
    generations = args.generations or plan(
        pop_size, args.steps, args.refine_fraction, args.budget)
    budget = StepBudget(args.budget)

    print(f"dataset {args.dataset} | pop {pop_size} | {args.steps} steps/genome "
          f"| refine {args.refine_fraction:.0%} | -> {generations} generations "
          f"within {args.budget:,} steps")

    pop = neat.Population(config, seed=args.seed)
    if not args.quiet:
        pop.add_reporter(neat.StdOutReporter(False))
    recorder = Recorder()
    pop.add_reporter(recorder)

    def evaluate(genomes, cfg):
        for _, genome in genomes:
            genome.fitness = evaluate_genome(
                genome, cfg, X_train, y_train, budget=budget, steps=args.steps)

    pop.run(evaluate, generations)

    with (run_dir / "fitness_history.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["generation", "best_fitness", "mean_fitness"])
        w.writerows(recorder.history)

    champ = recorder.champion
    if champ is not None:
        with (run_dir / "champion.pkl").open("wb") as fh:
            pickle.dump(champ, fh)
        net = GenomeNetwork(champ, config)
        nodes, conns = complexity(champ)
        print(f"\ndataset          : {args.dataset}")
        print(f"generations      : {len(recorder.history)}")
        print(f"budget           : {budget.summary()}")
        print(f"train accuracy   : {net.accuracy(X_train, y_train):.3f}")
        print(f"held-out accuracy: {net.accuracy(X_test, y_test):.3f}")
        print(f"champion size    : {nodes} nodes, {conns} enabled connections")
        print(f"wrote {run_dir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

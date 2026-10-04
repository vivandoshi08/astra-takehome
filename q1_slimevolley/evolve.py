"""Evolve a slime-volleyball agent with NEAT.

Usage:
    python evolve.py --generations 50
    python evolve.py --budget 2000000 --generations 20 --workers 4

Writes to runs/<run-id>/:
    fitness_history.csv   generation, best fitness, mean fitness
    champion.pkl          the best genome found
    budget_log.jsonl      step accounting (see budget.py)
"""

from __future__ import annotations

import argparse
import copy
import csv
import multiprocessing as mp
import os
import pickle
import statistics
import sys
from pathlib import Path

import neat

sys.path.insert(0, str(Path(__file__).parent))

import slimevolley  # noqa: E402
from budget import BudgetExhausted, BudgetedEnv, StepBudget, total_steps  # noqa: E402
from episode import play_episode  # noqa: E402

HERE = Path(__file__).parent
_W: dict = {}


def _init_worker(run_dir: str, config_path: str, per_process_cap: int) -> None:
    budget = StepBudget(Path(run_dir) / "budget_log.jsonl", per_process_cap)
    _W["budget"] = budget
    _W["env"] = BudgetedEnv(slimevolley.SlimeVolleyEnv(), budget)
    _W["config"] = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path,
    )


def make_net(genome, config):
    """Build the network class that matches the config's `feed_forward` setting.

    Reusable — call this from your own evaluation and replay code so it stays in
    step with whatever you put in neat_config.ini.
    """
    if config.genome_config.feed_forward:
        return neat.nn.FeedForwardNetwork.create(genome, config)
    return neat.nn.RecurrentNetwork.create(genome, config)


def _eval_one(genome_bytes: bytes) -> float:
    """Fitness for one genome: the score from a single episode."""
    genome = pickle.loads(genome_bytes)
    net = make_net(genome, _W["config"])
    if hasattr(net, "reset"):
        net.reset()  # recurrent nets carry state; clear it before each episode
    try:
        fitness = play_episode(net, _W["env"])
    except BudgetExhausted:
        _W["budget"].flush()
        raise
    _W["budget"].flush()
    return fitness


class BudgetReached(Exception):
    """Signals the driver to stop evolving."""


class RunRecorder(neat.reporting.BaseReporter):
    """Records per-generation fitness and stops the run when the budget is spent."""

    def __init__(self, budget_log: Path, limit: int):
        self.budget_log = budget_log
        self.limit = limit
        self.history: list[tuple[int, float, float]] = []
        self.champion = None
        self.champion_fitness = float("-inf")
        self._gen = 0

    def post_evaluate(self, config, population, species, best_genome):
        fitnesses = [g.fitness for g in population.values() if g.fitness is not None]
        if fitnesses:
            self.history.append(
                (self._gen, max(fitnesses), statistics.mean(fitnesses))
            )
        if best_genome is not None and best_genome.fitness is not None:
            if best_genome.fitness > self.champion_fitness:
                self.champion_fitness = best_genome.fitness
                # Snapshot: genomes survive into later generations and are
                # re-evaluated in place, which would overwrite .fitness here.
                self.champion = copy.deepcopy(best_genome)
        self._gen += 1

        spent = total_steps(self.budget_log)
        print(f"  [budget] {spent:,} / {self.limit:,} steps", flush=True)
        if spent >= self.limit:
            print("  [budget] limit reached — stopping", flush=True)
            raise BudgetReached


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--generations", type=int, default=50)
    ap.add_argument("--budget", type=int, default=100_000_000,
                    help="total environment steps across all workers")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--config", default=str(HERE / "neat_config.ini"))
    ap.add_argument("--run-id", default="run")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    run_dir = HERE / "runs" / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    budget_log = run_dir / "budget_log.jsonl"

    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        args.config,
    )
    pop = neat.Population(config, seed=args.seed)
    pop.add_reporter(neat.StdOutReporter(False))
    recorder = RunRecorder(budget_log, args.budget)
    pop.add_reporter(recorder)

    per_process_cap = max(1, args.budget // max(1, args.workers))
    stopped_early = False

    ctx = mp.get_context("spawn")
    pool = ctx.Pool(
        processes=args.workers,
        initializer=_init_worker,
        initargs=(str(run_dir), args.config, per_process_cap),
    )

    def evaluate(genomes, cfg):
        payload = [pickle.dumps(g) for _, g in genomes]
        results = pool.map(_eval_one, payload)
        for (_, genome), fitness in zip(genomes, results):
            genome.fitness = fitness

    try:
        pop.run(evaluate, args.generations)
    except BudgetReached:
        stopped_early = True
    except BudgetExhausted as exc:
        print(f"  [budget] {exc} — stopping", flush=True)
        stopped_early = True
    except KeyboardInterrupt:
        print("  interrupted — writing what we have", flush=True)
        stopped_early = True
    finally:
        pool.close()
        pool.join()

    with (run_dir / "fitness_history.csv").open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["generation", "best_fitness", "mean_fitness"])
        writer.writerows(recorder.history)

    if recorder.champion is not None:
        with (run_dir / "champion.pkl").open("wb") as fh:
            pickle.dump(recorder.champion, fh)

    print(f"\nwrote {run_dir}/")
    print(f"  generations completed: {len(recorder.history)}")
    print(f"  steps used:            {total_steps(budget_log):,}")
    if recorder.champion is not None:
        print(f"  champion training fitness: {recorder.champion_fitness:.2f}")
    if stopped_early:
        print("  (stopped before the requested generation count)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

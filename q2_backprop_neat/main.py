"""Evolve a classifier for one of the 2-D problems with backprop-NEAT.

    python main.py --dataset spiral --steps 10 --strategy lamarckian --lr 0.1
    python main.py --dataset spiral --allocation halving --strategy baldwinian --lr 0.1
    python main.py --dataset spiral --steps 0 --generations 30     # plain NEAT

Every generation, each genome is scored before refinement (forward passes are
free), the chosen genomes are refined by full-batch Adam on the fit split, and
fitness is fit-split accuracy after refinement minus the complexity penalty.
One gradient step = one full-batch gradient + one Adam update; the budget is
charged for the steps each refinement actually takes.

How the 30,000-step budget is allocated (--allocation):
    fixed    --steps per refined genome; the top --refine-fraction of the
             population by pre-refinement fitness is refined; generations
             come from budget.plan()
    halving  successive halving every generation: every genome gets
             --halving-base steps, the best 1/eta of them eta x more, and so
             on for --halving-rounds rounds; generations = budget // cost
             per generation

What refinement leaves behind (--strategy):
    lamarckian  refined parameters are written back into the genome, so
                offspring inherit them
    baldwinian  the genome keeps its own weights; the refined ones only set
                its fitness (and are stored as genome.refined_params so the
                champion is evaluated with them)
Elites are refined again every generation they are evaluated, under both.

Small networks (--max-conns N): every genome NEAT creates (initial, crossover,
mutation) is repaired at creation to at most N enabled connections, the count
complexity() reports (network.MaxConnsGenome: dead connections are disabled
first, then live ones by smallest |weight|; no random numbers drawn). So every
genome evaluated, refined, or chosen as champion is within the limit, and
--steps / --refine-fraction / plan() mean exactly what they mean without the
flag: no gradient step is spent on a genome that could not be a champion,
because none exists. The run fails (exit 1) rather than report a champion or
train-best genome over N. summary.json records the repairs.

Data: the training split is divided once, by a fixed seed shared by every run,
into a fit split (refinement and fitness) and a validation split (choosing the
champion, nothing else). The held-out split is only ever reported.

Writes to runs/<run-id>/:
    fitness_history.csv   per generation; the first three columns are the
                          scaffold's, then HISTORY_COLUMNS
    champion.pkl          champion genome (.refined_params if it was refined)
    summary.json          args, git commit, budget.summary(), champion stats
    genomes.jsonl         one row per genome per generation (--log-genomes)
With --max-conns, fitness_history.csv gains MAXCONNS_COLUMNS at the end and
genomes.jsonl rows gain repair_dead / repair_live; without it both files are
exactly as before.

Refuses to overwrite an existing run unless --overwrite is given.
Exit codes: 0 ok, 2 usage error, 4 over budget (logs written, run invalid);
anything else (1 included) is a crash.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import pickle
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import neat
import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

from budget import DEFAULT_BUDGET, BudgetExhausted, StepBudget, plan  # noqa: E402
from datasets import DATASETS, get_dataset  # noqa: E402
from network import (GenomeNetwork, champion_net, complexity,  # noqa: E402
                     load_config)

VAL_SEED = 20261004  # fixed: every run and seed shares one validation split
BUDGET_EXIT = 4      # exit code of an over-budget run (1 is Python's uncaught exception)

HISTORY_COLUMNS = [
    "generation", "best_fitness", "mean_fitness",          # scaffold
    "best_pre_acc", "mean_pre_acc", "mean_acc",             # fit split, before / after refinement
    "mean_gain_refined",                                    # mean (after - before) over refined genomes
    "n_refined", "gen_steps", "cum_steps",
    "n_species", "best_conns", "mean_conns",
    "best_val_acc", "champion_val_acc",                     # validation: this gen's best / champion so far
    "n_genome_changed", "n_clamped", "max_abs_param",       # inheritance evidence
]
# Appended only with --max-conns: genomes first seen this generation, and how
# many of them repair_to_max_conns had to cut down.
MAXCONNS_COLUMNS = ["n_new_genomes", "n_repaired"]


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


def split_validation(X, y, frac: float):
    """(X_fit, y_fit, X_val, y_val): stratified, the same for every run."""
    if frac <= 0:
        return X, y, X[:0], y[:0]
    rng = np.random.default_rng(VAL_SEED)
    val = np.zeros(len(y), dtype=bool)
    for label in (0.0, 1.0):
        idx = rng.permutation(np.flatnonzero(y == label))
        val[idx[:int(round(len(idx) * frac))]] = True
    return X[~val], y[~val], X[val], y[val]


def penalised(acc: float, conns: int, lam: float, form: str) -> float:
    """Fitness from accuracy and enabled-connection count.

    linear  acc - lam * conns
    sqrt    acc - lam * sqrt(conns)
    mult    1 - (1 - acc) * (1 + lam * sqrt(conns)): the error grows by a
            factor rising with sqrt(conns), one reading of the backprop-NEAT
            write-up's description
    """
    if lam == 0:
        return acc
    if form == "linear":
        return acc - lam * conns
    if form == "sqrt":
        return acc - lam * math.sqrt(conns)
    if form == "mult":
        return 1 - (1 - acc) * (1 + lam * math.sqrt(conns))
    raise ValueError(f"unknown penalty form {form!r}")


def halving_schedule(pop_size: int, base: int, eta: int, rounds: int):
    """[(genomes refined, steps each), ...] per round of one generation."""
    return [(max(1, int(round(pop_size / eta ** r))), base * eta ** r)
            for r in range(rounds)]


def genome_hash(genome) -> str:
    """Fingerprint of a genome's weights and biases (not its topology)."""
    vals = ([(k, c.weight) for k, c in sorted(genome.connections.items())]
            + [(k, n.bias) for k, n in sorted(genome.nodes.items())])
    return hashlib.sha1(repr(vals).encode()).hexdigest()[:12]


@dataclass
class Rec:
    """One genome's evaluation within a generation."""
    key: int
    genome: object
    net: GenomeNetwork
    nodes: int
    conns: int
    pre_acc: float
    acc: float
    hash_before: str
    pre_loss: float = float("nan")
    loss: float = float("nan")
    steps: int = 0
    state: object = None
    fitness: float = float("nan")
    val_acc: float = float("nan")
    hash_after: str = ""
    clamped: int = 0
    extra: dict = field(default_factory=dict)


class Evaluator:
    """NEAT fitness function: refine, score, pick the champion, log."""

    def __init__(self, args, config, X_fit, y_fit, X_val, y_val, budget,
                 genome_log=None):
        self.args = args
        self.pop_size = config.pop_size
        self.X, self.y = X_fit, y_fit
        self.X_val, self.y_val = X_val, y_val
        self.budget = budget
        self.refining = args.allocation == "halving" or args.steps > 0
        if self.refining:
            from refine import refine  # only needed when refining
            self._refine_fn = refine
        self.schedule = (halving_schedule(self.pop_size, args.halving_base,
                                          args.halving_eta, args.halving_rounds)
                         if args.allocation == "halving" else None)
        self.genome_log = genome_log
        self.pop = None  # set after the Population exists (species, ancestry)

        self.gen = 0
        self.history: list[dict] = []
        self.champion = None          # chosen on validation accuracy
        self.champion_key = (float("-inf"), float("-inf"))
        self.champion_gen = None
        self.trainbest = None         # chosen on fitness, for comparison
        self.trainbest_fitness = float("-inf")
        self.refine_calls = 0
        self.genomes_refined = 0

        # --max-conns bookkeeping (unused without it)
        self.max_conns = getattr(args, "max_conns", None)
        self._seen_keys: set = set()
        self.genomes_created = 0      # distinct genomes evaluated
        self.genomes_repaired = 0     # ... that repair had to cut down at creation
        self.repair_dead = 0          # connections repair disabled: dead ones
        self.repair_live = 0          # ... live ones
        self._gen_new = self._gen_repaired = 0

    # ---- scoring -----------------------------------------------------------
    def fitness_of(self, r: Rec) -> float:
        return penalised(r.acc, r.conns, self.args.penalty, self.args.penalty_form)

    def _rank(self, recs):
        """Best first: fitness, then lower loss, then lower key (determinism)."""
        return sorted(recs, key=lambda r: (-self.fitness_of(r),
                                           r.loss if self.refining else 0.0,
                                           r.key))

    def _step(self, r: Rec, steps: int) -> None:
        res = self._refine_fn(r.net, self.X, self.y, steps, self.args.lr,
                              tol=self.args.tol, state=r.state)
        r.state = res.state
        r.steps += res.steps
        r.loss = res.loss_after
        r.acc = r.net.accuracy(self.X, self.y)
        self.refine_calls += 1
        self.budget.charge(res.steps)  # raises BudgetExhausted past the limit

    def _refine(self, recs) -> None:
        if self.schedule is None:
            k = min(len(recs), max(1, int(round(self.pop_size * self.args.refine_fraction))))
            for r in self._rank(recs)[:k]:
                self._step(r, self.args.steps)
            return
        alive = recs
        for n, steps in self.schedule:
            alive = self._rank(alive)[:min(n, len(alive))]
            for r in alive:
                self._step(r, steps)

    # ---- the fitness function NEAT calls ------------------------------------
    def _check_size(self, key, g, conns) -> None:
        """--max-conns: no genome over the limit is ever scored; count repairs."""
        if conns > self.max_conns:
            raise RuntimeError(f"genome {key} has {conns} enabled connections, over "
                               f"--max-conns {self.max_conns}: repair failed")
        if key in self._seen_keys:  # an elite carried over: already counted
            return
        self._seen_keys.add(key)
        dead, live = getattr(g, "repair_dead", 0), getattr(g, "repair_live", 0)
        self.genomes_created += 1
        self._gen_new += 1
        if dead or live:
            self.genomes_repaired += 1
            self._gen_repaired += 1
        self.repair_dead += dead
        self.repair_live += live

    def __call__(self, genomes, config):
        recs = []
        self._gen_new = self._gen_repaired = 0
        for key, g in genomes:
            # Cleared first: an elite not refined this generation must not
            # carry last generation's vector.
            g.refined_params = None
            net = GenomeNetwork(g, config)
            nodes, conns = complexity(g)
            if self.max_conns is not None:
                self._check_size(key, g, conns)
            acc = net.accuracy(self.X, self.y)
            r = Rec(key, g, net, nodes, conns, pre_acc=acc, acc=acc,
                    hash_before=genome_hash(g))
            if self.refining:
                r.pre_loss = r.loss = net.loss(self.X, self.y)
            recs.append(r)

        if self.refining:
            self._refine(recs)

        for r in recs:
            r.fitness = self.fitness_of(r)
            r.genome.fitness = r.fitness
            if r.steps > 0:
                self.genomes_refined += 1
                r.genome.refined_params = r.net.get_params()
                if self.args.strategy == "lamarckian":
                    r.clamped = r.net.write_back(r.genome)
            r.hash_after = genome_hash(r.genome)

        best = self._rank(recs)[0]
        self._select(best)
        self._record(recs, best)
        self.gen += 1

    def _select(self, best: Rec) -> None:
        if best.fitness > self.trainbest_fitness:
            self.trainbest_fitness = best.fitness
            self.trainbest = copy.deepcopy(best.genome)
        if len(self.X_val):
            best.val_acc = best.net.accuracy(self.X_val, self.y_val)
            key = (best.val_acc, best.fitness)
        else:
            key = (best.fitness, best.fitness)
        if key > self.champion_key:  # strict: ties keep the earlier champion
            self.champion_key = key
            self.champion = copy.deepcopy(best.genome)
            self.champion_gen = self.gen

    def _record(self, recs, best: Rec) -> None:
        refined = [r for r in recs if r.steps > 0]
        fits = [r.fitness for r in recs]
        nets = [r.net for r in recs]
        row = {
            "generation": self.gen,
            "best_fitness": max(fits),
            "mean_fitness": statistics.mean(fits),
            "best_pre_acc": max(r.pre_acc for r in recs),
            "mean_pre_acc": statistics.mean(r.pre_acc for r in recs),
            "mean_acc": statistics.mean(r.acc for r in recs),
            "mean_gain_refined": (statistics.mean(r.acc - r.pre_acc for r in refined)
                                  if refined else ""),
            "n_refined": len(refined),
            "gen_steps": sum(r.steps for r in recs),
            "cum_steps": self.budget.spent,
            "n_species": len(self.pop.species.species) if self.pop else "",
            "best_conns": best.conns,
            "mean_conns": statistics.mean(r.conns for r in recs),
            "best_val_acc": best.val_acc,
            "champion_val_acc": self.champion_key[0] if len(self.X_val) else "",
            "n_genome_changed": sum(r.hash_before != r.hash_after for r in recs),
            "n_clamped": sum(r.clamped for r in recs),
            "max_abs_param": max(float(np.abs(n.get_params()).max()) for n in nets),
        }
        if self.max_conns is not None:
            row["n_new_genomes"] = self._gen_new
            row["n_repaired"] = self._gen_repaired
        self.history.append(row)

        if self.genome_log is not None:
            ancestors = self.pop.reproduction.ancestors if self.pop else {}
            g2s = self.pop.species.genome_to_species if self.pop else {}
            for r in recs:
                val = (r.net.accuracy(self.X_val, self.y_val)
                       if len(self.X_val) else None)
                entry = {
                    "gen": self.gen, "key": r.key,
                    "parents": list(ancestors.get(r.key, ())),
                    "species": g2s.get(r.key), "nodes": r.nodes, "conns": r.conns,
                    "steps": r.steps, "pre_acc": r.pre_acc, "acc": r.acc,
                    "pre_loss": r.pre_loss, "loss": r.loss, "fitness": r.fitness,
                    "val_acc": val, "hash_before": r.hash_before,
                    "hash_after": r.hash_after, "clamped": r.clamped,
                }
                if self.max_conns is not None:
                    entry["repair_dead"] = getattr(r.genome, "repair_dead", 0)
                    entry["repair_live"] = getattr(r.genome, "repair_live", 0)
                self.genome_log.write(json.dumps(entry) + "\n")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", choices=DATASETS, default="spiral")
    ap.add_argument("--steps", type=int, default=10,
                    help="gradient steps per refined genome (fixed allocation)")
    ap.add_argument("--refine-fraction", type=float, default=1.0,
                    help="share of the population you refine (1.0 = all)")
    ap.add_argument("--allocation", choices=("fixed", "halving"), default="fixed")
    ap.add_argument("--halving-base", type=int, default=2)
    ap.add_argument("--halving-eta", type=int, default=2)
    ap.add_argument("--halving-rounds", type=int, default=5)
    ap.add_argument("--strategy", choices=("lamarckian", "baldwinian"), default=None,
                    help="required when refining")
    ap.add_argument("--lr", type=float, default=None, help="Adam step size; required when refining")
    ap.add_argument("--tol", type=float, default=None,
                    help="stop refining a genome once its fit loss is below this")
    ap.add_argument("--val-frac", type=float, default=0.2,
                    help="share of the training split held back for champion selection")
    ap.add_argument("--penalty", type=float, default=0.0, help="complexity penalty strength")
    ap.add_argument("--penalty-form", choices=("linear", "sqrt", "mult"), default="linear")
    ap.add_argument("--max-conns", type=int, default=None,
                    help="small-network track: every genome is repaired at creation to at "
                         "most this many enabled connections (default: no limit)")
    ap.add_argument("--budget", type=int, default=DEFAULT_BUDGET)
    ap.add_argument("--generations", type=int, default=None,
                    help="override the derived generation count")
    ap.add_argument("--config", default=str(HERE / "neat_config.ini"))
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log-genomes", action="store_true",
                    help="write genomes.jsonl (one row per genome per generation)")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--overwrite", action="store_true",
                    help="allow replacing an existing runs/<run-id>/")
    return ap


def main() -> int:
    ap = build_parser()
    args = ap.parse_args()

    refining = args.allocation == "halving" or args.steps > 0
    if args.allocation == "fixed" and args.steps == 0 and args.generations is None:
        ap.error("--steps 0 (no refinement) needs --generations: "
                 "plan() divides by steps")
    if refining and (args.strategy is None or args.lr is None):
        ap.error("refining needs --strategy and --lr")
    if not 0 < args.refine_fraction <= 1:
        ap.error("--refine-fraction must be in (0, 1]")
    if not 0 <= args.val_frac < 0.5:
        ap.error("--val-frac must be in [0, 0.5)")
    if args.generations is not None and args.generations < 1:
        ap.error("--generations must be >= 1")
    if args.max_conns is not None and args.max_conns < 1:
        ap.error("--max-conns must be >= 1")

    run_dir = HERE / "runs" / (args.run_id or args.dataset)
    if (run_dir / "fitness_history.csv").exists() and not args.overwrite:
        ap.error(f"{run_dir} already holds a run; pick another --run-id "
                 "or pass --overwrite")
    run_dir.mkdir(parents=True, exist_ok=True)

    X_train, y_train, X_test, y_test = get_dataset(args.dataset)
    X_fit, y_fit, X_val, y_val = split_validation(X_train, y_train, args.val_frac)

    config = (load_config(args.config) if args.max_conns is None
              else load_config(args.config, max_conns=args.max_conns))
    pop_size = config.pop_size
    if args.allocation == "halving":
        per_gen = sum(n * s for n, s in halving_schedule(
            pop_size, args.halving_base, args.halving_eta, args.halving_rounds))
        planned = max(1, args.budget // per_gen)
    else:
        per_gen = max(1, int(round(pop_size * args.refine_fraction))) * args.steps
        planned = (plan(pop_size, args.steps, args.refine_fraction, args.budget)
                   if args.steps > 0 else None)
    generations = args.generations if args.generations is not None else planned
    budget = StepBudget(args.budget)

    alloc = (f"halving base {args.halving_base} eta {args.halving_eta} "
             f"rounds {args.halving_rounds}" if args.allocation == "halving" else
             f"{args.steps} steps/genome | refine {args.refine_fraction:.0%}")
    print(f"dataset {args.dataset} | pop {pop_size} | {alloc} | "
          f"{per_gen} steps/gen -> {generations} generations within "
          f"{args.budget:,} steps | strategy {args.strategy} | "
          f"fit {len(X_fit)} / val {len(X_val)}"
          + (f" | max {args.max_conns} enabled conns" if args.max_conns is not None else ""))
    if planned is not None and generations > planned:
        print(f"WARNING: --generations {generations} > the {planned} this "
              "allocation affords; the run will stop when the budget is spent")

    genome_log = (run_dir / "genomes.jsonl").open("w") if args.log_genomes else None
    evaluator = Evaluator(args, config, X_fit, y_fit, X_val, y_val, budget, genome_log)
    pop = neat.Population(config, seed=args.seed)
    evaluator.pop = pop
    if not args.quiet:
        pop.add_reporter(neat.StdOutReporter(False))

    started = time.time()
    commit = git_commit()
    exhausted = None
    try:
        pop.run(evaluator, generations)
    except BudgetExhausted as e:
        # Keep the partial logs, but the run does not count.
        exhausted = str(e)
        print(f"\nBUDGET EXCEEDED — this run is invalid: {e}")
    finally:
        if genome_log is not None:
            genome_log.close()

    columns = HISTORY_COLUMNS + (MAXCONNS_COLUMNS if args.max_conns is not None else [])
    with (run_dir / "fitness_history.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=columns)
        w.writeheader()
        w.writerows(evaluator.history)

    summary = {
        "args": vars(args),
        "git_commit": commit,
        "seconds": round(time.time() - started, 2),
        "generations_planned": generations,
        "generations_run": len(evaluator.history),
        "steps_per_generation": per_gen,
        "budget": budget.summary(),
        "budget_spent": budget.spent,
        "budget_limit": budget.limit,
        "budget_refinements": budget.refinements,
        "within_budget": exhausted is None and budget.spent <= budget.limit,
        "budget_error": exhausted,
        "refine_calls": evaluator.refine_calls,
        "genomes_refined": evaluator.genomes_refined,
        "n_fit": len(X_fit),
        "n_val": len(X_val),
        "heldout_majority_class": float(max(y_test.mean(), 1 - y_test.mean())),
    }
    if args.max_conns is not None:
        summary.update({
            "max_conns": args.max_conns,
            "max_conns_mechanism": "repair at creation (network.MaxConnsGenome): "
                                   "dead connections first, then smallest |weight|",
            "genomes_created": evaluator.genomes_created,
            "genomes_repaired": evaluator.genomes_repaired,
            "repair_disabled_dead": evaluator.repair_dead,
            "repair_disabled_live": evaluator.repair_live,
            "genomes_over_limit_evaluated": 0,  # any would have stopped the run
        })

    champ = evaluator.champion
    if champ is not None and args.max_conns is not None:
        # The guarantee the small-network result rests on, checked on the
        # genomes themselves before champion.pkl or summary.json is written.
        over = {name: complexity(g)[1] for name, g in (("champion", champ),
                                                       ("train-best", evaluator.trainbest))
                if complexity(g)[1] > args.max_conns}
        if over:
            raise RuntimeError(f"over --max-conns {args.max_conns}: {over}")
    if champ is not None:
        with (run_dir / "champion.pkl").open("wb") as fh:
            pickle.dump(champ, fh)
        net = champion_net(champ, config)
        nodes, conns = complexity(champ)
        tb = champion_net(evaluator.trainbest, config)
        summary.update({
            "champion_generation": evaluator.champion_gen,
            "champion_fitness": champ.fitness,
            "champion_refined": getattr(champ, "refined_params", None) is not None,
            "fit_acc": net.accuracy(X_fit, y_fit),
            "val_acc": net.accuracy(X_val, y_val) if len(X_val) else None,
            "train_acc": net.accuracy(X_train, y_train),
            "heldout_acc": net.accuracy(X_test, y_test),
            "nodes": nodes,
            "conns": conns,
            "max_abs_param": float(np.abs(net.get_params()).max()),
            "trainbest_heldout_acc": tb.accuracy(X_test, y_test),
            "trainbest_conns": complexity(evaluator.trainbest)[1],
        })
        print(f"\ndataset          : {args.dataset}")
        print(f"generations      : {len(evaluator.history)}")
        print(f"budget           : {budget.summary()}")
        print(f"fit / val acc    : {summary['fit_acc']:.3f} / "
              f"{summary['val_acc'] if summary['val_acc'] is None else round(summary['val_acc'], 3)}")
        print(f"held-out accuracy: {summary['heldout_acc']:.3f} "
              f"(majority class {summary['heldout_majority_class']:.3f})")
        print(f"champion size    : {nodes} nodes, {conns} enabled connections "
              f"(generation {evaluator.champion_gen})")

    with (run_dir / "summary.json").open("w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"wrote {run_dir}/")
    return BUDGET_EXIT if exhausted else 0


if __name__ == "__main__":
    raise SystemExit(main())

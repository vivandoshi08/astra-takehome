"""Evolve a slime-volleyball agent with NEAT.

Usage:
    python evolve.py --budget 5000000 --run-id sample      # short sample run
    python evolve.py --seed 0 --run-id best                # full 100M-step run

The run ends when the step budget is spent (--generations is only a safety cap).

Fitness is the mean score over --episodes seeded games, the same seeds for every
genome in a generation and fresh ones each generation. Because those seeds vary
in difficulty, training fitness is a noisy, biased-upward number: the champion
is instead the genome with the best mean on a fixed validation seed set
(--val-top genomes per generation, --val-episodes games each). Validation games
are real environment steps and are charged to the budget like any other.

--survival-bonus B shapes the *selection* fitness only: each episode counts as
score + B * steps/3000. Against this opponent ~90% of early games are 5-0
losses and the few points won are mostly decided by the serve, so raw score
gives selection almost nothing to work with; time survived does vary and
correlates with score. With B < 1 a point always outweighs any amount of
survival. The champion is still chosen on raw validation score, and every
reported number is raw score.

--return-bonus R adds R per *return* (our contact after which the ball reaches
the opponent's half), counting at most --return-cap returns per episode, to
each episode's selection fitness. The cap matters: winning points ends the
game sooner, so an uncapped bonus would pay a 0-0 rally to the time limit
more than a quick 5-0 win. B + R*cap must be < 1 so one point always
outweighs all shaping combined. Survival alone was
too weak (~0.7% of fitness variance) and raw contacts don't track score
(r = 0.05; a genome can juggle on its own side and still lose 5-0), whereas
returning the ball is the thing that has to happen before any point is won.

Writes to runs/<run-id>/ — see LOGS.md for every file and column.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import multiprocessing as mp
import os
import pickle
import random
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path

import neat

sys.path.insert(0, str(Path(__file__).parent))

import slimevolley  # noqa: E402
from budget import BudgetExhausted, BudgetedEnv, StepBudget, total_steps  # noqa: E402
from episode import run_episode  # noqa: E402

HERE = Path(__file__).parent
TRAIN_SEED_LIMIT = 1_000_000  # training seeds are drawn from [0, TRAIN_SEED_LIMIT)
HELDOUT_SEED_START = 1_000_000  # evaluate.py: HELDOUT_SEED_START + i
VAL_SEED_START = 2_000_000  # champion selection: VAL_SEED_START + i
_W: dict = {}

HISTORY_COLUMNS = [
    "generation", "best_fitness", "mean_fitness",
    "median_fitness", "std_fitness", "min_fitness", "n_species",
    "best_nodes", "best_conns", "mean_ep_steps", "timeout_frac",
    "gen_steps", "cum_steps", "val_best", "champion_val",
    "best_raw", "mean_raw",
    "mean_nodes", "mean_conns", "mean_contacts", "mean_returns",
    "best_raw_contacts", "mean_cancel", "mean_sat", "gen_seconds",
    "mean_moved", "mean_jumps", "jump_hit_rate", "zone_net_frac", "zone_back_frac",
]
ACT_KEYS = ("act_fwd", "act_back", "act_jump", "act_idle", "act_both")


def episode_summary(eps: list[dict]) -> dict:
    """Per-genome aggregates of run_episode stats (LOGS.md)."""
    n_steps = sum(e["steps"] for e in eps)
    return {
        "timeouts": sum(e["steps"] >= 3000 for e in eps),
        **{k: sum(e[k] for e in eps) / n_steps for k in ACT_KEYS},
        "out_sat": sum(e["out_sat"] for e in eps) / (3 * n_steps),
        "contacts": [e["contacts"] for e in eps],
        "opp_contacts": [e["opp_contacts"] for e in eps],
        "returns": [e["returns"] for e in eps],
        "moved": [round(e["moved"], 2) for e in eps],
        "jumps": [e["jumps"] for e in eps],
        "jump_hits": [e["jump_hits"] for e in eps],
        **{k: sum(e[k] for e in eps) / n_steps for k in ("zone_net", "zone_mid", "zone_back")},
    }


def _init_worker(run_dir: str, config_path: str, per_process_cap: int) -> None:
    budget = StepBudget(Path(run_dir) / "budget_log.jsonl", per_process_cap)
    _W["budget"] = budget
    _W["env"] = BudgetedEnv(slimevolley.SlimeVolleyEnv(), budget)
    _W["config"] = load_config(config_path)


def load_config(path) -> neat.Config:
    return neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                       neat.DefaultSpeciesSet, neat.DefaultStagnation, str(path))


def make_net(genome, config):
    """Build the network class that matches the config's `feed_forward` setting.

    Reusable — call this from your own evaluation and replay code so it stays in
    step with whatever you put in neat_config.ini.
    """
    if config.genome_config.feed_forward:
        return neat.nn.FeedForwardNetwork.create(genome, config)
    return neat.nn.RecurrentNetwork.create(genome, config)


def train_seeds(run_seed: int, generation: int, n: int) -> list[int]:
    """Episode seeds for one generation, shared by every genome in it.

    Sharing seeds within a generation means genomes are compared on the same
    serves (less selection noise); drawing fresh ones each generation stops the
    population overfitting a fixed set.
    """
    rng = random.Random(f"{run_seed}:{generation}")
    return [rng.randrange(TRAIN_SEED_LIMIT) for _ in range(n)]


def val_seeds(n: int) -> list[int]:
    return [VAL_SEED_START + i for i in range(n)]


def play_seeds(net, env, seeds: list[int]) -> list[dict]:
    """One seeded episode per seed; per-episode stats as dicts."""
    out = []
    for seed in seeds:
        env.seed(seed)
        if hasattr(net, "reset"):
            net.reset()  # recurrent nets carry state; clear it before each episode
        out.append(run_episode(net, env).to_dict())
    return out


def _eval_one(job: tuple[bytes, list[int]]) -> list[dict]:
    """Per-episode stats for one genome on the given seeds."""
    genome_bytes, seeds = job
    net = make_net(pickle.loads(genome_bytes), _W["config"])
    try:
        return play_seeds(net, _W["env"], seeds)
    finally:
        _W["budget"].flush()


def selection_fitness(ep: dict, args) -> float:
    """Per-episode fitness used for selection only (raw score + shaping)."""
    return (ep["score"] + args.survival_bonus * ep["steps"] / 3000
            + args.return_bonus * min(ep["returns"], args.return_cap))


def play_all(pool, genomes, seeds: list[int]) -> list[list[dict]]:
    """Every genome on every seed, one episode per job so the load spreads
    evenly over workers (a 20-episode validation job per genome used to pile
    onto 3 workers and hit their step caps early). Episodes are independent
    and seeded, so splitting changes nothing about the results."""
    blobs = [pickle.dumps(g) for g in genomes]
    flat = pool.map(_eval_one, [(b, [s]) for b in blobs for s in seeds], chunksize=1)
    n = len(seeds)
    return [[ep for job in flat[i * n:(i + 1) * n] for ep in job] for i in range(len(genomes))]


class BudgetReached(Exception):
    """Signals the driver to stop evolving."""


class RunRecorder(neat.reporting.BaseReporter):
    """Writes the per-generation logs, tracks the champion, enforces the budget."""

    def __init__(self, run_dir: Path, limit: int, per_process_cap: int, workers: int,
                 snapshot_every: int = 0):
        self.run_dir = run_dir
        self.budget_log = run_dir / "budget_log.jsonl"
        self.limit = limit
        self.per_process_cap = per_process_cap
        self.workers = workers
        self.snapshot_every = snapshot_every
        self.next_snapshot = snapshot_every
        self.validated_genomes: dict = {}           # key -> genome, this gen's validated
        self.history: list[list] = []
        self.episodes: dict[int, list[dict]] = {}  # genome key -> this gen's episodes
        self.validated: list[dict] = []             # this gen's validation rows
        self.champion = None
        self.champion_meta: dict = {"val_mean": float("-inf")}
        self._gen = 0
        self._last_steps = 0
        self._last_time = time.time()
        for name in ("genomes.jsonl", "validation.jsonl"):
            (run_dir / name).unlink(missing_ok=True)

    def consider(self, genome, val_mean: float, gen: int) -> None:
        """Champion = best validation mean; ties go to the earlier genome."""
        if val_mean > self.champion_meta["val_mean"]:
            nodes, conns = genome.size()
            # Snapshot: genomes survive into later generations and are
            # re-evaluated in place, which would overwrite .fitness here.
            self.champion = copy.deepcopy(genome)
            self.champion_meta = {"key": genome.key, "generation": gen,
                                  "train_fitness": genome.fitness,
                                  "val_mean": val_mean, "nodes": nodes, "conns": conns}

    def post_evaluate(self, config, population, species, best_genome):
        gen = self._gen
        genomes = [g for g in population.values() if g.fitness is not None]
        fits = [g.fitness for g in genomes]

        with (self.run_dir / "genomes.jsonl").open("a") as fh:
            for g in genomes:
                eps = self.episodes[g.key]
                nodes, conns = g.size()
                fh.write(json.dumps({
                    "gen": gen, "key": g.key,
                    "species": species.get_species_id(g.key),
                    "nodes": nodes, "conns": conns, "fitness": g.fitness,
                    "raw_fitness": statistics.mean(e["score"] for e in eps),
                    "scores": [e["score"] for e in eps],
                    "steps": [e["steps"] for e in eps],
                    "points_for": [e["points_for"] for e in eps],
                    "points_against": [e["points_against"] for e in eps],
                    **episode_summary(eps),
                }) + "\n")
        with (self.run_dir / "validation.jsonl").open("a") as fh:
            for row in self.validated:
                fh.write(json.dumps({"gen": gen, **row}) + "\n")

        all_eps = [e for g in genomes for e in self.episodes[g.key]]
        raw = [statistics.mean(e["score"] for e in self.episodes[g.key]) for g in genomes]
        spent = total_steps(self.budget_log)
        gen_steps = spent - self._last_steps
        self._last_steps = spent
        nodes, conns = best_genome.size()
        sizes = [g.size() for g in genomes]
        best_raw_g = genomes[max(range(len(genomes)), key=raw.__getitem__)]
        all_steps = sum(e["steps"] for e in all_eps)
        now = time.time()
        self.history.append([
            gen, max(fits), statistics.mean(fits),
            statistics.median(fits), statistics.pstdev(fits), min(fits),
            len(species.species), nodes, conns,
            statistics.mean(e["steps"] for e in all_eps),
            statistics.mean(e["steps"] >= 3000 for e in all_eps),
            gen_steps, spent,
            max(r["val_mean"] for r in self.validated),
            self.champion_meta["val_mean"],
            max(raw), statistics.mean(raw),
            statistics.mean(n for n, _ in sizes), statistics.mean(c for _, c in sizes),
            statistics.mean(e["contacts"] for e in all_eps),
            statistics.mean(e["returns"] for e in all_eps),
            statistics.mean(e["contacts"] for e in self.episodes[best_raw_g.key]),
            sum(e["act_both"] for e in all_eps) / all_steps,
            sum(e["out_sat"] for e in all_eps) / (3 * all_steps),
            round(now - self._last_time, 2),
            statistics.mean(e["moved"] for e in all_eps),
            statistics.mean(e["jumps"] for e in all_eps),
            sum(e["jump_hits"] for e in all_eps) / max(1, sum(e["jumps"] for e in all_eps)),
            sum(e["zone_net"] for e in all_eps) / all_steps,
            sum(e["zone_back"] for e in all_eps) / all_steps,
        ])
        self._last_time = now
        self.write_outputs()
        while self.snapshot_every and spent >= self.next_snapshot:
            self.write_snapshot(f"{self.next_snapshot / 1_000_000:g}M", gen, spent)
            self.next_snapshot += self.snapshot_every
        self._gen += 1

        h = dict(zip(HISTORY_COLUMNS, self.history[-1]))
        print(f"  [gen {gen}] train best {h['best_fitness']:+.2f} mean {h['mean_fitness']:+.2f}"
              f" (raw {h['best_raw']:+.2f}/{h['mean_raw']:+.2f})"
              f" | val best {h['val_best']:+.2f} champion {h['champion_val']:+.2f}"
              f" | species {h['n_species']} | best size {nodes}n/{conns}c"
              f" | ep len {h['mean_ep_steps']:.0f} timeouts {h['timeout_frac']:.0%}", flush=True)
        print(f"           contacts/ep {h['mean_contacts']:.2f} returns/ep {h['mean_returns']:.2f}"
              f" | cancel {h['mean_cancel']:.2f} saturated {h['mean_sat']:.2f}"
              f" | pop size {h['mean_nodes']:.1f}n/{h['mean_conns']:.1f}c | {h['gen_seconds']:.1f}s", flush=True)
        print(f"           moved/ep {h['mean_moved']:.0f} jumps/ep {h['mean_jumps']:.1f}"
              f" jump->hit {h['jump_hit_rate']:.2f} | at net {h['zone_net_frac']:.2f}"
              f" back {h['zone_back_frac']:.2f}", flush=True)
        print(f"  [budget] {spent:,} / {self.limit:,} steps (+{gen_steps:,} this gen)", flush=True)
        # Stop at a generation boundary rather than letting a worker hit its cap
        # mid-generation, which throws the whole partial generation away.
        # Workers don't share load perfectly, so also check the busiest worker
        # against its own cap (allowing it 1.5x its fair share of a generation).
        busiest = max(per_pid_steps(self.budget_log).values(), default=0)
        if (spent + gen_steps > self.limit
                or busiest + 1.5 * gen_steps / self.workers > self.per_process_cap):
            print("  [budget] another generation would not fit — stopping", flush=True)
            raise BudgetReached

    def write_snapshot(self, label: str, gen: int, spent: int) -> None:
        """Champion so far + this generation's best validated genome, for
        replay_snapshots.py (replays at fixed step intervals)."""
        d = self.run_dir / "snapshots" / label
        d.mkdir(parents=True, exist_ok=True)
        if self.champion is not None:
            (d / "champion.pkl").write_bytes(pickle.dumps(self.champion))
            (d / "champion.json").write_text(json.dumps(
                {**self.champion_meta, "snapshot_steps": spent, "snapshot_gen": gen}, indent=2))
        best = max(self.validated, key=lambda r: r["val_mean"])
        (d / "gen_best.pkl").write_bytes(pickle.dumps(self.validated_genomes[best["key"]]))
        (d / "gen_best.json").write_text(json.dumps(
            {"key": best["key"], "generation": gen, "val_mean": best["val_mean"],
             "train_fitness": best["train_fitness"], "snapshot_steps": spent}, indent=2))
        print(f"  [snapshot] {label}: champion + gen best saved to {d}", flush=True)

    def write_outputs(self) -> None:
        """Rewritten every generation so a crash or Ctrl-C loses nothing."""
        with (self.run_dir / "fitness_history.csv").open("w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(HISTORY_COLUMNS)
            writer.writerows(self.history)
        if self.champion is not None:
            with (self.run_dir / "champion.pkl").open("wb") as fh:
                pickle.dump(self.champion, fh)
            (self.run_dir / "champion.json").write_text(json.dumps(self.champion_meta, indent=2))


class Tee:
    """Write to the console and a log file at once."""

    def __init__(self, stream, path: Path):
        self.stream, self.file = stream, path.open("a")

    def write(self, text):
        self.stream.write(text)
        self.file.write(text)

    def flush(self):
        self.stream.flush()
        self.file.flush()


def per_pid_steps(budget_log: Path) -> dict[int, int]:
    """Latest cumulative step count per worker (read-only view of budget_log)."""
    latest: dict[int, int] = {}
    if budget_log.exists():
        for line in budget_log.read_text().splitlines():
            if line.strip():
                rec = json.loads(line)
                latest[rec["pid"]] = max(latest.get(rec["pid"], 0), rec["cum_steps"])
    return latest


def git_commit() -> str:
    try:
        sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=HERE, capture_output=True,
                             text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--", "."], cwd=HERE,
                               capture_output=True, text=True).stdout.strip()
        return sha + ("-dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--generations", type=int, default=100_000,
                    help="safety cap; the step budget normally ends the run")
    ap.add_argument("--budget", type=int, default=100_000_000,
                    help="total environment steps across all workers")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--config", default=str(HERE / "neat_config.ini"))
    ap.add_argument("--run-id", default="run")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--episodes", type=int, default=5,
                    help="seeded episodes per genome per generation; fitness is their mean")
    ap.add_argument("--val-top", type=int, default=3,
                    help="top genomes by training fitness re-scored on validation seeds each generation")
    ap.add_argument("--val-episodes", type=int, default=20,
                    help="fixed validation seeds used to pick the champion")
    ap.add_argument("--survival-bonus", type=float, default=0.0,
                    help="selection fitness per episode += B * steps/3000 (B < 1)")
    ap.add_argument("--return-bonus", type=float, default=0.0,
                    help="selection fitness per episode += R * min(returns, cap)")
    ap.add_argument("--snapshot-every", type=int, default=10_000_000,
                    help="save champion + generation best every N steps (0 = off); see replay_snapshots.py")
    ap.add_argument("--return-cap", type=int, default=9,
                    help="returns per episode that earn the bonus")
    args = ap.parse_args()
    if args.return_bonus < 0 or args.return_cap < 0:
        ap.error("--return-bonus and --return-cap must be >= 0")
    if args.survival_bonus + args.return_bonus * args.return_cap >= 1:
        ap.error("survival bonus + return bonus * cap must be < 1: a point must outweigh all shaping")
    if not 0 <= args.survival_bonus < 1:
        ap.error("--survival-bonus must be in [0, 1) so a point always outweighs survival")

    run_dir = HERE / "runs" / args.run_id
    if (run_dir / "budget_log.jsonl").exists():
        sys.exit(f"{run_dir} already has a budget log; pick a new --run-id")
    run_dir.mkdir(parents=True, exist_ok=True)
    sys.stdout = Tee(sys.stdout, run_dir / "stdout.log")  # keep NEAT's own report too
    shutil.copy(args.config, run_dir / "config.ini")
    config_path = run_dir / "config.ini"
    run_info = {
        "argv": sys.argv, "args": vars(args), "git_commit": git_commit(),
        "start_time": time.strftime("%Y-%m-%dT%H:%M:%S"), "python": sys.version.split()[0],
        "neat_version": getattr(neat, "__version__", "unknown"),
        "seed_ranges": {"train": [0, TRAIN_SEED_LIMIT],
                        "heldout_start": HELDOUT_SEED_START,
                        "heldout": [HELDOUT_SEED_START, VAL_SEED_START],
                        "val": [VAL_SEED_START, VAL_SEED_START + args.val_episodes]},
    }
    (run_dir / "run_args.json").write_text(json.dumps(run_info, indent=2))

    config = load_config(config_path)
    pop = neat.Population(config, seed=args.seed)
    pop.add_reporter(neat.StdOutReporter(False))
    per_process_cap = max(1, args.budget // max(1, args.workers))
    recorder = RunRecorder(run_dir, args.budget, per_process_cap, args.workers, args.snapshot_every)
    pop.add_reporter(recorder)

    stopped_early = False

    ctx = mp.get_context("spawn")
    pool = ctx.Pool(processes=args.workers, initializer=_init_worker,
                    initargs=(str(run_dir), str(config_path), per_process_cap))
    vseeds = val_seeds(args.val_episodes)
    generation = 0

    def evaluate(genomes, cfg):
        nonlocal generation
        seeds = train_seeds(args.seed, generation, args.episodes)
        results = play_all(pool, [g for _, g in genomes], seeds)
        recorder.episodes = {}
        for (_, genome), eps in zip(genomes, results):
            genome.fitness = statistics.mean(selection_fitness(e, args) for e in eps)
            recorder.episodes[genome.key] = eps

        top = sorted((g for _, g in genomes), key=lambda g: g.fitness, reverse=True)[:args.val_top]
        val_results = play_all(pool, top, vseeds)
        recorder.validated = []
        recorder.validated_genomes = {g.key: copy.deepcopy(g) for g in top}
        for genome, eps in zip(top, val_results):
            val_mean = statistics.mean(e["score"] for e in eps)
            recorder.validated.append({
                "key": genome.key, "train_fitness": genome.fitness, "val_mean": val_mean,
                "val_scores": [e["score"] for e in eps], "val_steps": [e["steps"] for e in eps],
                **{f"val_{k}": v for k, v in episode_summary(eps).items()},
            })
            recorder.consider(genome, val_mean, generation)
        generation += 1

    try:
        pop.run(evaluate, args.generations)
    except BudgetReached:
        stopped_early = True
    except BudgetExhausted as exc:
        print(f"  [budget] {exc} — stopping (partial generation discarded)", flush=True)
        stopped_early = True
    except KeyboardInterrupt:
        print("  interrupted — writing what we have", flush=True)
        stopped_early = True
    finally:
        pool.close()
        pool.join()

    recorder.write_outputs()
    if recorder.snapshot_every and recorder.history and recorder.validated:
        # Always snapshot the end state: the budget guard stops a run just short
        # of a round number (e.g. 19.4M of 20M), which would skip the last one.
        recorder.write_snapshot("end", len(recorder.history) - 1, total_steps(recorder.budget_log))
    run_info.update({"end_time": time.strftime("%Y-%m-%dT%H:%M:%S"),
                     "generations_completed": len(recorder.history),
                     "steps_used": total_steps(recorder.budget_log),
                     "stopped_early": stopped_early, "champion": recorder.champion_meta})
    (run_dir / "run_args.json").write_text(json.dumps(run_info, indent=2))

    print(f"\nwrote {run_dir}/")
    print(f"  generations completed: {len(recorder.history)}")
    print(f"  steps used:            {run_info['steps_used']:,} / {args.budget:,}")
    if recorder.champion is not None:
        m = recorder.champion_meta
        print(f"  champion: genome {m['key']} from gen {m['generation']}, "
              f"val mean {m['val_mean']:+.2f} over {args.val_episodes} games "
              f"(train fitness {m['train_fitness']:+.2f} — shaped/biased upward, don't report it)")
        print(f"  headline number: python evaluate.py --winner {run_dir / 'champion.pkl'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

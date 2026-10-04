"""Evaluate a champion on fresh, held-out-seed episodes (the Q1 headline number).

    python evaluate.py --winner runs/best/champion.pkl
    python evaluate.py --winner runs/best/champion.pkl --episodes 100 --out runs/best/eval.json

Seeds start at evolve.TRAIN_SEED_LIMIT, which no training episode ever uses, so
these games are never seen during evolution. Not budgeted: evaluation is not
training, and the budget log belongs to the training run.

Reports the mean score with a 95% confidence interval (t-distribution for the
mean, plus a percentile bootstrap as a check, since scores are bounded integers
in [-5, 5] and not normal).
"""

from __future__ import annotations

import argparse
import json
import pickle
import random
import statistics
import sys
from pathlib import Path

import neat

sys.path.insert(0, str(Path(__file__).parent))

import slimevolley  # noqa: E402
from episode import play_episode  # noqa: E402
from evolve import TRAIN_SEED_LIMIT, make_net  # noqa: E402

HERE = Path(__file__).parent

# Two-sided 95% t critical value for df = 99; close enough for df in 60..200.
T_95 = 1.984


def heldout_seeds(n: int) -> list[int]:
    return [TRAIN_SEED_LIMIT + i for i in range(n)]


def evaluate(genome, config, seeds: list[int]) -> list[float]:
    net = make_net(genome, config)
    env = slimevolley.SlimeVolleyEnv()
    scores = []
    for seed in seeds:
        env.seed(seed)
        if hasattr(net, "reset"):
            net.reset()
        scores.append(play_episode(net, env))
    return scores


def summarize(scores: list[float], boot: int = 10_000) -> dict:
    n = len(scores)
    mean = statistics.mean(scores)
    sd = statistics.stdev(scores) if n > 1 else 0.0
    half = T_95 * sd / n ** 0.5
    rng = random.Random(0)
    means = sorted(statistics.mean(rng.choices(scores, k=n)) for _ in range(boot))
    return {
        "n": n,
        "mean": mean,
        "sd": sd,
        "ci95_t": [mean - half, mean + half],
        "ci95_bootstrap": [means[int(0.025 * boot)], means[int(0.975 * boot) - 1]],
        "wins": sum(s > 0 for s in scores),
        "draws": sum(s == 0 for s in scores),
        "losses": sum(s < 0 for s in scores),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--winner", default=str(HERE / "runs" / "run" / "champion.pkl"))
    ap.add_argument("--config", default=str(HERE / "neat_config.ini"))
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--out", help="write scores + summary as JSON")
    args = ap.parse_args()

    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         args.config)
    with open(args.winner, "rb") as fh:
        blob = pickle.load(fh)
    genome = blob["genome"] if isinstance(blob, dict) else blob

    seeds = heldout_seeds(args.episodes)
    scores = evaluate(genome, config, seeds)
    s = summarize(scores)
    print(f"{s['n']} held-out episodes (seeds {seeds[0]}..{seeds[-1]})")
    print(f"  mean {s['mean']:+.3f}  sd {s['sd']:.3f}")
    print(f"  95% CI (t)         [{s['ci95_t'][0]:+.3f}, {s['ci95_t'][1]:+.3f}]")
    print(f"  95% CI (bootstrap) [{s['ci95_bootstrap'][0]:+.3f}, {s['ci95_bootstrap'][1]:+.3f}]")
    print(f"  W/D/L {s['wins']}/{s['draws']}/{s['losses']}")
    if args.out:
        Path(args.out).write_text(json.dumps({"seeds": seeds, "scores": scores, **s}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

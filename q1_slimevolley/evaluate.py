"""Evaluate a champion on fresh, held-out-seed episodes (the Q1 headline number).

    python evaluate.py --winner runs/best/champion.pkl
    python evaluate.py --winner runs/best/champion.pkl --episodes 100 --out runs/best/eval.json

Seeds are evolve.HELDOUT_SEED_START + i: outside both the training range and
the validation seeds used to pick the champion, so these games played no part
in producing or selecting it. Uses the run's saved config.ini when present.
Not budgeted: evaluation is not
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

sys.path.insert(0, str(Path(__file__).parent))

import slimevolley  # noqa: E402
from evolve import HELDOUT_SEED_START, load_config, make_net, play_seeds  # noqa: E402

HERE = Path(__file__).parent

def t_crit_95(df: int) -> float:
    """Two-sided 95% Student-t critical value (Cornish-Fisher expansion of the
    normal quantile; within 0.005 of exact for df >= 9: df=19 -> 2.093, df=99 -> 1.984)."""
    z = 1.959964
    return z + (z**3 + z) / (4 * df) + (5 * z**5 + 16 * z**3 + 3 * z) / (96 * df**2)


def heldout_seeds(n: int) -> list[int]:
    return [HELDOUT_SEED_START + i for i in range(n)]


def evaluate(genome, config, seeds: list[int]) -> list[dict]:
    """Per-episode stats (score, steps, points, actions) on each seed."""
    return play_seeds(make_net(genome, config), slimevolley.SlimeVolleyEnv(), seeds)


def summarize(scores: list[float], boot: int = 10_000) -> dict:
    n = len(scores)
    mean = statistics.mean(scores)
    sd = statistics.stdev(scores) if n > 1 else 0.0
    half = t_crit_95(n - 1) * sd / n ** 0.5 if n > 1 else 0.0
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

    run_config = Path(args.winner).parent / "config.ini"
    config_path = run_config if run_config.exists() else Path(args.config)
    print(f"config: {config_path}")
    config = load_config(config_path)
    with open(args.winner, "rb") as fh:
        blob = pickle.load(fh)
    genome = blob["genome"] if isinstance(blob, dict) else blob

    seeds = heldout_seeds(args.episodes)
    episodes = evaluate(genome, config, seeds)
    scores = [e["score"] for e in episodes]
    s = summarize(scores)
    steps = [e["steps"] for e in episodes]
    total = sum(steps)
    s["mean_steps"] = statistics.mean(steps)
    s["timeouts"] = sum(n >= 3000 for n in steps)
    s["points_for"] = sum(e["points_for"] for e in episodes)
    s["points_against"] = sum(e["points_against"] for e in episodes)
    s["action_frac"] = {k: sum(e[k] for e in episodes) / total
                        for k in ("act_fwd", "act_back", "act_jump", "act_idle", "act_both")}
    s["contacts_per_ep"] = statistics.mean(e["contacts"] for e in episodes)
    s["returns_per_ep"] = statistics.mean(e["returns"] for e in episodes)
    s["opp_contacts_per_ep"] = statistics.mean(e["opp_contacts"] for e in episodes)
    s["out_sat"] = sum(e["out_sat"] for e in episodes) / (3 * total)
    s["moved_per_ep"] = statistics.mean(e["moved"] for e in episodes)
    s["jumps_per_ep"] = statistics.mean(e["jumps"] for e in episodes)
    s["jump_hit_rate"] = sum(e["jump_hits"] for e in episodes) / max(1, sum(e["jumps"] for e in episodes))
    s["zones"] = {k: sum(e[k] for e in episodes) / total for k in ("zone_net", "zone_mid", "zone_back")}
    s["score_counts"] = {str(v): scores.count(v) for v in sorted(set(scores))}
    print(f"{s['n']} held-out episodes (seeds {seeds[0]}..{seeds[-1]})")
    print(f"  mean {s['mean']:+.3f}  sd {s['sd']:.3f}")
    print(f"  95% CI (t)         [{s['ci95_t'][0]:+.3f}, {s['ci95_t'][1]:+.3f}]")
    print(f"  95% CI (bootstrap) [{s['ci95_bootstrap'][0]:+.3f}, {s['ci95_bootstrap'][1]:+.3f}]")
    print(f"  W/D/L {s['wins']}/{s['draws']}/{s['losses']}   score counts {s['score_counts']}")
    print(f"  points won {s['points_for']}  lost {s['points_against']}")
    print(f"  episode length mean {s['mean_steps']:.0f}  timeouts (3000 steps) {s['timeouts']}")
    print(f"  per episode: contacts {s['contacts_per_ep']:.2f}  returns {s['returns_per_ep']:.2f}"
          f"  opponent contacts {s['opp_contacts_per_ep']:.2f}  | outputs saturated {s['out_sat']:.2f}")
    print(f"  movement: {s['moved_per_ep']:.0f} units/ep, {s['jumps_per_ep']:.1f} jumps/ep, "
          f"{s['jump_hit_rate']:.2f} of jumps hit the ball | time at net/mid/back "
          + "/".join(f"{v:.2f}" for v in s["zones"].values()))
    print("  action fractions " + "  ".join(f"{k[4:]} {v:.2f}" for k, v in s["action_frac"].items()))
    if args.out:
        Path(args.out).write_text(json.dumps(
            {"winner": args.winner, "config": str(config_path), "seeds": seeds,
             "episodes": episodes, **s}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

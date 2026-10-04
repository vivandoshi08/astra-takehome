"""Reward-hacking check: does shaped selection fitness rank behaviours like real skill?

    python reward_check.py [--episodes 20] [--champion runs/<id>/champion.pkl]

Plays fixed hand-written policies (degenerate and competent) plus optional
evolved champions, and scores each under several fitness formulas. A shaping
term is safe only if it never ranks a behaviour above one that wins more
points (raw score). Seeds are 3_000_000 + i: outside the training, held-out
and validation ranges, so this never touches the numbers we report.
"""

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

import slimevolley as sv  # noqa: E402
from episode import run_episode  # noqa: E402

PROBE_SEED_START = 3_000_000

# (label, survival bonus B, return bonus R, return cap C) — B + R*C < 1
FORMULAS = [
    ("raw", 0.0, 0.0, 0),
    ("B=0.5", 0.5, 0.0, 0),
    ("R=0.1,C=9", 0.0, 0.1, 9),
    ("B=0.3,R=0.07,C=9", 0.3, 0.07, 9),
]


class Fixed:
    """Outputs a constant button pattern (0.9 = pressed, 0.1 = not)."""

    def __init__(self, fwd, back, jump):
        self.out = [0.9 if b else 0.1 for b in (fwd, back, jump)]

    def activate(self, obs):
        return self.out


class Random:
    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)

    def activate(self, obs):
        return list(self.rng.random(3))


class Tracker:
    """Hand-written: stand slightly behind the ball, jump when it is close."""

    def activate(self, obs):
        x, bx, by = obs[0], obs[4], obs[5]
        target = bx + 0.15  # obs are /10; 'forward' (toward net) lowers x
        return [0.9 if x > target + 0.05 else 0.1,
                0.9 if x < target - 0.05 else 0.1,
                0.9 if abs(bx - x) < 0.35 and by < 0.6 else 0.1]


class Backline:
    """Degenerate: hide at the back wall, never move."""

    def activate(self, obs):
        return [0.1, 0.9, 0.1]


class BuiltIn:
    """The opponent's own 120-parameter network playing our side."""

    def __init__(self):
        self.p = sv.BaselinePolicy()

    def activate(self, obs):
        return [0.9 if a else 0.1 for a in self.p.predict(obs)]


def shaped(ep: dict, b: float, r: float, c: int) -> float:
    return ep["score"] + b * ep["steps"] / 3000 + r * min(ep["returns"], c)


def play(make, episodes: int) -> list[dict]:
    env = sv.SlimeVolleyEnv()
    out = []
    for i in range(episodes):
        env.seed(PROBE_SEED_START + i)
        net = make()
        if hasattr(net, "reset"):
            net.reset()
        out.append(run_episode(net, env).to_dict())
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--champion", action="append", default=[],
                    help="evolved champion.pkl to include (repeatable)")
    args = ap.parse_args()

    policies = {
        "idle": lambda: Fixed(0, 0, 0),
        "always jump": lambda: Fixed(0, 0, 1),
        "net camper (fwd+jump)": lambda: Fixed(1, 0, 1),
        "backline hider": Backline,
        "random": Random,
        "tracker (hand-written)": Tracker,
        "built-in policy": BuiltIn,
    }
    if args.champion:
        import pickle

        from evolve import load_config, make_net
        for path in args.champion:
            run = Path(path).parent
            genome = pickle.load(open(path, "rb"))
            config = load_config(run / "config.ini")
            policies[f"champion {run.name}"] = (
                lambda g=genome, c=config: make_net(g, c))

    rows = []
    for name, make in policies.items():
        eps = play(make, args.episodes)
        rows.append({
            "name": name,
            "raw": statistics.mean(e["score"] for e in eps),
            "steps": statistics.mean(e["steps"] for e in eps),
            "contacts": statistics.mean(e["contacts"] for e in eps),
            "returns": statistics.mean(e["returns"] for e in eps),
            "timeouts": sum(e["steps"] >= 3000 for e in eps),
            **{label: statistics.mean(shaped(e, b, r, c) for e in eps)
               for label, b, r, c in FORMULAS},
        })

    labels = [f[0] for f in FORMULAS]
    print(f"{args.episodes} probe episodes each (seeds {PROBE_SEED_START}+)\n")
    print(f"{'policy':28s} {'steps':>6s} {'cont':>5s} {'ret':>5s} {'t/o':>4s}  "
          + "  ".join(f"{label:>17s}" for label in labels))
    for r in sorted(rows, key=lambda r: r["raw"]):
        print(f"{r['name']:28s} {r['steps']:6.0f} {r['contacts']:5.1f} {r['returns']:5.1f} "
              f"{r['timeouts']:4d}  " + "  ".join(f"{r[label]:17.3f}" for label in labels))

    print("\nOrdering check: for every pair where A wins more points than B (raw score")
    print("higher by > 0.5, beyond probe noise), shaped fitness must also rank A above B.")
    ok = True
    for label in labels[1:]:
        bad = [(a["name"], b["name"]) for a in rows for b in rows
               if a["raw"] > b["raw"] + 0.5 and a[label] <= b[label]]
        ok &= not bad
        print(f"  {label:17s} {'OK' if not bad else 'VIOLATED: ' + str(bad)}")
    print("\nNote: shaping is *meant* to separate policies whose raw scores tie; this")
    print("check only guards against it overturning a real difference in points.")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

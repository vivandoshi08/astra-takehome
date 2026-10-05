"""What does a champion actually do? Possession, holding, and how points end.

    python behavior.py runs/<run-id> [--episodes 20] [--compare-builtin]

Plays held-out seeds (1e6+i, the same ones evaluate.py uses — nothing is
selected on them here) and measures, per point:
  - who touched the ball last, and where on our side a lost point landed
    (behind our agent = further from the net than it, or in front of it)
  - ball possession: share of steps the ball is on our half, and the longest
    unbroken stay there (a clock-running juggler would hold for 1000s of steps)
Writes behavior.md into the run dir. --compare-builtin adds the built-in
policy playing our side as a reference.
"""

from __future__ import annotations

import argparse
import pickle
import statistics
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import slimevolley as sv  # noqa: E402
from episode import _CONTACTS, net_inputs, policy_action  # noqa: E402
from evaluate import heldout_seeds  # noqa: E402
from evolve import load_config, make_net  # noqa: E402


def play(make_net_fn, seeds) -> dict:
    ours, holds, points = [], [], Counter()
    for seed in seeds:
        env = sv.SlimeVolleyEnv()
        env.seed(seed)
        obs = env.reset()
        env.policy.reset()
        net = make_net_fn()
        if hasattr(net, "reset"):
            net.reset()
        on_ours = steps = hold = longest = 0
        last_touch = None
        done = False
        while not done:
            before = _CONTACTS[1], _CONTACTS[-1]
            ball_x = env.game.ball.x
            me_x = env.game.agent_right.x
            obs, reward, done, _ = env.step(policy_action(net, net.activate(net_inputs(net, obs))))
            if _CONTACTS[1] > before[0]:
                last_touch = "us"
            elif _CONTACTS[-1] > before[1]:
                last_touch = "opp"
            if reward < 0:  # ball landed on our side
                where = "behind us" if ball_x > me_x else "in front of us"
                points[f"lost: landed {where}, last touch {last_touch or 'serve'}"] += 1
                last_touch = None
            elif reward > 0:
                points[f"won: last touch {last_touch or 'serve'}"] += 1
                last_touch = None
            steps += 1
            mine = env.game.ball.x > 0
            on_ours += mine
            hold = hold + 1 if mine else 0
            longest = max(longest, hold)
        ours.append(on_ours / steps)
        holds.append(longest)
    return {"possession": statistics.mean(ours), "hold_mean": statistics.mean(holds),
            "hold_max": max(holds), "points": points}


class BuiltIn:
    def __init__(self):
        self.p = sv.BaselinePolicy()

    def activate(self, obs):
        return [0.9 if a else 0.1 for a in self.p.predict(obs)]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run", type=Path)
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--compare-builtin", action="store_true")
    args = ap.parse_args()
    config = load_config(args.run / "config.ini")
    genome = pickle.loads((args.run / "champion.pkl").read_bytes())
    seeds = heldout_seeds(args.episodes)
    players = {"champion": lambda: make_net(genome, config)}
    if args.compare_builtin:
        players["built-in policy (our side)"] = BuiltIn
    lines = [f"# Behaviour: `{args.run.name}` champion", "",
             f"Held-out seeds {seeds[0]}..{seeds[-1]}.", "",
             "| player | ball on our half | longest hold, mean / max (steps) |", "|---|---|---|"]
    results = {name: play(fn, seeds) for name, fn in players.items()}
    for name, r in results.items():
        lines.append(f"| {name} | {r['possession']:.2f} | {r['hold_mean']:.0f} / {r['hold_max']} |")
    for name, r in results.items():
        total = sum(r["points"].values())
        lines += ["", f"How points ended — {name} ({total} points over {args.episodes} games):", "",
                  "| outcome | count | share |", "|---|---|---|"]
        for k, v in r["points"].most_common():
            lines.append(f"| {k} | {v} | {v / total:.0%} |")
    text = "\n".join(lines) + "\n"
    (args.run / "behavior.md").write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

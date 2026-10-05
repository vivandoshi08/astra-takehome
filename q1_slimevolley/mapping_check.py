"""Does an action mapping leave a fitness valley between the net camper and a ball tracker?

    python mapping_check.py

Hand-built policies, no evolution: a net camper (forward always on, jump
always on) plus a ball-tracking signal of strength k on the input bx - x,
wired the way each mapping allows:
  threshold  forward = sigmoid(5*bias) (stuck on); backward = sigmoid(5*k*(bx-x))
  signed     same two outputs; movement = sign(forward - backward)
  direction  one output = sigmoid(5*(bias - k*(bx-x))); movement from its value
k = 0 is the net camper itself. If phase-1 fitness (returns + 0.3*steps/3000)
rises steadily with k, every step toward tracking pays and selection can climb;
if it stays at the camper's level, tracking does nothing until a second change
happens (the valley). Probe seeds 3,000,000+ (outside train / val / held-out).
"""

from __future__ import annotations

import math
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import slimevolley as sv  # noqa: E402
from evolve import play_seeds  # noqa: E402

SEEDS = list(range(3_000_000, 3_000_010))
BIAS = 0.6  # camper's forward bias: sigmoid(5*0.6) = 0.95


def sig(z):
    return 1 / (1 + math.exp(-max(-60, min(60, 5 * z))))


class Policy:
    def __init__(self, mapping: str, k: float):
        self.action_mapping, self.k = mapping, k
        self.input_nodes = range(14)  # so net_inputs appends bx - x, by - y

    def activate(self, inp):
        dx = inp[12]  # bx - x (our frame, /10): > 0 means the ball is behind us
        if self.action_mapping == "direction":
            return [sig(BIAS - self.k * dx), 0.9]
        return [sig(BIAS), sig(self.k * dx), 0.9]


def phase1(mapping: str, k: float) -> tuple[float, float]:
    eps = play_seeds(Policy(mapping, k), sv.SlimeVolleyEnv(), SEEDS)
    return (statistics.mean(e["returns"] + 0.3 * e["steps"] / 3000 for e in eps),
            statistics.mean(e["contacts"] for e in eps))


def main() -> int:
    ks = [0, 0.5, 1, 2, 4, 8, 16]
    print(f"phase-1 fitness (contacts) vs tracking strength k, {len(SEEDS)} probe games each\n")
    print(f"{'mapping':10s} " + " ".join(f"{'k=' + str(k):>13s}" for k in ks))
    for mapping in ("threshold", "signed", "direction"):
        cells = [phase1(mapping, k) for k in ks]
        print(f"{mapping:10s} " + " ".join(f"{f:6.2f} ({c:4.1f})" for f, c in cells))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

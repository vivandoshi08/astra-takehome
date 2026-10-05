"""Compare two groups of loop runs by their champions, with the spread.

    python compare_groups.py a r            # group prefixes under runs/
    python compare_groups.py a b --set held_out

For each group: every champion's score (from run_meta.json), median and
range. Then an exact two-sided rank test (Mann-Whitney U, by enumerating
every way of splitting the pooled runs into two groups of these sizes), so
with five runs a side the smallest possible p is 2/252 ≈ 0.008. "Survives
the spread" here means p < 0.05 AND the two ranges do not overlap; anything
else is reported as not separated.
"""

from __future__ import annotations

import argparse
import itertools
import json
import statistics
from pathlib import Path

HERE = Path(__file__).parent
KEY = {"public": "best_public", "held_out": "best_held_out"}


def champions(prefix: str, key: str) -> dict[str, float]:
    out = {}
    for d in sorted((HERE / "runs").glob(f"{prefix}[0-9]*")):
        m = d / "run_meta.json"
        if m.exists():
            v = json.loads(m.read_text()).get(key)
            r = d / "held_out_rescore.json"
            if v is None and key == "best_held_out" and r.exists():
                v = json.loads(r.read_text()).get("held_out")   # run's own scoring timed out
            if v is not None:
                out[d.name] = v
    return out


def u_stat(x, y):
    return sum((a > b) + 0.5 * (a == b) for a in x for b in y)


def exact_p(x, y):
    """Two-sided exact p for U, by enumerating all splits of the pooled data."""
    pooled, n = list(x) + list(y), len(x)
    mu = len(x) * len(y) / 2
    obs = abs(u_stat(x, y) - mu)
    hits = total = 0
    for idx in itertools.combinations(range(len(pooled)), n):
        s = set(idx)
        xx = [pooled[i] for i in idx]
        yy = [pooled[i] for i in range(len(pooled)) if i not in s]
        total += 1
        hits += abs(u_stat(xx, yy) - mu) >= obs - 1e-12
    return hits / total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("g1")
    ap.add_argument("g2")
    ap.add_argument("--set", choices=KEY, default="public")
    args = ap.parse_args()
    key = KEY[args.set]
    a, b = champions(args.g1, key), champions(args.g2, key)
    for g, c in ((args.g1, a), (args.g2, b)):
        v = sorted(c.values())
        print(f"{g}: n={len(v)}  " + "  ".join(f"{k} {x:+.5f}" for k, x in c.items()))
        print(f"   median {statistics.median(v):+.5f}  range [{v[0]:+.5f}, {v[-1]:+.5f}]")
    x, y = list(b.values()), list(a.values())
    p = exact_p(x, y)
    U = u_stat(x, y)
    overlap = not (min(x) > max(y) or min(y) > max(x))
    print(f"\n{args.g2} vs {args.g1} on {args.set}: U = {U:.1f} of {len(x) * len(y)} "
          f"(how often a {args.g2} champion beats a {args.g1} one), exact two-sided p = {p:.3f}, "
          f"ranges {'overlap' if overlap else 'do not overlap'}")
    print("verdict:", "the difference survives the spread"
          if p < 0.05 and not overlap else "not separated by the spread")


if __name__ == "__main__":
    main()

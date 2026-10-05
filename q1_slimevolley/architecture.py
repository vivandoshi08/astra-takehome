"""Describe a champion's network: node / connection counts, activations, wiring.

    python architecture.py runs/<run-id>            # prints and writes architecture.md

Counts both the genome as stored and its *effective* network (nodes and
connections that lie on a path from an input to an output — NEAT keeps
dangling hidden nodes that cannot affect any action).
"""

from __future__ import annotations

import argparse
import collections
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from evolve import load_config  # noqa: E402

INPUT_NAMES = "x y vx vy bx by bvx bvy ox oy ovx ovy bx-x by-y".split()
OUTPUT_NAMES = {0: "forward", 1: "backward", 2: "jump"}


def effective(genome) -> tuple[set, list]:
    """Nodes/connections on some input -> output path (enabled connections only)."""
    en = [c for c in genome.connections.values() if c.enabled]
    reach_out = set(OUTPUT_NAMES)
    changed = True
    while changed:  # walk backwards from outputs
        changed = False
        for c in en:
            if c.key[1] in reach_out and c.key[0] not in reach_out:
                reach_out.add(c.key[0])
                changed = True
    conns = [c for c in en if c.key[1] in reach_out]
    return {k for k in reach_out if k >= 0}, conns


def describe(run: Path) -> str:
    config = load_config(run / "config.ini")
    genome = pickle.loads((run / "champion.pkl").read_bytes())
    n_in = config.genome_config.num_inputs
    names = {-1 - i: n for i, n in enumerate(INPUT_NAMES[:n_in])}
    name = lambda k: names.get(k, OUTPUT_NAMES.get(k, f"h{k}"))  # noqa: E731
    enabled = [c for c in genome.connections.values() if c.enabled]
    hidden = [k for k in genome.nodes if k not in OUTPUT_NAMES]
    eff_nodes, eff_conns = effective(genome)
    used_inputs = sorted({c.key[0] for c in eff_conns if c.key[0] < 0}, reverse=True)
    acts = collections.Counter(genome.nodes[k].activation for k in genome.nodes)
    lines = [
        f"# Champion architecture: `{run.name}` (genome {genome.key})", "",
        "| | count |", "|---|---|",
        f"| inputs | {n_in} |",
        f"| nodes (stored) | {len(genome.nodes)} = {len(OUTPUT_NAMES)} output + {len(hidden)} hidden |",
        f"| nodes (effective, on an input->output path) | {len(eff_nodes)} |",
        f"| enabled connections (stored) | {len(enabled)} ({len(genome.connections) - len(enabled)} disabled) |",
        f"| enabled connections (effective) | {len(eff_conns)} |",
        f"| activation functions | {dict(acts)} |",
        f"| aggregation | {dict(collections.Counter(n.aggregation for n in genome.nodes.values()))} |",
        f"| network type | {'feed-forward' if config.genome_config.feed_forward else 'recurrent'} |", "",
        f"Inputs used (effective): {', '.join(name(k) for k in used_inputs)}",
        f"Inputs unused: {', '.join(n for k, n in names.items() if k not in used_inputs)}", "",
        "Node evaluation (neat-python): out = sigmoid(5 * (bias + response * sum(w * in)));"
        f" responses here: {sorted({round(n.response, 3) for n in genome.nodes.values()})}.", "",
        "| node | bias | inputs (weight), strongest first | on a path to an output? |",
        "|---|---|---|---|",
    ]
    for k in list(OUTPUT_NAMES) + sorted(hidden):
        ins = sorted(((c.key[0], c.weight) for c in enabled if c.key[1] == k), key=lambda t: -abs(t[1]))
        lines.append(f"| {name(k)} | {genome.nodes[k].bias:+.2f} | "
                     + (", ".join(f"{name(i)} ({w:+.2f})" for i, w in ins) or "—")
                     + f" | {'yes' if k in eff_nodes else 'no (dead end)'} |")
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run", type=Path)
    args = ap.parse_args()
    text = describe(args.run)
    (args.run / "architecture.md").write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

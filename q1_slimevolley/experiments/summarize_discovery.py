"""Summarise experiments/discovery_hypotheses.sh: did each arm escape the net camper?

    python experiments/summarize_discovery.py

Per run: final / peak population contacts, best phase-1 fitness, champion
validation score, and the largest |weight| on the ball-tracking input (bx - x)
in the end-of-run champion and generation-best. 'Discovered' = peak mean
contacts >= 5 (random play 1.4; seed 0 of the full run was at 8 by 20M and 4.5
by 15M; seeds that stayed campers never exceeded 3.2).
"""

from __future__ import annotations

import csv
import json
import pickle
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

BXX = -13  # input key of bx - x in 14-input networks
ARMS = ["control", "h1wide", "h2pop", "h3episodes", "h4nodelete", "h5species"]
DISCOVERED = 5.0


def tracking_weight(pkl: Path) -> float:
    if not pkl.exists():
        return float("nan")
    g = pickle.loads(pkl.read_bytes())
    ws = [abs(c.weight) for c in g.connections.values() if c.enabled and c.key[0] == BXX]
    return max(ws, default=0.0)


def summarize(run: Path) -> dict | None:
    hist = run / "fitness_history.csv"
    if not hist.exists():
        return None
    rows = list(csv.DictReader(hist.open()))
    contacts = [float(r["mean_contacts"]) for r in rows]
    end = run / "snapshots" / "end"
    return {
        "gens": len(rows),
        "steps": int(rows[-1]["cum_steps"]),
        "contacts_end": contacts[-1],
        "contacts_peak": max(contacts),
        "best_fit": max(float(r["best_fitness"]) for r in rows),
        "champ_val": json.loads((run / "champion.json").read_text())["val_mean"],
        "w_champ": tracking_weight(end / "champion.pkl"),
        "w_genbest": tracking_weight(end / "gen_best.pkl"),
        "species_mean": statistics.mean(float(r["n_species"]) for r in rows),
        "done": "end_time" in json.loads((run / "run_args.json").read_text()),
    }


def main() -> int:
    runs = HERE / "runs"
    print(f"{'run':24s} {'gens':>4s} {'steps':>6s} {'cont end/peak':>13s} {'best fit':>8s} "
          f"{'champ val':>9s} {'|w bx-x| champ/genbest':>22s} {'species':>7s}  discovered")
    lines = ["| arm | seed | gens | contacts end / peak | best phase-1 fitness | champion val "
             "| tracking weight champ / gen-best | mean species | discovered |",
             "|---|---|---|---|---|---|---|---|---|"]
    tally = {}
    for arm in ARMS:
        for seed in (10, 11, 12):
            s = summarize(runs / f"dh_{arm}_s{seed}")
            if s is None:
                continue
            found = s["contacts_peak"] >= DISCOVERED
            if s["done"]:
                tally.setdefault(arm, []).append(found)
            print(f"dh_{arm}_s{seed:<8} {s['gens']:4d} {s['steps'] / 1e6:5.1f}M "
                  f"{s['contacts_end']:6.1f}/{s['contacts_peak']:<6.1f} {s['best_fit']:8.2f} "
                  f"{s['champ_val']:+9.2f} {s['w_champ']:10.2f}/{s['w_genbest']:<11.2f} "
                  f"{s['species_mean']:7.1f}  {'YES' if found else 'no'}{'' if s['done'] else ' (running)'}")
            lines.append(f"| {arm} | {seed} | {s['gens']} | {s['contacts_end']:.1f} / "
                         f"{s['contacts_peak']:.1f} | {s['best_fit']:.2f} | {s['champ_val']:+.2f} | "
                         f"{s['w_champ']:.2f} / {s['w_genbest']:.2f} | {s['species_mean']:.1f} | "
                         f"{'yes' if found else 'no'} |")
    print("\nDiscovered (peak mean contacts >= 5) per arm, finished runs:")
    for arm, found in tally.items():
        print(f"  {arm:11s} {sum(found)}/{len(found)}")
    (HERE / "experiments" / "discovery_results.md").write_text(
        "# Discovery hypotheses: results\n\n" + "\n".join(lines) + "\n\n"
        + "\n".join(f"- {arm}: {sum(f)}/{len(f)} discovered" for arm, f in tally.items()) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

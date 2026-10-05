"""Replay every champion snapshot of a run on the same held-out seeds.

    python replay_snapshots.py runs/<run-id>                 # all snapshots + final
    python replay_snapshots.py runs/<run-id> --episodes 20 --gifs 2

evolve.py saves runs/<id>/snapshots/<N>M/ every --snapshot-every steps:
`champion.pkl` (best validation mean so far) and `gen_best.pkl` (this
generation's best validated genome — differs from the champion when the
champion is old). For each snapshot this writes GIFs + contact sheets on the
first --gifs held-out seeds (identical seeds across snapshots, so frames are
directly comparable) and scores it on --episodes held-out seeds, then writes
snapshots/progress.md: score, contacts, returns, survival, timeouts and the
button pattern per snapshot, so learning — or a hack — is visible step by step.
"""

from __future__ import annotations

import argparse
import json
import pickle
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import slimevolley as sv  # noqa: E402
from evaluate import heldout_seeds, summarize  # noqa: E402
from evolve import load_config, make_net, play_seeds  # noqa: E402
from replay import record  # noqa: E402


def snapshot_dirs(run: Path) -> list[tuple[str, Path]]:
    snaps = sorted((run / "snapshots").glob("*M"), key=lambda p: float(p.name[:-1]))
    if (run / "snapshots" / "end").exists():
        snaps.append(run / "snapshots" / "end")
    out = [(p.name, p) for p in snaps]
    if (run / "champion.pkl").exists():
        out.append(("final", run))
    return out


def stats(genome, config, seeds) -> dict:
    eps = play_seeds(make_net(genome, config), sv.SlimeVolleyEnv(), seeds)
    s = summarize([e["score"] for e in eps])
    total = sum(e["steps"] for e in eps)
    return {
        "mean": s["mean"], "ci": s["ci95_t"],
        "steps": statistics.mean(e["steps"] for e in eps),
        "timeouts": sum(e["steps"] >= 3000 for e in eps),
        "contacts": statistics.mean(e["contacts"] for e in eps),
        "returns": statistics.mean(e["returns"] for e in eps),
        **{k: sum(e[k] for e in eps) / total for k in ("act_fwd", "act_back", "act_jump", "act_both")},
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run", type=Path)
    ap.add_argument("--episodes", type=int, default=20, help="held-out episodes scored per snapshot")
    ap.add_argument("--gifs", type=int, default=2, help="held-out seeds rendered per snapshot")
    args = ap.parse_args()

    config = load_config(args.run / "config.ini")
    seeds = heldout_seeds(args.episodes)
    lines = [f"# Snapshot progress: `{args.run.name}`", "",
             f"Each row: genome scored on held-out seeds {seeds[0]}..{seeds[-1]} "
             f"(same seeds for every row). GIFs/sheets use the first {args.gifs}.", "",
             "| snapshot | which | gen | val mean | held-out mean [95% CI] | steps | timeouts "
             "| contacts | returns | fwd | back | jump | both | replays |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for label, d in snapshot_dirs(args.run):
        meta = json.loads((d / "champion.json").read_text()) if (d / "champion.json").exists() else {}
        for which, pkl in (("champion", d / "champion.pkl"), ("gen best", d / "gen_best.pkl")):
            if not pkl.exists():
                continue
            genome = pickle.loads(pkl.read_bytes())
            out = args.run / "snapshots" / "replays"
            out.mkdir(parents=True, exist_ok=True)
            links = []
            for seed in seeds[:args.gifs]:
                stem = f"{label}_{which.replace(' ', '_')}_{seed}"
                net = make_net(genome, config)
                if hasattr(net, "reset"):
                    net.reset()
                r = record(net, str(out / f"{stem}.gif"), seed=seed, sheet=str(out / f"{stem}.png"))
                links.append(f"[{seed}](replays/{stem}.gif) ({r['score']:+.0f})")
            st = stats(genome, config, seeds)
            m = meta if which == "champion" else json.loads((d / "gen_best.json").read_text()) \
                if (d / "gen_best.json").exists() else {}
            row = (f"| {label} | {which} | {m.get('generation', '')} | "
                   f"{m.get('val_mean', float('nan')):+.2f} | {st['mean']:+.2f} "
                   f"[{st['ci'][0]:+.2f}, {st['ci'][1]:+.2f}] | {st['steps']:.0f} | {st['timeouts']} | "
                   f"{st['contacts']:.1f} | {st['returns']:.1f} | {st['act_fwd']:.2f} | "
                   f"{st['act_back']:.2f} | {st['act_jump']:.2f} | {st['act_both']:.2f} | "
                   + " ".join(links) + " |")
            lines.append(row)
            print(row, flush=True)
    lines += ["", "Reference (reward_check.py probe seeds): random 1.4 contacts / 0.1 returns; "
              "hand-written tracker 13.6 / 3.0; built-in policy 45 / 24."]
    (args.run / "snapshots").mkdir(exist_ok=True)
    (args.run / "snapshots" / "progress.md").write_text("\n".join(lines) + "\n")
    print(f"\nwrote {args.run / 'snapshots' / 'progress.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

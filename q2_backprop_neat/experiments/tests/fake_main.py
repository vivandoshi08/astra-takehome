"""Stand-in for main.py, used only by test_harness.py. Never produces results.

Honours main.py's output contract (writes into --run-id; fitness_history.csv,
champion.pkl, summary.json; exit 0 ok, 4 over budget, 2 usage error) and
misbehaves on request:

    --mode ok | over-budget | traceback | hang | random | no-champion | nan

FAKE_DEFAULT_STEPS in the environment changes the --steps default, to
simulate main.py's defaults changing between a run and its resume.
"""

import argparse
import json
import os
import pickle
import random
import sys
import time
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", required=True)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--run-id", required=True)
ap.add_argument("--quiet", action="store_true")
ap.add_argument("--overwrite", action="store_true")
ap.add_argument("--steps", type=int, default=int(os.environ.get("FAKE_DEFAULT_STEPS", 10)))
ap.add_argument("--budget", type=int, default=30000)
ap.add_argument("--mode", default="ok", choices=["ok", "over-budget", "traceback", "hang",
                                                  "random", "no-champion", "nan"])
ap.add_argument("--val", type=float, default=None, help="emit val_acc = val + 0.01 * seed")
ap.add_argument("--heldout", type=float, default=0.7, help="heldout_acc = this + 0.01 * seed")
ap.add_argument("--extra", type=float, default=None, help="emit an extra numeric field")
args = ap.parse_args()

run_dir = Path(args.run_id)
if (run_dir / "fitness_history.csv").exists() and not args.overwrite:
    ap.error(f"{run_dir} already holds a run")
run_dir.mkdir(parents=True, exist_ok=True)
(run_dir / "fitness_history.csv").write_text("generation,best_fitness,mean_fitness\n0,0.5,0.4\n")
print(f"fake run {args.dataset} seed {args.seed} mode {args.mode}", flush=True)

if args.mode == "hang":
    time.sleep(3600)
if args.mode == "traceback":
    raise RuntimeError("deliberate crash after fitness_history.csv was written")

over = args.mode == "over-budget"
spent = args.budget + 10 if over else args.steps * 100
summary = {
    "args": vars(args), "git_commit": "fake", "seconds": round(random.random(), 3),
    "budget": f"{spent}/{args.budget}", "budget_spent": spent, "budget_limit": args.budget,
    "within_budget": not over,
    "budget_error": f"gradient-step budget exceeded: {spent} / {args.budget}" if over else None,
}
if args.mode != "no-champion":
    heldout = random.random() if args.mode == "random" else args.heldout + 0.01 * args.seed
    summary.update({"train_acc": 0.9, "heldout_acc": heldout, "nodes": 3, "conns": 5})
    with (run_dir / "champion.pkl").open("wb") as fh:
        pickle.dump({"fake": True}, fh)
if args.val is not None:
    summary["val_acc"] = args.val + 0.01 * args.seed
if args.mode == "nan":
    summary["val_acc"] = float("nan")
if args.extra is not None:
    summary["my_extra_metric"] = args.extra
(run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
if over:
    print("BUDGET EXCEEDED — this run is invalid")
sys.exit(4 if over else 0)

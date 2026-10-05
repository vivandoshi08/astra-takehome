"""Re-score a run's champion on HELD_OUT when the run's own one-off scoring
timed out (wall-clock limit under machine load). Same sandbox and
evaluator, longer limit; the result goes to runs/<id>/held_out_rescore.json
and the run's own logs are left as written.

    python rescore_held_out.py l01 l02 --timeout 1200
"""
import argparse
import json
import time
from pathlib import Path

import data
from evo import loop
from tasks import ellipses

HERE = Path(__file__).parent

ap = argparse.ArgumentParser()
ap.add_argument("runs", nargs="+")
ap.add_argument("--timeout", type=float, default=1200.0)
args = ap.parse_args()
for run in args.runs:
    d = HERE / "runs" / run
    code = (d / "best.py").read_text()
    t = time.time()
    s, status, detail = loop.score_candidate(ellipses, code, True, args.timeout,
                                             images=data.HELD_OUT)
    rec = {"held_out": s if status == "scored" else None, "status": status,
           "detail": detail, "timeout": args.timeout,
           "seconds": round(time.time() - t, 1),
           "why": "the run's own HELD_OUT scoring hit the 180 s limit under load"}
    (d / "held_out_rescore.json").write_text(json.dumps(rec, indent=2) + "\n")
    print(run, rec)

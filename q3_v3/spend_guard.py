"""Stop running loop runs cleanly before the credit runs out (stop rule 4).

    python spend_guard.py --limit 9.50 &

Every 15 s it prices every runs/*/token_log.jsonl exactly as spend.py does
and, once the total reaches --limit, sends SIGTERM to every running
`evo.loop --client llm` process. evo.loop turns SIGTERM into a clean stop:
the response in hand is logged, best.py and run_meta.json are written, and
stop_reason says "interrupted". Exits when no loop process is left.

A call already in flight when the signal lands may be billed by the
provider without reaching token_log.jsonl; at ~$0.01 a call that is the
remaining gap, which is why the limit sits below $10.
"""

from __future__ import annotations

import argparse
import subprocess
import time
from pathlib import Path

import spend

HERE = Path(__file__).parent
PATTERN = "evo.loop --client llm"


def total_spend() -> float:
    total = 0.0
    for log in (HERE / "runs").glob("*/token_log.jsonl"):
        calls, _ = spend.read_calls(log)
        for c in calls:
            m = spend.model_of(c.get("tag", ""), spend.DEFAULT_MODEL)
            if m == "mock":
                continue
            rin, rout = spend.RATES[m]
            total += c["input_tokens"] / 1e6 * rin + c["output_tokens"] / 1e6 * rout
    return total


def running() -> bool:
    return subprocess.run(["pgrep", "-f", PATTERN],
                          capture_output=True).returncode == 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=float, default=9.50)
    ap.add_argument("--every", type=float, default=15.0)
    args = ap.parse_args()
    fired = False
    while running():
        t = total_spend()
        if t >= args.limit and not fired:
            print(f"[guard] ${t:.2f} >= ${args.limit:.2f}: stopping loop runs",
                  flush=True)
            subprocess.run(["pkill", "-TERM", "-f", PATTERN])
            fired = True
        time.sleep(args.every)
    print(f"[guard] no loop runs left; total ${total_spend():.2f}", flush=True)


if __name__ == "__main__":
    main()

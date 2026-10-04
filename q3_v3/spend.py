"""What your runs have cost so far.

    python spend.py                                   # prices every run in runs/
    python spend.py --model anthropic/claude-sonnet-5 # if you switched models

Reads the `token_log.jsonl` each run writes and prices it. Nothing here talks
to the network, and nothing here is enforcement — the hard cap lives in
`evo/budget.py`. This is just so you can see the bill.

Rates are USD per million tokens, as listed on OpenRouter. If you used a model
that isn't here, add it to RATES or pass --in-rate / --out-rate.

If you compare this against your balance on openrouter.ai, give it a minute:
calls settle asynchronously, so straight after a run their meter reads low.
It catches up, and then the two agree exactly.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

CREDIT = 10.00

RATES = {                                   # slug: (input $/Mtok, output $/Mtok)
    "anthropic/claude-haiku-4.5": (1.0, 5.0),
    "anthropic/claude-sonnet-5": (2.0, 10.0),
    "anthropic/claude-opus-5": (5.0, 25.0),
    "openai/gpt-4o-mini": (0.15, 0.60),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs", help="directory of run dirs")
    ap.add_argument("--model", default="anthropic/claude-haiku-4.5")
    ap.add_argument("--in-rate", type=float, default=None)
    ap.add_argument("--out-rate", type=float, default=None)
    args = ap.parse_args()

    if args.in_rate is not None and args.out_rate is not None:
        in_rate, out_rate = args.in_rate, args.out_rate
    elif args.model in RATES:
        in_rate, out_rate = RATES[args.model]
    else:
        raise SystemExit(f"unknown model {args.model!r}; add it to RATES in "
                         f"spend.py or pass --in-rate and --out-rate")

    logs = sorted(Path(args.runs).glob("*/token_log.jsonl"))
    if not logs:
        raise SystemExit(f"no token logs under {args.runs}/ — nothing to price")

    print(f"{'run':<16} {'calls':>6} {'in':>10} {'out':>9} {'$':>7}")
    total = 0.0
    for log in logs:
        calls = [json.loads(line) for line in log.read_text().splitlines() if line]
        tin = sum(c["input_tokens"] for c in calls)
        tout = sum(c["output_tokens"] for c in calls)
        cost = tin / 1e6 * in_rate + tout / 1e6 * out_rate
        total += cost
        print(f"{log.parent.name:<16} {len(calls):>6} {tin:>10,} {tout:>9,} "
              f"{cost:>7.2f}")

    print(f"{'':<16} {'':>6} {'':>10} {'total':>9} {total:>7.2f}")
    print(f"\n${total:.2f} of ${CREDIT:.2f} spent, ${CREDIT - total:.2f} left "
          f"(priced at {args.model} rates; mock runs are free and log nothing).")


if __name__ == "__main__":
    main()

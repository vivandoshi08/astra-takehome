"""What your runs have cost so far.

    python spend.py                       # prices every run in runs/
    python spend.py --model SLUG --in-rate 3 --out-rate 15   # a model not in RATES

Reads the `token_log.jsonl` each run writes and prices it. Nothing here talks
to the network, and nothing here is enforcement — the hard cap lives in
`evo/budget.py`. This is just so you can see the bill.

Every call is priced by the model recorded in its own tag
(`openrouter:<slug>|gen3`), so runs on different models are each priced at
their own rate. Mock calls (`mock|gen3`) are free and listed as such. The
shipped version priced every call at one `--model` rate and counted mock
runs as spend (AUDIT.md B3).

Rates are USD per million tokens, as listed on OpenRouter. A model that isn't
in RATES is an error, not a guess: add it, or pass --model with --in-rate and
--out-rate.

If you compare this against your balance on openrouter.ai, give it a minute:
calls settle asynchronously, so straight after a run their meter reads low.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

CREDIT = 10.00

RATES = {                                   # slug: (input $/Mtok, output $/Mtok)
    "anthropic/claude-haiku-4.5": (1.0, 5.0),
    "anthropic/claude-sonnet-5": (2.0, 10.0),
    "anthropic/claude-opus-5": (5.0, 25.0),
    "openai/gpt-4o-mini": (0.15, 0.60),
}
DEFAULT_MODEL = "anthropic/claude-haiku-4.5"


def model_of(tag: str, fallback: str) -> str:
    """'openrouter:<slug>|gen3' -> slug; 'mock|gen3' -> 'mock'; an untagged
    record -> fallback."""
    head = (tag or "").split("|", 1)[0]
    if head == "mock":
        return "mock"
    if head.startswith("openrouter:"):
        return head[len("openrouter:"):]
    return fallback


def read_calls(log: Path) -> tuple[list[dict], int]:
    """The parsed records, and how many lines could not be parsed (a run
    killed mid-write can leave a cut-off last line)."""
    calls, bad = [], 0
    for line in log.read_text().splitlines():
        if not line.strip():
            continue
        try:
            calls.append(json.loads(line))
        except json.JSONDecodeError:
            bad += 1
    return calls, bad


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=str(Path(__file__).parent / "runs"),
                    help="directory of run dirs (default: runs/ next to "
                         "this file)")
    ap.add_argument("--model", default=DEFAULT_MODEL,
                    help="model for records whose tag names none; with "
                         "--in-rate/--out-rate, the rate for this model")
    ap.add_argument("--in-rate", type=float, default=None)
    ap.add_argument("--out-rate", type=float, default=None)
    args = ap.parse_args()

    rates = dict(RATES)
    if (args.in_rate is None) != (args.out_rate is None):
        raise SystemExit("pass both --in-rate and --out-rate, or neither")
    if args.in_rate is not None:
        rates[args.model] = (args.in_rate, args.out_rate)

    logs = sorted(Path(args.runs).glob("*/token_log.jsonl"))
    if not logs:
        raise SystemExit(f"no token logs under {args.runs}/ — nothing to price")

    rows, unknown, warnings = [], set(), []
    for log in logs:
        calls, bad = read_calls(log)
        if bad:
            warnings.append(f"{log.parent.name}: skipped {bad} unparseable "
                            f"line(s)")
        by_model: dict[str, list[int]] = {}
        for c in calls:
            m = model_of(c.get("tag", ""), args.model)
            t = by_model.setdefault(m, [0, 0, 0])
            t[0] += 1
            t[1] += int(c["input_tokens"])
            t[2] += int(c["output_tokens"])
            if "usage-estimated" in c.get("tag", ""):
                warnings.append(f"{log.parent.name}: call {c.get('call')} "
                                f"has estimated usage (provider sent none)")
        for m, (n, tin, tout) in sorted(by_model.items()):
            if m == "mock":
                cost = 0.0
            elif m in rates:
                cost = tin / 1e6 * rates[m][0] + tout / 1e6 * rates[m][1]
            else:
                unknown.add(m)
                cost = float("nan")
            rows.append((log.parent.name, m, n, tin, tout, cost))

    if unknown:
        raise SystemExit(f"no rate for {sorted(unknown)}: add to RATES in "
                         f"spend.py, or pass --model SLUG --in-rate --out-rate")

    print(f"{'run':<16} {'model':<28} {'calls':>6} {'in':>10} {'out':>9} "
          f"{'$':>7}")
    total = 0.0
    for run, m, n, tin, tout, cost in rows:
        total += cost
        label = "mock (free)" if m == "mock" else m
        print(f"{run:<16} {label:<28} {n:>6} {tin:>10,} {tout:>9,} "
              f"{cost:>7.2f}")
    print(f"{'':<16} {'':<28} {'':>6} {'':>10} {'total':>9} {total:>7.2f}")
    for w in warnings:
        print(f"  note: {w}", file=sys.stderr)
    print(f"\n${total:.2f} of ${CREDIT:.2f} spent, ${CREDIT - total:.2f} left "
          f"(each call priced by the model in its tag; mock calls are free).")


if __name__ == "__main__":
    main()

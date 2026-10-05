"""Pre-flight check of the real API before any loop run (plan step 1).

    python probe_api.py --run-id probe-01                  # 2 real calls, ~$0.01
    python probe_api.py --run-id probe-mock --client mock  # free plumbing check

Sends the loop's real generation-0 prompt (the SEED alone in the
population) twice through `evo.llm_client.LLMClient`: once as shipped, once
with reasoning effort "low". Every call is charged through
`evo.budget.TokenBudget` into `runs/<run-id>/token_log.jsonl`, so `spend.py`
prices it like any run. For each call it records finish_reason, the
provider's raw usage (reasoning tokens, if any), whether a code block came
back, and that code's score in the sandbox. It also reads the key's own
usage meter from OpenRouter (free) before and after, for comparison with
`spend.py`.

Questions it answers (AUDIT.md F1, B5): does Haiku 4.5 accept a reasoning
setting, and does reasoning eat into MAX_TOKENS = 2048; how long are real
answers; what does one call cost.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.request
from pathlib import Path

from evo import loop
from evo.budget import TokenBudget
from tasks import ellipses

HERE = Path(__file__).parent
VARIANTS = [("default", None), ("reasoning-low", "low")]
RATES = {"anthropic/claude-haiku-4.5": (1.0, 5.0)}


def key_usage():
    """The key's spend so far according to OpenRouter (free endpoint)."""
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        return None
    req = urllib.request.Request("https://openrouter.ai/api/v1/key",
                                 headers={"Authorization": f"Bearer {key}"})
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            return json.load(r)["data"]["usage"]
    except Exception as exc:
        return f"unavailable: {exc!r}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--client", choices=("mock", "llm"), default="llm")
    args = ap.parse_args()

    run_dir = HERE / "runs" / args.run_id
    if run_dir.exists() and any(run_dir.iterdir()):
        print(f"refusing to reuse {run_dir}", file=sys.stderr)
        return 2
    run_dir.mkdir(parents=True, exist_ok=True)

    if args.client == "mock":
        from evo.mock_client import MockClient
        client = MockClient()
    else:
        from evo.llm_client import LLMClient
        client = LLMClient()

    seed_score, _, _ = loop.score_candidate(ellipses, ellipses.SEED)
    prompt = loop.build_prompt(ellipses, [(seed_score, ellipses.SEED)])
    budget = TokenBudget(limit=50_000, log_path=run_dir / "token_log.jsonl")
    before = key_usage() if args.client == "llm" else None

    rows = []
    for label, effort in VARIANTS:
        client.reasoning = effort
        t = time.time()
        rec = {"variant": label, "reasoning": effort, "prompt_chars": len(prompt)}
        try:
            r = client.complete(prompt, n=1)[0]
        except Exception as exc:
            rec.update(status="api_error", detail=repr(exc)[:500])
            rows.append(rec)
            continue
        budget.charge(r.input_tokens, r.output_tokens,
                      tag=f"{client.name}|probe-{label}")
        code = loop.extract_code(r.text)
        s, status, detail = ((float("-inf"), "no_code", "") if code is None
                             else loop.score_candidate(ellipses, code))
        usage = getattr(r, "usage", None) or {}
        details = usage.get("completion_tokens_details") or {}
        rec.update(status=status, score=s if s > float("-inf") else None,
                   detail=detail, seconds=round(time.time() - t, 1),
                   finish_reason=getattr(r, "finish_reason", None),
                   input_tokens=r.input_tokens, output_tokens=r.output_tokens,
                   reasoning_tokens=details.get("reasoning_tokens"),
                   usage=usage, text=r.text)
        rows.append(rec)

    with (run_dir / "responses.jsonl").open("w") as fh:
        for rec in rows:
            fh.write(json.dumps(rec, default=str) + "\n")

    time.sleep(5 if args.client == "llm" else 0)   # let the provider's meter settle a little
    after = key_usage() if args.client == "llm" else None
    meta = {"client": client.name, "model": getattr(client, "model", None),
            "max_tokens": getattr(client, "max_tokens", None),
            "seed_score": seed_score, "provider_usage_before": before,
            "provider_usage_after": after, "budget": budget.summary()}
    (run_dir / "run_meta.json").write_text(json.dumps(meta, indent=2) + "\n")

    rin, rout = RATES.get(getattr(client, "model", ""), (0.0, 0.0))
    print(f"{'variant':<14} {'status':<12} {'finish':<8} {'in':>6} {'out':>6} "
          f"{'reason':>6} {'secs':>5} {'score':>8} {'$':>7}")
    for rec in rows:
        if rec["status"] == "api_error":
            print(f"{rec['variant']:<14} api_error    {rec['detail'][:70]}")
            continue
        cost = rec["input_tokens"] / 1e6 * rin + rec["output_tokens"] / 1e6 * rout
        sc = f"{rec['score']:+.4f}" if rec["score"] is not None else "-"
        print(f"{rec['variant']:<14} {rec['status']:<12} "
              f"{str(rec['finish_reason']):<8} {rec['input_tokens']:>6} "
              f"{rec['output_tokens']:>6} {str(rec['reasoning_tokens']):>6} "
              f"{rec['seconds']:>5} {sc:>8} {cost:>7.4f}")
    print(f"\nbudget: {budget.summary()}")
    if args.client == "llm":
        print(f"OpenRouter meter: before {before}, after {after} "
              f"(settles asynchronously; re-check in a minute)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

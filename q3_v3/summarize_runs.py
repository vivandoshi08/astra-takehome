"""Summarise loop runs: champion scores, cost, failure modes, prompt diversity.

    python summarize_runs.py a01 a02 ...        # run ids under runs/
    python summarize_runs.py --group a          # every runs/a* directory

Per run: champion on PUBLIC and HELD_OUT (both from run_meta.json, HELD_OUT
scored once after the run), cost from token_log.jsonl at the model's rate,
answer outcomes from responses.jsonl (scored / no code / failed, and how
many were cut off at MAX_TOKENS), and two measures used to choose group B's
one change:

  dup3   generations whose prompt showed fewer than 3 distinct programs
         (the population is rebuilt from run_log.jsonl exactly as the loop
         keeps it: the top 8 of everything scored so far, stable order)
  flat   generations whose best, printed to 4 decimals as the prompt shows
         it, did not change from the previous generation

Then the group: each champion, median, range, spread.
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path

HERE = Path(__file__).parent
from spend import RATES  # one price table


def jsonl(p: Path):
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()] \
        if p.exists() else []


def norm(code: str) -> str:
    return "\n".join(l.rstrip() for l in code.strip().splitlines() if l.strip())


def summarise(run_id: str) -> dict:
    d = HERE / "runs" / run_id
    meta = json.loads((d / "run_meta.json").read_text())
    calls = jsonl(d / "token_log.jsonl")
    rin, rout = RATES.get(meta.get("model"), (0.0, 0.0))
    tin = sum(c["input_tokens"] for c in calls)
    tout = sum(c["output_tokens"] for c in calls)
    resp = jsonl(d / "responses.jsonl")
    status = Counter(r["status"] for r in resp)
    cut = sum(1 for r in resp if r.get("finish_reason") == "length")

    gens = [g for g in jsonl(d / "run_log.jsonl") if g["generation"] >= 0]
    seed = [g for g in jsonl(d / "run_log.jsonl") if g["generation"] == -1]
    pool = [(c["score"], c["code"]) for g in seed for c in g["candidates"]]
    dup3, flat, curve, prev = 0, 0, [], None
    for g in gens:
        top3 = sorted(pool, key=lambda t: -t[0])[:3]       # what this gen's prompt showed
        if len({norm(c) for _, c in top3}) < min(3, len(top3)) or len(top3) < 3:
            dup3 += 1
        pool += [(c["score"], c["code"]) for c in g["candidates"]]
        pool = sorted(pool, key=lambda t: -t[0])[:8]
        shown = f"{g['best']:.4f}"
        flat += prev == shown
        prev = shown
        curve.append(g["best"])
    return {"run": run_id, "model": meta.get("model"),
            "reasoning": meta.get("reasoning"),
            "public": meta.get("best_public"), "held_out": meta.get("best_held_out"),
            "flags": meta.get("champion_flags"), "stop": meta.get("stop_reason"),
            "calls": len(calls), "cost": tin / 1e6 * rin + tout / 1e6 * rout,
            "out_per_call": tout / max(len(calls), 1),
            "scored": status.get("scored", 0), "answers": len(resp),
            "status": dict(status), "cut": cut,
            "dup3": dup3, "flat": flat, "gens": len(gens), "curve": curve}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="*")
    ap.add_argument("--group")
    args = ap.parse_args()
    ids = list(args.runs)
    if args.group:
        ids += sorted(p.name for p in (HERE / "runs").glob(f"{args.group}*")
                      if (p / "run_meta.json").exists())
    rows = [summarise(r) for r in ids]

    print(f"{'run':<6} {'PUBLIC':>9} {'HELD_OUT':>9} {'gap':>8} {'$':>5} "
          f"{'valid':>6} {'cut':>4} {'dup3':>5} {'flat':>5}  failures")
    for r in rows:
        gap = (r["held_out"] - r["public"]) if r["held_out"] is not None else float("nan")
        fails = {k: v for k, v in r["status"].items() if k != "scored"}
        print(f"{r['run']:<6} {r['public']:+9.5f} {r['held_out']:+9.5f} {gap:+8.5f} "
              f"{r['cost']:5.2f} {r['scored']:>3}/{r['answers']:<2} {r['cut']:>4} "
              f"{r['dup3']:>2}/{r['gens']:<2} {r['flat']:>2}/{r['gens']:<2}  "
              f"{fails or '-'}{'  FLAGS ' + str(r['flags']) if r['flags'] else ''}"
              f"{'  STOP ' + r['stop'] if r['stop'] else ''}")
    if len(rows) > 1:
        for key in ("public", "held_out"):
            v = sorted(r[key] for r in rows)
            print(f"\n{key:9s} values {[round(x, 5) for x in v]}")
            print(f"{'':9s} median {statistics.median(v):+.5f}  range "
                  f"[{v[0]:+.5f}, {v[-1]:+.5f}]  spread {v[-1] - v[0]:.5f}")
        print(f"\ncost total ${sum(r['cost'] for r in rows):.2f}, "
              f"mean ${statistics.mean(r['cost'] for r in rows):.3f}/run")
    print("\nbest-so-far by generation:")
    for r in rows:
        print(f"  {r['run']:<6} " + " ".join(f"{x:+.4f}" for x in r["curve"]))


if __name__ == "__main__":
    main()

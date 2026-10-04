"""An LLM-as-optimizer loop, in the FunSearch / OpenEvolve / OPRO lineage.

Each generation: show the model the best candidates so far and their scores,
ask for new ones, evaluate them, keep the best. Per-generation aggregates and
every scored candidate are appended to `run_log.jsonl`.

    python -m evo.loop --client mock --generations 3
    python -m evo.loop --client llm  --generations 12

Start with `--client mock`: it runs the whole loop offline, for free.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

from evo.budget import BudgetExhausted, TokenBudget

HERE = Path(__file__).parent.parent
CODE_BLOCK = re.compile(r"```(?:python)?\s*(.*?)```", re.S)


def extract_code(text: str) -> str | None:
    m = CODE_BLOCK.search(text)
    return m.group(1).strip() if m else None


def compile_candidate(code: str, entry="heuristic", isolated: bool = False):
    """Turn candidate source into callables, or None if it won't compile.

    `entry` may be one function name or a tuple of names (the ellipses task
    uses ("encode", "decode")). One namespace for globals AND locals: with two
    dicts, a module-level `import math` binds into locals while the function's
    __globals__ is the other dict, so `math` is undefined at call time.

    With `isolated=True` (the ellipses task sets ISOLATED_NAMESPACES), each
    entry point is compiled from the source into its own fresh namespace.
    That makes the latent code the only channel between encode and decode:
    module-level state written by one is invisible to the other.
    """
    names = (entry,) if isinstance(entry, str) else tuple(entry)

    def load() -> dict | None:
        ns: dict = {"__builtins__": __builtins__}
        try:
            exec(compile(code, "<candidate>", "exec"), ns)
        except Exception:
            return None
        return ns

    if isolated:
        fns = []
        for name in names:
            ns = load()
            if ns is None:
                return None
            f = ns.get(name)
            if not callable(f):
                return None
            fns.append(f)
        return fns[0] if isinstance(entry, str) else tuple(fns)

    ns = load()
    if ns is None:
        return None
    fns = tuple(ns.get(name) for name in names)
    if not all(callable(f) for f in fns):
        return None
    return fns[0] if isinstance(entry, str) else fns


def build_prompt(task, population: list[tuple[float, str]], k: int = 3) -> str:
    """Show the top-k candidates with their scores and ask for a better one."""
    parts = [task.prompt_header()]
    if population:
        parts.append("\nHere are the best programs so far, with their scores "
                     "(higher is better):\n")
        for score, code in population[:k]:
            parts.append(f"--- score {score:.4f} ---\n```python\n{code}\n```\n")
        parts.append("\nWrite ONE new program that scores higher. "
                     "Think about why the best ones work.")
    else:
        parts.append("\nWrite your first attempt.")
    return "\n".join(parts)


def run(task="ellipses", client=None, generations: int = 12,
        candidates_per_gen: int = 4, population_size: int = 8,
        budget: TokenBudget | None = None, log_path=None, quiet: bool = False):
    """Run the loop. Returns (best_score, best_code, history)."""
    if isinstance(task, str):
        import importlib
        task = importlib.import_module(f"tasks.{task}")
    if client is None:
        from evo.mock_client import MockClient
        client = MockClient()
    if budget is None:
        budget = TokenBudget()

    log = Path(log_path) if log_path else None
    if log:
        log.parent.mkdir(parents=True, exist_ok=True)

    def record(**kw):
        if log:
            with log.open("a") as fh:
                fh.write(json.dumps({"wall_ts": time.time(), **kw}) + "\n")

    population: list[tuple[float, str]] = []
    history: list[dict] = []

    # Seed program, FunSearch-style: a minimal working candidate the model
    # can see and improve from generation 0. Costs no tokens.
    seed_code = getattr(task, "SEED", None)
    if seed_code:
        fns = compile_candidate(seed_code, getattr(task, "ENTRY_POINT",
                                                   "heuristic"),
                                isolated=getattr(task, "ISOLATED_NAMESPACES",
                                                 False))
        if fns is not None:
            s = task.evaluate(fns)
            if s > float("-inf"):
                population.append((s, seed_code))
                record(generation=-1, best=s, valid_this_gen=0, tokens=0,
                       candidates=[{"score": s, "code": seed_code}])
                if not quiet:
                    print(f"  seed     score {s:+.4f}", flush=True)

    for gen in range(generations):
        prompt = build_prompt(task, population)
        try:
            responses = client.complete(prompt, n=candidates_per_gen)
        except BudgetExhausted:
            raise
        for r in responses:
            # Tag with the client so a mock run's free tokens are never
            # mistaken for real spend when you add up what a run cost.
            budget.charge(r.input_tokens, r.output_tokens,
                          tag=f"{getattr(client, 'name', '?')}|gen{gen}")

        scored_this_gen = []
        for r in responses:
            code = extract_code(r.text)
            if code is None:
                continue
            fns = compile_candidate(code, getattr(task, "ENTRY_POINT", "heuristic"),
                                    isolated=getattr(task, "ISOLATED_NAMESPACES",
                                                     False))
            if fns is None:
                continue
            score = task.evaluate(fns)
            if score == float("-inf"):
                continue
            scored_this_gen.append((score, code))
            population.append((score, code))

        population.sort(key=lambda t: -t[0])
        del population[population_size:]

        best = population[0][0] if population else float("-inf")
        entry = {"generation": gen, "best": best,
                 "valid_this_gen": len(scored_this_gen),
                 "tokens": budget.total_tokens}
        history.append(entry)
        record(**entry, prompt_chars=len(prompt),
               candidates=[{"score": s, "code": c} for s, c in scored_this_gen])
        if not quiet:
            print(f"  gen {gen:3d}  best {best:+.4f}  "
                  f"valid {len(scored_this_gen)}/{candidates_per_gen}  "
                  f"tokens {budget.total_tokens:,}", flush=True)

    if not population:
        return float("-inf"), None, history
    return population[0][0], population[0][1], history


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--task", default="ellipses")
    ap.add_argument("--client", choices=("mock", "llm"), default="mock")
    ap.add_argument("--generations", type=int, default=12)
    ap.add_argument("--candidates", type=int, default=4)
    ap.add_argument("--cap", type=int, default=500_000)
    ap.add_argument("--run-id", default="run")
    args = ap.parse_args()

    run_dir = HERE / "runs" / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    if args.client == "mock":
        from evo.mock_client import MockClient
        client = MockClient()
    else:
        from evo.llm_client import LLMClient
        client = LLMClient()

    budget = TokenBudget(limit=args.cap, log_path=run_dir / "token_log.jsonl")
    print(f"task={args.task} client={client.name} cap={args.cap:,}")

    import importlib
    task = importlib.import_module(f"tasks.{args.task}")
    if hasattr(task, "baseline_scores"):
        print("baselines:", {k: round(v, 4)
                             for k, v in task.baseline_scores().items()})

    try:
        best, code, _ = run(task=task, client=client,
                            generations=args.generations,
                            candidates_per_gen=args.candidates,
                            budget=budget,
                            log_path=run_dir / "run_log.jsonl")
    except BudgetExhausted as exc:
        print(f"\n[budget] {exc}")
        return 0

    print(f"\nbest score: {best:+.4f}")
    print(f"budget    : {budget.summary()}")
    if code:
        (run_dir / "best.py").write_text(code + "\n")
        print(f"wrote {run_dir}/best.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

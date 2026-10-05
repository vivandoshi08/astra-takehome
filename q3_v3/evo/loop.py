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
import datetime
import json
import math
import re
import subprocess
import sys
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


def score_candidate(task, code: str, sandbox: bool = True,
                    timeout: float = 180.0, images=None):
    """Score one candidate: (score, status, detail), statuses as in
    evo/sandbox.py.

    encode/decode tasks are scored in fresh subprocesses (AUDIT.md A5-A7,
    C1-C3) and give exactly the score the shipped in-process path gives an
    honest candidate. `sandbox=False`, or any other task, uses the shipped
    path, with everything candidate code can raise caught (C1).
    """
    entry = getattr(task, "ENTRY_POINT", "heuristic")
    if sandbox and entry == ("encode", "decode"):
        from evo import sandbox as sbx
        return sbx.score(code, task, images=images, timeout=timeout)
    try:
        fns = compile_candidate(code, entry,
                                isolated=getattr(task, "ISOLATED_NAMESPACES",
                                                 False))
        if fns is None:
            return float("-inf"), "compile_fail", ""
        s = (task.evaluate(fns) if images is None
             else task.evaluate(fns, images=images))
    except KeyboardInterrupt:
        raise
    except BaseException as exc:       # OverflowError, SystemExit, ...
        return float("-inf"), "call_fail", repr(exc)[:300]
    return s, ("scored" if s > float("-inf") else "invalid"), ""


def _finite(x):
    return x if isinstance(x, float) and math.isfinite(x) else None


def run(task="ellipses", client=None, generations: int = 12,
        candidates_per_gen: int = 4, population_size: int = 8,
        budget: TokenBudget | None = None, log_path=None, quiet: bool = False,
        responses_path=None, sandbox: bool = True,
        eval_timeout: float = 180.0):
    """Run the loop. Returns (best_score, best_code, history).

    If the run stops early (the token cap raises BudgetExhausted, an API
    error survives the client's retries, Ctrl-C), the exception is re-raised
    with `exc.result = (best_score, best_code, history)` attached, after
    every paid response has been logged and scored, so the caller can still
    write best.py.

    Changes from the shipped loop, none of which change what it searches
    (AUDIT.md B1, B2, B5, C1-C4): one model call is made and charged at a
    time, so a failure can no longer leave billed calls out of
    token_log.jsonl; candidates are scored in fresh subprocesses; every
    response, including the ones that scored nothing, goes to
    `responses_path` with its finish_reason and the reason it failed. The
    prompt, the population rule and the code-extraction regex are as
    shipped.
    """
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
    resp_log = Path(responses_path) if responses_path else None
    if resp_log:
        resp_log.parent.mkdir(parents=True, exist_ok=True)

    def record(**kw):
        if log:
            with log.open("a") as fh:
                fh.write(json.dumps({"wall_ts": time.time(), **kw}) + "\n")

    def respond(**kw):
        if resp_log:
            with resp_log.open("a") as fh:
                fh.write(json.dumps({"wall_ts": time.time(), **kw},
                                    default=str) + "\n")

    population: list[tuple[float, str]] = []
    history: list[dict] = []

    def result():
        if not population:
            return float("-inf"), None, history
        return population[0][0], population[0][1], history

    # Seed program, FunSearch-style: a minimal working candidate the model
    # can see and improve from generation 0. Costs no tokens.
    seed_code = getattr(task, "SEED", None)
    if seed_code:
        s, status, detail = score_candidate(task, seed_code, sandbox,
                                            eval_timeout)
        if s > float("-inf"):
            population.append((s, seed_code))
            record(generation=-1, best=s, valid_this_gen=0, tokens=0,
                   candidates=[{"score": s, "code": seed_code}])
            if not quiet:
                print(f"  seed     score {s:+.4f}", flush=True)
        elif not quiet:
            print(f"  seed     did not score ({status}: {detail})", flush=True)

    try:
        for gen in range(generations):
            prompt = build_prompt(task, population)
            scored_this_gen = []
            stop = None
            for c in range(candidates_per_gen):
                try:
                    r = client.complete(prompt, n=1)[0]
                except Exception as exc:       # after the client's retries
                    respond(generation=gen, candidate=c, status="api_error",
                            detail=repr(exc)[:500])
                    stop = exc
                    break
                # Tag with the client so a mock run's free tokens are never
                # mistaken for real spend when you add up what a run cost.
                tag = f"{getattr(client, 'name', '?')}|gen{gen}"
                if getattr(r, "usage_estimated", False):
                    tag += "|usage-estimated"
                try:
                    budget.charge(r.input_tokens, r.output_tokens, tag=tag)
                except BudgetExhausted as exc:
                    stop = exc                 # logged and paid for: score it
                code = extract_code(r.text)
                if code is None:
                    s, status, detail = float("-inf"), "no_code", ""
                else:
                    s, status, detail = score_candidate(task, code, sandbox,
                                                        eval_timeout)
                if status == "scored":
                    scored_this_gen.append((s, code))
                    population.append((s, code))
                respond(generation=gen, candidate=c, status=status,
                        score=_finite(s), detail=detail,
                        finish_reason=getattr(r, "finish_reason", None),
                        input_tokens=r.input_tokens,
                        output_tokens=r.output_tokens,
                        usage=getattr(r, "usage", None), text=r.text)
                if stop is not None:
                    break

            population.sort(key=lambda t: -t[0])
            del population[population_size:]

            best = population[0][0] if population else float("-inf")
            entry = {"generation": gen, "best": best,
                     "valid_this_gen": len(scored_this_gen),
                     "tokens": budget.total_tokens}
            history.append(entry)
            record(**entry, prompt_chars=len(prompt),
                   candidates=[{"score": s, "code": c}
                               for s, c in scored_this_gen])
            if not quiet:
                print(f"  gen {gen:3d}  best {best:+.4f}  "
                      f"valid {len(scored_this_gen)}/{candidates_per_gen}  "
                      f"tokens {budget.total_tokens:,}", flush=True)
            if stop is not None:
                raise stop
    except BaseException as exc:
        exc.result = result()
        raise
    return result()


# Things an honest encode/decode pair has no reason to touch. A match is not
# proof of cheating; it is a reason to read the champion (AUDIT.md A5).
SUSPICIOUS = re.compile(
    r"\bimport\s+(?:data|tasks|evo|baselines|vae|student|race|sys|builtins|"
    r"gc|inspect|os|subprocess|socket|ctypes|importlib)\b|"
    r"\bfrom\s+(?:data|tasks|evo|sys|builtins|gc|inspect|os)\b|"
    r"__import__|__builtins__|_getframe|\bglobals\s*\(|\bopen\s*\(")


def _git_state():
    """(commit, dirty) for q3_v3, ignoring runs/; (None, None) without git."""
    try:
        sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=HERE,
                             capture_output=True, text=True,
                             timeout=10).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--", ".",
                                ":!runs"], cwd=HERE, capture_output=True,
                               text=True, timeout=10).stdout.strip()
    except Exception:
        return None, None
    return (sha or None), bool(dirty)


def _now():
    return datetime.datetime.now().astimezone().isoformat(timespec="seconds")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--task", default="ellipses")
    ap.add_argument("--client", choices=("mock", "llm"), default="mock")
    ap.add_argument("--generations", type=int, default=12)
    ap.add_argument("--candidates", type=int, default=4)
    ap.add_argument("--cap", type=int, default=500_000)
    ap.add_argument("--run-id", default="run")
    ap.add_argument("--eval-timeout", type=float, default=180.0,
                    help="seconds per scoring subprocess (shipped: none)")
    ap.add_argument("--no-sandbox", action="store_true",
                    help="score candidates in-process, as shipped (unsafe; "
                         "for comparison only)")
    args = ap.parse_args()

    run_dir = HERE / "runs" / args.run_id
    if run_dir.exists() and any(run_dir.iterdir()):
        # The shipped loop appended to the old logs (AUDIT.md B4).
        print(f"refusing to reuse {run_dir}: it already holds a run. "
              f"Pick a new --run-id.", file=sys.stderr)
        return 2
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

    # Every setting that defines the run, so a reader of runs/ never has to
    # guess what produced it (AUDIT.md C5).
    sha, dirty = _git_state()
    meta = {"run_id": args.run_id, "started": _now(), "argv": sys.argv,
            "task": args.task, "client": client.name,
            "model": getattr(client, "model", None),
            "reasoning": getattr(client, "reasoning", None),
            "max_tokens": getattr(client, "max_tokens", None),
            "generations": args.generations, "candidates": args.candidates,
            "population_size": 8, "cap": args.cap,
            "sandbox": not args.no_sandbox, "eval_timeout": args.eval_timeout,
            "git_commit": sha, "git_dirty": dirty,
            "python": sys.version.split()[0]}
    meta_path = run_dir / "run_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")

    stop_reason = None
    try:
        best, code, _ = run(task=task, client=client,
                            generations=args.generations,
                            candidates_per_gen=args.candidates,
                            budget=budget,
                            log_path=run_dir / "run_log.jsonl",
                            responses_path=run_dir / "responses.jsonl",
                            sandbox=not args.no_sandbox,
                            eval_timeout=args.eval_timeout)
    except BaseException as exc:
        if isinstance(exc, BudgetExhausted):
            print(f"\n[budget] {exc}")
            stop_reason = f"budget: {exc}"
        elif isinstance(exc, KeyboardInterrupt):
            print("\n[interrupted]")
            stop_reason = "interrupted"
        elif isinstance(exc, Exception):
            print(f"\n[stopped] {exc!r}")
            stop_reason = f"error: {exc!r}"[:500]
        else:
            raise
        best, code, _ = getattr(exc, "result", (float("-inf"), None, []))

    print(f"\nbest score: {best:+.4f}")
    print(f"budget    : {budget.summary()}")
    meta.update({"finished": _now(), "stop_reason": stop_reason,
                 "best_public": _finite(best), "budget": budget.summary(),
                 "calls": budget.calls, "input_tokens": budget.input_tokens,
                 "output_tokens": budget.output_tokens})
    if code:
        (run_dir / "best.py").write_text(code + "\n")
        print(f"wrote {run_dir}/best.py")
        if hasattr(task, "HELD_OUT"):
            # Never shown to the optimizer: scored once, after the run.
            held, status, detail = score_candidate(
                task, code, not args.no_sandbox, args.eval_timeout,
                images=task.HELD_OUT)
            meta["best_held_out"] = _finite(held)
            print(f"held-out  : {held:+.4f}"
                  + ("" if status == "scored" else f" ({status}: {detail})"))
        flags = sorted({m.group(0) for m in SUSPICIOUS.finditer(code)})
        meta["champion_flags"] = flags
        if flags:
            print(f"[check]   champion touches {flags}: read best.py")
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    return 0 if stop_reason is None or stop_reason.startswith("budget") else 2


if __name__ == "__main__":
    raise SystemExit(main())

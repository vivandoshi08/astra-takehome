"""Accounting under failure: what reaches token_log.jsonl / run_log.jsonl when
the cap trips mid-generation, when one API call inside `complete()` raises,
and when the provider's response has `usage=None` or no choices.

Fake clients only; all logs go to out/accounting/.
"""
from _common import (FENCED_SEED, FailingClient, ScriptedClient, fresh_dir,
                     read_jsonl)

from evo import loop
from evo.budget import BudgetExhausted, TokenBudget


def show(title, client, tok_log, run_log, exc):
    toks = read_jsonl(tok_log)
    runs = read_jsonl(run_log)
    billed = sum(a + b for a, b in client.billed)
    logged = sum(t["input_tokens"] + t["output_tokens"] for t in toks)
    print(f"\n=== {title}")
    print(f"  raised            : {type(exc).__name__ if exc else None}: {exc}")
    print(f"  calls billed      : {len(client.billed)} ({billed:,} tokens)")
    print(f"  calls in token_log: {len(toks)} ({logged:,} tokens)")
    print(f"  UNLOGGED billed   : {len(client.billed) - len(toks)} calls, "
          f"{billed - logged:,} tokens")
    print(f"  run_log records   : generations "
          f"{[r['generation'] for r in runs]}")


def scenario_cap_mid_generation():
    d = fresh_dir("accounting/cap_mid_gen")
    tok, rl = d / "token_log.jsonl", d / "run_log.jsonl"
    # 1500 tokens/call, 4 calls/gen: gen0 = 6000; gen1 call 6 -> 9000 >= 8000.
    client = ScriptedClient([FENCED_SEED], per_call=(1000, 500))
    budget = TokenBudget(limit=8000, log_path=tok)
    exc = None
    try:
        loop.run(client=client, generations=3, candidates_per_gen=4,
                 budget=budget, log_path=rl, quiet=True)
    except BudgetExhausted as e:
        exc = e
    show("A. cap crossed at response 2 of 4 in gen 1", client, tok, rl, exc)
    print(f"  budget.summary()  : {budget.summary()}")


def scenario_cap_exactly_at_limit():
    d = fresh_dir("accounting/cap_exact")
    tok = d / "token_log.jsonl"
    b = TokenBudget(limit=3000, log_path=tok)
    b.charge(1000, 500)
    exc = None
    try:
        b.charge(1000, 500)            # total == limit exactly
    except BudgetExhausted as e:
        exc = e
    print("\n=== B. charge that lands exactly ON the cap")
    print(f"  raised: {exc!r}; logged calls: {len(read_jsonl(tok))}; "
          f"remaining(): {b.remaining()}")
    b2 = TokenBudget(limit=1000, log_path=None)
    try:
        b2.charge(900, 400)
    except BudgetExhausted:
        pass
    print(f"  overshoot summary: {b2.summary()}")


def scenario_api_error_mid_generation():
    d = fresh_dir("accounting/api_error")
    tok, rl = d / "token_log.jsonl", d / "run_log.jsonl"
    # gen0 = calls 1-4 (fine); gen1: calls 5,6 succeed (billed), call 7 raises.
    client = FailingClient([FENCED_SEED], fail_at=7, per_call=(1000, 500))
    budget = TokenBudget(limit=500_000, log_path=tok)
    exc = None
    try:
        loop.run(client=client, generations=3, candidates_per_gen=4,
                 budget=budget, log_path=rl, quiet=True)
    except Exception as e:
        exc = e
    show("C. API call 3 of 4 raises in gen 1 (calls 1-2 of gen 1 billed)",
         client, tok, rl, exc)


class _Obj:
    def __init__(self, **kw):
        self.__dict__.update(kw)


class FakeOpenAI:
    """Stands in for openai.OpenAI; `plan` lists what each call returns."""

    def __init__(self, plan):
        self.plan = list(plan)
        self.calls = 0
        self.chat = _Obj(completions=_Obj(create=self._create))
        self.kwargs_seen = []

    def _create(self, **kw):
        self.kwargs_seen.append(kw)
        kind = self.plan[self.calls]
        self.calls += 1
        msg = _Obj(content=FENCED_SEED)
        ok_usage = _Obj(prompt_tokens=1000, completion_tokens=500)
        if kind == "ok":
            return _Obj(choices=[_Obj(message=msg, finish_reason="stop")],
                        usage=ok_usage)
        if kind == "no_usage":
            return _Obj(choices=[_Obj(message=msg, finish_reason="stop")],
                        usage=None)
        if kind == "no_choices":          # provider error relayed with HTTP 200
            return _Obj(choices=None, usage=ok_usage)
        if kind == "none_content":
            return _Obj(choices=[_Obj(message=_Obj(content=None),
                                      finish_reason="stop")], usage=ok_usage)
        if kind == "truncated":
            cut = FENCED_SEED[: len(FENCED_SEED) // 2]
            return _Obj(choices=[_Obj(message=_Obj(content=cut),
                                      finish_reason="length")],
                        usage=_Obj(prompt_tokens=1000, completion_tokens=2048))
        raise RuntimeError(kind)


def make_llm_client(plan, reasoning=None):
    """An LLMClient built without __init__: no env var, no key, no network."""
    from evo.llm_client import LLMClient
    c = LLMClient.__new__(LLMClient)
    c.model = "anthropic/claude-haiku-4.5"
    c.reasoning = reasoning
    c._client = FakeOpenAI(plan)
    c.name = f"openrouter:{c.model}"
    return c


def scenario_llm_client_edge_cases():
    print("\n=== D. LLMClient with a fake openai object")
    for plan in (["ok", "ok", "no_usage", "ok"],
                 ["ok", "no_choices", "ok", "ok"],
                 ["ok", "none_content", "truncated", "ok"]):
        c = make_llm_client(plan)
        try:
            rs = c.complete("prompt", n=4)
            print(f"  plan {plan}: returned {len(rs)} responses; texts empty/"
                  f"unfenced -> extract_code None: "
                  f"{[loop.extract_code(r.text) is None for r in rs]}")
        except Exception as e:
            print(f"  plan {plan}: complete() raised {type(e).__name__}: {e}"
                  f" after {c._client.calls} provider calls -> "
                  f"{c._client.calls - 1} earlier billed responses discarded "
                  f"(+ the failing call, which the provider also billed if it "
                  f"returned a completion)")
    c = make_llm_client(["ok"], reasoning="high")
    c.complete("prompt", n=1)
    kw = c._client.kwargs_seen[0]
    print(f"  reasoning=high kwargs: max_tokens={kw['max_tokens']}, "
          f"extra_body={kw.get('extra_body')}")
    r = make_llm_client(["truncated"]).complete("p")[0]
    print(f"  Response attributes kept: {sorted(vars(r))} "
          f"(finish_reason='length' is discarded; truncated text -> "
          f"extract_code None: {loop.extract_code(r.text) is None})")


def scenario_full_run_with_usage_none():
    d = fresh_dir("accounting/usage_none_run")
    tok, rl = d / "token_log.jsonl", d / "run_log.jsonl"
    c = make_llm_client(["ok"] * 4 + ["ok", "no_usage", "ok", "ok"] + ["ok"] * 8)
    budget = TokenBudget(limit=500_000, log_path=tok)
    exc = None
    try:
        loop.run(client=c, generations=3, candidates_per_gen=4, budget=budget,
                 log_path=rl, quiet=True)
    except Exception as e:
        exc = e
    print("\n=== E. full loop, real LLMClient code, usage=None on gen1 call 2")
    print(f"  raised: {type(exc).__name__}: {exc}")
    print(f"  provider calls made: {c._client.calls}; token_log calls: "
          f"{len(read_jsonl(tok))}; run_log gens: "
          f"{[r['generation'] for r in read_jsonl(rl)]}")


if __name__ == "__main__":
    scenario_cap_mid_generation()
    scenario_cap_exactly_at_limit()
    scenario_api_error_mid_generation()
    scenario_llm_client_edge_cases()
    scenario_full_run_with_usage_none()

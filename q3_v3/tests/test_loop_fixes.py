"""Checks for the fixes made to the loop after the audit (AUDIT.md).

    python tests/test_loop_fixes.py

Each check reproduces a shipped failure and asserts it no longer happens, or
asserts that a fix left honest behaviour exactly as shipped. Fake clients
only, no network. Run dirs go to a temp directory; nothing under runs/ is
touched.
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.dont_write_bytecode = True
Q3 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Q3))
sys.path.insert(0, str(Q3 / "audit" / "loop"))

import data  # noqa: E402
from evo import loop, sandbox  # noqa: E402
from evo.budget import BudgetExhausted, TokenBudget  # noqa: E402
from evo.mock_client import CANNED, MockResponse  # noqa: E402
from tasks import ellipses as T  # noqa: E402
import check_isolation  # noqa: E402  (the audit's exploit candidates)
import programs  # noqa: E402         (the audit's realistic candidates)

HONEST_OPTIMUM = -0.0096      # -sigma^2 (1 - 6/256), AUDIT.md A3
FENCED = "```python\n{}\n```"
GOOD = FENCED.format(CANNED[3])                       # scores -0.0162
OVERFLOW = FENCED.format("def encode(img):\n    return [10**400] + [0]*5\n"
                         "def decode(z):\n    return [0.0]*256")
HANG_CODE = ("def encode(img):\n    while True:\n        pass\n"
             "def decode(z):\n    return [0.0]*256")
HANG = FENCED.format(HANG_CODE)


class Scripted:
    """Returns `texts` in order; `fail_at` (1-based call number) raises, as
    a 402 or a dropped connection would after the SDK's retries."""

    name = "openrouter:test/fake"

    def __init__(self, texts, per_call=(1000, 500), fail_at=None):
        self.texts, self.per_call, self.fail_at = list(texts), per_call, fail_at
        self.billed = 0

    def complete(self, prompt, n=1, **_):
        out = []
        for _ in range(n):
            if self.fail_at and self.billed + 1 == self.fail_at:
                raise ConnectionError("simulated 402: insufficient credit")
            out.append(MockResponse(self.texts[self.billed % len(self.texts)],
                                    *self.per_call))
            self.billed += 1
        return out


def jsonl(path):
    p = Path(path)
    return [json.loads(l) for l in p.read_text().splitlines()] if p.exists() else []


def quiet_run(**kw):
    with contextlib.redirect_stdout(io.StringIO()):
        return loop.run(task=T, quiet=True, **kw)


# ---------------------------------------------------------------- scoring
def test_honest_scores_unchanged():
    """Sandbox scores equal the shipped in-process scores bit for bit."""
    progs = {"SEED": T.SEED, **{f"canned{i}": c for i, c in enumerate(CANNED)},
             "MOMENTS": programs.MOMENTS, "REFINED": programs.REFINED}
    for name, code in progs.items():
        fns = loop.compile_candidate(code, T.ENTRY_POINT, isolated=True)
        for imgs in (data.PUBLIC, data.HELD_OUT):
            ref = T.evaluate(fns, images=imgs)
            got, status, _ = sandbox.score(code, T, images=imgs)
            assert got == ref and status == "scored", (name, ref, got, status)


def test_side_channels_closed():
    """None of the audit's exploits beats an honest model any more, and
    none of them can change how a later candidate is scored."""
    for name, code in check_isolation.EXPLOITS.items():
        s, status, detail = sandbox.score(code, T, timeout=20)
        assert s < HONEST_OPTIMUM, (name, s, status, detail)
    # patch_evaluate / mutate_public_in_place ran above; the SEED is unaffected
    seed, _, _ = sandbox.score(T.SEED, T)
    assert seed == T.evaluate(loop.compile_candidate(T.SEED, T.ENTRY_POINT,
                                                     isolated=True))


def test_key_not_visible_to_candidates():
    code = ("def encode(img):\n    import os\n"
            "    return [0.0]*5 + [1.0 if os.environ.get('OPENROUTER_API_KEY') else 0.0]\n"
            "def decode(z):\n    return [z[5]] * 256\n")
    os.environ["OPENROUTER_API_KEY"] = "sk-test-canary"
    try:
        s, _, _ = sandbox.score(code, T)
    finally:
        del os.environ["OPENROUTER_API_KEY"]
    zeros = T.evaluate((lambda img: [0.0] * 6, lambda z: [0.0] * 256))
    assert s == zeros, (s, zeros)        # decode saw 0.0, not 1.0


def test_nondeterminism_rejected():
    s, status, _ = sandbox.score(check_isolation.EXPLOITS["nondeterministic_encode"], T)
    assert status == "nondeterministic" and s == float("-inf")


def test_timeout():
    t = time.time()
    s, status, _ = sandbox.score(HANG_CODE, T, timeout=2)
    assert status == "timeout" and time.time() - t < 10


# ------------------------------------------------------------- accounting
def test_api_error_loses_no_billed_call():
    """Shipped: call 7 of 8 raises -> calls 5-6 billed but never logged, run
    dies with no result. Now every billed call is logged and the best so
    far comes back on the exception."""
    with tempfile.TemporaryDirectory() as d:
        client = Scripted([GOOD], fail_at=7)
        budget = TokenBudget(log_path=Path(d) / "token_log.jsonl")
        try:
            quiet_run(client=client, generations=3, budget=budget,
                      responses_path=Path(d) / "responses.jsonl")
        except ConnectionError as exc:
            best, code, _ = exc.result
        else:
            raise AssertionError("the API error was swallowed")
        assert len(jsonl(Path(d) / "token_log.jsonl")) == client.billed == 6
        assert code is not None and best > -0.02
        statuses = [r["status"] for r in jsonl(Path(d) / "responses.jsonl")]
        assert statuses == ["scored"] * 6 + ["api_error"], statuses


def test_cap_mid_generation_loses_no_billed_call():
    with tempfile.TemporaryDirectory() as d:
        client = Scripted([GOOD])
        budget = TokenBudget(limit=9_000, log_path=Path(d) / "token_log.jsonl")
        try:
            quiet_run(client=client, generations=5, budget=budget)
        except BudgetExhausted as exc:
            best, code, hist = exc.result
        else:
            raise AssertionError("cap did not trip")
        assert len(jsonl(Path(d) / "token_log.jsonl")) == client.billed == 6
        assert code is not None and len(hist) == 2


def test_bad_candidate_does_not_kill_run():
    """Shipped: a latent of 10**400 raised OverflowError out of evaluate and
    killed the run; an infinite loop hung it."""
    with tempfile.TemporaryDirectory() as d:
        best, code, hist = quiet_run(
            client=Scripted([OVERFLOW, HANG, GOOD, "no code here"]),
            generations=1, budget=TokenBudget(log_path=None),
            responses_path=Path(d) / "r.jsonl", eval_timeout=2)
        statuses = [r["status"] for r in jsonl(Path(d) / "r.jsonl")]
        assert statuses == ["call_fail", "timeout", "scored", "no_code"], statuses
        assert hist[0]["valid_this_gen"] == 1 and best > -0.02


def test_run_log_schema_unchanged():
    with tempfile.TemporaryDirectory() as d:
        quiet_run(client=Scripted([GOOD]), generations=2,
                  budget=TokenBudget(log_path=None), log_path=Path(d) / "run_log.jsonl")
        recs = jsonl(Path(d) / "run_log.jsonl")
        assert set(recs[0]) == {"wall_ts", "generation", "best", "valid_this_gen",
                                "tokens", "candidates"}
        assert set(recs[1]) == {"wall_ts", "generation", "best", "valid_this_gen",
                                "tokens", "prompt_chars", "candidates"}


# ------------------------------------------------------------------ main
def run_main(tmp, *argv, client_cls=None):
    """evo.loop.main() with runs/ redirected into `tmp`."""
    import evo.mock_client as mc
    saved = (loop.HERE, sys.argv, mc.MockClient)
    loop.HERE = Path(tmp)
    sys.argv = ["evo.loop", "--client", "mock", *argv]
    if client_cls:
        mc.MockClient = client_cls
    try:
        with contextlib.redirect_stdout(io.StringIO()) as out, \
                contextlib.redirect_stderr(io.StringIO()) as err:
            code = loop.main()
    finally:
        loop.HERE, sys.argv, mc.MockClient = saved
    return code, out.getvalue() + err.getvalue()


def test_main_mock_run_writes_everything():
    with tempfile.TemporaryDirectory() as d:
        code, _ = run_main(d, "--generations", "2", "--run-id", "m1")
        rd = Path(d) / "runs" / "m1"
        assert code == 0
        assert {p.name for p in rd.iterdir()} == {
            "run_log.jsonl", "token_log.jsonl", "responses.jsonl",
            "run_meta.json", "best.py"}
        meta = json.loads((rd / "run_meta.json").read_text())
        for k in ("model", "reasoning", "generations", "candidates", "cap",
                  "sandbox", "git_commit", "stop_reason", "best_public",
                  "best_held_out", "champion_flags"):
            assert k in meta, k
        assert meta["stop_reason"] is None and meta["sandbox"] is True


def test_main_writes_best_py_when_api_fails():
    class Failing(Scripted):
        name = "mock"

        def __init__(self, seed=0):
            super().__init__([GOOD], fail_at=6)

    with tempfile.TemporaryDirectory() as d:
        code, out = run_main(d, "--generations", "3", "--run-id", "f1",
                             client_cls=Failing)
        rd = Path(d) / "runs" / "f1"
        meta = json.loads((rd / "run_meta.json").read_text())
        assert code == 2 and (rd / "best.py").exists(), out
        assert meta["stop_reason"].startswith("error") and meta["calls"] == 5


def test_main_refuses_to_reuse_run_id():
    with tempfile.TemporaryDirectory() as d:
        run_main(d, "--generations", "1", "--run-id", "r")
        before = (Path(d) / "runs" / "r" / "token_log.jsonl").read_text()
        code, out = run_main(d, "--generations", "1", "--run-id", "r")
        assert code == 2 and "refusing" in out
        assert (Path(d) / "runs" / "r" / "token_log.jsonl").read_text() == before


# ---------------------------------------------------------------- client
def test_llm_client_keeps_finish_reason_and_survives_missing_usage():
    """Shipped: usage=None raised AttributeError and choices=None TypeError,
    losing the generation; finish_reason was thrown away."""
    from types import SimpleNamespace as NS
    from evo.llm_client import LLMClient

    def completion(usage, choices):
        return NS(usage=usage, choices=choices)

    ok_usage = NS(prompt_tokens=900, completion_tokens=2048,
                  model_dump=lambda: {"prompt_tokens": 900,
                                      "completion_tokens": 2048})
    plan = [completion(ok_usage, [NS(message=NS(content="```python\nx"),
                                     finish_reason="length")]),
            completion(None, [NS(message=NS(content="hi"), finish_reason="stop")]),
            completion(ok_usage, None)]
    os.environ["OPENROUTER_API_KEY"] = "sk-test-not-real"
    try:
        c = LLMClient(model="test/fake")
    finally:
        del os.environ["OPENROUTER_API_KEY"]
    c._client = NS(chat=NS(completions=NS(create=lambda **kw: plan.pop(0))))
    cut, nousage, nochoice = c.complete("prompt " * 100, n=3)
    assert cut.finish_reason == "length" and cut.output_tokens == 2048
    assert nousage.usage_estimated and nousage.input_tokens > 0
    assert nochoice.text == "" and nochoice.input_tokens == 900
    assert c._client is not None and c.max_tokens == 2048


# ----------------------------------------------------------------- spend
def test_spend_prices_each_call_by_its_model():
    with tempfile.TemporaryDirectory() as d:
        def log(run, tag, n, tin, tout, tail=""):
            p = Path(d) / run
            p.mkdir()
            lines = [json.dumps({"call": i + 1, "tag": tag, "input_tokens": tin,
                                 "output_tokens": tout}) for i in range(n)]
            (p / "token_log.jsonl").write_text("\n".join(lines) + "\n" + tail)

        log("a_mock", "mock|gen0", 10, 100_000, 10_000)
        log("b_haiku", "openrouter:anthropic/claude-haiku-4.5|gen0", 2, 100_000, 20_000,
            tail='{"call": 3, "tag": "openr')                   # cut-off line
        log("c_sonnet", "openrouter:anthropic/claude-sonnet-5|gen0", 1, 100_000, 50_000)
        out = subprocess.run([sys.executable, str(Q3 / "spend.py"), "--runs", d],
                             capture_output=True, text=True)
        assert out.returncode == 0, out.stderr
        # haiku 2*(0.1 + 0.1) = 0.40; sonnet 0.2 + 0.5 = 0.70; mock free
        assert "$1.10 of $10.00" in out.stdout, out.stdout
        assert "unparseable" in out.stderr

        log("d_unknown", "openrouter:someone/new-model|gen0", 1, 10, 10)
        out = subprocess.run([sys.executable, str(Q3 / "spend.py"), "--runs", d],
                             capture_output=True, text=True)
        assert out.returncode != 0 and "no rate" in (out.stderr + out.stdout)


def main():
    tests = [(k, v) for k, v in globals().items() if k.startswith("test_")]
    failed = 0
    for name, fn in tests:
        t = time.time()
        try:
            fn()
            print(f"  ok    {name}  ({time.time() - t:.1f}s)")
        except Exception as exc:
            failed += 1
            print(f"  FAIL  {name}: {exc!r}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

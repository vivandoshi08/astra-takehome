"""`run()` population logic and the run_log audit trail, with the real
MockClient and with scripted fake clients. Logs go to out/population/."""
import hashlib
import json
import subprocess
import sys

from _common import FENCED_SEED, SEED, ScriptedClient, fresh_dir, read_jsonl
from programs import MOMENTS

from evo import loop
from evo.budget import TokenBudget
from evo.mock_client import CANNED, MockClient

GOOD = f"```python\n{MOMENTS.strip()}\n```"


def h(code):
    return hashlib.md5(" ".join(code.split()).encode()).hexdigest()[:6]


def shown_programs(prompt):
    """The programs build_prompt put in front of the model."""
    parts = prompt.split("```python\n")[1:]
    return [p.split("\n```")[0] for p in parts]


class Recorder:
    """Wraps a client and keeps every prompt it was sent."""

    def __init__(self, inner):
        self.inner, self.prompts = inner, []
        self.name = inner.name

    def complete(self, prompt, n=1, **kw):
        self.prompts.append(prompt)
        return self.inner.complete(prompt, n=n, **kw)


def duplicates_with_mock():
    d = fresh_dir("population/mock12")
    c = Recorder(MockClient())
    loop.run(client=c, generations=12, candidates_per_gen=4,
             budget=TokenBudget(log_path=d / "token_log.jsonl"),
             log_path=d / "run_log.jsonl", quiet=True)
    print("=== A. real MockClient, 12 generations: programs shown in each prompt")
    for g, p in enumerate(c.prompts):
        progs = shown_programs(p)
        print(f"  gen {g:2d}: shown {[h(x) for x in progs]} "
              f"unique {len(set(map(h, progs)))}/{len(progs)}")
    print(f"  seed hash {h(SEED)}; CANNED[2] (centroid, fixed shape) hash "
          f"{h(CANNED[2])}; same score as seed? see run_log")
    rl = read_jsonl(d / "run_log.jsonl")
    print(f"  valid_this_gen per gen: {[r['valid_this_gen'] for r in rl]}")
    print(f"  seed record: generation={rl[0]['generation']} valid_this_gen="
          f"{rl[0]['valid_this_gen']} len(candidates)={len(rl[0]['candidates'])}"
          f" keys={sorted(rl[0])}")


def duplicates_scripted():
    d = fresh_dir("population/same_good")
    c = Recorder(ScriptedClient([GOOD]))
    _, best, _ = loop.run(client=c, generations=3, candidates_per_gen=4,
                          budget=TokenBudget(log_path=None),
                          log_path=d / "run_log.jsonl", quiet=True)
    progs = shown_programs(c.prompts[-1])
    print("\n=== B. model returns the same good program 4x per generation")
    print(f"  last prompt shows {[h(x) for x in progs]} -> "
          f"{len(set(map(h, progs)))} unique of {len(progs)}; the seed "
          f"({h(SEED)}) shown: {h(SEED) in map(h, progs)}")


def audit_trail():
    d = fresh_dir("population/audit")
    texts = [GOOD,
             "Sorry, I can't see the images clearly; here is my reasoning ...",
             "```py\n" + MOMENTS + "\n```",            # dropped by the regex
             "```python\ndef encode(img):\n    return [0]*5\n"
             "def decode(z):\n    return [0]*256\n```"]  # -inf (5-D latent)
    c = ScriptedClient(texts)
    tl = d / "token_log.jsonl"
    loop.run(client=c, generations=1, candidates_per_gen=4,
             budget=TokenBudget(log_path=tl), log_path=d / "run_log.jsonl",
             quiet=True)
    rl = read_jsonl(d / "run_log.jsonl")
    blob = (d / "run_log.jsonl").read_text() + tl.read_text()
    print("\n=== C. audit trail of failed responses (1 good, 3 failing)")
    print(f"  gen0 valid_this_gen={rl[1]['valid_this_gen']}, logged candidates="
          f"{len(rl[1]['candidates'])}, token_log calls={len(read_jsonl(tl))}")
    print(f"  refusal text in any log: {'Sorry' in blob}; '```py' answer in any "
          f"log: {'```py' in blob}; 5-D latent program in any log: "
          f"{'[0]*5' in blob}")
    print(f"  gen0 record 'tokens' field = {rl[1]['tokens']} (cumulative run "
          f"total, not this generation); prompt_chars = {rl[1]['prompt_chars']}"
          f" (characters of ONE prompt, sent {len(texts)}x)")


INTEGRITY_CHILD = r'''
import sys; sys.path.insert(0, sys.argv[1])
from _common import FENCED_SEED, ScriptedClient, read_jsonl, fresh_dir
from evo import loop
from evo.budget import TokenBudget
kind = sys.argv[2]
patch = {
 "evaluate": "import tasks.ellipses as _E\n_E.evaluate = lambda fns, images=None: 0.0\n",
 "budget": "import evo.budget as _B\n_B.TokenBudget.charge = lambda self, *a, **k: None\n",
}[kind]
bad = "```python\n" + patch + "def encode(img):\n    return [0.0]*6\ndef decode(z):\n    return [0.0]*256\n```"
d = fresh_dir("population/integrity_" + kind)
c = ScriptedClient([bad, FENCED_SEED, FENCED_SEED, FENCED_SEED])
b = TokenBudget(limit=10_000 if kind == "budget" else 500_000, log_path=d / "token_log.jsonl")
best, code, hist = loop.run(client=c, generations=4, candidates_per_gen=4,
                            budget=b, log_path=d / "run_log.jsonl", quiet=True)
rl = read_jsonl(d / "run_log.jsonl")
print("scores per gen:", [[round(x["score"], 4) for x in r["candidates"]] for r in rl[1:]])
print("best:", best, "| best program is the patcher:", "lambda" in code)
print("provider calls billed:", len(c.billed), "| token_log lines:",
      len(read_jsonl(d / "token_log.jsonl")), "| budget.total_tokens:",
      b.total_tokens, "(billed", sum(x+y for x,y in c.billed), ")")
'''


def integrity_in_loop():
    from _common import HERE
    print("\n=== D. a candidate that patches the evaluator / the budget at "
          "compile time, inside a real loop run")
    for kind in ("evaluate", "budget"):
        p = subprocess.run([sys.executable, "-c", INTEGRITY_CHILD, str(HERE),
                            kind], capture_output=True, text=True,
                           env={"PYTHONDONTWRITEBYTECODE": "1",
                                "PATH": "/usr/bin:/bin"})
        print(f"  [{kind}]")
        for line in (p.stdout + p.stderr).strip().splitlines()[-4:]:
            print(f"    {line}")


def score_resolution():
    print("\n=== E. score resolution in the prompt: two programs 3e-5 apart")
    pop = [(-0.010410, SEED), (-0.010440, SEED)]
    from tasks import ellipses
    p = loop.build_prompt(ellipses, pop)
    print("  ", [l for l in p.splitlines() if l.startswith("--- score")])


if __name__ == "__main__":
    duplicates_with_mock()
    duplicates_scripted()
    audit_trail()
    integrity_in_loop()
    score_resolution()

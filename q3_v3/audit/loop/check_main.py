"""`evo.loop.main()` end to end (fake clients, runs/ redirected to out/main/):
what is written and what the exit code is on success, on cap, on an API
error, when a candidate raises SystemExit, when a --run-id is reused, and
what a reader of the logs can learn about the model / reasoning setting."""
import os
import subprocess
import sys

from _common import HERE, OUT, Q3, fresh_dir, read_jsonl

PY = sys.executable
ENV = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
ENV.pop("OPENROUTER_API_KEY", None)


def drive(scenario, root, *args, env=None):
    p = subprocess.run([PY, str(HERE / "drive_main.py"), scenario, str(root),
                        *args], capture_output=True, text=True,
                       env=env or ENV, cwd=Q3)
    return p


def files(run_dir):
    return sorted(x.name for x in run_dir.iterdir()) if run_dir.exists() else []


def report(title, p, run_dir):
    print(f"\n=== {title}")
    print(f"  exit code: {p.returncode}")
    tail = (p.stdout.strip().splitlines() + p.stderr.strip().splitlines())[-4:]
    for line in tail:
        print(f"  | {line}")
    print(f"  files: {files(run_dir)}")
    rl = read_jsonl(run_dir / "run_log.jsonl")
    tl = read_jsonl(run_dir / "token_log.jsonl")
    print(f"  run_log gens: {[r['generation'] for r in rl]}; token_log calls: "
          f"{[t['call'] for t in tl]}")
    if (run_dir / "best.py").exists():
        print(f"  best.py first line: "
              f"{(run_dir / 'best.py').read_text().splitlines()[0]!r}")
    if rl:
        best_logged = max(c["score"] for r in rl for c in r["candidates"])
        print(f"  best score recoverable from run_log: {best_logged:+.5f}")


root = fresh_dir("main")

p = drive("ok", root, "--generations", "2", "--run-id", "ok")
report("ok: 2 generations", p, root / "runs" / "ok")

p = drive("cap", root, "--generations", "5", "--cap", "8000", "--run-id", "cap")
report("cap: --cap 8000, trips at call 6 (gen 1)", p, root / "runs" / "cap")

p = drive("crash", root, "--generations", "5", "--run-id", "crash")
report("crash: API error on call 7 (gen 1)", p, root / "runs" / "crash")

p = drive("candidate_exit", root, "--generations", "5",
          "--run-id", "cand_exit")
report("candidate_exit: gen-1 candidate has module-level `raise SystemExit(0)`",
       p, root / "runs" / "cand_exit")

# --run-id reuse: two runs into the same directory
for _ in range(2):
    p = drive("ok", root, "--generations", "2", "--run-id", "reuse")
report("reuse: same --run-id twice", p, root / "runs" / "reuse")
rl = read_jsonl(root / "runs" / "reuse" / "run_log.jsonl")
tl = read_jsonl(root / "runs" / "reuse" / "token_log.jsonl")
print(f"  run_log has {sum(r['generation'] == -1 for r in rl)} seed lines; "
      f"token_log call numbers: {[t['call'] for t in tl]}")
sp = subprocess.run([PY, str(Q3 / "spend.py"), "--runs", str(root / "runs")],
                    capture_output=True, text=True, env=ENV, cwd=OUT)
print("  spend.py over these runs:")
for line in sp.stdout.splitlines():
    print(f"  | {line}")

# What a reader can learn: model name, reasoning effort
p = drive("llm", root, "--client", "llm", "--generations", "1",
          "--run-id", "llm_default")
p2 = drive("llm", root, "--client", "llm", "--generations", "1",
           "--run-id", "llm_reasoning", env={**ENV, "EVO_REASONING": "high"})
for rid in ("llm_default", "llm_reasoning"):
    d = root / "runs" / rid
    tl = read_jsonl(d / "token_log.jsonl")
    rl = read_jsonl(d / "run_log.jsonl")
    print(f"\n=== {rid}: exit {p.returncode if rid == 'llm_default' else p2.returncode}")
    print(f"  token_log tags: {sorted({t['tag'] for t in tl})}")
    print(f"  token_log keys: {sorted(tl[0])}")
    print(f"  run_log keys  : {sorted(set().union(*map(set, rl)))}")
    blob = (d / "token_log.jsonl").read_text() + (d / "run_log.jsonl").read_text()
    print(f"  'reasoning'/'high' anywhere in logs: "
          f"{'reasoning' in blob or 'high' in blob}")

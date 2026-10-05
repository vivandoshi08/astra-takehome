"""spend.py against a synthetic runs/ directory: a mock run, a Haiku run and a
Sonnet run (model visible in each token_log tag), plus edge cases."""
import json
import os
import subprocess
import sys

from _common import OUT, Q3, REPO, fresh_dir

PY = sys.executable
ENV = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}


def write_log(d, tag_prefix, calls, tin=2000, tout=1000, partial=False):
    d.mkdir(parents=True)
    lines, cum = [], 0
    for i in range(calls):
        cum += tin + tout
        lines.append(json.dumps({"call": i + 1, "tag": f"{tag_prefix}|gen{i//4}",
                                 "input_tokens": tin, "output_tokens": tout,
                                 "cumulative": cum, "wall_ts": 0.0}))
    text = "\n".join(lines) + "\n"
    if partial:
        text += '{"call": 99, "tag": "x|gen9", "input_tok'   # crash mid-write
    (d / "token_log.jsonl").write_text(text)


def spend(runs, *args, cwd=OUT):
    p = subprocess.run([PY, str(Q3 / "spend.py"), "--runs", str(runs), *args],
                       capture_output=True, text=True, env=ENV, cwd=cwd)
    return (p.stdout + p.stderr).strip(), p.returncode


root = fresh_dir("spend/runs")
write_log(root / "a_mock", "mock", 12)
write_log(root / "b_haiku", "openrouter:anthropic/claude-haiku-4.5", 48)
write_log(root / "c_sonnet", "openrouter:anthropic/claude-sonnet-5", 48)

print("=== A. default (Haiku rates): 2000 in + 1000 out per call; mock 12 calls, others 48")
print(spend(root)[0])
print("\n  true cost: mock $0, haiku 96K*1+48K*5 = $0.336, sonnet "
      "96K*2+48K*10 = $0.672 -> $1.01 total")
print("\n=== B. --model anthropic/claude-sonnet-5 (prices ALL runs at Sonnet)")
print(spend(root, "--model", "anthropic/claude-sonnet-5")[0])

print("\n=== C. --in-rate given without --out-rate")
print(spend(root, "--in-rate", "100")[0].splitlines()[-1])

print("\n=== D. a token_log whose last line was cut mid-write")
r2 = fresh_dir("spend/partial")
write_log(r2 / "p", "mock", 4, partial=True)
out, rc = spend(r2)
print(f"  exit {rc}: {out.splitlines()[-1]}")

print("\n=== E. run from the repo root as `python q3_v3/spend.py` (default "
      "--runs is CWD-relative)")
p = subprocess.run([PY, str(Q3 / "spend.py")], capture_output=True, text=True,
                   env=ENV, cwd=REPO)
print(f"  exit {p.returncode}: {(p.stdout + p.stderr).strip()}")

print("\n=== F. does spend.py read the tag at all?")
src = (Q3 / "spend.py").read_text()
print(f"  'tag' appears in spend.py source: {'tag' in src}")

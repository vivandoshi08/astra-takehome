"""Smaller checks: sanity_check's cap test, stray token_log from run(budget=None),
loop.py length vs README, budget edge cases, silent seed failure, __name__."""
import ast
import io
import os
import re
import subprocess
import sys
import tokenize
import types

from _common import HERE, OUT, Q3, SEED, fresh_dir, read_jsonl

from evo import loop
from evo.budget import DEFAULT_CAP, BudgetExhausted, TokenBudget
from evo.mock_client import MockClient
from tasks import ellipses

print("=== A. sanity_check.sh cap test: TokenBudget(limit=1_000) + MockClient")
b = TokenBudget(limit=1_000, log_path=None)
try:
    loop.run(client=MockClient(), generations=25, budget=b, quiet=True)
except BudgetExhausted as e:
    print(f"  trips after {b.calls} call(s): {e}  (first call alone exceeds "
          f"the cap, so the test only proves 'charge() raises')")

print("\n=== B. run(budget=None) creates TokenBudget() -> ./token_log.jsonl in CWD")
d = fresh_dir("misc/cwd")
code = (f"import sys; sys.path.insert(0, {str(HERE)!r}); import _common, os; "
        f"os.chdir({str(d)!r}); from evo import loop; "
        f"loop.run(generations=1, quiet=True)")
subprocess.run([sys.executable, "-c", code], env={"PYTHONDONTWRITEBYTECODE": "1",
               "PATH": "/usr/bin:/bin"}, check=True)
print(f"  files in CWD afterwards: {sorted(p.name for p in d.iterdir())}; "
      f"lines: {len(read_jsonl(d / 'token_log.jsonl'))}")

print("\n=== C. loop.py length (README: 'about 150 lines')")
src = (Q3 / "evo" / "loop.py").read_text()
lines = src.splitlines()
doc_lines = set()
for node in ast.walk(ast.parse(src)):
    if isinstance(node, (ast.Module, ast.FunctionDef)) and ast.get_docstring(node):
        ds = node.body[0]
        doc_lines.update(range(ds.lineno, ds.end_lineno + 1))
comment = {t.start[0] for t in tokenize.generate_tokens(io.StringIO(src).readline)
           if t.type == tokenize.COMMENT and lines[t.start[0] - 1].strip().startswith("#")}
code_lines = [i for i, l in enumerate(lines, 1)
              if l.strip() and i not in doc_lines and i not in comment]
print(f"  total {len(lines)}, non-blank {sum(1 for l in lines if l.strip())}, "
      f"code only (no blank/comment/docstring) {len(code_lines)}")

print("\n=== D. budget edge cases")
print(f"  main() --cap default 500_000 == budget.DEFAULT_CAP {DEFAULT_CAP}: "
      f"{'default=500_000' in src and DEFAULT_CAP == 500_000} (duplicated literal)")
try:
    TokenBudget(limit=0, log_path=None).summary()
except ZeroDivisionError as e:
    print(f"  TokenBudget(limit=0).summary() -> ZeroDivisionError: {e}")
try:
    TokenBudget(log_path=None).charge(None, 10)
except TypeError as e:
    print(f"  charge(None, 10) -> TypeError ({e}); nothing logged")

print("\n=== E. a broken SEED is skipped silently")
fake = types.SimpleNamespace(
    SEED="def encode(img):\n    return [0]*5\ndef decode(z):\n    return [0.0]*256",
    ENTRY_POINT=ellipses.ENTRY_POINT, ISOLATED_NAMESPACES=True,
    evaluate=ellipses.evaluate, prompt_header=ellipses.prompt_header)
d = fresh_dir("misc/badseed")
import contextlib
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    loop.run(task=fake, client=MockClient(), generations=1,
             budget=TokenBudget(log_path=None), log_path=d / "run_log.jsonl")
print(f"  stdout: {buf.getvalue().strip().splitlines()}")
print(f"  run_log gens: {[r['generation'] for r in read_jsonl(d / 'run_log.jsonl')]}"
      f" (no seed line, no warning)")

print("\n=== F. names visible to candidate code")
probe = ("def encode(img):\n    return [float(__name__ == '__main__'), "
         "float('__file__' in globals()), 0, 0, 0, 0]\n"
         "def decode(z):\n    return [z[0]] * 256")
e, _ = loop.compile_candidate(probe, ellipses.ENTRY_POINT, isolated=True)
z = e([0.0] * 256)
print(f"  __name__ == '__main__': {bool(z[0])} (so `if __name__ == '__main__':` "
      f"test code in an answer does not run); __name__ resolves to "
      f"{e.__globals__.get('__name__', 'builtins (via __builtins__)')!r}")

print("\n=== G. dead code: does any client raise BudgetExhausted?")
for f in ("evo/llm_client.py", "evo/mock_client.py"):
    print(f"  {f}: mentions BudgetExhausted/budget: "
          f"{bool(re.search(r'Budget|budget', (Q3 / f).read_text().split(chr(34)*3, 2)[-1]))}")

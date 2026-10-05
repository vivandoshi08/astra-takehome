"""Shared helpers for the Q3 loop audit. Never writes inside the repo.

Every script imports this first: it puts q3_v3 on sys.path, disables .pyc
writing (so importing repo modules leaves no __pycache__ in the repo), and
provides fake clients that never touch the network.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.dont_write_bytecode = True
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

HERE = Path(__file__).resolve().parent          # q3_v3/audit/loop
Q3 = HERE.parents[1]
REPO = Q3.parent
# scratch output (run dirs, logs, canaries) never goes inside the repo
import tempfile  # noqa: E402
OUT = Path(os.environ.get("Q3_AUDIT_OUT",
                          Path(tempfile.gettempdir()) / "q3_audit")) / "loop"
OUT.mkdir(parents=True, exist_ok=True)

if str(Q3) not in sys.path:
    sys.path.insert(0, str(Q3))
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
os.chdir(Q3)   # same CWD as `python -m evo.loop`; we never write relative paths

from evo.mock_client import MockResponse  # noqa: E402
from tasks import ellipses  # noqa: E402

SEED = ellipses.SEED.strip()
FENCED_SEED = f"```python\n{SEED}\n```"


class ScriptedClient:
    """Returns the scripted texts in order (cycling), with fixed token counts.

    `per_call` is (input_tokens, output_tokens) reported for every response.
    """

    def __init__(self, texts, per_call=(1000, 500), name="fake"):
        self.texts = list(texts)
        self.per_call = per_call
        self.name = name
        self.i = 0
        self.billed = []          # what the "provider" billed, call by call

    def complete(self, prompt, n=1, **_):
        out = []
        for _ in range(n):
            t = self.texts[self.i % len(self.texts)]
            self.i += 1
            self.billed.append(self.per_call)
            out.append(MockResponse(t, *self.per_call))
        return out


class FailingClient(ScriptedClient):
    """Like ScriptedClient, but call number `fail_at` (1-based, global) raises
    after the earlier calls of the same `complete()` have been billed."""

    def __init__(self, texts, fail_at, exc=None, **kw):
        super().__init__(texts, **kw)
        self.fail_at = fail_at
        self.exc = exc or ConnectionError("simulated 429 / network error")

    def complete(self, prompt, n=1, **_):
        out = []
        for _ in range(n):
            if len(self.billed) + 1 == self.fail_at:
                raise self.exc
            t = self.texts[self.i % len(self.texts)]
            self.i += 1
            self.billed.append(self.per_call)
            out.append(MockResponse(t, *self.per_call))
        return out


def read_jsonl(path):
    import json
    p = Path(path)
    if not p.exists():
        return []
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def fresh_dir(name):
    import shutil
    d = OUT / name
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True)
    return d

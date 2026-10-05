"""Run the real `evo.loop.main()` with a fake client and with `runs/` redirected
into the audit temp dir (never the repo). Usage:

    python drive_main.py <scenario> <runs_root> [extra loop CLI args...]

scenarios: ok | cap | crash | candidate_exit | llm
"""
import sys
from pathlib import Path

ROOT = Path(sys.argv[2]).resolve()      # resolve BEFORE _common changes cwd

import _common  # noqa: E402,F401  (sys.path, no .pyc, cwd)
if _common.REPO in ROOT.parents or ROOT == _common.REPO:
    raise SystemExit(f"refusing to write inside the repo: {ROOT}")
from _common import FENCED_SEED, FailingClient, ScriptedClient
from programs import MOMENTS

from pathlib import Path

import evo.mock_client
from evo import loop

scenario, root = sys.argv[1], ROOT
extra = sys.argv[3:]

loop.HERE = root                      # main() writes HERE / "runs" / run_id
GOOD = f"Here you go.\n```python\n{MOMENTS}\n```"

if scenario == "ok":
    factory = lambda: ScriptedClient([FENCED_SEED, GOOD], name="mock")
elif scenario == "cap":
    factory = lambda: ScriptedClient([FENCED_SEED, GOOD], per_call=(1000, 500),
                                     name="mock")
elif scenario == "crash":
    factory = lambda: FailingClient([GOOD, FENCED_SEED], fail_at=7, name="mock")
elif scenario == "candidate_exit":
    bad = "```python\nraise SystemExit(0)\n" + MOMENTS + "\n```"
    factory = lambda: ScriptedClient([GOOD, GOOD, GOOD, GOOD, bad], name="mock")
elif scenario == "llm":
    import os
    import evo.llm_client
    from check_accounting import make_llm_client
    _real = evo.llm_client.LLMClient
    reasoning = os.environ.get("EVO_REASONING")
    def _fake():
        evo.llm_client.LLMClient = _real     # make_llm_client needs the class
        c = make_llm_client(["ok"] * 100, reasoning=reasoning)
        return c
    evo.llm_client.LLMClient = _fake
    factory = None
else:
    raise SystemExit(f"unknown scenario {scenario}")

if factory is not None:
    evo.mock_client.MockClient = factory   # main() imports it at call time

sys.argv = ["evo.loop"] + extra
raise SystemExit(loop.main())

#!/usr/bin/env bash
# Confirms your environment is set up correctly. Should finish in a few seconds.
#
# If any line fails, stop and email us — that's our bug to fix, not yours.

set -uo pipefail
cd "$(dirname "$0")"

PY="${PYTHON:-python3}"
fail=0

ok()   { printf '  \033[32mOK\033[0m    %s\n' "$1"; }
bad()  { printf '  \033[31mFAIL\033[0m  %s\n' "$1"; fail=1; }
skip() { printf '  --    %s (not present yet)\n' "$1"; }

echo "ASTRA take-home — environment check"
echo

# ---------------------------------------------------------------- dependencies
if "$PY" - <<'EOF' 2>/dev/null
import gymnasium, neat, cv2, numpy, matplotlib, openai
EOF
then ok "dependencies import"
else bad "dependencies import — run: pip install -r requirements.txt"
fi

"$PY" -c 'import sys; sys.exit(0 if sys.version_info >= (3,10) else 1)' \
  && ok "python $("$PY" -c 'import sys;print(".".join(map(str,sys.version_info[:3])))')" \
  || bad "python 3.10+ required (found $("$PY" -c 'import sys;print(sys.version.split()[0])'))"

# ------------------------------------------------------------------------- Q1
if "$PY" - <<'EOF' 2>/dev/null
import sys, configparser
sys.path.insert(0, "q1_slimevolley")
import slimevolley

env = slimevolley.SlimeVolleyEnv()
env.reset()
for _ in range(20):
    env.step(env.action_space.sample())

assert env.observation_space.shape == (12,), env.observation_space
assert env.action_space.n == 3, env.action_space
assert hasattr(slimevolley, "BaselinePolicy")

cfg = configparser.ConfigParser()
cfg.read("q1_slimevolley/neat_config.ini")
assert cfg["DefaultGenome"]["num_inputs"] == "12"
assert cfg["DefaultGenome"]["num_outputs"] == "3"
EOF
then ok "Q1 env + NEAT config"
else bad "Q1 env + NEAT config"
fi

# ------------------------------------------------------------------------- Q2
if [ -f q2_backprop_neat/datasets.py ]; then
  if "$PY" - <<'EOF' 2>/dev/null
import sys
sys.path.insert(0, "q2_backprop_neat")
import datasets
for name in datasets.DATASETS:
    X_tr, y_tr, X_te, y_te = datasets.get_dataset(name)
    assert X_tr.ndim == 2 and X_tr.shape[1] == 2, (name, X_tr.shape)
    assert y_tr.ndim == 1 and len(y_tr) == len(X_tr), (name, y_tr.shape)
    assert len(X_te) > 0 and len(X_te) == len(y_te), (name, X_te.shape)
EOF
  then ok "Q2 datasets"
  else bad "Q2 datasets"
  fi
else
  skip "Q2 datasets"
fi

# ------------------------------------------------------------------------- Q3
if [ -f q3_v3/evo/loop.py ]; then
  if "$PY" - <<'EOF' 2>/dev/null
import sys
sys.path.insert(0, "q3_v3")
from evo.budget import TokenBudget, BudgetExhausted
from evo.mock_client import MockClient
from evo import loop

budget = TokenBudget(limit=50_000, log_path=None)
best, code, hist = loop.run(task="ellipses", client=MockClient(),
                            generations=2, budget=budget, quiet=True)
assert budget.total_tokens > 0, "token counter did not move"
assert best > float("-inf"), "no valid candidate scored"

tight = TokenBudget(limit=1_000, log_path=None)
try:
    loop.run(task="ellipses", client=MockClient(), generations=25,
             budget=tight, quiet=True)
except BudgetExhausted:
    pass
else:
    raise AssertionError("token cap did not trip")

import data
assert data.PUBLIC.shape == (128, 256) and data.HELD_OUT.shape == (256, 256)

import student
from tasks import ellipses
s = ellipses.evaluate(student.fns(), images=data.PUBLIC[:8])
assert s > float("-inf"), "student.py placeholder did not score"

from vae import VAE
m = VAE(ambient=256, hidden=(32,), latent=6, beta=0.01, seed=0)
h = m.train(data.PUBLIC[:64], epochs=5)
assert h[0] > h[-1] > 0.0, "VAE training loss did not decrease"
EOF
  then ok "Q3 ellipses loop + student stub + VAE (offline, no API calls)"
  else bad "Q3 ellipses loop + student stub + VAE"
  fi
else
  skip "Q3 ellipses loop"
fi

echo
if [ "$fail" -eq 0 ]; then
  echo "All checks passed. You're set up — start the clock."
else
  echo "Something failed above. Email us before spending more time on it."
fi
exit "$fail"

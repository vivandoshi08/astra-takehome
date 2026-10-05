"""Does retraining the VAE baseline from its fixed seed reproduce the shipped
cache (baselines.py docstring: "you will get identical weights")?

Run from q3_v3/:
    python <this file> [n_runs]

Never writes into the repo: uses cache=False and saves retrained weights into
this script's directory.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, ".")
import baselines  # noqa: E402
import data  # noqa: E402
from tasks import ellipses  # noqa: E402
from vae import VAE  # noqa: E402

from _paths import OUT  # noqa: E402
n_runs = int(sys.argv[1]) if len(sys.argv) > 1 else 1

shipped = VAE.load(baselines.CACHE / "vae_k6.npz")
s_ship_pub = ellipses.evaluate(baselines.vae_fns(shipped))
s_ship_ho = ellipses.evaluate(baselines.vae_fns(shipped), images=data.HELD_OUT)
print(f"shipped cache: PUBLIC {s_ship_pub:+.6f}  HELD_OUT {s_ship_ho:+.6f}")

models = []
for r in range(n_runs):
    t0 = time.time()
    m = baselines.vae_model(cache=False)
    dt = time.time() - t0
    m.save(OUT / f"vae_retrained_run{r}.npz")
    sp = ellipses.evaluate(baselines.vae_fns(m))
    sh = ellipses.evaluate(baselines.vae_fns(m), images=data.HELD_OUT)
    diffs = [float(np.max(np.abs(a - b))) for a, b in zip(m._params, shipped._params)]
    print(f"retrain {r}: {dt:.1f}s  PUBLIC {sp:+.6f}  HELD_OUT {sh:+.6f}  "
          f"max|w - w_shipped| = {max(diffs):.3e}  "
          f"bitwise identical to shipped: {all(np.array_equal(a, b) for a, b in zip(m._params, shipped._params))}")
    models.append(m)

for r in range(1, len(models)):
    same = all(np.array_equal(a, b) for a, b in zip(models[0]._params, models[r]._params))
    md = max(float(np.max(np.abs(a - b))) for a, b in zip(models[0]._params, models[r]._params))
    print(f"run0 vs run{r}: bitwise identical {same}, max diff {md:.3e}")

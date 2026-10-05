"""Misc checks: how much of PUBLIC's noise each baseline reproduces, the
remaining plotting helpers, and a student that simply reuses the shipped VAE.

Run from q3_v3/:
    python <this file>
PNGs go to this script's directory (race_out/), never into the repo.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, ".")
import baselines  # noqa: E402
import data  # noqa: E402
import plots  # noqa: E402
from evo.loop import compile_candidate  # noqa: E402
from tasks import ellipses  # noqa: E402

from _paths import OUT as _O; OUT = _O / "race_out"
OUT.mkdir(exist_ok=True)


def recon(fns, X):
    e, d = fns
    return np.array([d(list(e([float(v) for v in x]))) for x in X])


print("Fraction of the image's own noise that the reconstruction reproduces")
print("  f = <recon - clean, noise> / <noise, noise>; an honest 6-param fit is ~6/256 = 0.023")
for nm, fns in (("mean", baselines.mean_fns()), ("pca (k=6)", baselines.pca_fns()),
                ("vae (k=6)", baselines.vae_fns())):
    out = []
    for X, C in ((data.PUBLIC, data.PUBLIC_CLEAN), (data.HELD_OUT, data.HELD_OUT_CLEAN)):
        R = recon(fns, X)
        n = X - C
        f = np.sum((R - C) * n) / np.sum(n * n)
        mse_clean = np.mean((R - C) ** 2)
        out.append(f"f={f:+.3f} MSE-vs-clean={mse_clean:.5f}")
    print(f"  {nm:10s} PUBLIC {out[0]}   HELD_OUT {out[1]}")

print("\nA student.py that just returns baselines.vae_fns() (trained on PUBLIC):")
s = ellipses.evaluate(baselines.vae_fns())
print(f"  race score on PUBLIC {s:+.4f} -> Bonus tier (>= -0.0105): {s >= -0.0105}; "
      f"on HELD_OUT {ellipses.evaluate(baselines.vae_fns(), images=data.HELD_OUT):+.4f}")

seed = compile_candidate(ellipses.SEED, ellipses.ENTRY_POINT, isolated=True)
for f in (plots.plot_latents(seed, str(OUT / "latents_seed.png")),
          plots.plot_latents(baselines.vae_fns(), str(OUT / "latents_vae.png")),
          plots.plot_traversal(seed, str(OUT / "traversal_seed.png")),
          plots.plot_loss([1.0, 0.5, 0.25], str(OUT / "loss.png")),
          plots.plot_candidate(baselines.vae_fns(), str(OUT / "vae_heldout.png"),
                               images=data.HELD_OUT)):
    print("  wrote", Path(f).name)

"""Does the un-masked logvar clip (vae.py:65 forward clip, :125 backward
ignores it) ever engage in practice? Train the shipped architecture on PUBLIC
for several betas (cache=False, nothing written) and report the pre-clip
logvar range at the end, and the scores.

Run from q3_v3/:
    python <this file>
"""
from __future__ import annotations
import sys
import numpy as np
sys.path.insert(0, ".")
import baselines, data  # noqa: E401,E402
from tasks import ellipses  # noqa: E402

for beta in (0.01, 1e-3, 1e-4, 0.0):
    m = baselines.vae_model(beta=beta, cache=False)
    acts, mu, lv = m._enc_forward(data.PUBLIC)
    raw = acts[-1][:, m.latent:]
    fr = np.mean(np.abs(raw) > 10)
    s = ellipses.evaluate(baselines.vae_fns(m))
    sh = ellipses.evaluate(baselines.vae_fns(m), images=data.HELD_OUT)
    print(f"beta={beta:<7}: pre-clip logvar [{raw.min():8.2f}, {raw.max():8.2f}]  "
          f"fraction beyond +-10: {fr:.3f}   PUBLIC {s:+.5f}  HELD_OUT {sh:+.5f}")

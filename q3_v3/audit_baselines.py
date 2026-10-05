"""Audit the three startup baselines against the noise floor, on PUBLIC and HELD_OUT.

    python audit_baselines.py

The images are clean renders plus N(0, 0.1^2) pixel noise, and the score is
MSE against the *noisy* image. Reconstructing the true clean ellipse exactly
therefore scores about -SIGMA^2 = -0.0100. A per-image least-squares fit of
the 6 parameters does slightly better, -SIGMA^2 (1 - 6/256) = -0.0096,
because it also absorbs the noise along the 6 directions it can move in.
That is the honest optimum for a smooth decoder; a smooth model that scores
well above it on PUBLIC has memorised PUBLIC's noise, and HELD_OUT (never
trained on) shows it. (Float64 latents can carry more than 6 numbers' worth
of information if bit-packed — see AUDIT.md A4 — but the VAE is a smooth
tanh network and does not.)
"""

from __future__ import annotations

import numpy as np

import baselines
import data
from tasks import ellipses


def main():
    fns = {"mean": baselines.mean_fns(),
           "pca (k=6)": baselines.pca_fns(),
           "vae (k=6)": baselines.vae_fns()}
    sets = [("PUBLIC", data.PUBLIC, data.PUBLIC_CLEAN),
            ("HELD_OUT", data.HELD_OUT, data.HELD_OUT_CLEAN)]

    print(f"{'':12s}" + "".join(f"{name:>12s}" for name, _, _ in sets))
    floor = [-float(np.mean((X - C) ** 2)) for _, X, C in sets]
    print(f"{'clean truth':12s}" + "".join(f"{f:+12.4f}" for f in floor))
    for k, f in fns.items():
        row = [ellipses.evaluate(f, images=X) for _, X, _ in sets]
        print(f"{k:12s}" + "".join(f"{s:+12.4f}" for s in row))
    print(f"\n-SIGMA^2 = {-data.SIGMA ** 2:+.4f}")


if __name__ == "__main__":
    main()

"""Deliverable 4: is the VAE's -0.0051 a plausible score for a 6-number model?

    python vae_verdict.py            # ~2 min on a laptop; writes nothing to the repo

The shipped baseline trains `vae.VAE(hidden=(64,), latent=6, beta=0.01,
seed=0)` for 4000 epochs on PUBLIC (128 images) and is scored on PUBLIC.
This retrains the identical recipe on sets that are NOT the scored one and
scores each model three ways: on its own training images, on PUBLIC, and on
HELD_OUT. It also measures how much of each image's own noise the
reconstruction reproduces,

    f = <recon - clean, noise> / <noise, noise>,

which is ~6/256 = 0.023 for an honest 6-parameter fit (it can only follow
the noise along its six directions) and approaches 1 for a model that
copies the image.

If the incumbent's number comes from memorising PUBLIC, then every model
here should beat the honest optimum (-0.0096) only on its own training
images, and score PUBLIC like HELD_OUT.
"""

from __future__ import annotations

import time

import numpy as np

import baselines
import data
from tasks import ellipses
from vae import VAE

HONEST = -data.SIGMA ** 2 * (1 - data.LATENT / data.AMBIENT)


def noise_fraction(model, noisy, clean):
    recon = model.reconstruct(noisy)
    noise = noisy - clean
    return float(np.sum((recon - clean) * noise) / np.sum(noise * noise))


def score(model, images):
    return ellipses.evaluate(baselines.vae_fns(model), images=images)


def train(images, epochs=baselines.VAE_EPOCHS):
    m = VAE(ambient=data.AMBIENT, hidden=baselines.VAE_HIDDEN,
            latent=data.LATENT, beta=baselines.VAE_BETA, seed=baselines.VAE_SEED)
    m.train(images, epochs=epochs)
    return m


def main():
    sets = [("PUBLIC (shipped)", data.PUBLIC_PARAMS, data.PUBLIC_CLEAN, data.PUBLIC, None)]
    for n, seed in ((128, 11), (512, 12), (2048, 13)):
        p, c, x = data.train_images(n, seed=seed)
        sets.append((f"fresh n={n}", p, c, x, None))

    print(f"honest optimum for a 6-number fit: {HONEST:+.4f}; "
          f"noise fraction of an honest fit ~ {6 / 256:.3f}\n")
    print(f"{'trained on':<18} {'own train':>10} {'PUBLIC':>9} {'HELD_OUT':>9} "
          f"{'f(train)':>9} {'f(HELD)':>8} {'secs':>5}")
    for name, _, clean, noisy, _ in sets:
        t = time.time()
        m = baselines.vae_model() if name.startswith("PUBLIC") else train(noisy)
        print(f"{name:<18} {score(m, noisy):+10.4f} {score(m, data.PUBLIC):+9.4f} "
              f"{score(m, data.HELD_OUT):+9.4f} "
              f"{noise_fraction(m, noisy, clean):9.3f} "
              f"{noise_fraction(m, data.HELD_OUT, data.HELD_OUT_CLEAN):8.3f} "
              f"{time.time() - t:5.0f}")


if __name__ == "__main__":
    main()

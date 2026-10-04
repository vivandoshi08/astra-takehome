"""Reference models, scored on the same terms as any candidate — including
yours: each exposes an (encode, decode) pair over a 6-number latent.

    mean_fns()   decode to the mean PUBLIC image, ignore the input
    pca_fns()    project onto the first 6 principal components of PUBLIC
    vae_fns()    a variational autoencoder (vae.py) with a 6-D latent

The VAE trains once (about a minute on a laptop) and is cached to
baseline_cache/vae_k6.npz; delete the file to retrain it from its fixed seed
and you will get identical weights. Train your own variants directly with
vae.py — different widths, latent sizes, betas, or training sets are a few
lines.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

import data
from vae import VAE

CACHE = Path(__file__).parent / "baseline_cache"

# Training configuration for the cached baseline. Fixed; see vae.py.
VAE_HIDDEN = (64,)
VAE_BETA = 0.01
VAE_EPOCHS = 4000
VAE_SEED = 0


def mean_fns():
    mu = [float(v) for v in data.PUBLIC.mean(axis=0)]

    def encode(img):
        return [0.0] * data.LATENT

    def decode(z):
        return list(mu)

    return encode, decode


def pca_fns(k: int = data.LATENT):
    X = data.PUBLIC
    mu = X.mean(axis=0)
    _, _, vt = np.linalg.svd(X - mu, full_matrices=False)
    comps = vt[:k]                      # (k, ambient)

    def encode(img):
        return list((np.asarray(img, dtype=float) - mu) @ comps.T)

    def decode(z):
        return list(np.asarray(z, dtype=float) @ comps + mu)

    return encode, decode


def vae_model(hidden=VAE_HIDDEN, latent: int = data.LATENT,
              beta: float = VAE_BETA, epochs: int = VAE_EPOCHS,
              seed: int = VAE_SEED, cache: bool = True) -> VAE:
    """Train (or load from cache) the VAE baseline. Deterministic given the
    arguments."""
    name = (f"vae_h{'x'.join(str(h) for h in hidden)}"
            f"_k{latent}_b{beta}_e{epochs}_s{seed}.npz")
    path = CACHE / ("vae_k6.npz" if (hidden, latent, beta, epochs, seed) ==
                    (VAE_HIDDEN, data.LATENT, VAE_BETA, VAE_EPOCHS, VAE_SEED)
                    else name)
    if cache and path.exists():
        return VAE.load(path)
    model = VAE(ambient=data.AMBIENT, hidden=hidden, latent=latent,
                beta=beta, seed=seed)
    model.train(data.PUBLIC, epochs=epochs)
    if cache:
        CACHE.mkdir(exist_ok=True)
        model.save(path)
    return model


def vae_fns(model: VAE | None = None):
    model = model or vae_model()

    def encode(img):
        return [float(v) for v in model.encode(np.asarray(img, dtype=float))[0]]

    def decode(z):
        return [float(v) for v in model.decode(np.asarray(z, dtype=float))[0]]

    return encode, decode

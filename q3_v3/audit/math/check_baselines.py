"""Audit baselines.py: train-on-test leakage of mean / PCA / VAE, and the
cache logic.

Run from q3_v3/:
    python <this file> [--vae-control]
Writes nothing into the repo (cache=False everywhere; any npz goes to this
script's directory).
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
# Safety: every baselines.vae_model() call in this script sees a scratch copy
# of the cache, never the repo's baseline_cache/.
import shutil  # noqa: E402
_REPO_CACHE_FILE = baselines.CACHE / "vae_k6.npz"
_GLOBAL_SANDBOX = OUT / "cache_sandbox_global"
_GLOBAL_SANDBOX.mkdir(exist_ok=True)
shutil.copy(_REPO_CACHE_FILE, _GLOBAL_SANDBOX / "vae_k6.npz")
baselines.CACHE = _GLOBAL_SANDBOX
EV = ellipses.evaluate
SETS = (("PUBLIC", data.PUBLIC), ("HELD_OUT", data.HELD_OUT))
_, _, TRAIN = data.train_images()          # fresh 2048, seed 7


def hdr(s):
    print("\n" + "=" * 72 + "\n" + s + "\n" + "=" * 72)


def mean_from(X):
    mu = [float(v) for v in np.asarray(X).mean(axis=0)]
    return (lambda img: [0.0] * 6, lambda z: list(mu))


def pca_from(X, k=6):
    X = np.asarray(X)
    mu = X.mean(axis=0)
    _, _, vt = np.linalg.svd(X - mu, full_matrices=False)
    comps = vt[:k]
    return (lambda img: list((np.asarray(img, dtype=float) - mu) @ comps.T),
            lambda z: list(np.asarray(z, dtype=float) @ comps + mu))


def row(name, fns):
    s = [EV(fns, images=X) for _, X in SETS]
    print(f"  {name:34s} PUBLIC {s[0]:+.5f}   HELD_OUT {s[1]:+.5f}")
    return s


# --------------------------------------------------------------------------
hdr("1. Mean and PCA baselines: fit on PUBLIC (shipped) vs fit elsewhere")
row("mean, shipped (fit on PUBLIC)", baselines.mean_fns())
row("mean, fit on HELD_OUT", mean_from(data.HELD_OUT))
row("mean, fit on train_images(2048)", mean_from(TRAIN))
row("pca, shipped (fit on PUBLIC)", baselines.pca_fns())
row("pca, fit on HELD_OUT", pca_from(data.HELD_OUT))
row("pca, fit on train_images(2048)", pca_from(TRAIN))
_, _, T128 = data.sample(128, seed=2024)
row("pca, fit on a fresh 128 (seed 2024)", pca_from(T128))
s_ship = EV(baselines.pca_fns())
s_tr = EV(pca_from(TRAIN))
print(f"  PCA in-sample advantage on PUBLIC: {s_ship - s_tr:+.5f} "
      f"(T2 'beats PCA' threshold is inflated by this much)")

# --------------------------------------------------------------------------
hdr("2. VAE cache: stored config vs the VAE_* constants")
path = _REPO_CACHE_FILE          # read-only
d = np.load(path)
print(f"  {path.name}: keys {sorted(d.files)}")
print(f"  meta (ambient, latent, seed) = {d['meta'].tolist()}; beta = {d['beta'].tolist()}; "
      f"hidden = {d['hidden'].tolist()}")
print(f"  constants: VAE_HIDDEN={baselines.VAE_HIDDEN} VAE_BETA={baselines.VAE_BETA} "
      f"VAE_EPOCHS={baselines.VAE_EPOCHS} VAE_SEED={baselines.VAE_SEED} LATENT={data.LATENT}")
print(f"  epochs stored in the file: {'epochs' in d.files}; training-set identity stored: False "
      "(no key) -> the default path is returned without any check (baselines.py:67-71)")
print("  shapes:", {k: d[k].shape for k in d.files if k[1] in 'Wb'})
d.close()

# stale-cache demos. baselines.CACHE is redirected to a scratch copy FIRST,
# so nothing can be written into the repo's baseline_cache/.
import shutil
SCR = OUT / "cache_sandbox"
if SCR.exists():
    shutil.rmtree(SCR)
SCR.mkdir()
shutil.copy(path, SCR / "vae_k6.npz")
REAL_CACHE = baselines.CACHE    # already the global sandbox
baselines.CACHE = SCR
saved_consts = (baselines.VAE_HIDDEN, baselines.VAE_BETA, baselines.VAE_EPOCHS, baselines.VAE_SEED)
saved_defaults = baselines.vae_model.__defaults__
try:
    # (a) someone edits VAE_BETA = 0.05 in baselines.py: both the constant and
    #     the default argument change together (simulated by patching both).
    baselines.VAE_BETA = 0.05
    baselines.vae_model.__defaults__ = ((64,), 6, 0.05, 4000, 0, True)
    m = baselines.vae_model()
    print(f"  (a) VAE_BETA edited to 0.05: vae_model() silently LOADS vae_k6.npz; "
          f"returned model.beta = {m.beta} (the old config), no retrain, no warning")
    baselines.VAE_EPOCHS = 100
    baselines.VAE_BETA = 0.01
    baselines.vae_model.__defaults__ = ((64,), 6, 0.01, 100, 0, True)
    m = baselines.vae_model()
    print(f"      VAE_EPOCHS edited to 100: also loads vae_k6.npz (epochs not stored at all)")
    # (b) runtime patch of the module constant only (default args were bound
    #     at def time): the default call no longer matches and RETRAINS + WRITES.
    baselines.VAE_EPOCHS = 4000
    baselines.vae_model.__defaults__ = saved_defaults
    baselines.VAE_BETA = 0.5
    before = set(p.name for p in SCR.iterdir())
    m = baselines.vae_model()
    after = set(p.name for p in SCR.iterdir())
    print(f"  (b) baselines.VAE_BETA = 0.5 at runtime: vae_model() retrains with beta={m.beta} "
          f"and writes {sorted(after - before)} into the cache dir (this is how this audit "
          f"accidentally created a file in the repo on its first run)")
finally:
    (baselines.VAE_HIDDEN, baselines.VAE_BETA, baselines.VAE_EPOCHS,
     baselines.VAE_SEED) = saved_consts
    baselines.vae_model.__defaults__ = saved_defaults
    baselines.CACHE = REAL_CACHE

# --------------------------------------------------------------------------
hdr("3. Cache filename logic for non-default configs (no training, no writes)")
def path_for(hidden=baselines.VAE_HIDDEN, latent=data.LATENT, beta=baselines.VAE_BETA,
             epochs=baselines.VAE_EPOCHS, seed=baselines.VAE_SEED):
    name = (f"vae_h{'x'.join(str(h) for h in hidden)}"
            f"_k{latent}_b{beta}_e{epochs}_s{seed}.npz")
    return ("vae_k6.npz" if (hidden, latent, beta, epochs, seed) ==
            (baselines.VAE_HIDDEN, data.LATENT, baselines.VAE_BETA,
             baselines.VAE_EPOCHS, baselines.VAE_SEED) else name)
for kw in ({}, {"hidden": [64]}, {"hidden": (64, 32)}, {"hidden": ()},
           {"beta": 1e-5}, {"beta": 0.010}, {"epochs": 4000.0}, {"latent": 3}):
    print(f"  {str(kw):24s} -> {path_for(**kw)}")
try:
    path_for(hidden=64)
except TypeError as e:
    print(f"  hidden=64 (int)          -> TypeError: {e}")
print("  NOTE: baselines.vae_model mirrors this logic exactly (copied from baselines.py:65-69)")
print("  NOTE: hidden=[64] (a list) is the default config but misses the default cache and"
      " trains + writes a second file vae_h64_k6_b0.01_e4000_s0.npz into baseline_cache/")
print("  NOTE: vae_model(... cache=True) for any non-default config writes into q3_v3/baseline_cache/")

# --------------------------------------------------------------------------
hdr("4. VAE trained on fresh data (train_images, 2048) — the leak control")
if "--vae-control" in sys.argv:
    for epochs in (250, 1000):
        t0 = time.time()
        m = VAE(ambient=256, hidden=baselines.VAE_HIDDEN, latent=6,
                beta=baselines.VAE_BETA, seed=baselines.VAE_SEED)
        m.train(TRAIN, epochs=epochs)
        m.save(OUT / f"vae_train2048_e{epochs}.npz")
        row(f"vae fit on train_images, {epochs} ep ({time.time() - t0:.0f}s)",
            baselines.vae_fns(m))
    m = VAE(ambient=256, hidden=baselines.VAE_HIDDEN, latent=6,
            beta=baselines.VAE_BETA, seed=baselines.VAE_SEED)
    m.train(data.HELD_OUT, epochs=4000)
    row("vae fit on HELD_OUT (4000 ep)", baselines.vae_fns(m))
else:
    print("  (pass --vae-control to train it; ~1-2 min)")
row("vae, shipped (fit on PUBLIC)", baselines.vae_fns())

# logvar statistics of the shipped VAE
m = baselines.vae_model()
e_acts, mu, lv = m._enc_forward(data.PUBLIC)
raw_lv = e_acts[-1][:, m.latent:]
print(f"\n  shipped VAE posterior on PUBLIC: pre-clip logvar range [{raw_lv.min():.2f}, "
      f"{raw_lv.max():.2f}], fraction clipped at +-10: {np.mean(np.abs(raw_lv) > 10):.4f}; "
      f"posterior std median {np.median(np.exp(0.5 * lv)):.4f}; mu std per dim "
      f"{np.round(mu.std(0), 3).tolist()}")

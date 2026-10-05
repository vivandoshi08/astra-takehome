"""Audit vae.py: finite-difference gradient check of every parameter, the
logvar clip, Adam, in-place aliasing, save/load, and encode == mu.

Run from q3_v3/:
    python <this file>
Writes only into this script's directory.
"""
from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, ".")
from vae import VAE  # noqa: E402

from _paths import OUT  # noqa: E402
B1, B2, EE = 0.9, 0.999, 1e-8


def hdr(s):
    print("\n" + "=" * 72 + "\n" + s + "\n" + "=" * 72)


class FixedRNG:
    """Stand-in for model._rng so the reparameterisation noise is fixed."""
    def __init__(self, eps):
        self.eps = eps
    def standard_normal(self, shape):
        assert shape == self.eps.shape
        return self.eps.copy()
    def permutation(self, n):
        return np.arange(n)


def loss_fn(m, X, T, eps):
    """The loss exactly as vae.py defines it, with fixed eps."""
    _, mu, lv = m._enc_forward(X)
    z = mu + np.exp(0.5 * lv) * eps
    out = m._dec_forward(z)[-1]
    n = len(X)
    recon = np.mean((out - T) ** 2)
    kl = 0.5 * np.sum(mu ** 2 + np.exp(lv) - 1.0 - lv) / (n * m.ambient)
    return recon + m.beta * kl


def analytic_grads(m, X, T, eps):
    """Read the gradients vae.py computes: one _step at lr=0 on a fresh-Adam
    copy leaves params unchanged and m_1 = (1 - b1) * g."""
    c = copy.deepcopy(m)
    c._m = [np.zeros_like(p) for p in c._params]
    c._v = [np.zeros_like(p) for p in c._params]
    c._t = 0
    c._rng = FixedRNG(eps)
    loss, _, _ = c._step(X, T, lr=0.0)
    return [mm / (1 - B1) for mm in c._m], loss


def names(m):
    return ([f"eW{i}" for i in range(len(m.eW))] + [f"eb{i}" for i in range(len(m.eb))]
            + [f"dW{i}" for i in range(len(m.dW))] + [f"db{i}" for i in range(len(m.db))])


def fd_check(m, X, T, eps, h=1e-6, label=""):
    g_an, loss_step = analytic_grads(m, X, T, eps)
    L0 = loss_fn(m, X, T, eps)
    print(f"  {label} loss returned by _step {loss_step:.12f} == mirror loss {L0:.12f}: "
          f"{abs(loss_step - L0) < 1e-13}")
    worst = 0.0
    for nm, p, ga in zip(names(m), m._params, g_an):
        gn = np.zeros_like(p)
        it = np.nditer(p, flags=["multi_index"])
        for _ in it:
            idx = it.multi_index
            old = p[idx]
            p[idx] = old + h; Lp = loss_fn(m, X, T, eps)
            p[idx] = old - h; Lm = loss_fn(m, X, T, eps)
            p[idx] = old
            gn[idx] = (Lp - Lm) / (2 * h)
        err = np.max(np.abs(gn - ga)) / max(np.max(np.abs(gn)), np.max(np.abs(ga)), 1e-12)
        worst = max(worst, err)
        print(f"    {nm:4s} shape {str(p.shape):9s} max rel err {err:.2e}  "
              f"(|g| max {np.max(np.abs(ga)):.2e})")
    return worst, g_an


rng = np.random.default_rng(0)
AMB, HID, K, N = 12, (7, 5), 3, 9
X = rng.normal(size=(N, AMB))
T = rng.normal(size=(N, AMB))          # targets != inputs, to exercise `targets`
EPS = rng.normal(size=(N, K))

# --------------------------------------------------------------------------
hdr("1. Finite-difference check of every parameter (beta large so KL matters)")
m = VAE(ambient=AMB, hidden=HID, latent=K, beta=3.0, seed=1)
worst, _ = fd_check(m, X, T, EPS, label="2 hidden layers, beta=3:")
print(f"  WORST relative error: {worst:.2e}")
m1 = VAE(ambient=AMB, hidden=(), latent=K, beta=0.5, seed=2)
worst1, _ = fd_check(m1, X, T, EPS, label="no hidden layer (linear), beta=0.5:")
print(f"  WORST relative error: {worst1:.2e}")

# --------------------------------------------------------------------------
hdr("2. logvar clip at +-10: does backward respect it?")
mc = VAE(ambient=AMB, hidden=HID, latent=K, beta=3.0, seed=1)
mc.eb[-1][K + 0] = 25.0      # logvar dim 0 pushed far above +10 for every row
mc.eb[-1][K + 1] = -25.0     # logvar dim 1 far below -10
raw = mc._enc_forward(X)[0][-1][:, K:]
print(f"  pre-clip logvar dim0 range [{raw[:, 0].min():.1f}, {raw[:, 0].max():.1f}], "
      f"dim1 [{raw[:, 1].min():.1f}, {raw[:, 1].max():.1f}]")
g_an, _ = analytic_grads(mc, X, T, EPS)
h = 1e-6
for j in (0, 1, 2):
    b = mc.eb[-1]
    old = b[K + j]
    b[K + j] = old + h; Lp = loss_fn(mc, X, T, EPS)
    b[K + j] = old - h; Lm = loss_fn(mc, X, T, EPS)
    b[K + j] = old
    print(f"  d loss / d (logvar bias {j}): finite diff {(Lp - Lm) / (2 * h):+.6e}   "
          f"vae.py analytic {g_an[names(mc).index(f'eb{len(mc.eb) - 1}')][K + j]:+.6e}")

# --------------------------------------------------------------------------
hdr("3. Adam: compare 6 steps against an independent reference implementation")
ma = VAE(ambient=AMB, hidden=HID, latent=K, beta=0.3, seed=4)
ref = [p.copy() for p in ma._params]
rm = [np.zeros_like(p) for p in ref]
rv = [np.zeros_like(p) for p in ref]
ma._rng = FixedRNG(EPS)
lr = 1e-2
for t in range(1, 7):
    # reference: gradient at the reference params via a throwaway model
    tmp = copy.deepcopy(ma)
    for dst, src in zip(tmp._params, ref):
        dst[...] = src
    g, _ = analytic_grads(tmp, X, T, EPS)
    for i in range(len(ref)):
        rm[i] = B1 * rm[i] + (1 - B1) * g[i]
        rv[i] = B2 * rv[i] + (1 - B2) * g[i] ** 2
        mh = rm[i] / (1 - B1 ** t)
        vh = rv[i] / (1 - B2 ** t)
        ref[i] = ref[i] - lr * mh / (np.sqrt(vh) + EE)
    ma._step(X, T, lr)
    d = max(float(np.max(np.abs(a - b))) for a, b in zip(ma._params, ref))
    print(f"  step {t}: max |vae.py params - reference| = {d:.2e}")

# --------------------------------------------------------------------------
hdr("4. In-place updates alias the weight lists")
mb = VAE(ambient=AMB, hidden=HID, latent=K, beta=0.3, seed=5)
before = [w.copy() for w in mb.eW + mb.dW]
mb._rng = FixedRNG(EPS)
mb._step(X, T, 1e-2)
alias = all(a is b for a, b in zip(mb._params, [*mb.eW, *mb.eb, *mb.dW, *mb.db]))
moved = all(not np.array_equal(a, b) for a, b in zip(before, mb.eW + mb.dW))
print(f"  _params[i] is eW/eb/dW/db[i]: {alias}; weights actually changed: {moved}")

# --------------------------------------------------------------------------
hdr("5. encode returns mu; reconstruct is deterministic")
_, mu, lv = mb._enc_forward(X)
print(f"  encode(X) == mu: {np.array_equal(mb.encode(X), mu)}; shape {mb.encode(X).shape}; "
      f"reconstruct twice identical: {np.array_equal(mb.reconstruct(X), mb.reconstruct(X))}; "
      f"1-D input handled: {mb.encode(X[0]).shape}")

# KL closed form vs Monte Carlo
mu1, s1 = 0.7, 0.3
zg = np.linspace(mu1 - 12 * s1, mu1 + 12 * s1, 2_000_001)
q = np.exp(-(zg - mu1) ** 2 / (2 * s1 ** 2)) / np.sqrt(2 * np.pi * s1 ** 2)
logratio = (-0.5 * np.log(2 * np.pi * s1 ** 2) - (zg - mu1) ** 2 / (2 * s1 ** 2)
            + 0.5 * np.log(2 * np.pi) + zg ** 2 / 2)
mc_kl = np.sum(q * logratio) * (zg[1] - zg[0])
cf = 0.5 * (mu1 ** 2 + s1 ** 2 - 1 - np.log(s1 ** 2))
print(f"  KL(N(0.7,0.3^2)||N(0,1)): closed form used by vae.py {cf:.8f}, numerical integral {mc_kl:.8f}")

# --------------------------------------------------------------------------
hdr("6. save / load round trip, and resuming training after load")
mt = VAE(ambient=AMB, hidden=HID, latent=K, beta=0.3, seed=6)
mt.train(X, T, epochs=10, batch=4)
p = OUT / "roundtrip.npz"
mt.save(p)
ml = VAE.load(p)
same = all(np.array_equal(a, b) for a, b in zip(mt._params, ml._params))
print(f"  params equal after load: {same}; encode equal: {np.array_equal(mt.encode(X), ml.encode(X))}; "
      f"config (ambient, hidden, latent, beta, seed) equal: "
      f"{(mt.ambient, mt.hidden, mt.latent, mt.beta, mt.seed) == (ml.ambient, ml.hidden, ml.latent, ml.beta, ml.seed)}")
alias = all(a is b for a, b in zip(ml._params, [*ml.eW, *ml.eb, *ml.dW, *ml.db]))
print(f"  after load _params alias the weight lists: {alias}; arrays writeable: "
      f"{all(a.flags.writeable for a in ml._params)}; dtype {ml._params[0].dtype}")
print(f"  Adam state after load: t={ml._t} (trained model t={mt._t}); m all zero: "
      f"{all(not np.any(a) for a in ml._m)}; _rng reset to the seed stream")
ml.train(X, T, epochs=10, batch=4)
mt.train(X, T, epochs=10, batch=4)
d = max(float(np.max(np.abs(a - b))) for a, b in zip(mt._params, ml._params))
print(f"  10 more epochs: uninterrupted vs save/load/resume differ by max {d:.3e} "
      f"(optimizer state and RNG are not saved)")

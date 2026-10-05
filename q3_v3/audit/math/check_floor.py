"""What is the best score an honest 6-number model can get, and where do the
graded thresholds sit relative to it?

(1) Per-image least-squares fit of the true generator (data.render) — a
    legitimate deterministic encoder: it only measures the one image. Scored
    with the real evaluator on PUBLIC and HELD_OUT. Theory for a smooth 6-D
    model fit to each image: E[MSE] ~= sigma^2 * (1 - 6/256).
(2) Same fit from the oracle (true) parameters, as a check that the legit
    encoder reaches the global optimum.
(3) A float64 bit-packing encoder: the 6 numbers are float64s, so they carry
    ~318 bits. Pack quantised fit parameters + the signs of the residual
    noise. Shows the "6 numbers cannot carry the noise" premise needs a
    smooth / finite-precision decoder assumption.

Run from q3_v3/:
    python <this file>
Writes nothing.
"""
from __future__ import annotations

import sys
import time

import numpy as np

sys.path.insert(0, ".")
import data  # noqa: E402
from tasks import ellipses  # noqa: E402

S2 = data.SIGMA ** 2


def lm_fit(X, P0, iters=200, h=1e-6):
    """Batched Levenberg-Marquardt on render(P) ~ X. X (n,256), P0 (n,6)."""
    X = np.atleast_2d(X)
    P = np.array(P0, dtype=float, copy=True)
    lam = np.full(len(P), 1e-2)
    def cost(P):
        return np.sum((data.render(P) - X) ** 2, axis=1)
    c = cost(P)
    for _ in range(iters):
        R = data.render(P) - X
        J = np.empty((len(P), 256, 6))
        for k in range(6):
            Q = P.copy(); Q[:, k] += h
            J[:, :, k] = (data.render(Q) - data.render(P)) / h
        JTJ = np.einsum("nik,nil->nkl", J, J)
        g = np.einsum("nik,ni->nk", J, R)
        D = np.einsum("nkk->nk", JTJ)[:, :, None] * np.eye(6)
        step = -np.linalg.solve(JTJ + lam[:, None, None] * (D + 1e-9 * np.eye(6)),
                                g[:, :, None])[:, :, 0]
        Pn = P + step
        Pn[:, 2:4] = np.maximum(Pn[:, 2:4], 0.3)
        cn = cost(Pn)
        ok = cn < c
        P[ok], c[ok] = Pn[ok], cn[ok]
        lam = np.where(ok, lam * 0.3, lam * 10.0)
        lam = np.clip(lam, 1e-12, 1e12)
    return P, c


def moments_init(img):
    """Image-only initial guess: thresholded centroid and second moments."""
    img = np.asarray(img, dtype=float).reshape(16, 16)
    w = np.where(img > 0.3, img, 0.0)
    if w.sum() <= 0:
        return np.array([8.0, 8.0, 3.0, 2.0, 0.0, 0.8])
    yy, xx = np.mgrid[0:16, 0:16]
    m = w.sum()
    cx, cy = (w * xx).sum() / m, (w * yy).sum() / m
    cxx = (w * (xx - cx) ** 2).sum() / m
    cyy = (w * (yy - cy) ** 2).sum() / m
    cxy = (w * (xx - cx) * (yy - cy)).sum() / m
    ev, evec = np.linalg.eigh(np.array([[cxx, cxy], [cxy, cyy]]))
    a = 2 * np.sqrt(max(ev[1], 0.25)); b = 2 * np.sqrt(max(ev[0], 0.25))
    th = np.arctan2(evec[1, 1], evec[0, 1]) % np.pi
    A = np.median(img[img > 0.5 * img.max()]) if np.any(img > 0.5 * img.max()) else 0.8
    return np.array([cx, cy, a, b, th, A])


def fit_encode(img):
    """Legit encoder: moments init + 3 theta restarts, keep the best LM fit."""
    x = np.asarray(img, dtype=float)[None]
    p0 = moments_init(img)
    starts = [p0]
    for dth in (np.pi / 3, 2 * np.pi / 3):
        q = p0.copy(); q[4] = (q[4] + dth) % np.pi; starts.append(q)
    q = p0.copy(); q[2], q[3] = q[3], q[2]; starts.append(q)   # swapped axes
    best = None
    for s in starts:
        P, c = lm_fit(x, s[None], iters=120)
        if best is None or c[0] < best[1]:
            best = (P[0], c[0])
    return [float(v) for v in best[0]]


def render_decode(z):
    return [float(v) for v in data.render(np.asarray(z, dtype=float))[0]]


# --------------------------------------------------------------------------
print("=" * 72)
print("(1)+(2) per-image least-squares fit of data.render")
print("=" * 72)
res = {}
for nm, X, C, Ptrue in (("PUBLIC", data.PUBLIC, data.PUBLIC_CLEAN, data.PUBLIC_PARAMS),
                        ("HELD_OUT", data.HELD_OUT, data.HELD_OUT_CLEAN, data.HELD_OUT_PARAMS)):
    t0 = time.time()
    s_fit = ellipses.evaluate((fit_encode, render_decode), images=X)
    t1 = time.time()
    Por, cor = lm_fit(X, Ptrue, iters=200)
    s_or = -cor.sum() / X.size
    s_clean = -np.mean((X - C) ** 2)
    # theory: fitting 6 params absorbs ~6/256 of the noise variance per image
    theory = s_clean + 6 * S2 / 256
    res[nm] = (s_clean, s_fit, s_or)
    print(f"  {nm:8s}: clean truth {s_clean:+.6f} | legit LM fit (evaluate) {s_fit:+.6f} "
          f"[{t1 - t0:.0f}s] | oracle-init LM fit {s_or:+.6f} | clean + 6*sigma^2/256 = {theory:+.6f}")
    # how often does the legit fit miss the oracle optimum?
    Z = np.array([fit_encode(x) for x in X[:64]])
    cz = np.sum((data.render(Z) - X[:64]) ** 2, axis=1)
    miss = cz - cor[:64]
    print(f"            legit vs oracle optimum on first 64: worse by >1e-3 SSE in "
          f"{np.sum(miss > 1e-3)} / 64 images (max excess SSE {miss.max():.4f})")
print(f"  expected value sigma^2 (1 - 6/256) = {S2 * (1 - 6 / 256):.6f}")

# --------------------------------------------------------------------------
print("\n" + "=" * 72)
print("(3) float64 bit-packing: params (6 x 16 bits) + 222 residual-sign bits")
print("=" * 72)
LO = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
HI = np.array([16.0, 16.0, 8.0, 8.0, np.pi, 2.0])
QB, SB = 16, 37            # 16 + 37 = 53 bits per float: exact as a float64 int
NS = 6 * SB                # 222 pixels get a residual sign
C_SIGN = data.SIGMA * np.sqrt(2 / np.pi)   # E|N(0, sigma^2)|


def quant(p):
    p = np.array(p, dtype=float)
    p[4] %= np.pi
    q = np.clip(np.round((p - LO) / (HI - LO) * (2 ** QB - 1)), 0, 2 ** QB - 1)
    return q.astype(np.int64)


def dequant(q):
    return LO + np.asarray(q, dtype=float) / (2 ** QB - 1) * (HI - LO)


def pack_encode(img):
    p = fit_encode(img)
    q = quant(p)
    base = data.render(dequant(q))[0]
    signs = (np.asarray(img)[:NS] > base[:NS]).astype(np.int64)
    z = []
    for k in range(6):
        bits = 0
        for s in signs[k * SB:(k + 1) * SB][::-1]:
            bits = (bits << 1) | int(s)
        z.append(float((int(q[k]) << SB) | bits))   # < 2**53: exact
    return z


def pack_decode(z):
    q, signs = [], []
    for v in z:
        n = int(v)
        q.append(n >> SB)
        bits = n & ((1 << SB) - 1)
        signs.extend((bits >> i) & 1 for i in range(SB))
    img = data.render(dequant(q))[0]
    img[:NS] += C_SIGN * (2 * np.array(signs[:NS]) - 1)
    return [float(v) for v in img]


z = pack_encode(data.PUBLIC[0])
print(f"  example latent: {[int(v) for v in z]}  (max < 2**53: {max(z) < 2 ** 53})")
for nm, X in (("PUBLIC", data.PUBLIC), ("HELD_OUT", data.HELD_OUT)):
    s = ellipses.evaluate((pack_encode, pack_decode), images=X)
    cl, fit, _ = res[nm]
    pred = fit * (NS * (1 - 2 / np.pi) + (256 - NS)) / 256
    print(f"  {nm:8s}: bit-packed score {s:+.6f}  (predicted ~{pred:+.6f}; "
          f"clean truth {cl:+.6f}; VAE PUBLIC -0.005051 / HELD_OUT -0.034619)")

# --------------------------------------------------------------------------
print("\n" + "=" * 72)
print("Graded thresholds vs the honest optimum")
print("=" * 72)
cl, fit, _ = res["PUBLIC"]
for nm, t in (("T3", -0.0140), ("Bonus", -0.0105)):
    print(f"  {nm:5s} {t:+.4f}: slack below legit LM fit {fit - t:.5f} "
          f"({(t / fit - 1) * 100:.1f}% worse MSE allowed), below clean truth {cl - t:.5f}")
print(f"  PCA {-0.0173:+.4f}, SEED {-0.0272:+.4f}, mean {-0.0454:+.4f}")

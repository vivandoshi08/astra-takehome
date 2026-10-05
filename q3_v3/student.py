"""Our model: fit the generator's six parameters to each image.

Every image is `data.render(cx, cy, a, b, theta, A)` plus N(0, 0.1^2) pixel
noise, so the six numbers that matter are known, and so is the exact drawing
formula. The model is analysis by synthesis:

  encode(img)  1. A first guess from the image's moments: centre = mean
                  position of the bright pixels; axes and tilt from their
                  spread (a filled ellipse with semi-axes a, b has variances
                  a^2/4 and b^2/4 along its axes); brightness by least
                  squares given that shape.
               2. Refine all six by Levenberg-Marquardt: draw the ellipse,
                  compare with the image pixel by pixel, step the six
                  numbers to shrink the squared error, repeat. Three
                  starting tilts, because the tilt of a near-circle is
                  barely visible and a single start can stall.
               3. Return the fitted (cx, cy, a, b, theta, A).
  decode(z)    Draw the ellipse with `data.render` — the generator itself.

Rules: encode sees one image; decode sees only the six numbers; numpy only
(no scipy in requirements.txt); deterministic; nothing is learned, so
nothing was fit to PUBLIC. The few settings below (threshold, iterations,
starts) were chosen on fresh `data.train_images()` samples, never on PUBLIC
or HELD_OUT.

Where the error comes from: the per-image fit can do no better than about
-sigma^2 (1 - 6/256) = -0.0096 on average — the noise it cannot see past,
less the share it absorbs along its six directions. What is left above
that is fits that stop in the wrong valley; see `report.md`.
"""

from __future__ import annotations

import math

import numpy as np

import data

# ---------------------------------------------------------------------------
# Our model (replaces the shipped placeholder).
# ---------------------------------------------------------------------------

THRESHOLD = 0.3          # pixels above this count as "bright" for the first guess
ITERS = 40               # Levenberg-Marquardt iterations per start
STARTS = (0.0, math.pi / 3, -math.pi / 3)   # tilt offsets tried from the first guess
H = 1e-5                 # finite-difference step for the Jacobian
MIN_AXIS = 0.3           # keeps render() away from division by zero


def _render(p: np.ndarray) -> np.ndarray:
    """data.render with the axes kept positive. p: (k, 6) -> (k, 256)."""
    p = np.array(p, dtype=float, ndmin=2)
    p[:, 2:4] = np.maximum(np.abs(p[:, 2:4]), MIN_AXIS)
    return data.render(p)


def _first_guess(x: np.ndarray) -> np.ndarray:
    img = x.reshape(data.IMG, data.IMG)
    w = np.where(img > THRESHOLD, img, 0.0)
    m = w.sum()
    if m <= 0.0:
        return np.array([7.5, 7.5, 3.0, 2.0, 0.0, 0.8])
    ys, xs = np.mgrid[0:data.IMG, 0:data.IMG]
    cx, cy = (w * xs).sum() / m, (w * ys).sum() / m
    dx, dy = xs - cx, ys - cy
    sxx, syy, sxy = ((w * dx * dx).sum() / m, (w * dy * dy).sum() / m,
                     (w * dx * dy).sum() / m)
    half_tr = (sxx + syy) / 2.0
    root = math.sqrt(max(0.0, half_tr ** 2 - (sxx * syy - sxy ** 2)))
    a = 2.0 * math.sqrt(max(half_tr + root, 0.1))
    b = 2.0 * math.sqrt(max(half_tr - root, 0.1))
    th = 0.5 * math.atan2(2.0 * sxy, sxx - syy)
    p = np.array([cx, cy, a, b, th, 1.0])
    shape = _render(p)[0]
    p[5] = float(shape @ x) / max(float(shape @ shape), 1e-9)   # best brightness
    return p


def _lm(x: np.ndarray, p: np.ndarray) -> tuple[np.ndarray, float]:
    """Levenberg-Marquardt on the squared pixel error, from p."""
    r = _render(p)[0] - x
    cost = float(r @ r)
    lam = 1e-3
    eye = np.eye(6)
    for _ in range(ITERS):
        J = (_render(p + H * eye) - _render(p)) / H        # (6, 256)
        g, A = J @ r, J @ J.T
        improved = False
        while lam < 1e8:
            step = np.linalg.solve(A + lam * np.diag(np.diag(A) + 1e-9), -g)
            q = p + step
            rq = _render(q)[0] - x
            cq = float(rq @ rq)
            if cq < cost:
                gain = cost - cq
                p, r, cost = q, rq, cq
                lam = max(lam / 4.0, 1e-9)
                improved = True
                break
            lam *= 4.0
        if not improved or gain < 1e-10:
            break
    return p, cost


def encode(img):
    """img: list of 256 floats. Return the six fitted parameters."""
    x = np.asarray(img, dtype=float)
    p0 = _first_guess(x)
    best_p, best_cost = None, float("inf")
    for dth in STARTS:
        start = p0.copy()
        start[4] += dth
        p, cost = _lm(x, start)
        if cost < best_cost:
            best_p, best_cost = p, cost
    p = best_p.copy()
    p[2:4] = np.maximum(np.abs(p[2:4]), MIN_AXIS)
    p[4] = math.remainder(p[4], math.pi)                    # tilt in (-pi/2, pi/2]
    return [float(v) for v in p]


def decode(z):
    """z: the six numbers from encode. Return 256 floats."""
    return [float(v) for v in _render(np.asarray(z, dtype=float))[0]]

# ---------------------------------------------------------------------------
# end of our model
# ---------------------------------------------------------------------------


def fns():
    """The (encode, decode) pair race.py and the evaluator consume."""
    return encode, decode

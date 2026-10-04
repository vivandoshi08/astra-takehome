"""Data for the ellipses task: 16x16 images of one soft-edged ellipse each.

Every image is generated, not stored: an ellipse with center (cx, cy),
semi-axes a >= b, orientation theta, and intensity A, rendered with a soft
rim, plus Gaussian pixel noise with standard deviation SIGMA. Six numbers
fully determine the clean image.

PUBLIC is the sample the optimizer is scored on. HELD_OUT is a second,
larger sample — different ellipses, different noise — that is shipped but
never shown to the optimizer.

Images are flattened row-major: pixel (row i, col j) is index i * 16 + j.
"""

from __future__ import annotations

import numpy as np

IMG = 16                 # image side length
AMBIENT = IMG * IMG      # 256 pixels
LATENT = 6               # latent numbers candidates must use
SIGMA = 0.1              # pixel noise standard deviation

CX_RANGE = (5.0, 11.0)
CY_RANGE = (5.0, 11.0)
A_RANGE = (2.2, 4.4)     # semi-major axis
B_MIN = 1.4              # semi-minor axis: uniform in (B_MIN, a)
THETA_RANGE = (0.0, np.pi)
AMP_RANGE = (0.6, 1.0)   # intensity of the ellipse interior

PUBLIC_N = 128
HELD_OUT_N = 256
PUBLIC_SEED = 1
HELD_OUT_SEED = 999

PARAM_NAMES = ["cx", "cy", "a", "b", "theta", "A"]

_YY, _XX = np.mgrid[0:IMG, 0:IMG]


def render(params: np.ndarray) -> np.ndarray:
    """Clean ellipse images for parameter rows (cx, cy, a, b, theta, A),
    shape (n, 256).

    In the frame rotated by theta, rho is the elliptical radius
    sqrt((u/a)^2 + (v/b)^2). Pixel value = A * clip((1 - rho) * b + 0.5,
    0, 1): intensity ramps from A to 0 across roughly one pixel at the rim
    (exactly one pixel along the minor axis).
    """
    params = np.atleast_2d(np.asarray(params, dtype=float))
    cx = params[:, 0, None, None]
    cy = params[:, 1, None, None]
    a = params[:, 2, None, None]
    b = params[:, 3, None, None]
    th = params[:, 4, None, None]
    amp = params[:, 5, None, None]
    dx = _XX[None] - cx
    dy = _YY[None] - cy
    u = dx * np.cos(th) + dy * np.sin(th)
    v = -dx * np.sin(th) + dy * np.cos(th)
    rho = np.sqrt((u / a) ** 2 + (v / b) ** 2)
    img = amp * np.clip((1.0 - rho) * b + 0.5, 0.0, 1.0)
    return img.reshape(len(params), AMBIENT)


def sample(n: int, seed: int, sigma: float = SIGMA):
    """Return (params, clean, noisy): ellipse parameters, clean renders,
    observed images."""
    rng = np.random.default_rng(seed)
    a = rng.uniform(*A_RANGE, n)
    b = rng.uniform(B_MIN, a)
    params = np.stack([
        rng.uniform(*CX_RANGE, n),
        rng.uniform(*CY_RANGE, n),
        a,
        b,
        rng.uniform(*THETA_RANGE, n),
        rng.uniform(*AMP_RANGE, n),
    ], axis=1)
    clean = render(params)
    noisy = clean + sigma * rng.standard_normal(clean.shape)
    return params, clean, noisy


PUBLIC_PARAMS, PUBLIC_CLEAN, PUBLIC = sample(PUBLIC_N, PUBLIC_SEED)
HELD_OUT_PARAMS, HELD_OUT_CLEAN, HELD_OUT = sample(HELD_OUT_N, HELD_OUT_SEED)


def train_images(n: int = 2048, seed: int = 7, sigma: float = SIGMA):
    """Fresh generator samples, for training models of your own.
    Returns (params, clean, noisy)."""
    return sample(n, seed, sigma)


# ---------------------------------------------------------------------------
# Extension hook: alternative sprites.
#
# `render` draws one ellipse. If you want to probe how the task changes with
# a different family (discs, two shapes, sharper rims), add a renderer here
# and build your own point sets from `sample`'s pattern. The shipped
# PUBLIC / HELD_OUT sets above are fixed — leave them as they are so your
# scores stay comparable to ours.
# ---------------------------------------------------------------------------

"""YOUR model. This is the piece you implement.

Write an `encode`/`decode` pair, in whatever style you like — a
hand-designed estimator, a fitted model, something trained with `vae.py`,
anything — and race it against the loop and the baselines with `race.py`.
It is scored by exactly the same evaluator as every loop candidate
(`tasks/ellipses.py::evaluate`), under the same rules:

  - encode(img) sees ONLY the 256 floats of one image (plus any fixed
    constants or precomputed model weights you build in). It returns
    exactly 6 finite numbers.
  - decode(z) sees ONLY those 6 numbers. The latent is the only channel
    between the two functions.
  - Deterministic. No looking up the answer: the model must work because
    it measures the image, not because it recognises it.

Unlike loop candidates you are not limited to the standard library —
numpy, `data.render`, and models you train yourself are all fair game, as
long as the two rules above hold.

The placeholder below is the loop's seed program: it finds the position
and nothing else. Replace it.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# TODO: replace everything between these markers with your model.
# ---------------------------------------------------------------------------

def encode(img):
    """img: list of 256 floats. Return a list of exactly 6 numbers."""
    sx = sy = m = 0.0
    for i in range(16):
        for j in range(16):
            v = img[i * 16 + j]
            if v > 0.3:
                sx += v * j
                sy += v * i
                m += v
    if m == 0.0:
        return [8.0, 8.0, 3.0, 2.0, 0.0, 0.8]
    return [sx / m, sy / m, 3.0, 2.0, 0.0, 0.8]


def decode(z):
    """z: the 6-number list produced by encode. Return 256 floats."""
    import math
    cx, cy, a, b, th, amp = z
    ct, st = math.cos(th), math.sin(th)
    img = []
    for i in range(16):
        for j in range(16):
            dx = j - cx
            dy = i - cy
            u = dx * ct + dy * st
            v = -dx * st + dy * ct
            rho = ((u / a) ** 2 + (v / b) ** 2) ** 0.5
            p = (1.0 - rho) * b + 0.5
            img.append(amp * (0.0 if p < 0.0 else (1.0 if p > 1.0 else p)))
    return img

# ---------------------------------------------------------------------------
# end of your model
# ---------------------------------------------------------------------------


def fns():
    """The (encode, decode) pair race.py and the evaluator consume."""
    return encode, decode

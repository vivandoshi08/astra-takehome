"""Ellipse reconstruction — the default task for the optimizer loop.

A candidate is a pair of Python functions: an encoder that maps a 16x16
image (a flat list of 256 floats) to a 6-number latent code, and a decoder
that maps the code back to an image.

    def encode(img):
        return [cx, cy, a, b, theta, A]     # any 6 numbers you like

    def decode(z):
        return [...]                        # 256 floats

Scoring: negative mean squared reconstruction error per pixel on the
PUBLIC images, so higher is better and 0.0 is perfect.

encode and decode are compiled into SEPARATE namespaces (see
evo/loop.py::compile_candidate): the 6 latent numbers are the only channel
between them. The evaluator rejects any latent that is not exactly 6
finite numbers.

Your own model in `student.py` is scored by exactly this evaluator, under
exactly these rules — see `race.py`.

PUBLIC is what the loop optimises against. HELD_OUT is a second, larger
sample from the same generator — different ellipses, different noise —
that is shipped but never shown to the optimizer. Both are fixed by seed;
see `data.py` for how they are drawn.
"""

from __future__ import annotations

import math

import data

ENTRY_POINT = ("encode", "decode")
ISOLATED_NAMESPACES = True

IMG = data.IMG
AMBIENT = data.AMBIENT
LATENT = data.LATENT
PUBLIC = data.PUBLIC
HELD_OUT = data.HELD_OUT

CLIP = 1e6

# How many PUBLIC images are printed in the prompt (as digit grids, rounded
# to tenths). The model reasons about the structure from these; the score is
# still computed on all of PUBLIC.
SHOW_IMAGES = 3


def evaluate(fns, images=None) -> float:
    """Negative per-pixel MSE. Higher is better; 0.0 is perfect."""
    encode, decode = fns
    imgs = PUBLIC if images is None else images
    total = 0.0
    count = 0
    for row in imgs:
        img = [float(a) for a in row]
        try:
            z = encode(list(img))
        except Exception:
            return float("-inf")
        try:
            z = [float(a) for a in z]
        except (TypeError, ValueError):
            return float("-inf")
        if len(z) != LATENT or not all(math.isfinite(a) for a in z):
            return float("-inf")
        try:
            q = decode(list(z))
        except Exception:
            return float("-inf")
        if q is None:
            return float("-inf")
        try:
            q = [float(a) for a in q]
        except (TypeError, ValueError):
            return float("-inf")
        if len(q) != len(img):
            return float("-inf")
        for a, b in zip(q, img):
            err = a - b
            if not math.isfinite(err):
                return float("-inf")
            total += min(err * err, CLIP)
            count += 1
    return -total / count


def baseline_scores() -> dict[str, float]:
    """Reference points printed at the start of a run. The loop's job — and
    yours — is to beat the strongest of these."""
    import baselines
    return {
        "mean": evaluate(baselines.mean_fns()),
        "pca (k=6)": evaluate(baselines.pca_fns()),
        "vae (k=6)": evaluate(baselines.vae_fns()),
    }


def _digit_grid(img) -> str:
    """One image as 16 rows of digits: pixel value rounded to tenths,
    clipped to 0..9."""
    rows = []
    for i in range(IMG):
        row = img[i * IMG:(i + 1) * IMG]
        rows.append("  " + "".join(
            str(min(9, max(0, round(float(v) * 10)))) for v in row))
    return "\n".join(rows)


def _images_block(k: int) -> str:
    return "\n\n".join(_digit_grid(PUBLIC[i]) for i in range(k))


PROMPT_HEADER = """\
You are writing a model of a fixed set of small grayscale images.

Each image is 16x16, flattened row-major into a list of 256 floats
(index i*16 + j is row i, column j). Every image shows ONE bright filled
ellipse on a dark background: interior intensity is a constant A between
about 0.6 and 1.0 (it differs from image to image), the background is
about 0.0, and the rim fades over roughly one pixel. The ellipse has
semi-axes roughly 1.4 to 4.4 pixels, any orientation, and its center
varies from image to image. Every pixel carries independent noise of
standard deviation about 0.1.

Write TWO Python functions in ONE ```python block:

    def encode(img):
        # img is a list of 256 floats. Return a list of exactly 6
        # numbers: the latent code for this image.

    def decode(z):
        # z is the 6-number list produced by encode. Return a list of
        # 256 floats: the reconstructed image.

You are scored by the negative mean squared reconstruction error of
decode(encode(img)) over the images, per pixel: higher is better, 0.0 is
perfect. encode and decode run in separate namespaces — the 6 numbers are
the only thing decode receives. Deterministic standard-library Python
only; import math yourself if you use it. Answer with the single
```python block; keep any commentary brief.
"""

# The seed program, FunSearch-style: a minimal working candidate placed in
# the population before generation 0 (see evo/loop.py). Its decoder works
# and its encoder finds the position; everything else about the shape is
# deliberately left constant.
SEED = '''\
def encode(img):
    # Weighted centroid of the bright region. TODO: measure the shape too
    # (axes, orientation, brightness), not just the position.
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
'''

BLIND_FOOTER = """\
You do not get to see the images, only your score and the scores of
previous attempts.
"""


def prompt_header() -> str:
    if SHOW_IMAGES <= 0:
        return PROMPT_HEADER + "\n" + BLIND_FOOTER
    return (PROMPT_HEADER
            + f"\nHere are {min(SHOW_IMAGES, len(PUBLIC))} of the "
              f"{len(PUBLIC)} images, pixel values rounded to tenths "
              f"(0-9; the evaluator uses the raw floats):\n\n"
            + _images_block(SHOW_IMAGES) + "\n")

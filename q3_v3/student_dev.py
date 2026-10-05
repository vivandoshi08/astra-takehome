"""Develop and diagnose student.py on FRESH images only (never PUBLIC/HELD_OUT).

    python student_dev.py [--n 256] [--seeds 7 2024]

For each fresh sample from the generator it reports, through the task's own
evaluator:
  - student.py's score;
  - the clean truth (the noise floor before fitting) and the theory value for
    a perfect 6-parameter fit, -sigma^2 (1 - 6/256) = -0.00977 in expectation;
  - an "oracle-start" fit: the same Levenberg-Marquardt, started from the
    true parameters. Caution (independent check): where student beats it,
    it is almost always a swapped-axis fit (|b| > |a|) that matches the
    noisy image better and the clean one worse, not an escaped local
    minimum; the theory value is a linear-fit approximation, not a bound.
  - the worst images and their shapes (a/b, size), to see where it fails.
"""

from __future__ import annotations

import argparse
import time

import numpy as np

import data
import student
from tasks import ellipses


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--seeds", type=int, nargs="+", default=[7, 2024])
    args = ap.parse_args()

    for seed in args.seeds:
        assert seed not in (data.PUBLIC_SEED, data.HELD_OUT_SEED)
        params, clean, noisy = data.train_images(args.n, seed=seed)
        t = time.time()
        s = ellipses.evaluate(student.fns(), images=noisy)
        dt = time.time() - t

        per_img, oracle = [], []
        for p, x in zip(params, noisy):
            z = student.encode(list(x))
            per_img.append(float(np.mean((np.array(student.decode(z)) - x) ** 2)))
            po, _ = student._lm(x, p.copy())
            oracle.append(float(np.mean((student._render(po)[0] - x) ** 2)))
        per_img, oracle = np.array(per_img), np.array(oracle)
        truth = float(np.mean((clean - noisy) ** 2))

        print(f"seed {seed} ({args.n} fresh images, {dt:.1f}s, "
              f"{1000 * dt / args.n:.1f} ms/image)")
        print(f"  student            {s:+.6f}")
        print(f"  oracle-start fit   {-oracle.mean():+.6f}")
        print(f"  clean truth        {-truth:+.6f}")
        print(f"  theory (expected)  {-data.SIGMA ** 2 * (1 - 6 / 256):+.6f}")
        excess = per_img - oracle
        bad = np.argsort(-excess)[:5]
        # excess < 0: student found a lower minimum than the oracle start did
        print(f"  student - oracle-start, mean per image: {excess.mean():+.7f}; "
              f"worse by >1e-4 MSE on {int((excess > 1e-4).sum())}/{args.n}, "
              f"better by >1e-4 on {int((excess < -1e-4).sum())}/{args.n}")
        for i in bad:
            cx, cy, a, b, th, A = params[i]
            print(f"    img {i:3d}: excess {excess[i]:.5f}  a/b {a / b:4.2f}  "
                  f"a {a:.2f} b {b:.2f}  A {A:.2f}")
        print()


if __name__ == "__main__":
    main()

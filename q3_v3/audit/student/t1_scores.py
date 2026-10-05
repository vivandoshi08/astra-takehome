"""Headline numbers, fresh seeds, transformed PUBLIC images.

For each set: student score via tasks.ellipses.evaluate, clean-truth MSE,
per-set "theory" = -(clean_mse - 6 sigma^2/256), expected -sigma^2(1-6/256),
and the gap student - per-set theory.
"""
import time
import numpy as np
import data
import student
from tasks import ellipses

S2 = data.SIGMA ** 2
EXP = -S2 * (1 - 6 / 256)


def row(name, noisy, clean):
    t = time.time()
    s = ellipses.evaluate(student.fns(), images=noisy)
    dt = time.time() - t
    cm = float(np.mean((noisy - clean) ** 2))
    th = -cm + 6 * S2 / 256
    print(f"{name:28s} n={len(noisy):4d} student {s:+.6f} | clean {-cm:+.6f} | "
          f"per-set theory {th:+.6f} | student-theory {s - th:+.2e} | "
          f"{1000 * dt / len(noisy):.1f} ms/img", flush=True)
    return s, th


print(f"expected optimum -sigma^2(1-6/256) = {EXP:+.6f}\n")
print("== graded / held-out ==")
row("PUBLIC", data.PUBLIC, data.PUBLIC_CLEAN)
row("HELD_OUT", data.HELD_OUT, data.HELD_OUT_CLEAN)

print("\n== DEVLOG dev seeds ==")
for seed in (7, 2024):
    p, c, x = data.train_images(256, seed=seed)
    row(f"fresh seed {seed}", x, c)

print("\n== fresh seeds never used ==")
gaps, scores = [], []
for seed in (3, 11, 42, 100, 314, 2718, 31337, 55555, 8, 123456):
    assert seed not in (1, 999, 7, 2024)
    p, c, x = data.train_images(256, seed=seed)
    s, th = row(f"fresh seed {seed}", x, c)
    gaps.append(s - th); scores.append(s)
gaps = np.array(gaps)
print(f"fresh seeds: score mean {np.mean(scores):+.6f} (expected {EXP:+.6f}); "
      f"student - per-set theory: mean {gaps.mean():+.2e}, sd {gaps.std(ddof=1):.2e}, "
      f"se {gaps.std(ddof=1) / np.sqrt(len(gaps)):.2e}, positive in {int((gaps > 0).sum())}/{len(gaps)}")

print("\n== PUBLIC transformed (same noise values, permuted; exact model-family images) ==")
X = data.PUBLIC.reshape(-1, 16, 16)
C = data.PUBLIC_CLEAN.reshape(-1, 16, 16)
for nm, f in (("flip left-right", lambda a: a[:, :, ::-1]),
              ("flip up-down", lambda a: a[:, ::-1, :]),
              ("transpose", lambda a: a.transpose(0, 2, 1)),
              ("rot90", lambda a: np.rot90(a, 1, axes=(1, 2))),
              ("rot180", lambda a: a[:, ::-1, ::-1])):
    row(nm, f(X).reshape(-1, 256).copy(), f(C).reshape(-1, 256).copy())

# integer shifts: keep only images whose clean render shifts without wrap
for (dy, dx) in ((0, 1), (1, 0), (-1, -1), (2, -2)):
    keep = []
    for k in range(len(X)):
        P = data.PUBLIC_PARAMS[k].copy(); P[0] += dx; P[1] += dy
        if np.allclose(np.roll(C[k], (dy, dx), axis=(0, 1)).ravel(), data.render(P)[0], atol=1e-12):
            keep.append(k)
    keep = np.array(keep)
    xs = np.roll(X[keep], (dy, dx), axis=(1, 2)).reshape(-1, 256).copy()
    cs = np.roll(C[keep], (dy, dx), axis=(1, 2)).reshape(-1, 256).copy()
    s_orig = ellipses.evaluate(student.fns(), images=data.PUBLIC[keep])
    s, th = row(f"shift dy={dy} dx={dx}", xs, cs)
    print(f"{'':28s} same {len(keep)} images unshifted: {s_orig:+.6f}")

# PUBLIC clean + brand-new noise
rng = np.random.default_rng(424242)
Xn = data.PUBLIC_CLEAN + data.SIGMA * rng.standard_normal(data.PUBLIC_CLEAN.shape)
row("PUBLIC_CLEAN + new noise", Xn, data.PUBLIC_CLEAN)

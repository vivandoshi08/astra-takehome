"""Is the shipped setting special on PUBLIC (sign of tuning on PUBLIC)?
And what do the swapped-axes fits do to the CLEAN image?"""
import math
import warnings
import numpy as np
import data
import student
from tasks import ellipses

warnings.simplefilter("ignore")
F = [data.train_images(256, seed=s) for s in (5, 6)]
FX = np.concatenate([f[2] for f in F])

base = dict(THRESHOLD=student.THRESHOLD, ITERS=student.ITERS, STARTS=student.STARTS, H=student.H,
            MIN_AXIS=student.MIN_AXIS)
settings = [
    {}, {"THRESHOLD": 0.2}, {"THRESHOLD": 0.4}, {"ITERS": 20}, {"ITERS": 80},
    {"STARTS": (0.0,)}, {"STARTS": (0.0, math.pi / 2)},
    {"STARTS": tuple(np.arange(6) * math.pi / 6)}, {"H": 1e-7}, {"H": 1e-4}, {"MIN_AXIS": 0.1},
]
print(f"{'setting':42s} {'PUBLIC':>10s} {'HELD_OUT':>10s} {'fresh 5+6':>10s}")
for s in settings:
    for k, v in base.items():
        setattr(student, k, v)
    for k, v in s.items():
        setattr(student, k, v)
    r = [ellipses.evaluate(student.fns(), images=X) for X in (data.PUBLIC, data.HELD_OUT, FX)]
    lab = ", ".join(f"{k}={v if not isinstance(v, tuple) else len(v)}" for k, v in s.items()) or "shipped"
    print(f"{lab:42s} {r[0]:+.6f} {r[1]:+.6f} {r[2]:+.6f}", flush=True)
for k, v in base.items():
    setattr(student, k, v)

# swapped (|b|>|a|) fits vs clean truth, fresh seed 5
P, C, X = F[0]
Z = np.array([student.encode(list(x)) for x in X])
R = data.render(np.c_[Z[:, :2], np.maximum(np.abs(Z[:, 2:4]), 0.3), Z[:, 4:]])
sw = np.abs(Z[:, 3]) > np.abs(Z[:, 2])
noisy = np.mean((R - X) ** 2, axis=1); clean = np.mean((R - C) ** 2, axis=1)
ab = P[:, 2] / P[:, 3]
print(f"\nseed 5: swapped fits {sw.sum()}/256; true a/b of those: median {np.median(ab[sw]):.2f} "
      f"(others {np.median(ab[~sw]):.2f})")
print(f"  MSE vs NOISY: swapped {noisy[sw].mean():.6f}, unswapped {noisy[~sw].mean():.6f}")
print(f"  MSE vs CLEAN (denoising error): swapped {clean[sw].mean():.2e}, unswapped {clean[~sw].mean():.2e}")
# restrict to the generator's branch: from each swapped fit, refit from its a>=b twin
fixed = []
for k in np.where(sw)[0]:
    z = Z[k].copy(); q = z.copy(); q[2], q[3] = z[3], z[2]; q[4] += np.pi / 2
    old = student.ITERS; student.ITERS = 300
    pq, cq = student._lm(X[k], q)
    student.ITERS = old
    fixed.append((cq / 256, np.mean((student._render(pq)[0] - C[k]) ** 2), abs(pq[3]) > abs(pq[2])))
fixed = np.array(fixed)
print(f"  same images refit in the a>=b branch: MSE vs noisy {fixed[:,0].mean():.6f}, "
      f"vs clean {fixed[:,1].mean():.2e}; still swapped after refit {int(fixed[:,2].sum())}")

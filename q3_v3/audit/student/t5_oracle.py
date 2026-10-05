"""Is the 'oracle-start' a fair reference, and how far is student from the
global least-squares optimum?  (fresh seeds only)"""
import sys
import warnings
import numpy as np
import data
import student

warnings.simplefilter("ignore")
S2 = data.SIGMA ** 2
seeds = [int(s) for s in sys.argv[1:]] or [7, 2024, 3, 11]


def lm(x, p, iters):
    old = student.ITERS
    student.ITERS = iters
    try:
        return student._lm(x, p.copy())
    finally:
        student.ITERS = old


def swapped(p):
    q = p.copy(); q[2], q[3] = p[3], p[2]; q[4] += np.pi / 2
    return q


for seed in seeds:
    P, C, X = data.train_images(256, seed=seed)
    n = len(X)
    s_stu, s_o40, s_o1k, s_strong, s_stu_polish = (np.empty(n) for _ in range(5))
    stu_swapped = np.zeros(n, bool)
    o_swapped = np.zeros(n, bool)
    for k, (p, x) in enumerate(zip(P, X)):
        z = np.array(student.encode(list(x)))
        s_stu[k] = np.sum((np.array(student.decode(list(z))) - x) ** 2)
        stu_swapped[k] = abs(z[3]) > abs(z[2])
        po, c40 = lm(x, p, 40); s_o40[k] = c40
        o_swapped[k] = abs(po[3]) > abs(po[2])
        _, s_o1k[k] = lm(x, p, 1000)
        _, s_stu_polish[k] = lm(x, z, 1000)
        starts = [p, swapped(p), z, swapped(z)]
        g = student._first_guess(x)
        for dth in np.arange(12) * np.pi / 12:
            q = g.copy(); q[4] += dth; starts.append(q); starts.append(swapped(q))
        for dth in (np.pi / 3, -np.pi / 3):
            q = p.copy(); q[4] += dth; starts.append(q)
        best = min(lm(x, s, 300)[1] for s in starts)
        s_strong[k] = min(best, s_stu[k])
    cm = np.mean((C - X) ** 2)
    th = -cm + 6 * S2 / 256
    f = lambda a: -a.mean() / 256
    print(f"seed {seed}: per-set theory {th:+.6f}")
    print(f"  student {f(s_stu):+.6f} | oracle-start 40 it {f(s_o40):+.6f} | oracle-start 1000 it "
          f"{f(s_o1k):+.6f} | student polished 1000 it {f(s_stu_polish):+.6f} | strong multistart {f(s_strong):+.6f}")
    ex_o = (s_stu - s_o40) / 256
    ex_s = (s_stu - s_strong) / 256
    print(f"  student - oracle40 per image: mean {ex_o.mean():+.2e}; worse >1e-4: {(ex_o > 1e-4).sum()}, "
          f"better >1e-4: {(ex_o < -1e-4).sum()}")
    print(f"  student - strong per image:   mean {ex_s.mean():+.2e}; worse >1e-4: {(ex_s > 1e-4).sum()}, "
          f"worse >1e-5: {(ex_s > 1e-5).sum()}, max {ex_s.max():.2e}")
    big = np.where(ex_o < -1e-4)[0]
    print(f"  images where student beats oracle40 by >1e-4: {len(big)}; of these student-fit swapped (|b|>|a|): "
          f"{stu_swapped[big].sum()}, oracle40 swapped: {o_swapped[big].sum()}; "
          f"oracle 1000 it closes the gap on {(np.abs(s_o1k[big] - s_stu[big]) / 256 < 1e-5).sum()}")
    print(f"  swapped (|b|>|a|) student fits overall: {stu_swapped.sum()}/{n}; oracle40 fits: {o_swapped.sum()}/{n}")
    print(f"  strong - per-set theory {f(s_strong) - th:+.2e}; oracle40 - theory {f(s_o40) - th:+.2e}", flush=True)

"""Maths checks for student.py."""
import math
import numpy as np
import data
import student

rng = np.random.default_rng(0)

# 1. filled ellipse: variance a^2/4 along major axis, b^2/4 along minor -------
print("== 1. variance of a uniformly filled ellipse ==")
for a, b in ((4.0, 1.5), (3.0, 3.0), (2.2, 1.4)):
    n = 4_000_000
    u = rng.uniform(-a, a, n); v = rng.uniform(-b, b, n)
    keep = (u / a) ** 2 + (v / b) ** 2 <= 1
    u, v = u[keep], v[keep]
    print(f"  a={a} b={b}: var_u={u.var():.5f} (a^2/4={a*a/4:.5f})  "
          f"var_v={v.var():.5f} (b^2/4={b*b/4:.5f})")
# analytic: E[u^2] = a^2 E[r^2 cos^2] with r^2~U(0,1)? r has density 2r -> E[r^2]=1/2, E[cos^2]=1/2 -> a^2/4

# 2. first guess on CLEAN renders -------------------------------------------
print("\n== 2. first guess on clean renders (no noise), fresh seed 77, n=2000 ==")
P, C, X = data.train_images(2000, seed=77)
G = np.array([student._first_guess(c) for c in C])
def angdiff(t1, t2):
    d = (t1 - t2) % np.pi
    return np.minimum(d, np.pi - d)
ecc = P[:, 2] / P[:, 3] > 1.2
print(f"  centre error: median |dcx| {np.median(np.abs(G[:,0]-P[:,0])):.4f} |dcy| {np.median(np.abs(G[:,1]-P[:,1])):.4f}")
print(f"  a_hat/a median {np.median(G[:,2]/P[:,2]):.4f} [5%,95%] {np.percentile(G[:,2]/P[:,2],[5,95]).round(4)}; "
      f"b_hat/b median {np.median(G[:,3]/P[:,3]):.4f} [5%,95%] {np.percentile(G[:,3]/P[:,3],[5,95]).round(4)}")
print(f"  theta error mod pi (a/b>1.2, n={ecc.sum()}): median {np.degrees(np.median(angdiff(G[ecc,4],P[ecc,4]))):.3f} deg, "
      f"max {np.degrees(np.max(angdiff(G[ecc,4],P[ecc,4]))):.3f} deg")
# what would the WRONG sign convention give? (theta -> -theta)
print(f"  same with theta negated (wrong convention would match this): median "
      f"{np.degrees(np.median(angdiff(-G[ecc,4],P[ecc,4]))):.2f} deg")
# moments of the soft render itself, no threshold, dense check of 2*sqrt(lambda)
print(f"  A_hat/A median {np.median(G[:,5]/P[:,5]):.4f}")

# 3. brightness least squares ------------------------------------------------
print("\n== 3. brightness least squares ==")
x = X[5]
p = student._first_guess(x)
def cost_A(A):
    q = p.copy(); q[5] = A
    return float(np.sum((student._render(q)[0] - x) ** 2))
As = np.linspace(p[5] - 0.05, p[5] + 0.05, 20001)
best = As[np.argmin([cost_A(A) for A in As])]
lin = np.max(np.abs(data.render(np.r_[p[:5], 0.7])[0] - 0.7 * data.render(np.r_[p[:5], 1.0])[0]))
print(f"  closed form A={p[5]:.6f}, grid argmin A={best:.6f}; render linear in A: max|r(0.7)-0.7 r(1)|={lin:.1e}")

# 4. render(theta) == render(theta + k pi); math.remainder range ------------
print("\n== 4. theta periodicity and remainder ==")
Pq = P[:500].copy()
mx = 0.0
for k in (1, -1, 2, 7, -13):
    Q = Pq.copy(); Q[:, 4] += k * np.pi
    mx = max(mx, np.max(np.abs(data.render(Q) - data.render(Pq))))
print(f"  max |render(theta+k pi) - render(theta)| over 500 params, k in (1,-1,2,7,-13): {mx:.2e}")
tt = np.concatenate([rng.uniform(-50, 50, 100000), [math.pi / 2, -math.pi / 2, 3 * math.pi / 2, math.pi, 0.0]])
r = np.array([math.remainder(t, math.pi) for t in tt])
print(f"  remainder range [{r.min():.17f}, {r.max():.17f}] vs pi/2={math.pi/2:.17f}; "
      f"remainder(-pi/2)={math.remainder(-math.pi/2, math.pi)!r}")

# 5. encode's z decodes to exactly what the fit evaluated --------------------
print("\n== 5. decode(encode(x)) vs the fit's own render (MIN_AXIS clamp, remainder) ==")
mx, mxc = 0.0, 0.0
nclamp = nswap = nneg = 0
Pf, Cf, Xf = data.train_images(256, seed=4242)
for x in Xf:
    p0 = student._first_guess(x)
    best_p, best_c = None, np.inf
    for dth in student.STARTS:
        s = p0.copy(); s[4] += dth
        pp, c = student._lm(x, s)
        if c < best_c:
            best_p, best_c = pp, c
    nneg += int(np.any(best_p[2:4] < 0)); nclamp += int(np.any(np.abs(best_p[2:4]) < student.MIN_AXIS))
    nswap += int(abs(best_p[3]) > abs(best_p[2]))
    z = student.encode(list(x))
    d = np.array(student.decode(z))
    fit_img = student._render(best_p)[0]
    mx = max(mx, np.max(np.abs(d - fit_img)))
    mxc = max(mxc, abs(float(np.sum((d - x) ** 2)) - best_c))
print(f"  max |decode(encode(x)) - render(best_p)| = {mx:.2e}; max |SSE(decode) - fit cost| = {mxc:.2e}")
print(f"  fits with a negative axis: {nneg}/256, an axis below MIN_AXIS: {nclamp}/256, |b|>|a| (swapped): {nswap}/256")

# 6. swapped axes are NOT a symmetry of render (rim width uses b) -----------
print("\n== 6. is (a,b,th) == (b,a,th+pi/2) in render? ==")
p = np.array([8.0, 8.0, 4.0, 2.0, 0.3, 1.0]); q = np.array([8.0, 8.0, 2.0, 4.0, 0.3 + np.pi / 2, 1.0])
print(f"  max |render(p)-render(q)| = {np.max(np.abs(data.render(p)-data.render(q))):.3f}  (rim ramp scales with b)")

# 7. finite-difference Jacobian accuracy -------------------------------------
print("\n== 7. FD Jacobian (H=1e-5, forward) vs central difference H=1e-7 / complex-free check ==")
errs = []
for k in range(50):
    p = P[k].copy()
    J = (student._render(p + student.H * np.eye(6)) - student._render(p)) / student.H
    h2 = 1e-6
    Jc = (student._render(p + h2 * np.eye(6)) - student._render(p - h2 * np.eye(6))) / (2 * h2)
    errs.append(np.max(np.abs(J - Jc)) / np.max(np.abs(Jc)))
print(f"  relative max |J_fwd - J_central|: median {np.median(errs):.1e}, max {np.max(errs):.1e} "
      "(spikes come from pixels at the clip kink)")

# 8. expected optimum: sigma^2 (1 - p/n) for a linear LS fit, sanity -------
print("\n== 8. linear-model check of E[RSS]/n = sigma^2 (1 - p/n) ==")
n, pdim, s = 256, 6, 0.1
Jl = rng.standard_normal((n, pdim))
Pr = Jl @ np.linalg.pinv(Jl)
e = s * rng.standard_normal((40000, n))
res = e - e @ Pr.T
print(f"  simulated {np.mean(res**2):.7f} vs sigma^2(1-6/256) = {s*s*(1-6/256):.7f}")

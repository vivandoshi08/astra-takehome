"""Audit data.py (generator, render) and the claims made about it in the
docstrings and in the prompt the LLM sees (tasks/ellipses.py PROMPT_HEADER,
digit grids, SEED).

Run from q3_v3/:
    python <this file>
Writes nothing.
"""
from __future__ import annotations

import sys

import numpy as np

sys.path.insert(0, ".")
import data  # noqa: E402
from evo.loop import compile_candidate  # noqa: E402
from tasks import ellipses  # noqa: E402


def hdr(s):
    print("\n" + "=" * 72 + "\n" + s + "\n" + "=" * 72)


def render_xy(params, X, Y):
    """data.render's formula at arbitrary (x, y) points (x = column, y = row)."""
    cx, cy, a, b, th, amp = params
    dx, dy = X - cx, Y - cy
    u = dx * np.cos(th) + dy * np.sin(th)
    v = -dx * np.sin(th) + dy * np.cos(th)
    rho = np.sqrt((u / a) ** 2 + (v / b) ** 2)
    return amp * np.clip((1.0 - rho) * b + 0.5, 0.0, 1.0)


P_all = np.concatenate([data.PUBLIC_PARAMS, data.HELD_OUT_PARAMS])
big_params, big_clean, big_noisy = data.sample(200_000, seed=12345)

# --------------------------------------------------------------------------
hdr("1. render() matches its own formula on the grid; pixel-centre convention")
yy, xx = np.mgrid[0:16, 0:16]
mx = max(np.max(np.abs(render_xy(p, xx, yy).ravel() - data.render(p)[0]))
         for p in P_all)
print(f"max |render_xy - data.render| over PUBLIC+HELD_OUT params: {mx:.2e}")
print("pixel (row i, col j) sits at (x=j, y=i), integer coordinates 0..15; "
      "image centre is (7.5, 7.5)")
print(f"render(PUBLIC_PARAMS) == PUBLIC_CLEAN: "
      f"{np.array_equal(data.render(data.PUBLIC_PARAMS), data.PUBLIC_CLEAN)}")

# SEED decoder vs data.render
seed_enc, seed_dec = compile_candidate(ellipses.SEED, ellipses.ENTRY_POINT,
                                       isolated=True)
mx = max(np.max(np.abs(np.array(seed_dec(list(p))) - data.render(p)[0]))
         for p in P_all)
print(f"max |SEED.decode(p) - data.render(p)| over PUBLIC+HELD_OUT params: {mx:.2e}")

# --------------------------------------------------------------------------
hdr("2. Rim ramp width (docstring: 'exactly one pixel along the minor axis')")
for p in [(8, 8, 4.4, 1.4, 0.0, 1.0), (8, 8, 3.0, 3.0, 0.0, 1.0),
          (8, 8, 4.0, 2.0, 0.7, 0.8)]:
    cx, cy, a, b, th, amp = p
    t = np.linspace(0, 8, 800_001)
    # along the major axis direction (cos th, sin th), minor (-sin th, cos th)
    for name, (ex, ey) in (("major", (np.cos(th), np.sin(th))),
                           ("minor", (-np.sin(th), np.cos(th)))):
        val = render_xy(p, cx + t * ex, cy + t * ey) / amp
        ramp = t[(val > 1e-9) & (val < 1 - 1e-9)]
        print(f"  a={a} b={b}: {name}-axis ramp from r={ramp.min():.4f} to "
              f"{ramp.max():.4f}  width={ramp.max() - ramp.min():.4f} px "
              f"(theory: {'a/b = %.4f' % (a / b) if name == 'major' else '1'})")
ab = P_all[:, 2] / P_all[:, 3]
ab_big = big_params[:, 2] / big_params[:, 3]
print(f"major-axis ramp width a/b on PUBLIC+HELD_OUT: min {ab.min():.3f} "
      f"median {np.median(ab):.3f} p90 {np.percentile(ab, 90):.3f} max {ab.max():.3f}")
print(f"  generator (200k draws): median {np.median(ab_big):.3f}, "
      f"P(a/b > 1.5) = {np.mean(ab_big > 1.5):.3f}, P(a/b > 2) = {np.mean(ab_big > 2):.3f}, "
      f"sup = 4.4/1.4 = {4.4 / 1.4:.3f}")
print("half-intensity contour is exactly rho = 1 (the nominal ellipse): "
      f"{render_xy((8, 8, 3, 2, 0, 1), np.array([11.0]), np.array([8.0]))[0]:.3f}"
      f" at u=a, {render_xy((8, 8, 3, 2, 0, 1), np.array([8.0]), np.array([10.0]))[0]:.3f} at v=b")

# Plateau ('interior intensity is a constant A') vs ramp pixel counts
C = big_clean[:20000]
A = big_params[:20000, 5:6]
plateau = np.isclose(C, A, rtol=0, atol=1e-12)
ramp = (C > 1e-12) & ~plateau
print(f"pixels at the full value A per image: mean {plateau.sum(1).mean():.1f}, "
      f"ramp (0<v<A) pixels: mean {ramp.sum(1).mean():.1f}; "
      f"ramp share of non-zero pixels: {ramp.sum() / (ramp.sum() + plateau.sum()):.3f}")
print(f"images with zero plateau pixels: {np.mean(plateau.sum(1) == 0):.4f}; "
      f"min plateau count {plateau.sum(1).min()}")
print(f"signal energy in ramp pixels: {np.sum(C[ramp]**2) / np.sum(C**2):.3f}")

# --------------------------------------------------------------------------
hdr("3. Parameter ranges, a >= b, theta range, degeneracies")
for nm, (lo, hi) in (("cx", data.CX_RANGE), ("cy", data.CY_RANGE),
                     ("a", data.A_RANGE), ("theta", data.THETA_RANGE),
                     ("A", data.AMP_RANGE)):
    j = data.PARAM_NAMES.index(nm)
    v = big_params[:, j]
    print(f"  {nm:5s} range [{v.min():.4f}, {v.max():.4f}]  declared [{lo:.4f}, {hi:.4f})")
print(f"  b in [{big_params[:, 3].min():.4f}, {big_params[:, 3].max():.4f}]; "
      f"b < a always: {np.all(big_params[:, 3] < big_params[:, 2])}")
print(f"  marginal of b is not uniform: mean b {big_params[:, 3].mean():.3f} "
      f"(uniform on [1.4,4.4] would give 2.9); P(b<2) {np.mean(big_params[:, 3] < 2):.3f}")
p = np.array([8.3, 7.1, 3.7, 2.1, 0.4, 0.9])
q = p.copy(); q[4] += np.pi
print(f"  theta -> theta+pi gives identical image: "
      f"max diff {np.max(np.abs(data.render(p) - data.render(q))):.2e}")
q = p.copy(); q[2], q[3], q[4] = p[3], p[2], p[4] + np.pi / 2
print(f"  swapping (a,b) and theta+pi/2 is NOT the same image (rim uses b): "
      f"max diff {np.max(np.abs(data.render(p) - data.render(q))):.3f}")
for thr in (1.05, 1.1, 1.2):
    print(f"  near-circular a/b < {thr}: PUBLIC {np.mean(data.PUBLIC_PARAMS[:, 2] / data.PUBLIC_PARAMS[:, 3] < thr):.3f}"
          f"  generator {np.mean(ab_big < thr):.3f}  (theta weakly identified)")
# theta wrap: images near theta=0 and theta=pi are the same; a regression
# target theta has a discontinuity there
print("  theta wrap-around: theta=0.01 and theta=pi-0.01 differ by "
      f"{np.max(np.abs(data.render([8, 8, 4, 2, 0.01, 1]) - data.render([8, 8, 4, 2, np.pi - 0.01, 1]))):.3f} max pixel"
      " (tiny), yet their theta values differ by ~3.12")

# --------------------------------------------------------------------------
hdr("4. Border clipping: how often does the ellipse leave the 16x16 grid?")
# render on an extended grid and see how much non-zero mass falls outside 0..15
ext = np.arange(-10, 26)
EY, EX = np.meshgrid(ext, ext, indexing="ij")
inside = (EX >= 0) & (EX <= 15) & (EY >= 0) & (EY <= 15)
def clip_stats(P, label):
    lost_any, lost_frac, sides = [], [], np.zeros(4, int)
    for p in P:
        img = render_xy(p, EX, EY)
        out = img * ~inside
        lost_any.append(out.sum() > 0)
        lost_frac.append(out.sum() / img.sum())
        nz = img > 0
        sides += [np.any(nz & (EX < 0)), np.any(nz & (EX > 15)),
                  np.any(nz & (EY < 0)), np.any(nz & (EY > 15))]
    lost_any, lost_frac = np.array(lost_any), np.array(lost_frac)
    print(f"  {label}: {lost_any.mean():.3f} of images have non-zero ellipse "
          f"pixels outside the grid; mass lost: mean {lost_frac.mean():.4f}, "
          f"max {lost_frac.max():.3f}; images losing >5% of mass: {np.mean(lost_frac > 0.05):.3f}")
    print(f"     side counts (left x<0, right x>15, top y<0, bottom y>15): {sides.tolist()}")
    return lost_any, lost_frac
clip_stats(data.PUBLIC_PARAMS, "PUBLIC  ")
clip_stats(data.HELD_OUT_PARAMS, "HELD_OUT")
clip_stats(big_params[:5000], "gen 5000")
print(f"  centre range [5, 11] is centred on 8, not on the image centre 7.5: "
      f"margin to left/top edge 5 px, to right/bottom edge 4 px")

# --------------------------------------------------------------------------
hdr("5. Noise and the clean-truth score")
for nm, X, Cc in (("PUBLIC", data.PUBLIC, data.PUBLIC_CLEAN),
                  ("HELD_OUT", data.HELD_OUT, data.HELD_OUT_CLEAN)):
    r = X - Cc
    N = r.size
    mse = float(np.mean(r ** 2))
    se = data.SIGMA ** 2 * np.sqrt(2.0 / N)
    print(f"  {nm:8s}: noise std {r.std():.5f}, mean {r.mean():+.5f}; clean-truth MSE "
          f"{mse:.6f}  (sigma^2 = {data.SIGMA**2:.6f}, sampling s.e. {se:.6f}, z = {(mse - 0.01) / se:+.2f})")
print(f"  pixel value range PUBLIC: [{data.PUBLIC.min():.3f}, {data.PUBLIC.max():.3f}]; "
      f"fraction negative {np.mean(data.PUBLIC < 0):.3f}")
print(f"  PUBLIC/HELD_OUT/train_images use seeds {data.PUBLIC_SEED}/{data.HELD_OUT_SEED}/7: "
      f"independent streams")

# --------------------------------------------------------------------------
hdr("6. Prompt claims vs generator")
print("  'interior intensity A between about 0.6 and 1.0'  -> AMP_RANGE", data.AMP_RANGE)
print("  'semi-axes roughly 1.4 to 4.4'  -> b in (1.4, a), a in", data.A_RANGE,
      "(semi-major never below 2.2: not stated)")
print("  'rim fades over roughly one pixel' -> 1 px along minor axis, a/b px along "
      f"major axis (median {np.median(ab):.2f}, max {ab.max():.2f} on PUBLIC+HELD_OUT)")
print("  'background about 0.0', 'noise sd about 0.1' -> exact 0.0 and 0.1: OK")
print("  centre range [5,11]^2 and the frequent border clipping are not mentioned")

# --------------------------------------------------------------------------
hdr("7. Digit grids shown to the LLM (PUBLIC[0:3])")
for k in range(ellipses.SHOW_IMAGES):
    p = data.PUBLIC_PARAMS[k]
    img = data.PUBLIC[k]
    v10 = img * 10
    at9 = np.sum(v10 >= 9.5)
    at0 = np.sum(v10 < -0.5)
    exact_half = np.sum(np.isclose(v10 - np.floor(v10), 0.5, atol=1e-9))
    grid = ellipses._digit_grid(img)
    digits = np.array([[int(c) for c in line.strip()] for line in grid.splitlines()])
    recon = digits.ravel() / 10
    print(f"  image {k}: cx={p[0]:.2f} cy={p[1]:.2f} a={p[2]:.2f} b={p[3]:.2f} "
          f"theta={p[4]:.2f} A={p[5]:.2f} a/b={p[2]/p[3]:.2f}; "
          f"clipped-to-9 (v>=0.95): {at9}, clipped-to-0 (v<-0.05): {at0}, "
          f"exact .x5 ties: {exact_half}; |grid/10 - v| max {np.max(np.abs(recon - img)):.3f}")
lost9 = [np.sum(data.PUBLIC[k] * 10 >= 9.5) for k in range(128)]
print(f"  across PUBLIC: images with any pixel shown as 9 though >= 0.95: "
      f"{np.mean(np.array(lost9) > 0):.3f}; 9 means 'v >= 0.85' (anything brighter is hidden)")
print(f"  round(2.5) = {round(2.5)}, round(3.5) = {round(3.5)} (banker's rounding; "
      "ties are measure-zero for noisy floats)")
print(f"  PUBLIC[0:3] params vs PUBLIC distribution: a pct "
      f"{[int(np.mean(data.PUBLIC_PARAMS[:, 2] <= data.PUBLIC_PARAMS[k, 2]) * 100) for k in range(3)]}, "
      f"A pct {[int(np.mean(data.PUBLIC_PARAMS[:, 5] <= data.PUBLIC_PARAMS[k, 5]) * 100) for k in range(3)]}")
print("\n--- prompt as the LLM sees it (images part) ---")
print(ellipses.prompt_header().split("rounded to tenths")[1][:1200])

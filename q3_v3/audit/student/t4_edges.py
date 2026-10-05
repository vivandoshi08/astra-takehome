"""Edge cases for encode and decode."""
import math
import warnings
import numpy as np
import data
import student
from tasks import ellipses

warnings.simplefilter("ignore")
rng = np.random.default_rng(1)


def try_encode(name, img):
    try:
        z = student.encode(list(map(float, img)))
        ok = len(z) == 6 and all(math.isfinite(v) for v in z)
        d = student.decode(z)
        okd = len(d) == 256 and all(math.isfinite(v) for v in d)
        mse = float(np.mean((np.array(d) - np.asarray(img, float)) ** 2)) if okd else float("nan")
        print(f"  {name:38s} z finite {ok}, decode finite {okd}, mse {mse:.4g}, z={np.round(z, 3).tolist()}")
    except Exception as e:
        print(f"  {name:38s} EXCEPTION {type(e).__name__}: {e}")


print("== encode on odd images ==")
try_encode("all zeros", np.zeros(256))
try_encode("all ones", np.ones(256))
try_encode("all 0.31 (just above threshold)", np.full(256, 0.31))
try_encode("all -1", -np.ones(256))
try_encode("single bright pixel (0,0)", np.eye(1, 256, 0)[0])
try_encode("single bright pixel (15,15)", np.eye(1, 256, 255)[0])
try_encode("single bright pixel (7,8)", np.eye(1, 256, 7 * 16 + 8)[0])
try_encode("checkerboard 0/1", (np.indices((16, 16)).sum(0) % 2).ravel())
try_encode("horizontal bar row 8", np.r_[np.zeros(128), np.ones(16), np.zeros(112)])
try_encode("two blobs (corners)", (data.render([3, 3, 2, 2, 0, 1]) + data.render([12, 12, 2, 2, 0, 1]))[0])
try_encode("ellipse cut by border", data.render([0.5, 8, 4, 2, 0.3, 0.9])[0])
try_encode("tiny ellipse a=b=0.5", data.render([8, 8, 0.5, 0.5, 0, 1])[0])
try_encode("huge 1e6 everywhere", np.full(256, 1e6))
try_encode("1e150 everywhere", np.full(256, 1e150))
try_encode("1e160 everywhere (squares overflow)", np.full(256, 1e160))
try_encode("negative ellipse A=-0.8 + noise", data.render([8, 8, 4, 2, 0.3, -0.8])[0] + 0.1 * rng.standard_normal(256))
for s in range(3):
    try_encode(f"pure noise sigma 0.1 seed {s}", 0.1 * np.random.default_rng(s).standard_normal(256))
    try_encode(f"pure noise sigma 1 seed {s}", 1.0 * np.random.default_rng(s).standard_normal(256))
try_encode("NaN in one pixel (invalid input)", np.r_[np.nan, np.zeros(255)])

print("\n== pure-noise images through the evaluator (1000 images, sigma 0.1) ==")
N = 0.1 * np.random.default_rng(9).standard_normal((1000, 256))
s = ellipses.evaluate(student.fns(), images=N)
print(f"  score {s:+.6f}  (decode-to-zero would score {-np.mean(N**2):+.6f})")

print("\n== decode on extreme finite latents (200k random + hand-picked) ==")
bad = 0
cases = [
    [0, 0, 0, 0, 0, 0], [8, 8, -3, -2, 0, 1], [8, 8, 1e-320, 1e-320, 0, 1],
    [1e308, -1e308, 1e308, 1e308, 1e308, 1e308], [8, 8, 1e308, 1e308, 0, 1e308],
    [8, 8, 0.3, 0.3, 1e300, -1e308], [-1e308, 8, 0.3, 1e308, 0, 1],
    [8, 8, 1e-300, 1e308, 0.7, 1], [5e-324, -5e-324, -0.0, 0.0, -0.0, 5e-324],
    [1.7976931348623157e308] * 6, [-1.7976931348623157e308] * 6,
]
rnd = np.exp(rng.uniform(-745, 709, (200000, 6))) * rng.choice([-1, 1], (200000, 6))
for z in cases + rnd.tolist():
    try:
        d = student.decode(list(z))
        if len(d) != 256 or not all(math.isfinite(v) for v in d):
            bad += 1
            if bad <= 5:
                print("  non-finite decode for z =", z)
    except Exception as e:
        bad += 1
        if bad <= 5:
            print("  EXCEPTION", type(e).__name__, e, "for z =", z)
print(f"  decode non-finite or raised on {bad} of {len(cases) + len(rnd)} latents")

print("\n== can best_p stay None on a VALID-range image? cost is SSE of bounded values: ==")
mx = 0.0
for x in np.concatenate([data.train_images(500, seed=5)[2], N[:200]]):
    p0 = student._first_guess(x)
    for dth in student.STARTS:
        s0 = p0.copy(); s0[4] += dth
        _, c = student._lm(x, s0)
        mx = max(mx, c)
        assert math.isfinite(c)
print(f"  all 2100 start costs finite; max start cost {mx:.3f}")

print("\n== evaluator with the invalid-input images ==")
for nm, img in (("1e160", np.full((1, 256), 1e160)), ("NaN pixel", np.r_[np.nan, np.zeros(255)][None])):
    print(f"  {nm}: evaluate -> {ellipses.evaluate(student.fns(), images=img)}")

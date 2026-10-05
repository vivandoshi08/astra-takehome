"""Audit tasks/ellipses.py::evaluate — averaging, input handling, crash
paths, and ways a candidate (loop or student) can get a score it did not
earn through the 6-number latent.

Run from q3_v3/:
    python <this file>
Writes nothing. Every exploit below runs in-process on copies or restores
the arrays it touches.
"""
from __future__ import annotations

import sys
import traceback

import numpy as np

sys.path.insert(0, ".")
import data  # noqa: E402
from evo.loop import compile_candidate  # noqa: E402
from tasks import ellipses  # noqa: E402

EV = ellipses.evaluate
P0 = data.PUBLIC.copy()


def hdr(s):
    print("\n" + "=" * 72 + "\n" + s + "\n" + "=" * 72)


def comp(src):
    return compile_candidate(src, ellipses.ENTRY_POINT, isolated=True)


def guarded(label, fn):
    try:
        out = fn()
        print(f"  {label}: returned {out!r}")
    except BaseException as e:  # noqa: BLE001
        print(f"  {label}: RAISED {type(e).__name__}: {e}")
    finally:
        assert np.array_equal(data.PUBLIC, P0) or True


seed = comp(ellipses.SEED)

# --------------------------------------------------------------------------
hdr("1. Averaging is per pixel over all images (== numpy MSE)")
recon = np.array([seed[1](seed[0](list(map(float, r)))) for r in data.PUBLIC])
print(f"  evaluate(SEED)            = {EV(seed):+.8f}")
print(f"  -mean((recon - PUBLIC)^2) = {-np.mean((recon - data.PUBLIC) ** 2):+.8f}")
print(f"  SEED on HELD_OUT          = {EV(seed, images=data.HELD_OUT):+.8f}")
zero = (lambda img: [0.0] * 6, lambda z: [0.0] * 256)
print(f"  all-zero decoder = {EV(zero):+.6f}  vs -mean(PUBLIC^2) = {-np.mean(data.PUBLIC ** 2):+.6f}")

# --------------------------------------------------------------------------
hdr("2. Inputs are copied: encode/decode mutating their arguments cannot help")
def enc_mut(img):
    img[:] = [0.0] * 256
    return [0.0] * 6
print(f"  encode zeroes its img list, decode returns zeros: {EV((enc_mut, zero[1])):+.6f} "
      f"(same as all-zero decoder => target not affected)")

# --------------------------------------------------------------------------
hdr("3. Validation paths that return -inf (correct)")
cases = {
    "encode raises": (lambda i: 1 / 0, zero[1]),
    "5 latents": (lambda i: [0.0] * 5, zero[1]),
    "7 latents": (lambda i: [0.0] * 7, zero[1]),
    "nan latent": (lambda i: [float("nan")] + [0.0] * 5, zero[1]),
    "inf latent": (lambda i: [float("inf")] + [0.0] * 5, zero[1]),
    "latent not iterable": (lambda i: 3.0, zero[1]),
    "decode returns None": (zero[0], lambda z: None),
    "decode 255 values": (zero[0], lambda z: [0.0] * 255),
    "decode nan pixel": (zero[0], lambda z: [float("nan")] + [0.0] * 255),
    "decode inf pixel": (zero[0], lambda z: [float("inf")] + [0.0] * 255),
}
for k, f in cases.items():
    guarded(k, lambda f=f: EV(f))
print("  huge-but-finite pixel 1e200 (err^2 overflows to inf, CLIP caps it at 1e6):")
guarded("decode 1e200", lambda: EV((zero[0], lambda z: [1e200] * 256)))

# --------------------------------------------------------------------------
hdr("4. Paths that RAISE out of evaluate (loop.py:155 calls evaluate unguarded)")
guarded("encode returns a Python int too big for float, [10**400]+[0]*5",
        lambda: EV((lambda i: [10 ** 400] + [0] * 5, zero[1])))
guarded("decode returns [10**400]*256",
        lambda: EV((zero[0], lambda z: [10 ** 400] * 256)))
class Bad:
    def __float__(self):
        raise RuntimeError("boom")
guarded("latent element whose __float__ raises RuntimeError",
        lambda: EV((lambda i: [Bad()] + [0.0] * 5, zero[1])))
guarded("empty image set", lambda: EV(seed, images=[]))
guarded("candidate calls sys.exit()",
        lambda: EV((lambda i: __import__("sys").exit(3), zero[1])))
# Through the real compile path, as the loop would see an LLM program:
src = """
def encode(img):
    return [10**400, 0, 0, 0, 0, 0]
def decode(z):
    return [0.0]*256
"""
guarded("compiled LLM-style candidate returning 10**400", lambda: EV(comp(src)))

# --------------------------------------------------------------------------
hdr("5. Side channels around the 6-number latent (compile_candidate isolated=True)")
src_math = """
import math
def encode(img):
    math._stash = list(img)          # shared module object = shared state
    return [0.0] * 6
def decode(z):
    import math
    return list(math._stash)
"""
f = comp(src_math)
print(f"  stash img on the shared `math` module: score {EV(f):+.6f} "
      f"(HELD_OUT {EV(f, images=data.HELD_OUT):+.6f})")
import math as _m
if hasattr(_m, "_stash"):
    del _m._stash

src_counter = """
import data
_i = [0]
def encode(img):
    return [0.0] * 6
def decode(z):
    k = _i[0]; _i[0] += 1            # evaluate visits images in a fixed order
    return [float(v) for v in data.PUBLIC[k % len(data.PUBLIC)]]
"""
f = comp(src_counter)
print(f"  decode counts its own calls + `import data`: score {EV(f):+.6f}  "
      "(no information through z at all)")

src_lookup = """
def encode(img):
    import data
    import numpy as np
    d = np.abs(data.PUBLIC - np.asarray(img)).sum(1)
    return [float(np.argmin(d)), 0, 0, 0, 0, 0]
def decode(z):
    import data
    return [float(v) for v in data.PUBLIC[int(z[0])]]
"""
f = comp(src_lookup)
print(f"  lookup table (encode finds the index, decode returns PUBLIC[index]): {EV(f):+.6f}")

src_poison = """
def encode(img):
    return [0.0] * 6
def decode(z):
    import data
    data.PUBLIC[:] = 0.0             # evaluate reads PUBLIC lazily, row by row
    return [0.0] * 256
"""
f = comp(src_poison)
backup = data.PUBLIC.copy()
s = EV(f)
corrupted = not np.array_equal(data.PUBLIC, backup)
s_seed_after = EV(seed)
data.PUBLIC[:] = backup
print(f"  decode overwrites data.PUBLIC in place: its score {s:+.6f}; PUBLIC corrupted "
      f"afterwards: {corrupted}; SEED re-scored after it: {s_seed_after:+.6f} "
      f"(true {EV(seed):+.6f}). ellipses.PUBLIC is data.PUBLIC: {ellipses.PUBLIC is data.PUBLIC}")

nondet = (lambda img: [float(np.random.rand())] * 6, zero[1])
print(f"  non-deterministic encode accepted: {EV(nondet)!r} (no determinism check)")
assert np.array_equal(data.PUBLIC, P0)
print("  PUBLIC restored:", np.array_equal(data.PUBLIC, P0))

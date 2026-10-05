"""Does encode/decode depend on anything but its argument?

1. Proxy the `data` module seen by student.py and log every attribute read.
2. Bit-identity of encode under: garbage overwrite of every data.* set/params/
   clean array (and tasks.ellipses.PUBLIC/HELD_OUT), data.sample/train_images
   replaced by raisers, shuffled order, repeats, reversed order.
3. student module globals identical before/after many calls (no hidden state).
4. encode does not mutate its input; decode depends only on z; z from another
   image decodes to exactly render(z).
5. Dump bytes of encode/decode outputs to a file for the cross-process test.
"""
import hashlib
import sys
import types
import copy
import pickle
import numpy as np

import data
import student
from tasks import ellipses

out = sys.argv[1] if len(sys.argv) > 1 else None

IMGS = np.concatenate([data.PUBLIC[:64], data.HELD_OUT[:32],
                       data.train_images(32, seed=4321)[2]])
IMGS_L = [list(map(float, r)) for r in IMGS]


def enc_all(order=None):
    order = range(len(IMGS_L)) if order is None else order
    res = {}
    for k in order:
        res[k] = np.array(student.encode(list(IMGS_L[k])))
    return np.array([res[k] for k in range(len(IMGS_L))])


def h(a):
    return hashlib.sha256(np.ascontiguousarray(a, dtype=np.float64).tobytes()).hexdigest()[:16]


Z0 = enc_all()
D0 = np.array([student.decode(list(z)) for z in Z0])
print("baseline encode hash", h(Z0), " decode hash", h(D0))

# --- 1. attribute-access log on the data module as seen by student --------
real_data = student.data
accessed = []


class Proxy(types.ModuleType):
    def __getattr__(self, name):
        accessed.append(name)
        return getattr(real_data, name)


student.data = Proxy("data_proxy")
Z1 = enc_all()
D1 = np.array([student.decode(list(z)) for z in Z0])
student.data = real_data
print("data.* attributes read by encode+decode:", sorted(set(accessed)),
      "| bit-identical:", np.array_equal(Z0, Z1) and np.array_equal(D0, D1))

# --- 2. garbage-overwrite every dataset array ------------------------------
saved = {}
rng = np.random.default_rng(0)
for name in ("PUBLIC", "PUBLIC_PARAMS", "PUBLIC_CLEAN", "HELD_OUT",
             "HELD_OUT_PARAMS", "HELD_OUT_CLEAN"):
    arr = getattr(data, name)
    saved[name] = arr.copy()
    arr[...] = rng.standard_normal(arr.shape) * 1e3     # in place (shared refs)
    setattr(data, name, np.full_like(arr, np.nan))      # and rebind
ellipses.PUBLIC = np.full((128, 256), np.nan)
ellipses.HELD_OUT = np.full((256, 256), np.nan)


def boom(*a, **k):
    raise RuntimeError("encode/decode called data.sample/train_images")


data.sample = boom
data.train_images = boom
Z2 = enc_all()
D2 = np.array([student.decode(list(z)) for z in Z0])
print("after garbage-overwriting PUBLIC/HELD_OUT/*_PARAMS/*_CLEAN and disabling "
      "data.sample/train_images: bit-identical encode", np.array_equal(Z0, Z2),
      " decode", np.array_equal(D0, D2))
# IMGS was built from the arrays before overwrite: confirm we really fed copies
assert not np.isnan(IMGS).any()

# --- 3. order / repetition --------------------------------------------------
perm = np.random.default_rng(5).permutation(len(IMGS_L))
Z3 = enc_all(perm)
Z4 = enc_all(list(range(len(IMGS_L)))[::-1])
Z5 = enc_all()
print("shuffled order bit-identical:", np.array_equal(Z0, Z3),
      "| reversed:", np.array_equal(Z0, Z4), "| repeat:", np.array_equal(Z0, Z5))

# interleave: encode image k right after encoding a very different image
Z6 = []
for k in range(len(IMGS_L)):
    student.encode([0.0] * 256); student.encode([1.0] * 256)
    student.decode([1e300, -1e300, 1e-300, 1e300, 1e9, -5.0])
    Z6.append(student.encode(list(IMGS_L[k])))
print("interleaved with junk calls bit-identical:", np.array_equal(Z0, np.array(Z6)))

# --- 4. module state ---------------------------------------------------------
def snapshot(mod):
    snap = {}
    for k, v in vars(mod).items():
        if isinstance(v, (types.ModuleType, types.FunctionType, type)):
            snap[k] = id(v)
        else:
            try:
                snap[k] = pickle.dumps(v)
            except Exception:
                snap[k] = repr(v)
    return snap


s_before = snapshot(student)
for k in range(20):
    student.encode(list(IMGS_L[k]))
    student.decode(list(Z0[k]))
s_after = snapshot(student)
print("student module globals unchanged by calls:", s_before == s_after,
      "| function defaults:", student._lm.__defaults__, student.encode.__defaults__,
      "| closures:", student.encode.__closure__, student.decode.__closure__)

# input not mutated
inp = list(IMGS_L[3]); inp_c = list(inp)
student.encode(inp)
arr_in = np.array(IMGS[3]); arr_c = arr_in.copy()
student.encode(arr_in)
zin = list(Z0[3]); zin_c = list(zin)
student.decode(zin)
print("inputs not mutated: list", inp == inp_c, " ndarray", np.array_equal(arr_in, arr_c),
      " z", zin == zin_c)

# decode = render(z) exactly, for z of other images
errs = []
for k in range(len(Z0)):
    zz = Z0[(k * 37 + 11) % len(Z0)]
    d = np.array(student.decode(list(zz)))
    p = zz.copy(); p[2:4] = np.maximum(np.abs(p[2:4]), 0.3)
    errs.append(np.max(np.abs(d - real_data.render(p)[0])))
print("decode(z_other) == render(z): max abs diff", max(errs))

# restore
for name, arr in saved.items():
    setattr(data, name, arr)

if out:
    np.savez(out, Z=Z0, D=D0)
    print("wrote", out, "encode hash", h(Z0), "decode hash", h(D0))

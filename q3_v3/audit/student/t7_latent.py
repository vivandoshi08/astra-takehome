"""Does the latent carry only a measurement (no packed bits)? Parameter
recovery on PUBLIC vs fresh. Dev seeds disjoint from PUBLIC/HELD_OUT."""
import numpy as np
import data
import student
from tasks import ellipses


def canon(z):
    """map a fit to the generator's convention a>=b, theta in [0, pi)."""
    z = np.array(z, float)
    a, b = abs(z[2]), abs(z[3]); th = z[4]
    if b > a:
        a, b = b, a; th += np.pi / 2
    return np.array([z[0], z[1], a, b, th % np.pi, z[5]])


def recovery(name, P, X):
    Z = np.array([canon(student.encode(list(x))) for x in X])
    d = np.abs(Z - P)
    d[:, 4] = np.minimum(d[:, 4], np.pi - d[:, 4])
    ecc = P[:, 2] / P[:, 3] > 1.3
    print(f"  {name:12s} median |err| cx {np.median(d[:,0]):.3f} cy {np.median(d[:,1]):.3f} "
          f"a {np.median(d[:,2]):.3f} b {np.median(d[:,3]):.3f} theta(a/b>1.3) {np.degrees(np.median(d[ecc,4])):.2f}deg "
          f"A {np.median(d[:,5]):.4f}")
    return Z


print("== parameter recovery (canonicalised to a>=b) ==")
recovery("PUBLIC", data.PUBLIC_PARAMS, data.PUBLIC)
recovery("HELD_OUT", data.HELD_OUT_PARAMS, data.HELD_OUT)
for s in (7, 2024, 31):
    P, C, X = data.train_images(256, seed=s)
    recovery(f"fresh {s}", P, X)

print("\n== quantising the latent (no information hides in low-order bits) ==")
enc, dec = student.fns()
for nm, q in (("float64 (shipped)", lambda z: z),
              ("float32", lambda z: [float(np.float32(v)) for v in z]),
              ("4 decimals", lambda z: [round(v, 4) for v in z]),
              ("3 decimals", lambda z: [round(v, 3) for v in z])):
    for setname, X in (("PUBLIC", data.PUBLIC), ("HELD_OUT", data.HELD_OUT)):
        s = ellipses.evaluate((lambda img, q=q: q(enc(img)), dec), images=X)
        print(f"  {nm:18s} {setname:8s} {s:+.7f}")

print("\n== dev seeds disjoint from PUBLIC/HELD_OUT ==")
ref = np.concatenate([data.PUBLIC_PARAMS, data.HELD_OUT_PARAMS])
refx = np.concatenate([data.PUBLIC, data.HELD_OUT])
for s in (7, 2024):
    P, C, X = data.train_images(256, seed=s)
    dp = min(np.min(np.abs(ref - p).sum(1)) for p in P)
    dx = min(np.min(((refx - x) ** 2).sum(1)) for x in X)
    print(f"  seed {s}: min L1 param distance to any PUBLIC/HELD_OUT row {dp:.3f}; min squared image distance {dx:.3f}")

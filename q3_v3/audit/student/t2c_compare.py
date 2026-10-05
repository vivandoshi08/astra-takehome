"""Compare encode/decode outputs written by three separate t2_leakage.py processes."""
import numpy as np
a, b, c = [np.load(f"out/run_{k}.npz") for k in "abc"]
print("cross-process bit-identical encode a==b", np.array_equal(a["Z"], b["Z"]), "a==c", np.array_equal(a["Z"], c["Z"]),
      " decode a==b", np.array_equal(a["D"], b["D"]), "a==c", np.array_equal(a["D"], c["D"]))

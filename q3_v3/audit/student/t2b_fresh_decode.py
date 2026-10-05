"""Fresh process: decode only (no encode ever called), compare with the warm-process decode."""
import sys, numpy as np
import student
f = np.load(sys.argv[1])
D = np.array([student.decode(list(z)) for z in f["Z"]])
print("fresh-process decode bit-identical to warm-process decode:", np.array_equal(D, f["D"]))
# encode a single image first in this process, out of order, compare
import data
Zr = np.array(student.encode(list(map(float, data.HELD_OUT[31]))))
print("fresh-process encode of 1 image (HELD_OUT[31]) bit-identical:", np.array_equal(Zr, f["Z"][64 + 31]))

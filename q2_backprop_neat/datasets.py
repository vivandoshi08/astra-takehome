"""The three classification problems, with frozen train / held-out splits.

    from datasets import get_dataset
    X_train, y_train, X_test, y_test = get_dataset("spiral3")

    xor      four Gaussian blobs, diagonal pairs share a label   (easy)
    spiral   two interleaved arms, 2.6 pi turns                  (hard)
    spiral3  the same arms, 3.0 pi turns                         (hardest)

Inputs are 2-D, labels are 0/1. Splits are fixed by seed so everyone is
measured on the same data — don't regenerate them with a different seed.
"""

from __future__ import annotations

import numpy as np

N_PER_CLASS = 250
TEST_FRACTION = 0.3
SEED = 20260731

DATASETS = ("xor", "spiral", "spiral3")


def _xor(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Four Gaussian blobs; diagonal pairs share a label."""
    centers = [(-0.6, -0.6), (0.6, 0.6), (-0.6, 0.6), (0.6, -0.6)]
    labels = [0, 0, 1, 1]
    n = N_PER_CLASS // 2
    Xs, ys = [], []
    for (cx, cy), lab in zip(centers, labels):
        Xs.append(rng.normal([cx, cy], 0.25, size=(n, 2)))
        ys.append(np.full(n, lab, dtype=float))
    return np.vstack(Xs), np.concatenate(ys)


def _spiral(rng: np.random.Generator, turns: float = 2.6
            ) -> tuple[np.ndarray, np.ndarray]:
    """Two interleaved spiral arms. `turns` is in units of pi."""
    Xs, ys = [], []
    for label in (0, 1):
        t = np.sqrt(rng.uniform(0, 1, N_PER_CLASS)) * turns * np.pi
        r = t / (turns * np.pi) * 1.1
        phase = label * np.pi
        pts = np.c_[r * np.cos(t + phase), r * np.sin(t + phase)]
        Xs.append(pts + rng.normal(0, 0.035, pts.shape))
        ys.append(np.full(N_PER_CLASS, label, dtype=float))
    return np.vstack(Xs), np.concatenate(ys)


_BUILDERS = {
    "xor": (_xor, {}),
    "spiral": (_spiral, {"turns": 2.6}),
    "spiral3": (_spiral, {"turns": 3.0}),
}


def get_dataset(name: str):
    """Return (X_train, y_train, X_test, y_test) for one of DATASETS."""
    if name not in _BUILDERS:
        raise ValueError(f"unknown dataset {name!r}; expected one of {DATASETS}")
    build, kwargs = _BUILDERS[name]
    rng = np.random.default_rng(SEED + sum(map(ord, name)))
    X, y = build(rng, **kwargs)

    order = rng.permutation(len(X))
    X, y = X[order], y[order]
    n_test = int(len(X) * TEST_FRACTION)
    return X[n_test:], y[n_test:], X[:n_test], y[:n_test]


if __name__ == "__main__":
    for name in DATASETS:
        Xtr, ytr, Xte, yte = get_dataset(name)
        print(f"{name:8} train {Xtr.shape} test {Xte.shape} "
              f"class balance {ytr.mean():.2f}/{yte.mean():.2f}")

"""What our model and the loop's champions actually do to the images.

    env -u OPENROUTER_API_KEY python compare_models.py      # ~5 min (searching encoders are slow)

1. figures/compare_models.png — the same eight PUBLIC images through our
   model and three champions (best shipped a02, best hint b07, best Luna
   l03): reconstruction, then |reconstruction - noisy input|.
2. A table of each model's error against the CLEAN image (the noise taken
   out, so only the model's own mistakes are left), split by region of the
   true ellipse: interior (clean >= 0.98 A), rim (between), background
   (clean <= 0.02 A). Written to figures/compare_models.txt.

Champion code runs in this process; run it without the API key, as for
decompose_champions.py. None of these champions has run_meta champion_flags.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import data
import student
from evo.loop import compile_candidate
from tasks import ellipses

HERE = Path(__file__).parent
FIG = HERE / "figures"
SHOW = [0, 1, 2, 3, 4, 6, 7, 9]          # PUBLIC images in the figure


def champion(run):
    return compile_candidate((HERE / "runs" / run / "best.py").read_text(),
                             ellipses.ENTRY_POINT, isolated=True)


MODELS = [("our model", student.fns()),
          ("shipped best (a02)", champion("a02")),
          ("hint best (b07)", champion("b07")),
          ("Luna best (l03)", champion("l03"))]


def reconstruct(fns, X):
    enc, dec = fns
    return np.array([dec(enc([float(v) for v in x])) for x in X], dtype=float)


def main():
    X, C, P = data.PUBLIC, data.PUBLIC_CLEAN, data.PUBLIC_PARAMS
    A = P[:, 5:6]
    regions = {"interior": C >= 0.98 * A,
               "rim": (C > 0.02 * A) & (C < 0.98 * A),
               "background": C <= 0.02 * A}
    recon = {name: reconstruct(fns, X) for name, fns in MODELS}

    lines = [f"{'model':<20} {'score':>8} {'model error vs clean, mean per pixel':>40}",
             f"{'':<20} {'':>8} {'interior':>12} {'rim':>12} {'background':>12}   "
             f"share of error on the rim"]
    for name, R in recon.items():
        score = -float(np.mean((R - X) ** 2))
        e = (R - C) ** 2
        cols = [float(e[m].mean()) for m in regions.values()]
        rim_share = float(e[regions["rim"]].sum() / e.sum())
        lines.append(f"{name:<20} {score:+8.4f} " + " ".join(f"{c:12.5f}" for c in cols)
                     + f"   {rim_share:6.0%}")
    frac = {k: float(m.mean()) for k, m in regions.items()}
    lines.append(f"\npixels per region: " + ", ".join(f"{k} {v:.0%}" for k, v in frac.items()))
    lines.append("noise variance (what every model pays on top): 0.0100")
    text = "\n".join(lines)
    print(text)
    (FIG / "compare_models.txt").write_text(text + "\n")

    rows = 1 + 2 * len(MODELS)
    fig, axes = plt.subplots(rows, len(SHOW), figsize=(1.35 * len(SHOW), 1.4 * rows))
    labels = ["input (noisy)"]
    for name, _ in MODELS:
        labels += [f"{name}", f"|error| {name.split(' (')[0]}"]
    for j, i in enumerate(SHOW):
        axes[0, j].imshow(X[i].reshape(16, 16), cmap="gray", vmin=-0.2, vmax=1.2)
        for k, (name, _) in enumerate(MODELS):
            R = recon[name][i]
            axes[1 + 2 * k, j].imshow(R.reshape(16, 16), cmap="gray", vmin=-0.2, vmax=1.2)
            axes[2 + 2 * k, j].imshow(np.abs(R - X[i]).reshape(16, 16),
                                      cmap="magma", vmin=0, vmax=0.5)
    for r in range(rows):
        for j in range(len(SHOW)):
            axes[r, j].set_xticks([]); axes[r, j].set_yticks([])
        axes[r, 0].set_ylabel(labels[r], fontsize=6, rotation=0, ha="right", va="center")
    fig.suptitle("Same PUBLIC images: our model vs the loop's best champions "
                 "(error maps share one colour scale)", fontsize=8)
    fig.tight_layout()
    out = FIG / "compare_models.png"
    fig.savefig(out, dpi=130)
    print("wrote", out)


if __name__ == "__main__":
    main()

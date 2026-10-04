"""Plotting helpers. Every function writes a PNG and returns its path.

    python plots.py            # plot the data and the shipped baselines

These work on anything with an (encode, decode) pair — loop champions,
baselines, or your own model. Load a champion straight from a run:

    from evo.loop import compile_candidate
    code = open("runs/r1/best.py").read()
    fns = compile_candidate(code, ("encode", "decode"), isolated=True)
    plot_candidate(fns, "champion.png")
"""

from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import data


def _apply(fns, images: np.ndarray) -> np.ndarray:
    encode, decode = fns
    return np.array([decode(list(encode([float(a) for a in img])))
                     for img in images])


def _grid(ax, img, title=None, vmin=-0.2, vmax=1.2):
    ax.imshow(np.asarray(img, dtype=float).reshape(data.IMG, data.IMG),
              cmap="gray", vmin=vmin, vmax=vmax, interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=7)


def plot_images(path: str = "images.png", n: int = 10) -> str:
    """The first n PUBLIC images: noisy observation over clean ellipse."""
    fig, axes = plt.subplots(2, n, figsize=(1.1 * n, 2.6))
    for j in range(n):
        _grid(axes[0, j], data.PUBLIC[j],
              "noisy" if j == 0 else None)
        _grid(axes[1, j], data.PUBLIC_CLEAN[j],
              "clean" if j == 0 else None)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_candidate(fns, path: str = "candidate.png", images=None,
                   n: int = 10) -> str:
    """What a candidate actually does: input, reconstruction, and the
    per-pixel error of each, for the first n images."""
    images = data.PUBLIC if images is None else np.asarray(images)
    recon = _apply(fns, images[:n])
    err = recon - images[:n]
    fig, axes = plt.subplots(3, n, figsize=(1.1 * n, 3.9))
    for j in range(n):
        _grid(axes[0, j], images[j], "input" if j == 0 else None)
        _grid(axes[1, j], recon[j], "reconstruction" if j == 0 else None)
        axes[2, j].imshow(np.abs(err[j]).reshape(data.IMG, data.IMG),
                          cmap="magma", vmin=0.0, vmax=0.5,
                          interpolation="nearest")
        axes[2, j].set_xticks([])
        axes[2, j].set_yticks([])
        if j == 0:
            axes[2, j].set_title("|error|", fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_traversal(fns, path: str = "traversal.png", steps: int = 9) -> str:
    """The decoder's image as each latent coordinate sweeps its observed
    range while the others sit at their observed median."""
    encode, decode = fns
    Z = np.array([encode([float(a) for a in img]) for img in data.PUBLIC])
    med = np.median(Z, axis=0)
    fig, axes = plt.subplots(data.LATENT, steps,
                             figsize=(1.1 * steps, 1.15 * data.LATENT))
    axes = np.atleast_2d(axes)
    for i in range(data.LATENT):
        lo, hi = np.percentile(Z[:, i], [2, 98])
        for j, v in enumerate(np.linspace(lo, hi, steps)):
            z = med.copy()
            z[i] = v
            _grid(axes[i, j], decode(list(z)),
                  f"z[{i}]={v:.2f}" if j in (0, steps - 1) else None)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_latents(fns, path: str = "latents.png") -> str:
    """Each latent coordinate against each true ellipse parameter, on
    PUBLIC. A model that recovered the structure shows clean 1-1 curves
    (possibly permuted or rescaled) — except where the parameter itself is
    degenerate, which is worth thinking about."""
    encode, _ = fns
    Z = np.array([encode([float(a) for a in img]) for img in data.PUBLIC])
    names = data.PARAM_NAMES
    fig, axes = plt.subplots(data.LATENT, len(names),
                             figsize=(2.2 * len(names), 1.9 * data.LATENT))
    axes = np.atleast_2d(axes)
    for i in range(data.LATENT):
        for j in range(len(names)):
            axes[i, j].scatter(data.PUBLIC_PARAMS[:, j], Z[:, i], s=5)
            axes[i, j].set_xlabel(names[j], fontsize=8)
            axes[i, j].set_ylabel(f"z[{i}]", fontsize=8)
            axes[i, j].tick_params(labelsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_loss(history, path: str = "loss.png") -> str:
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot(history)
    ax.set_xlabel("epoch")
    ax.set_ylabel("training loss")
    ax.set_yscale("log")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


if __name__ == "__main__":
    import baselines
    print(plot_images())
    print(plot_candidate(baselines.pca_fns(), "baseline_pca.png"))
    print(plot_candidate(baselines.vae_fns(), "baseline_vae.png"))
    print(plot_traversal(baselines.vae_fns(), "baseline_vae_traversal.png"))

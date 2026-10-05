"""figures/smallnet_sine_representative.png: median-seed Lamarckian champions
with sine/Gaussian units and at most six connections (spiral, spiral3).

    python experiments/make_smallnet_figure.py

Median seed = lower median of the 10 seeds' held-out accuracy (ties to the
lower seed), as in make_figures.py. Champions are rebuilt with
champion_net() under neat_config_sine.ini and drawn with viz.plot_boundary
on the held-out split.
"""

from __future__ import annotations

import json
import pickle
import sys
import tempfile
from pathlib import Path

from PIL import Image

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

from datasets import get_dataset  # noqa: E402
from network import champion_net, complexity, load_config  # noqa: E402
from viz import plot_boundary  # noqa: E402


def main() -> int:
    config = load_config(HERE / "neat_config_sine.ini")
    panels = []
    with tempfile.TemporaryDirectory() as tmp:
        for d in ("spiral", "spiral3"):
            runs = sorted((json.loads((HERE / f"runs/smallnet/lam_sine/{d}_s{s}/summary.json")
                                      .read_text())["heldout_acc"], s) for s in range(10))
            acc, seed = runs[(len(runs) - 1) // 2]
            run = HERE / f"runs/smallnet/lam_sine/{d}_s{seed}"
            genome = pickle.loads((run / "champion.pkl").read_bytes())
            net = champion_net(genome, config)
            _, _, X, y = get_dataset(d)
            assert abs(net.accuracy(X, y) - acc) < 1e-12, (d, seed)
            nodes, conns = complexity(genome)
            out = Path(tmp) / f"{d}.png"
            plot_boundary(net.predict, X, y, out=str(out),
                          title=f"{d} — held-out {acc:.3f}, {nodes} nodes / {conns} conns")
            panels.append(Image.open(out).copy())
    fig = Image.new("RGB", (sum(p.width for p in panels), max(p.height for p in panels)), "white")
    x = 0
    for p in panels:
        fig.paste(p, (x, 0))
        x += p.width
    path = HERE / "figures" / "smallnet_sine_representative.png"
    fig.save(path)
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

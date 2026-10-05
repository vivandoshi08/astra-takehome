"""Where does a loop champion's error come from: its encoder or its decoder?

    env -u OPENROUTER_API_KEY python decompose_champions.py a01 a02 ...

For each champion (runs/<id>/best.py), on PUBLIC and HELD_OUT:

  as is          its own encode -> its own decode (the loop's score)
  true decoder   its encode -> the generator's own formula (the SEED's
                 decode, = data.render). Did the champion's decoder edits
                 help, or compensate for a biased encoder?
  best for its   the six numbers that fit each image best *through the
  decoder        champion's own decode*, found by Levenberg-Marquardt from
                 its encoder's output: the best this decoder can do with
                 any encoder. as-is minus this = what its encoder loses.
  floor          student.py: best encoder with the true decoder.

So  as_is - floor = (as_is - best_for_decoder)    encoder loss
                  + (best_for_decoder - floor)     decoder loss.

Champion code is executed in this process (the LM needs thousands of decode
calls); run it without the API key in the environment, and only on
champions whose run_meta.json has no `champion_flags`.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

import data
import student
from evo.loop import compile_candidate
from tasks import ellipses

HERE = Path(__file__).parent
SEED_DECODE = compile_candidate(ellipses.SEED, ellipses.ENTRY_POINT, isolated=True)[1]


def mse(decode, Z, X):
    R = np.array([decode(list(z)) for z in Z], dtype=float)
    return float(np.mean((R - X) ** 2))


def lm_through(decode, x, p, iters=25, h=1e-4):
    """Fit six numbers so that decode(z) matches x, from p (LM, finite diffs)."""
    def f(q):
        try:
            out = np.asarray(decode(list(q)), dtype=float)
            return out if out.shape == x.shape and np.all(np.isfinite(out)) else None
        except Exception:
            return None
    p = np.asarray(p, dtype=float)
    r = f(p) - x
    cost, lam = float(r @ r), 1e-3
    for _ in range(iters):
        cols = []
        for k in range(6):
            q = p.copy(); q[k] += h
            fq = f(q)
            cols.append(np.zeros_like(x) if fq is None else (fq - x - r) / h)
        J = np.array(cols)
        g, A = J @ r, J @ J.T
        improved = False
        while lam < 1e8:
            q = p + np.linalg.solve(A + lam * np.diag(np.diag(A) + 1e-9), -g)
            fq = f(q)
            if fq is not None and float((fq - x) @ (fq - x)) < cost:
                gain = cost - float((fq - x) @ (fq - x))
                p, r, cost = q, fq - x, float((fq - x) @ (fq - x))
                lam = max(lam / 4, 1e-9); improved = True
                break
            lam *= 4
        if not improved or gain < 1e-10:
            break
    return cost / x.size


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    args = ap.parse_args()

    floors = {n: -ellipses.evaluate(student.fns(), images=X)
              for n, X in (("PUBLIC", data.PUBLIC), ("HELD_OUT", data.HELD_OUT))}
    print(f"{'run':<5} {'set':<9} {'as is':>8} {'true dec':>9} {'best/dec':>9} "
          f"{'floor':>8} {'enc loss':>9} {'dec loss':>9}")
    out = []
    for run in args.runs:
        meta = json.loads((HERE / "runs" / run / "run_meta.json").read_text())
        if meta.get("champion_flags"):
            print(f"{run}: skipped, flagged {meta['champion_flags']}")
            continue
        enc, dec = compile_candidate((HERE / "runs" / run / "best.py").read_text(),
                                     ellipses.ENTRY_POINT, isolated=True)
        for name, X in (("PUBLIC", data.PUBLIC), ("HELD_OUT", data.HELD_OUT)):
            Z = [enc(list(map(float, x))) for x in X]
            as_is = mse(dec, Z, X)
            true_dec = mse(SEED_DECODE, Z, X)
            best = float(np.mean([lm_through(dec, x, z) for x, z in zip(X, Z)]))
            fl = floors[name]
            row = dict(run=run, set=name, as_is=as_is, true_dec=true_dec,
                       best_dec=best, floor=fl, enc_loss=as_is - best,
                       dec_loss=best - fl)
            out.append(row)
            print(f"{run:<5} {name:<9} {-as_is:+8.4f} {-true_dec:+9.4f} "
                  f"{-best:+9.4f} {-fl:+8.4f} {as_is - best:9.4f} {best - fl:9.4f}")
    (HERE / "runs" / "decompose_champions.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()

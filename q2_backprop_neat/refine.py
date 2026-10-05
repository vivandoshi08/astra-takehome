"""Full-batch Adam refinement of a genome's weights.

    from refine import refine
    net = GenomeNetwork(genome, config)
    result = refine(net, X, y, steps=20, lr=0.1)
    budget.charge(result.steps)            # the caller charges, once per genome

One gradient step = one gradient of the full-batch training loss
(`net.loss_and_grad`, exactly one call) plus one Adam update, projected onto
[net.lower, net.upper] so `write_back` stores what was evaluated. Forward
passes are free: the `tol` check and `loss_before` / `loss_after` use
`net.loss`, never a gradient.

Resumable, for refining in rounds (successive halving): pass the returned
`result.state` back and Adam continues exactly as one longer call would,

    r1 = refine(net, X, y, 2, lr)
    r2 = refine(net, X, y, 4, lr, state=r1.state)   # == refine(net, X, y, 6, lr)

bit for bit when tol is None. The state is tied to this net's get_params()
order: resume on the same net, or one rebuilt from the same genome after
write_back (same topology, so the same order). `result.steps` counts this
call only; `state.t` counts every step since the state was created.

`refine` reads and writes only `net`; it never touches a genome and never
charges the budget. Afterwards `net` holds `result.params`. Lamarckian code
then calls `net.write_back(genome)`; Baldwinian code does not. No randomness:
the same net, data, settings and state give bit-identical parameters.
"""

from __future__ import annotations

from dataclasses import dataclass
from operator import index

import numpy as np


@dataclass
class AdamState:
    m: np.ndarray         # first-moment estimate, in get_params() order
    v: np.ndarray         # second-moment estimate
    t: int                # Adam steps taken so far (the bias-correction exponent)


@dataclass
class RefineResult:
    steps: int            # gradient steps actually taken (= gradients computed)
    loss_before: float    # training loss at the starting parameters
    loss_after: float     # training loss at `params`
    params: np.ndarray    # final parameters, inside [net.lower, net.upper]
    state: AdamState      # pass back as `state=` to continue


def refine(net, X, y, steps, lr, betas=(0.9, 0.999), eps=1e-8, tol=None,
           state=None) -> RefineResult:
    """Run up to `steps` full-batch Adam steps on `net`'s parameters.

    Stops early, before computing another gradient, once the training loss
    is below `tol` (if given); `result.steps` is what to charge. `state`
    continues an earlier call's Adam moments (None starts fresh); it is
    copied, never modified. The starting point is projected into the bounds
    first (a genome's own values always are inside, so this is a no-op for
    nets built from genomes).
    """
    steps = index(steps)
    if steps < 0:
        raise ValueError(f"steps must be >= 0, got {steps}")
    if not lr > 0:
        raise ValueError(f"lr must be > 0, got {lr}")
    b1, b2 = betas
    if not (0 <= b1 < 1 and 0 <= b2 < 1):
        raise ValueError(f"betas must be in [0, 1), got {betas}")

    theta = np.clip(net.get_params(), net.lower, net.upper)
    net.set_params(theta)
    loss_before = cur = net.loss(X, y)

    if state is None:
        m, v, t = np.zeros_like(theta), np.zeros_like(theta), 0
    else:
        m, v, t = (np.array(state.m, dtype=float), np.array(state.v, dtype=float),
                   index(state.t))
        if m.shape != theta.shape or v.shape != theta.shape or t < 0:
            raise ValueError(f"state does not fit this net: m {m.shape}, v {v.shape}, "
                             f"t {t}; net has {theta.shape[0]} params")

    taken = 0
    while taken < steps and (tol is None or cur >= tol):
        _, g = net.loss_and_grad(X, y)
        taken += 1
        t += 1
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * g * g
        m_hat = m / (1 - b1 ** t)
        v_hat = v / (1 - b2 ** t)
        theta = np.clip(theta - lr * m_hat / (np.sqrt(v_hat) + eps),
                        net.lower, net.upper)
        net.set_params(theta)
        if tol is not None:
            cur = net.loss(X, y)

    loss_after = cur if tol is not None else net.loss(X, y)
    return RefineResult(steps=taken, loss_before=loss_before, loss_after=loss_after,
                        params=theta, state=AdamState(m=m, v=v, t=t))

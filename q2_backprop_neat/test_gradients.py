"""Checks for the gradient engine: GenomeNetwork.loss / loss_and_grad, refine().

    python test_gradients.py

The gradient is checked against central finite differences of a loss built
only from forward(): binary cross-entropy of its probabilities, clipped to
[1e-12, 1 - 1e-12]. That reference shares no code with loss_and_grad.

Pass rule for one component, gradient g and finite difference f:

    |g - f| <= RTOL * max(|g|, |f|) + atol,   atol = 10 * u * kappa / H

atol is 10x the reference's own round-off: forward() returns p near 1 only to
absolute precision u, so -log(1 - p) on a y=0 point is off by about u/(1-p),
and kappa = 1 + loss + mean over y=0 points of 1/(1-p). Below H = 1e-5 the
observed error grows like 1/H (round-off), above it like H^2 (truncation).
Where that atol would exceed 1e-7, 1 - p is instead taken from forward() of
the net with its output node negated (sigmoid(-z)): still forward() only,
with no cancellation, so kappa = 1 + loss.

Skipped and counted, not hidden: genomes with a relu pre-activation within
1e-5 of 0 (or a hidden sigmoid within 1e-5 of its +-60 clip), and genomes
where the reference's clip hides a misclassified point. A component whose
+-H step moves any point across a kink is flagged and left out.

Nothing is written to disk.
"""

from __future__ import annotations

import copy
import sys
import time
from collections import Counter
from pathlib import Path

import neat
import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

from datasets import DATASETS, get_dataset  # noqa: E402
from network import GenomeNetwork, _act, _act_grad, load_config  # noqa: E402
from refine import AdamState, RefineResult, refine  # noqa: E402

CONFIG = load_config(HERE / "neat_config.ini")
GC = CONFIG.genome_config
U = np.finfo(float).eps
H = 1e-5           # finite-difference step
RTOL = 1e-6        # truncation allowance, relative
SAFETY = 10        # atol = SAFETY * reference round-off bound
MAX_ATOL = 1e-7    # a reference noisier than this cannot check the gradient
KINK = 1e-5        # pre-activation this close to a kink: skip the genome
LR = 0.3           # refinement tests: lr_probe.py's recommendation


def check(name, ok, detail=""):
    print(f"  {'OK  ' if ok else 'FAIL'} {name}{' — ' + detail if detail else ''}")
    if not ok:
        check.failed += 1


check.failed = 0


# ---------------------------------------------------------------- reference
def mirror(net):
    """Copy of net whose output node has bias and incoming weights negated:
    its forward() is sigmoid(-z) = 1 - forward(), computed without cancellation
    (the +-60 clip is symmetric)."""
    out = net.output_keys[0]
    theta = net.get_params()
    for ci, (_, dst) in enumerate(net.conns):
        if dst == out:
            theta[ci] = -theta[ci]
    theta[len(net.weights) + net._bias_ix[out]] *= -1
    m = copy.copy(net)
    m.set_params(theta)
    return m


def ref_loss(net, X, y, mirrored=False):
    """BCE from forward()'s probabilities only.

    mirrored: take 1 - p from forward() of mirror(net) instead of
    subtracting, for nets where p rounds near 1 on a y=0 point.
    """
    p = np.clip(net.forward(X), 1e-12, 1 - 1e-12)
    q = np.clip(mirror(net).forward(X), 1e-12, 1 - 1e-12) if mirrored else 1 - p
    return float(np.mean(-y * np.log(p) - (1 - y) * np.log(q)))


def ref_atol(net, X, y, mirrored=False):
    """SAFETY x the round-off of ref_loss's central difference."""
    p = np.clip(net.forward(X), 1e-12, 1 - 1e-12)
    kappa = 1 + ref_loss(net, X, y, mirrored)
    if not mirrored:
        kappa += np.mean((1 - y) / (1 - p))
    return SAFETY * U * kappa / H


def preacts(net, X):
    """Every live node's pre-activation, recomputed from the net's arrays.

    Used only to locate kinks, never to compute the reference loss.
    """
    vals = {k: X[:, i] for i, k in enumerate(net.input_keys)}
    pre = {}
    for n in net.eval_order:
        j = net._bias_ix[n]
        z = np.full(len(X), net.biases[j])
        for ci, src in net._incoming[n]:
            if src in vals:
                z = z + net.weights[ci] * vals[src]
        pre[n] = z
        vals[n] = _act(net.activations[j], z)
    return pre


def nonsmooth(net, X):
    """(distance of the nearest pre-activation to a kink, side of every kink)."""
    pre = preacts(net, X)
    out = net.output_keys[0]
    margin, sides = np.inf, []
    for n in net.eval_order:
        act, z = net.activations[net._bias_ix[n]], pre[n]
        if act == "relu":
            margin = min(margin, float(np.abs(z).min()))
            sides.append(z > 0)
        elif act == "sigmoid" and n != out:
            margin = min(margin, float(np.abs(np.abs(z) - 60).min()))
            sides.append(np.abs(z) < 60)
    p = net.forward(X)
    sides.append((p < 1e-12) | (p > 1 - 1e-12))  # the reference's own clip
    return margin, np.concatenate(sides)


def fd_check(net, X, y):
    """Compare loss_and_grad with central differences of ref_loss.

    Returns a dict with 'status' in {ok, fail, kink, saturated, illcond}.
    """
    theta = net.get_params()
    _, g = net.loss_and_grad(X, y)
    margin, base = nonsmooth(net, X)
    p = net.forward(X)
    if margin < KINK:
        return {"status": "kink"}
    if np.any(((p < 1e-12) | (p > 1 - 1e-12)) & (np.abs(p - y) > 0.5)):
        return {"status": "saturated"}
    mirrored = False
    atol = ref_atol(net, X, y)
    if atol > MAX_ATOL:  # too noisy: take 1 - p without cancellation
        mirrored = True
        atol = ref_atol(net, X, y, mirrored)
    if atol > MAX_ATOL:
        return {"status": "illcond", "atol": atol}

    f = np.empty_like(theta)
    crossed = np.zeros(len(theta), dtype=bool)
    for j in range(len(theta)):
        side = []
        for s in (+1, -1):
            t = theta.copy()
            t[j] += s * H
            net.set_params(t)
            side.append(ref_loss(net, X, y, mirrored))
            crossed[j] |= not np.array_equal(nonsmooth(net, X)[1], base)
        f[j] = (side[0] - side[1]) / (2 * H)
    net.set_params(theta)

    m = np.maximum(np.abs(g), np.abs(f))
    err = np.abs(g - f)
    use = ~crossed
    tol = RTOL * m + atol
    rel = np.where(m > 0, err / np.where(m > 0, m, 1), 0.0)
    return {
        "status": "ok" if np.all(err[use] <= tol[use]) else "fail",
        "mirrored": mirrored,
        "crossed": int(crossed.sum()),
        "n": int(use.sum()),
        "usage": float((err[use] / tol[use]).max()) if use.any() else 0.0,
        "abs": float(err[use].max()) if use.any() else 0.0,
        "atol": atol,
        # worst relative error among components at least this large
        "rel": {c: float(rel[use & (m >= c)].max()) if (use & (m >= c)).any() else 0.0
                for c in (1e-2, 1e-4, 1e-6)},
    }


def summarise(results):
    st = Counter(r["status"] for r in results)
    ok = [r for r in results if r["status"] in ("ok", "fail")]
    worst = {c: max((r["rel"][c] for r in ok), default=0.0) for c in (1e-2, 1e-4, 1e-6)}
    return st, ok, worst


# ---------------------------------------------------------------- genomes
def collect_genomes(seeds=(0, 1), generations=80, keep=(0, 5, 20, 50, 79), per_gen=10):
    """Copies of genomes from plain NEAT runs (accuracy fitness) on every dataset.

    Per kept generation: a strided sample plus the largest genome, so hidden
    nodes and mixed activations are present.
    """
    found = []
    for name in DATASETS:
        X, y = get_dataset(name)[:2]
        for seed in seeds:
            gen = 0

            def ev(genomes, cfg):
                nonlocal gen
                for _, g in genomes:
                    g.fitness = GenomeNetwork(g, cfg).accuracy(X, y)
                if gen in keep:
                    largest = max(genomes, key=lambda kg: len(kg[1].connections))
                    pick = genomes[::max(1, len(genomes) // per_gen)][:per_gen - 1]
                    for _, g in pick + [largest]:
                        found.append((name, copy.deepcopy(g)))
                gen += 1

            neat.Population(CONFIG, seed=seed).run(ev, generations)
    return found


def build(nodes, conns):
    """Hand-built genome. nodes: {key: (activation, bias)};
    conns: [(src, dst, weight)] or [(src, dst, weight, enabled)]."""
    g = CONFIG.genome_type(0)
    for k, (act, b) in nodes.items():
        n = g.create_node(GC, k)
        n.activation, n.bias = act, b
        g.nodes[k] = n
    for i, c in enumerate(conns):
        a, b, w = c[:3]
        cg = g.create_connection(GC, a, b, 100_000 + i)
        cg.weight, cg.enabled = w, (c[3] if len(c) > 3 else True)
        g.connections[(a, b)] = cg
    return g


def dead_indices(net):
    live = set(net.eval_order)
    return [i for i, (_, dst) in enumerate(net.conns) if dst not in live]


# ---------------------------------------------------------------- tests
def test_act_grad():
    print("activation derivatives match finite differences of _act")
    z = np.r_[np.linspace(-70, 70, 2801), [-1e6, -1e3, 1e3, 1e6]]
    z = z[(np.abs(z) > 1e-3) & (np.abs(np.abs(z) - 60) > 1e-3)]  # off the kinks
    h = 1e-6 * np.maximum(1.0, np.abs(z))  # relative step: ulp(1e6) is 1e-10
    for name in ("sigmoid", "tanh", "relu", "identity"):
        f = (_act(name, z + h) - _act(name, z - h)) / (2 * h)
        err = np.abs(f - _act_grad(name, z)).max()
        check(f"{name}: max |analytic - FD| over z in [-70, 70] and +-1e6",
              err < 1e-8, f"{err:.1e}")
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        ok = all(np.isfinite(_act_grad(n, np.array([-1e300, 1e300]))).all()
                 for n in ("sigmoid", "tanh", "relu", "identity"))
    check("no overflow or NaN at |z| = 1e300", ok)
    check("sigmoid derivative is 0 beyond the +-60 clip, relu's is 0 at z = 0",
          _act_grad("sigmoid", np.array([60.5, -61.0])).tolist() == [0.0, 0.0]
          and _act_grad("relu", np.array([0.0]))[0] == 0.0)


def test_real_genomes(genomes):
    print(f"finite differences on {len(genomes)} genomes from NEAT runs, "
          "as evolved and after 300 Adam steps at lr 1.0")
    stats = Counter()
    acts = Counter()
    for name, g in genomes:
        net = GenomeNetwork(g, CONFIG)
        hidden = [n for n in net.eval_order if n not in net.output_keys]
        stats["with live hidden nodes"] += bool(hidden)
        stats["with a bias neuron"] += any(not any(s in net._bias_ix or s in net.input_keys
                                                   for _, s in net._incoming[n])
                                           for n in hidden)
        stats["with dead connections"] += bool(dead_indices(net))
        acts.update(net.activations[net._bias_ix[n]] for n in hidden)
    sizes = [GenomeNetwork(g, CONFIG).n_params for _, g in genomes]
    print(f"    sample: {dict(stats)}; live hidden activations {dict(acts)}; "
          f"params median {int(np.median(sizes))}, max {max(sizes)}")

    results = {"evolved": [], "refined": []}
    for name, g in genomes:
        X, y = get_dataset(name)[:2]
        net = GenomeNetwork(g, CONFIG)
        results["evolved"].append(fd_check(net, X, y))
        refine(net, X, y, 300, 1.0)
        results["refined"].append(fd_check(net, X, y))

    for label, res in results.items():
        st, ok, worst = summarise(res)
        crossed = sum(r.get("crossed", 0) for r in ok)
        usage = max((r["usage"] for r in ok), default=0.0)
        n_mirror = sum(r["mirrored"] for r in ok)
        check(f"{label}: every checked component within tolerance",
              st["fail"] == 0 and len(ok) > 0,
              f"checked {len(ok)} ({n_mirror} with the mirrored reference), failed "
              f"{st['fail']}, skipped: relu/clip kink {st['kink']}, reference "
              f"saturated {st['saturated']}, reference too noisy {st['illcond']}; "
              f"components flagged for crossing a kink {crossed}; worst "
              f"error/tolerance {usage:.2f}")
        print(f"    {label}: worst relative error |g-f|/max(|g|,|f|) over components "
              f">= 1e-2: {worst[1e-2]:.1e}, >= 1e-4: {worst[1e-4]:.1e}, "
              f">= 1e-6: {worst[1e-6]:.1e}; worst absolute "
              f"{max(r['abs'] for r in ok):.1e}; atol range "
              f"[{min(r['atol'] for r in ok):.0e}, {max(r['atol'] for r in ok):.0e}]")
    return results


def test_edge_cases():
    print("hand-built edge cases")
    X, y = get_dataset("spiral")[:2]

    cases = {
        "initial genome, no hidden nodes": build(
            {0: ("sigmoid", 0.3)}, [(-1, 0, 0.8), (-2, 0, -1.2)]),
        "all connections disabled": build(
            {0: ("sigmoid", 0.4)}, [(-1, 0, 0.8, False), (-2, 0, -1.2, False)]),
        "bias neuron feeding the output": build(
            {0: ("sigmoid", -0.2), 5: ("tanh", 0.7), 6: ("relu", 0.5)},
            [(-1, 5, 1.0, False), (5, 0, 1.5), (6, 0, -0.8), (-1, 0, 0.9), (-2, 0, 0.4)]),
        "dead branch, and the output feeding a hidden node": build(
            {0: ("sigmoid", 0.1), 5: ("tanh", 0.2), 6: ("relu", 0.3), 7: ("sigmoid", 0.4),
             8: ("tanh", -0.5)},
            [(-1, 5, 1.1), (5, 0, -0.7), (-2, 6, 0.9), (6, 7, 1.3), (0, 8, 2.0),
             (-1, 8, 0.5), (-2, 0, 0.6)]),
        "deep chain, mixed activations, skip connections": build(
            {0: ("sigmoid", 0.1), 10: ("tanh", 0.3), 11: ("relu", 0.6), 12: ("identity", -0.2),
             13: ("sigmoid", 0.1), 14: ("tanh", -0.4), 15: ("identity", 0.2)},
            [(-1, 10, 1.2), (10, 11, 0.9), (11, 12, -1.1), (12, 13, 1.4), (13, 14, 2.0),
             (14, 15, -1.3), (15, 0, 1.7), (-2, 12, 0.8), (10, 14, -0.6), (-1, 0, 0.5),
             (-2, 11, 0.7)]),
        "one hidden node of each activation, hidden-to-hidden edges": build(
            {0: ("sigmoid", 0.0), 5: ("sigmoid", 0.2), 6: ("tanh", -0.3), 7: ("relu", 0.4),
             8: ("identity", 0.1)},
            [(-1, 5, 2.1), (-2, 5, -1.3), (-1, 6, 0.7), (-2, 6, 1.9), (-1, 7, -1.5),
             (-2, 7, 1.1), (-1, 8, 0.6), (-2, 8, -0.9), (5, 7, 1.2), (6, 8, -1.4),
             (5, 0, 1.3), (6, 0, -1.1), (7, 0, 0.9), (8, 0, 1.6), (-1, 0, -0.4)]),
        "large weights: a hidden sigmoid past its +-60 clip on some points": build(
            {0: ("sigmoid", -0.5), 5: ("sigmoid", 30.0), 6: ("tanh", 0.0)},
            [(-1, 5, 30.0), (-2, 5, 30.0), (-2, 6, 25.0), (5, 0, 2.0), (6, 0, -1.5),
             (-1, 0, 0.3)]),
    }
    for label, g in cases.items():
        net = GenomeNetwork(g, CONFIG)
        r = fd_check(net, X, y)
        detail = r["status"]
        if r["status"] in ("ok", "fail"):
            detail += (f", {r['n']} params, worst rel {r['rel'][1e-6]:.1e}, "
                       f"error/tolerance {r['usage']:.2f}")
        check(label, r["status"] == "ok", detail)

    net = GenomeNetwork(cases["large weights: a hidden sigmoid past its +-60 clip on some points"],
                        CONFIG)
    z5 = preacts(net, X)[5]
    check("  ...that case really has points past the clip and points inside it",
          (z5 > 60).any() and (z5 < 60).any(), f"{int((z5 > 60).sum())} of {len(z5)} past 60")

    # Closed forms for networks with no hidden node (logistic regression).
    for label in ("initial genome, no hidden nodes", "all connections disabled"):
        net = GenomeNetwork(cases[label], CONFIG)
        L, gr = net.loss_and_grad(X, y)
        A = np.c_[X[:, [0, 1]][:, [net.input_keys.index(s) for s, _ in net.conns]],
                  np.ones(len(X))]
        z = A @ net.get_params()
        Lc = np.mean(np.log1p(np.exp(-np.abs(z))) + np.maximum(z, 0) - y * z)
        gc = A.T @ (1 / (1 + np.exp(-z)) - y) / len(X)
        check(f"{label}: loss and gradient equal the logistic-regression closed form",
              abs(L - Lc) < 1e-14 and np.allclose(gr, gc, rtol=1e-12, atol=1e-15),
              f"{net.n_params} params, |dL| {abs(L - Lc):.1e}, |dg| {np.abs(gr - gc).max():.1e}")

    net = GenomeNetwork(cases["bias neuron feeding the output"], CONFIG)
    _, gr = net.loss_and_grad(X, y)
    nw = len(net.weights)
    check("bias neuron: first in eval_order, its bias and outgoing weight get gradient",
          net.eval_order[0] == 5 and gr[nw + net._bias_ix[5]] != 0
          and gr[net.conns.index((5, 0))] != 0, f"eval_order {net.eval_order}")

    net = GenomeNetwork(cases["dead branch, and the output feeding a hidden node"], CONFIG)
    _, gr = net.loss_and_grad(X, y)
    dead = dead_indices(net)
    check("dead branch: 4 dead connections, gradient exactly 0, dead nodes have no bias param",
          len(dead) == 4 and np.all(gr[dead] == 0.0)
          and not {6, 7, 8} & set(net.node_keys),
          f"dead {[net.conns[i] for i in dead]}")

    # relu exactly at its kink: node 6 has bias 0 and only an always-off relu
    # feeding it, so z6 = 0 on every point. Subgradient 0 is the left derivative.
    g = build({0: ("sigmoid", 0.2), 5: ("relu", -1.0), 6: ("relu", 0.0)},
              [(-1, 5, 0.5), (5, 6, 2.0), (6, 0, 1.5), (-1, 0, 0.7), (-2, 0, -0.6)])
    net = GenomeNetwork(g, CONFIG)
    _, gr = net.loss_and_grad(X, y)
    j = len(net.weights) + net._bias_ix[6]
    th = net.get_params()
    L0 = ref_loss(net, X, y)
    t = th.copy(); t[j] -= H; net.set_params(t); Lm = ref_loss(net, X, y)
    t = th.copy(); t[j] += H; net.set_params(t); Lp = ref_loss(net, X, y)
    net.set_params(th)
    left, right = (L0 - Lm) / H, (Lp - L0) / H
    check("relu at z = 0 exactly: the test skips it, and the gradient is the left "
          "derivative (subgradient 0)",
          fd_check(net, X, y)["status"] == "kink" and gr[j] == 0.0 and abs(left) < 1e-9
          and abs(right) > 1e-3,
          f"grad {gr[j]}, left FD {left:.1e}, right FD {right:.2e}")

    # Output far past forward()'s +-60 clip: the loss keeps its slope.
    g = build({0: ("sigmoid", 30.0)}, [(-1, 0, 30.0), (-2, 0, 30.0)])
    net = GenomeNetwork(g, CONFIG)
    z = 30 + 30 * X[:, 0] + 30 * X[:, 1]
    L, gr = net.loss_and_grad(X, y)
    th = net.get_params()
    f = np.empty_like(th)
    for k in range(len(th)):
        t = th.copy(); t[k] += H; net.set_params(t); lp = net.loss(X, y)
        t = th.copy(); t[k] -= H; net.set_params(t); lm = net.loss(X, y)
        f[k] = (lp - lm) / (2 * H)
    net.set_params(th)
    check("output z beyond +-60 (forward() saturates): loss is still "
          "mean(softplus(z) - y z) and its gradient matches FD of loss()",
          (np.abs(z) > 60).any()
          and abs(L - np.mean(np.logaddexp(0, z) - y * z)) < 1e-12
          and np.allclose(gr, f, rtol=1e-6, atol=1e-9),
          f"{int((np.abs(z) > 60).sum())} points past 60; max |g - FD| {np.abs(gr - f).max():.1e}"
          " (reference here is loss() itself: forward()-based BCE is clipped flat)")

    g = build({0: ("tanh", 0.0)}, [(-1, 0, 1.0)])
    try:
        GenomeNetwork(g, CONFIG).loss_and_grad(X, y)
        raised = ""
    except ValueError as e:
        raised = str(e)
    check("a non-sigmoid output raises a clear error", "sigmoid" in raised, raised[:70])

    net = GenomeNetwork(cases["initial genome, no hidden nodes"], CONFIG)
    try:
        net.loss(X, y[:, None])
        raised = ""
    except ValueError as e:
        raised = str(e)
    check("y of shape (N, 1) is refused, not broadcast to (N, N)", "shape" in raised, raised)


def test_structure(genomes):
    print("structure: dead connections, missing sources, side effects")
    n_dead, n_dead_nonzero, missing, out_missing = 0, 0, 0, 0
    changed, loss_mismatch, ref_gap = 0, 0, 0.0
    mirror_ratio, mirror_sum = 0.0, 0.0
    for name, g in genomes:
        X, y = get_dataset(name)[:2]
        net = GenomeNetwork(g, CONFIG)
        out_missing += net.output_keys[0] not in net.eval_order
        pos = {n: i for i, n in enumerate(net.eval_order)}
        for n in net.eval_order:  # a source forward() would read as 0
            missing += sum(1 for _, s in net._incoming[n]
                           if s not in net.input_keys and pos.get(s, len(pos)) >= pos[n])
        th, p0 = net.get_params(), net.forward(X)
        L, gr = net.loss_and_grad(X, y)
        dead = dead_indices(net)
        n_dead += len(dead)
        n_dead_nonzero += int(np.count_nonzero(gr[dead]))
        changed += not (np.array_equal(net.get_params(), th)
                        and np.array_equal(net.forward(X), p0))
        loss_mismatch += net.loss(X, y) != L
        if np.all((p0 > 1e-12) & (p0 < 1 - 1e-12)):
            ref_gap = max(ref_gap, abs(L - ref_loss(net, X, y)) / max(L, 1e-300))
            # the mirrored reference agrees with the plain one, to the plain one's round-off
            gap = abs(ref_loss(net, X, y, True) - ref_loss(net, X, y))
            mirror_ratio = max(mirror_ratio, gap / (ref_atol(net, X, y) * H / SAFETY))
            mirror_sum = max(mirror_sum, np.abs(mirror(net).forward(X) + p0 - 1).max())
    check(f"dead connections get exactly 0 gradient ({n_dead} dead connections "
          f"over {len(genomes)} genomes)", n_dead > 0 and n_dead_nonzero == 0,
          f"{n_dead_nonzero} nonzero")
    check("the output node is in eval_order for every genome", out_missing == 0)
    check("no live edge has a source forward() would read as 0", missing == 0,
          f"{missing} such edges (values.get default never fires)")
    check("loss_and_grad changes neither params nor forward()", changed == 0)
    check("loss() equals loss_and_grad()'s loss exactly", loss_mismatch == 0)
    check("loss() agrees with the forward()-based BCE (unclipped genomes)",
          ref_gap < 1e-12, f"max relative gap {ref_gap:.1e}")
    check("mirrored reference: forward(mirror) = 1 - forward, and its loss agrees with "
          "the plain reference within the plain one's round-off bound",
          mirror_sum < 4 * U and mirror_ratio <= 1.0,
          f"max |q + p - 1| {mirror_sum:.1e}; max gap / bound {mirror_ratio:.2f}")


def genome_state(g):
    return ({k: (c.weight, c.enabled) for k, c in g.connections.items()},
            {k: (n.bias, n.activation, n.response, n.aggregation) for k, n in g.nodes.items()},
            getattr(g, "refined_params", "unset"))


def test_refine(genomes):
    print(f"refine (Adam, lr {LR})")
    for name in ("xor", "spiral"):
        X, y = get_dataset(name)[:2]
        sample = [g for d, g in genomes if d == name]
        before, after, worse, flat = [], [], 0, 0
        for g in sample:
            net = GenomeNetwork(g, CONFIG)
            r = refine(net, X, y, 50, LR)
            before.append(r.loss_before)
            after.append(r.loss_after)
            worse += r.loss_after > r.loss_before
            flat += r.loss_after == r.loss_before
        check(f"{name}: 50 steps lower the training loss on every one of "
              f"{len(sample)} genomes", worse == 0 and flat == 0,
              f"median {np.median(before):.3f} -> {np.median(after):.3f}; "
              f"increased {worse}, unchanged {flat}")

    X, y = get_dataset("spiral")[:2]
    g = max((g for d, g in genomes if d == "spiral"), key=lambda g: len(g.connections))
    state = copy.deepcopy(genome_state(g))

    net = GenomeNetwork(g, CONFIG)
    calls = [0]
    orig = net.loss_and_grad

    def counted(X_, y_):
        calls[0] += 1
        return orig(X_, y_)

    net.loss_and_grad = counted
    r = refine(net, X, y, 37, LR)
    check("exactly `steps` gradients computed, and result.steps says so",
          calls[0] == 37 and r.steps == 37 and isinstance(r, RefineResult),
          f"{calls[0]} calls, steps {r.steps}")
    check("net holds result.params; loss_after is the loss there",
          np.array_equal(net.get_params(), r.params) and r.loss_after == net.loss(X, y))

    calls[0] = 0
    net.set_params(GenomeNetwork(g, CONFIG).get_params())
    tol = r.loss_before - 0.3 * (r.loss_before - r.loss_after)
    r2 = refine(net, X, y, 37, LR, tol=tol)
    check("early stop: fewer steps, one gradient per step, loss below tol",
          0 < r2.steps < 37 and calls[0] == r2.steps and r2.loss_after < tol
          and r2.loss_before >= tol,
          f"{r2.steps} steps, {calls[0]} calls, loss {r2.loss_after:.4f} < tol {tol:.4f}")
    calls[0] = 0
    r3 = refine(net, X, y, 37, LR, tol=net.loss(X, y) + 1.0)
    check("tol above the starting loss: 0 steps, no gradient, params unchanged",
          r3.steps == 0 and calls[0] == 0 and np.array_equal(r3.params, r2.params))

    runs = [refine(GenomeNetwork(g, CONFIG), X, y, 60, LR) for _ in range(2)]
    check("deterministic: two runs give bit-identical params and losses",
          np.array_equal(runs[0].params, runs[1].params)
          and runs[0].loss_after == runs[1].loss_after)

    net = GenomeNetwork(g, CONFIG)
    r = refine(net, X, y, 20, 1e6)
    inside = np.all((r.params >= net.lower) & (r.params <= net.upper))
    at_bound = int(np.sum(np.abs(r.params) == 30.0))
    check("lr 1e6: params stay inside [lower, upper]", inside,
          f"{at_bound} of {len(r.params)} at +-30")

    check("the genome is untouched (weights, biases, activations, refined_params)",
          genome_state(g) == state)

    # Adam steps 1 and 2 against the textbook update. Step 1 moves every
    # parameter with a nonzero gradient by lr * g / (|g| + eps): bias
    # correction makes it lr in size, whatever the gradient's scale.
    net = GenomeNetwork(g, CONFIG)
    th0 = net.get_params()
    _, g0 = net.loss_and_grad(X, y)
    lr, (b1, b2), eps = 0.01, (0.9, 0.999), 1e-8
    th1 = th0 - lr * g0 / (np.abs(g0) + eps)
    net.set_params(th1)
    _, g1 = net.loss_and_grad(X, y)
    m = b1 * (1 - b1) * g0 + (1 - b1) * g1
    v = b2 * (1 - b2) * g0 ** 2 + (1 - b2) * g1 ** 2
    th2 = th1 - lr * (m / (1 - b1 ** 2)) / (np.sqrt(v / (1 - b2 ** 2)) + eps)
    r1 = refine(GenomeNetwork(g, CONFIG), X, y, 1, lr)
    r2 = refine(GenomeNetwork(g, CONFIG), X, y, 2, lr)
    zero = g0 == 0
    check("Adam steps 1 and 2 match the textbook update; zero-gradient params do not move",
          np.allclose(r1.params, th1, rtol=0, atol=1e-14)
          and np.allclose(r2.params, th2, rtol=0, atol=1e-14)
          and np.array_equal(r2.params[zero], th0[zero]),
          f"{int(zero.sum())} zero-gradient params (dead connections)")

    net = GenomeNetwork(g, CONFIG)
    th = net.get_params()
    th[0] = 55.0
    net.set_params(th)
    r = refine(net, X, y, 0, LR)
    check("steps=0: no update, but the start is projected into bounds",
          r.steps == 0 and r.params[0] == 30.0
          and np.array_equal(r.params[1:], th[1:]))


def test_refine_resume(genomes):
    print("refine resumed from its state (successive halving: 2, then 4, then 8 steps)")
    rounds = (2, 4, 8)
    same, same_lamarck, n = 0, 0, 0
    for name, g in genomes[::15]:
        X, y = get_dataset(name)[:2]
        whole = refine(GenomeNetwork(g, CONFIG), X, y, sum(rounds), LR)

        net, state = GenomeNetwork(g, CONFIG), None
        for k in rounds:
            r = refine(net, X, y, k, LR, state=state)
            state = r.state
        same += (np.array_equal(r.params, whole.params) and r.loss_after == whole.loss_after
                 and np.array_equal(state.m, whole.state.m)
                 and np.array_equal(state.v, whole.state.v) and state.t == whole.state.t)

        # Lamarckian rounds: write back, rebuild the net from the genome, resume.
        gl, state = copy.deepcopy(g), None
        for k in rounds:
            net = GenomeNetwork(gl, CONFIG)
            r = refine(net, X, y, k, LR, state=state)
            clamped = net.write_back(gl)
            state = r.state
        same_lamarck += (clamped == 0 and np.array_equal(
            GenomeNetwork(gl, CONFIG).get_params(), whole.params))
        n += 1
    check(f"rounds {'+'.join(map(str, rounds))} on one net == one {sum(rounds)}-step call, "
          "bit for bit (params, m, v, t, loss)", same == n, f"{same}/{n} genomes")
    check("same through write_back and a rebuilt net between rounds (Lamarckian)",
          same_lamarck == n, f"{same_lamarck}/{n} genomes")

    name, g = genomes[-1]
    X, y = get_dataset(name)[:2]
    net = GenomeNetwork(g, CONFIG)
    r1 = refine(net, X, y, 3, LR)
    kept = copy.deepcopy(r1.state)
    calls = [0]
    orig = net.loss_and_grad

    def counted(X_, y_):
        calls[0] += 1
        return orig(X_, y_)

    net.loss_and_grad = counted
    r2 = refine(net, X, y, 5, LR, state=r1.state)
    check("resumed call: result.steps and gradients count this call, state.t all calls; "
          "the state passed in is not modified",
          r2.steps == 5 and calls[0] == 5 and r2.state.t == 8 and isinstance(r2.state, AdamState)
          and np.array_equal(r1.state.m, kept.m) and np.array_equal(r1.state.v, kept.v)
          and r1.state.t == kept.t and r2.state.m is not r1.state.m)
    try:
        refine(GenomeNetwork(genomes[0][1], CONFIG), X, y, 1, LR,
               state=AdamState(np.zeros(1000), np.zeros(1000), 3))
        raised = ""
    except ValueError as e:
        raised = str(e)
    check("a state of the wrong size is refused", "does not fit" in raised, raised[:60])


def test_timing(genomes):
    print("timing on the 350 training points")
    X, y = get_dataset("spiral")[:2]
    nets = sorted((GenomeNetwork(g, CONFIG) for _, g in genomes), key=lambda n: n.n_params)
    for label, net in (("median", nets[len(nets) // 2]), ("largest", nets[-1])):
        times = {}
        for fn, call in (("forward", lambda: net.forward(X)),
                         ("loss_and_grad", lambda: net.loss_and_grad(X, y))):
            call()
            reps = 300
            t = time.perf_counter()
            for _ in range(reps):
                call()
            times[fn] = (time.perf_counter() - t) / reps * 1e3
        print(f"    {label:7} genome ({len(net.conns)} enabled conns, "
              f"{len(net.eval_order)} live nodes, {net.n_params} params): "
              f"loss_and_grad {times['loss_and_grad']:.3f} ms, forward "
              f"{times['forward']:.3f} ms -> 30,000 steps ~ "
              f"{times['loss_and_grad'] * 30:.0f} s")


def main() -> int:
    test_act_grad()
    genomes = collect_genomes()
    test_real_genomes(genomes)
    test_edge_cases()
    test_structure(genomes)
    test_refine(genomes)
    test_refine_resume(genomes)
    test_timing(genomes)
    print(f"\n{'all checks passed' if not check.failed else f'{check.failed} FAILED'}")
    return 1 if check.failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Audit of the Q2 scaffold's assumptions.

    python audit_scaffold.py

Checks the assumptions the rest of the work rests on: the data, the network
rig against neat-python's own network, the budget arithmetic, and how the NEAT
loop treats elites. Prints one line per check; nothing is written to disk.

Written against the shipped scaffold. It builds genomes with neat's plain
DefaultGenome on purpose, so the output-activation numbers describe the
shipped behaviour; the write_back and forward() checks report whatever
network.py currently does (both were fixed after this audit; see
test_scaffold_fixes.py).
"""

from __future__ import annotations

import copy
import math
import sys
import time
from pathlib import Path

import neat
import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

from budget import BudgetExhausted, StepBudget, plan  # noqa: E402
from datasets import DATASETS, get_dataset  # noqa: E402
from network import GenomeNetwork, complexity  # noqa: E402

CONFIG = str(HERE / "neat_config.ini")


def load_config():
    return neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                       neat.DefaultSpeciesSet, neat.DefaultStagnation, CONFIG)


def section(title):
    print(f"\n== {title}")


# ---------------------------------------------------------------- datasets
def audit_datasets():
    section("datasets")
    for name in DATASETS:
        Xtr, ytr, Xte, yte = get_dataset(name)
        Xtr2, _, Xte2, _ = get_dataset(name)
        frozen = np.array_equal(Xtr, Xtr2) and np.array_equal(Xte, Xte2)
        overlap = len({tuple(r) for r in Xtr} & {tuple(r) for r in Xte})
        maj_te = max(yte.mean(), 1 - yte.mean())

        d = ((Xte[:, None, :] - Xtr[None, :, :]) ** 2).sum(-1)
        order = np.argsort(d, axis=1)
        knn = {}
        for k in (1, 5, 15):
            vote = ytr[order[:, :k]].mean(1)
            knn[k] = ((vote > 0.5) == yte).mean()
        print(f"{name:8} train {len(Xtr)} / held-out {len(Xte)} | frozen={frozen} "
              f"| train/test duplicates={overlap} | class-1 share train "
              f"{ytr.mean():.3f} held-out {yte.mean():.3f} | majority-class "
              f"held-out {maj_te:.3f} | k-NN held-out k1 {knn[1]:.3f} "
              f"k5 {knn[5]:.3f} k15 {knn[15]:.3f}")

    # Spiral geometry, from datasets._spiral: angle t in [0, turns*pi],
    # radius r = 1.1 * t / (turns*pi), arms offset by pi.
    for name, turns in (("spiral", 2.6), ("spiral3", 3.0)):
        revs = turns * math.pi / (2 * math.pi)
        gap = 1.1 * math.pi / (turns * math.pi)  # radial distance between arms on a ray
        print(f"{name:8} 'turns={turns}' is {turns} pi radians = {revs:.2f} "
              f"revolutions per arm; arm-to-arm radial gap {gap:.3f} "
              f"= {gap / 0.035:.1f} noise sd")

    # xor Bayes accuracy: blobs at +-0.6, sd 0.25; best boundary is the axes.
    p = 0.5 * math.erfc(0.6 / 0.25 / math.sqrt(2))  # P(one coordinate crosses 0)
    print(f"xor      Bayes-optimal accuracy (axis boundary) = {1 - 2 * p * (1 - p):.4f}")


# ---------------------------------------------------------------- network
def collect_genomes(config, generations=40, seed=0, dataset="spiral"):
    """Plain NEAT as shipped; keep a copy of every genome it evaluates."""
    Xtr, ytr, _, _ = get_dataset(dataset)
    seen, per_gen = [], []

    def ev(genomes, cfg):
        per_gen.append([(gid, g) for gid, g in genomes])
        for _, g in genomes:
            g.fitness = GenomeNetwork(g, cfg).accuracy(Xtr, ytr)
            seen.append(copy.deepcopy(g))

    pop = neat.Population(config, seed=seed)
    pop.run(ev, generations)
    return seen, per_gen


def audit_network(config, genomes):
    section("network.py vs neat-python's own FeedForwardNetwork")
    X = get_dataset("spiral")[0]

    def max_diff(cfg):
        worst = 0.0
        for g in genomes[::7]:
            ours = GenomeNetwork(g, cfg).forward(X)
            ref = neat.nn.FeedForwardNetwork.create(g, cfg)
            theirs = np.array([ref.activate(x)[0] for x in X])
            worst = max(worst, float(np.abs(ours - theirs).max()))
        return worst

    print(f"default neat activations       : max |ours - neat| = {max_diff(config):.3g}")
    plain = load_config()
    gc = plain.genome_config
    gc.activation_defs.add("sigmoid", lambda z: 1 / (1 + math.exp(-max(-60, min(60, z)))))
    gc.activation_defs.add("tanh", lambda z: math.tanh(z))
    print(f"neat with unscaled sigmoid/tanh: max |ours - neat| = {max_diff(plain):.3g}")
    print("  -> topology/ordering match; the only difference is neat-python's "
          "sigmoid(5z) and tanh(2.5z) gains, which network.py drops")

    section("output node: what forward() returns, and what predict() thresholds")
    acts = {}
    for g in genomes:
        acts[g.nodes[0].activation] = acts.get(g.nodes[0].activation, 0) + 1
    total = sum(acts.values())
    print("output activation across all evaluated genomes: " +
          ", ".join(f"{k} {v / total:.0%}" for k, v in sorted(acts.items())))
    for name in ("sigmoid", "tanh", "relu"):
        g = next((g for g in genomes if g.nodes[0].activation == name), None)
        if g is None:
            continue
        out = GenomeNetwork(g, config).forward(X)
        print(f"  {name:7} output: forward() range [{out.min():+.3f}, {out.max():+.3f}]; "
              f"predict() = forward() > 0.5")
    says_logits = "Logits" in (GenomeNetwork.forward.__doc__ or "")
    print("  -> forward() returns the output activation, not a logit "
          f"(docstring says 'Logits': {says_logits}); the 0.5 threshold means "
          f"pre-activation z > 0 for sigmoid, z > {math.atanh(0.5):.3f} for tanh, "
          "z > 0.5 for relu")

    section("parameter round-trips")
    worst_rt, worst_wb = 0.0, 0.0
    for g in genomes[::11]:
        g = copy.deepcopy(g)
        net = GenomeNetwork(g, config)
        th = net.get_params()
        net.set_params(th)
        worst_rt = max(worst_rt, float(np.abs(net.get_params() - th).max()))
        th2 = th + np.random.default_rng(0).normal(0, 0.3, th.shape)
        net.set_params(th2)
        net.write_back(g)
        net2 = GenomeNetwork(g, config)
        worst_wb = max(worst_wb, float(np.abs(net2.get_params() - th2).max()),
                       float(np.abs(net2.forward(X) - net.forward(X)).max()))
    print(f"set_params(get_params()) max err {worst_rt:.3g}; "
          f"write_back -> rebuild max err {worst_wb:.3g}")

    g = copy.deepcopy(genomes[-1])
    net = GenomeNetwork(g, config)
    th = net.get_params()
    th[0] = 50.0
    net.set_params(th)
    net.write_back(g)
    key = net.conns[0]
    stored = g.connections[key].weight
    rng_state = __import__("random").getstate()
    __import__("random").seed(1)
    mutated = None
    for _ in range(50):
        gene = copy.deepcopy(g.connections[key])
        gene.mutate(config.genome_config)
        if gene.weight != stored:
            mutated = gene.weight
            break
    __import__("random").setstate(rng_state)
    verdict = ("clamp only applies on mutation, so a refined weight past +-30 "
               "silently jumps back the first time it mutates" if stored > 30
               else "write_back clamps to the config range")
    print(f"write_back of weight 50.0 stores {stored} (config max 30); first "
          f"mutation of that gene gives {mutated:.3f} -> {verdict}")

    section("dead parameters (enabled genes the forward pass never uses)")
    dead_conn, dead_node, n_conn, n_node, any_dead = 0, 0, 0, 0, 0
    for g in genomes:
        net = GenomeNetwork(g, config)
        used = set(net.node_keys)
        dc = sum(1 for (a, b) in net.conns if b not in used)
        dn = sum(1 for n in g.nodes if n not in used)
        dead_conn += dc
        dead_node += dn
        n_conn += len(net.conns)
        n_node += len(g.nodes)
        any_dead += (dc + dn) > 0
    print(f"{dead_conn / n_conn:.1%} of enabled connections and {dead_node / n_node:.1%} "
          f"of nodes feed nothing that reaches the output; {any_dead / len(genomes):.1%} "
          "of genomes have some. complexity() counts them; their gradient is 0")

    section("forward() cost")
    sizes = [complexity(g)[1] for g in genomes]
    med = genomes[int(np.argsort(sizes)[len(sizes) // 2])]
    big = genomes[int(np.argmax(sizes))]
    for label, g in (("median", med), ("largest", big)):
        net = GenomeNetwork(g, config)
        t = time.perf_counter()
        for _ in range(200):
            net.forward(X)
        ms = (time.perf_counter() - t) / 200 * 1e3
        print(f"{label:7} genome ({complexity(g)[1]} conns): {ms:.3f} ms per "
              f"forward on 350 points -> 30,000 steps ~ {ms * 30_000 * 3 / 1e3:.0f} s "
              "if a step costs ~3 forwards")


# ---------------------------------------------------------------- budget
def audit_budget():
    section("budget.py")
    for args in ((100, 10, 1.0), (100, 20, 1.0), (100, 40, 0.25), (100, 7, 1.0),
                 (100, 100, 0.1), (100, 300, 0.05)):
        g = plan(*args)
        per = max(1, int(round(args[0] * args[2]))) * args[1]
        print(f"plan{args} -> {g} generations, uses {g * per:,} / 30,000 "
              f"(unused {30_000 - g * per:,})")
    try:
        plan(100, 0, 1.0)
        print("plan(100, 0, 1.0) returned")
    except ZeroDivisionError:
        print("plan(100, 0, 1.0) -> ZeroDivisionError: a 0-step (no refinement) "
              "baseline needs --generations")
    b = StepBudget(100)
    b.charge(100)
    ok_at_limit = b.spent == 100
    try:
        b.charge(1)
        over = "no exception"
    except BudgetExhausted:
        over = f"BudgetExhausted, spent left at {b.spent}"
    print(f"charging exactly the limit ok={ok_at_limit}; one more -> {over}; "
          f"refinements counts calls ({b.refinements}), not steps")


# ---------------------------------------------------------------- NEAT loop
def audit_loop(per_gen):
    section("NEAT loop: who gets evaluated each generation")
    sizes = [len(g) for g in per_gen]
    carried, same_obj = [], []
    for prev, cur in zip(per_gen, per_gen[1:]):
        prev_ids = {gid: id(g) for gid, g in prev}
        keys = [gid for gid, _ in cur if gid in prev_ids]
        carried.append(len(keys))
        same_obj.append(sum(1 for gid, g in cur if prev_ids.get(gid) == id(g)))
    print(f"population size per generation: min {min(sizes)} max {max(sizes)}")
    print(f"genomes re-evaluated from the previous generation (elites): mean "
          f"{np.mean(carried):.1f}, max {max(carried)}; same Python object: "
          f"{sum(same_obj)}/{sum(carried)}")
    print("  -> elitism=2 is per species; every elite is the same object and is "
          "re-evaluated (and would be re-refined and re-charged) every generation")


def main() -> int:
    config = load_config()
    audit_datasets()
    genomes, per_gen = collect_genomes(config)
    audit_network(config, genomes)
    audit_budget()
    audit_loop(per_gen)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

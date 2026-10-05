"""Checks for the small-network track: --max-conns and the repair behind it.

    python test_smallnet.py                 # everything (about 30 s)
    python test_smallnet.py default grid    # only these groups (see GROUPS)

1. network.live_connections / repair_to_max_conns on hand-built genomes and on
   genomes evolved without a limit; each MaxConnsGenome entry point
   (configure_new, configure_crossover, mutate) on its own, with exact counts.
2. NEAT populations built with load_config(..., max_conns=N): no genome over
   N, repairs really happen, seeds reproduce.
3. Real short main.py runs (3 datasets x 2 strategies x 2 allocations x
   seeds, both configs) checked from their own logs: champion, train-best and
   every evaluated genome within the limit; the repair counts in summary.json
   match genomes.jsonl; budget exact; full-budget and over-budget runs.
4. Determinism with --max-conns.
5. Without --max-conns, main.py is bit-identical to the code before the flag
   existed: the BASELINE commit's files (read with `git show`) and the current
   ones run the same arguments into a temp dir and their outputs are compared.
6. experiments/grids/smallnet.json: parses, and a run of each arm records
   exactly the arguments the harness asks for.

Run dirs go to a temp directory; nothing under runs/ is touched. Every
constrained main.py run uses seeds >= 100, so no test reproduces (or runs the
first generations of) a cell of a graded grid, whose seeds are 0-9 and 20-29.
"""

from __future__ import annotations

import copy
import csv
import dataclasses
import json
import os
import pickle
import random
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import neat
import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "experiments"))

from datasets import get_dataset  # noqa: E402
from network import (GenomeNetwork, MaxConnsGenome, SigmoidOutputGenome,  # noqa: E402
                     champion_net, complexity, live_connections, load_config,
                     repair_to_max_conns)
from test_gradients import build, collect_genomes  # noqa: E402

PY = sys.executable
CONFIG_PATH = HERE / "neat_config.ini"
SINE_PATH = HERE / "neat_config_sine.ini"
GC_CONFIG = load_config(CONFIG_PATH)
GC = GC_CONFIG.genome_config
# The commit this track was branched from: main.py's behaviour without
# --max-conns must equal it. Override with Q2_BASELINE=<rev>.
BASELINE = os.environ.get("Q2_BASELINE", "ebccd07")
BASELINE_FILES = ("main.py", "network.py", "refine.py", "budget.py", "datasets.py",
                  "neat_config.ini")
WORKERS = 6
SEED0 = 100  # constrained runs use SEED0 + i: off every graded grid's seeds


def check(name, ok, detail=""):
    print(f"  {'OK  ' if ok else 'FAIL'} {name}{' — ' + str(detail) if detail != '' else ''}")
    if not ok:
        check.failed += 1


check.failed = 0
check.skipped = 0


def skip(name, why):
    print(f"  SKIP {name} — {why}")
    check.skipped += 1


def enabled(g):
    return sorted(k for k, c in g.connections.items() if c.enabled)


def state(g):
    """Everything about a genome except the enabled flags."""
    return ({k: c.weight for k, c in g.connections.items()},
            {k: (n.bias, n.activation, n.response, n.aggregation) for k, n in g.nodes.items()})


def net_live(net):
    """The (src, dst) connections GenomeNetwork's own pass reads."""
    _, _, live = net._pass(np.zeros((1, 2)))
    return {net.conns[ci] for node in net.eval_order for ci, _ in live[node]}


# ------------------------------------------------------------ 1. repair unit
def test_repair_handbuilt():
    print("repair_to_max_conns on hand-built genomes")
    X = get_dataset("spiral")[0]

    # 5 live (two inputs -> tanh 5 -> output, plus direct), 3 dead (node 6
    # feeds node 7, which feeds nothing).
    g0 = build({0: ("sigmoid", 0.1), 5: ("tanh", 0.2), 6: ("relu", 0.3), 7: ("sigmoid", 0.4)},
               [(-1, 5, 1.1), (-2, 5, -0.9), (5, 0, 1.3), (-1, 0, 0.7), (-2, 0, -0.6),
                (-1, 6, 0.01), (-2, 6, 0.02), (6, 7, 0.03)])
    check("live_connections: the 5 live, not the 3 dead",
          live_connections(g0, GC) == {(-1, 5), (-2, 5), (5, 0), (-1, 0), (-2, 0)}
          and live_connections(g0, GC) == net_live(GenomeNetwork(g0, GC_CONFIG)))

    g = copy.deepcopy(g0)
    before = (copy.deepcopy(state(g)), enabled(g))
    r = repair_to_max_conns(g, GC, 8)
    check("at the limit: nothing changes", r == (0, 0) and (state(g), enabled(g)) == before)

    p0 = GenomeNetwork(g0, GC_CONFIG).forward(X)
    for n, want in ((6, (2, 0)), (5, (3, 0))):
        g = copy.deepcopy(g0)
        r = repair_to_max_conns(g, GC, n)
        check(f"N={n}: dead connections go first (smallest key first), "
              "forward() bit-identical, weights untouched",
              r == want and len(enabled(g)) == n
              and np.array_equal(GenomeNetwork(g, GC_CONFIG).forward(X), p0)
              and state(g) == state(g0)
              and (n != 6 or enabled(g) == sorted(set(enabled(g0)) - {(-2, 6), (-1, 6)})),
              f"disabled {r}, enabled {enabled(g)}")
    g = copy.deepcopy(g0)
    r = repair_to_max_conns(g, GC, 4)
    check("N=4: then the live connection with the smallest |weight| ((-2, 0), 0.6)",
          r == (3, 1) and enabled(g) == [(-2, 5), (-1, 0), (-1, 5), (5, 0)],
          f"disabled {r}, enabled {enabled(g)}")

    # Cascade: the smallest live weight is a hidden node's only way out; once
    # it is gone the node's inputs are dead and go next, not the direct links.
    g1 = build({0: ("sigmoid", 0.1), 5: ("tanh", 0.2)},
               [(-1, 5, 3.0), (-2, 5, -2.5), (5, 0, 0.05), (-1, 0, 1.0), (-2, 0, -1.5)])
    g = copy.deepcopy(g1)
    r = repair_to_max_conns(g, GC, 2)
    check("cascade: cutting a node's output makes its inputs dead; they go before "
          "any other live connection",
          r == (2, 1) and enabled(g) == [(-2, 0), (-1, 0)], f"disabled {r}, enabled {enabled(g)}")

    g2 = build({0: ("sigmoid", 0.1)}, [(-1, 0, 0.5), (-2, 0, -0.5)])
    g = copy.deepcopy(g2)
    r = repair_to_max_conns(g, GC, 1)
    check("equal |weight|: the smaller key goes", r == (0, 1) and enabled(g) == [(-1, 0)],
          enabled(g))
    g = copy.deepcopy(g2)
    check("N=0 disables everything; the network is sigmoid(bias)",
          repair_to_max_conns(g, GC, 0) == (0, 2) and enabled(g) == []
          and np.allclose(GenomeNetwork(g, GC_CONFIG).forward(X), 1 / (1 + np.exp(-0.1))))


def test_entry_points():
    print("MaxConnsGenome: each entry point repairs on its own, and counts exactly")
    cfg6 = load_config(CONFIG_PATH, max_conns=6).genome_config
    cfg1 = load_config(CONFIG_PATH, max_conns=1).genome_config
    big = build({0: ("sigmoid", 0.1), 5: ("tanh", 0.2), 6: ("relu", 0.3)},
                [(-1, 5, 1.1), (-2, 5, -0.9), (5, 0, 1.3), (-1, 0, 0.7), (-2, 0, -0.6),
                 (-1, 6, 0.4), (-2, 6, 0.5), (6, 0, 0.8)])  # 8 enabled, all live
    big.fitness = 1.0

    child = MaxConnsGenome(1)
    child.configure_crossover(big, big, cfg6)  # a clone: 8 enabled before repair
    check("configure_crossover alone: an 8-connection clone comes out with 6, and the "
          "genome records exactly 2 disabled",
          len(enabled(child)) == 6 and (child.repair_dead, child.repair_live) == (0, 2),
          f"{len(enabled(child))} enabled, recorded ({child.repair_dead}, {child.repair_live})")

    # mutate alone, on a genome over the limit (built directly, bypassing
    # repair). The same mutation on a plain SigmoidOutputGenome, same random
    # seed and fresh configs (repair draws no random numbers, so both see the
    # same draws), says how many connections the repair had to cut.
    bad, cut = [], 0
    for seed in range(30):
        out = {}
        for cls, path_n in ((SigmoidOutputGenome, None), (MaxConnsGenome, 6)):
            gc = load_config(CONFIG_PATH, max_conns=path_n).genome_config
            gc.innovation_tracker = neat.innovation.InnovationTracker()
            m = cls(2)
            m.connections = copy.deepcopy(big.connections)
            m.nodes = copy.deepcopy(big.nodes)
            random.seed(seed)
            m.mutate(gc)
            out[cls] = m
        plain, rep_ = out[SigmoidOutputGenome], out[MaxConnsGenome]
        e = len(enabled(plain))
        cut += max(0, e - 6)
        if not (len(enabled(rep_)) == min(e, 6)
                and rep_.repair_dead + rep_.repair_live == max(0, e - 6)
                and set(enabled(rep_)) <= set(enabled(plain))
                and set(rep_.connections) == set(plain.connections)):
            bad.append(seed)
    check("mutate alone (30 seeds, genome starting at 8): comes out with min(E, 6) "
          "connections, E = what the same mutation leaves without repair; the genome "
          "records exactly E - 6 cut", not bad and cut > 0,
          f"{cut} connections cut in total" + (f"; wrong on seeds {bad}" if bad else ""))

    g = MaxConnsGenome(3)
    cfg1.innovation_tracker = neat.innovation.InnovationTracker()
    g.configure_new(cfg1)
    g6 = MaxConnsGenome(4)
    cfg6.innovation_tracker = neat.innovation.InnovationTracker()
    g6.configure_new(cfg6)
    check("configure_new alone: full_direct's 2 connections cut to 1 at N=1 (1 counted); "
          "untouched and nothing counted at N=6",
          len(enabled(g)) == 1 and g.repair_dead + g.repair_live == 1
          and len(enabled(g6)) == 2 and (g6.repair_dead, g6.repair_live) == (0, 0)
          and g6.nodes[0].activation == "sigmoid")


def test_repair_evolved(genomes):
    print(f"repair_to_max_conns on {len(genomes)} genomes evolved without a limit")
    n_live_match = sum(live_connections(g, GC) == net_live(GenomeNetwork(g, GC_CONFIG))
                       for _, g in genomes)
    check("live_connections == the connections GenomeNetwork's pass reads, every genome",
          n_live_match == len(genomes), f"{n_live_match}/{len(genomes)}")

    sizes = Counter(len(enabled(g)) for _, g in genomes)
    bad = Counter()
    n_cut, n_free = 0, 0
    for name, g in genomes:
        X = get_dataset(name)[0]
        p0 = GenomeNetwork(g, GC_CONFIG).forward(X)
        dead0 = len(enabled(g)) - len(live_connections(g, GC))
        for n in range(0, 9):
            h = copy.deepcopy(g)
            rs = random.getstate()
            d, l = repair_to_max_conns(h, GC, n)
            excess = max(0, len(enabled(g)) - n)
            bad["random state consumed"] += random.getstate() != rs
            bad["count != min(before, N)"] += len(enabled(h)) != min(len(enabled(g)), n)
            bad["disabled != excess"] += d + l != excess
            bad["something other than enabled changed"] += state(h) != state(g)
            bad["enabled a disabled gene"] += not set(enabled(h)) <= set(enabled(g))
            h2 = copy.deepcopy(g)
            repair_to_max_conns(h2, GC, n)
            bad["not deterministic"] += enabled(h2) != enabled(h)
            if excess and excess <= dead0:  # enough dead ones: must be free
                n_free += 1
                bad["live disabled while dead ones remained"] += l != 0
                bad["forward() changed by a dead-only repair"] += not np.array_equal(
                    GenomeNetwork(h, GC_CONFIG).forward(X), p0)
            n_cut += excess > 0
    check("for N = 0..8: count = min(before, N), exactly the excess disabled, only "
          "enabled flags True -> False, deterministic, no random numbers drawn",
          not +bad, f"{n_cut} repairs over sizes {dict(sorted(sizes.items()))}"
          + (f"; violations {dict(+bad)}" if +bad else ""))
    check("repairs that only needed dead connections leave forward() bit-identical",
          n_free > 0 and bad["forward() changed by a dead-only repair"] == 0,
          f"{n_free} such repairs")


# ------------------------------------------------------- 2. NEAT populations
def run_population(path, n, seed, generations=40, dataset="spiral"):
    """NEAT with fitness = accuracy + 0.02 x conns (pushes against the limit).

    Returns (per-generation signatures, max enabled seen, genomes created,
    genomes repaired, output activations seen, genome types seen)."""
    X, y = get_dataset(dataset)[:2]
    config = load_config(path, max_conns=n)
    sigs, max_seen, outs, types = [], 0, set(), set()
    created, repaired, repaired_gen0 = set(), set(), 0

    def ev(genomes, cfg):
        nonlocal max_seen, repaired_gen0
        sig = []
        for key, g in genomes:
            conns = complexity(g)[1]
            max_seen = max(max_seen, conns)
            outs.add(g.nodes[0].activation)
            types.add(type(g).__name__)
            created.add(key)
            if g.repair_dead or g.repair_live:
                repaired.add(key)
                repaired_gen0 += not sigs
            g.fitness = GenomeNetwork(g, cfg).accuracy(X, y) + 0.02 * conns
            sig.append((key, tuple((k, c.weight, c.enabled) for k, c in sorted(g.connections.items())),
                        tuple((k, nd.bias, nd.activation) for k, nd in sorted(g.nodes.items()))))
        sigs.append(sig)

    neat.Population(config, seed=seed).run(ev, generations)
    return sigs, max_seen, len(created), len(repaired), outs, types, repaired_gen0


def test_populations():
    print("NEAT populations with load_config(..., max_conns=N)")
    plain = load_config(CONFIG_PATH)
    check("without max_conns: SigmoidOutputGenome, no max_conns on the genome config",
          plain.genome_type is SigmoidOutputGenome
          and not hasattr(plain.genome_config, "max_conns"))
    for bad in (-1, 2.5):
        try:
            load_config(CONFIG_PATH, max_conns=bad)
            raised = False
        except ValueError:
            raised = True
        check(f"max_conns={bad!r} is refused", raised)

    for path in (CONFIG_PATH, SINE_PATH):
        for n in (2, 6):
            sigs, max_seen, created, repaired, outs, types, gen0 = run_population(path, n, seed=3)
            check(f"{path.name}, N={n}: 40 generations, no genome over {n}; repairs happen, "
                  "but not to every genome and none in generation 0; outputs stay sigmoid",
                  max_seen <= n and 0 < repaired < created and gen0 == 0
                  and outs == {"sigmoid"} and types == {"MaxConnsGenome"},
                  f"max {max_seen}; {repaired}/{created} genomes repaired at creation")
    a = run_population(SINE_PATH, 6, seed=11, generations=25)
    b = run_population(SINE_PATH, 6, seed=11, generations=25)
    c = run_population(SINE_PATH, 6, seed=12, generations=25)
    check("same seed, fresh configs: identical genomes every generation; another seed differs",
          a[0] == b[0] and a[0] != c[0], f"{len(a[0])} generations")


# ----------------------------------------------------------- 3. main.py runs
def run_main(tmp, rid, *args, main=HERE / "main.py", log=True):
    out = Path(tmp) / rid
    cmd = [PY, str(main), "--quiet", "--run-id", str(out), *args]
    if log:
        cmd.append("--log-genomes")
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=Path(main).parent)
    summary = (json.loads((out / "summary.json").read_text())
               if (out / "summary.json").exists() else None)
    rows = ([json.loads(line) for line in (out / "genomes.jsonl").open()]
            if (out / "genomes.jsonl").exists() else [])
    hist = (list(csv.DictReader((out / "fitness_history.csv").open()))
            if (out / "fitness_history.csv").exists() else [])
    return r, summary, rows, hist, out


ARMS = {  # (strategy, steps, refine fraction)
    "lam_std": ("lamarckian", 10, 1.0),
    "bald_std": ("baldwinian", 100, 0.1),
    "lam_deep": ("lamarckian", 100, 0.1),
    "bald_shallow": ("baldwinian", 10, 1.0),
}


def run_cells(tmp, cells):
    with ThreadPoolExecutor(WORKERS) as ex:
        futs = {rid: ex.submit(run_main, tmp, rid, *args) for rid, args in cells.items()}
        return {rid: f.result() for rid, f in futs.items()}


def check_cell(rid, res, n, gens, strategy, steps, frac):
    """Every property of one constrained run, from its own files. Returns
    (list of failed property names, repairs seen)."""
    r, s, rows, hist, out = res
    fails = []
    if r.returncode != 0 or s is None:
        return [f"exit {r.returncode}: {r.stderr.strip()[-300:]}"], 0
    k = max(1, int(round(100 * frac)))
    champ = pickle.loads((out / "champion.pkl").read_bytes())
    ds = s["args"]["dataset"]
    Xtr, ytr, Xte, yte = get_dataset(ds)
    cfg = load_config(HERE / s["args"]["config"], max_conns=n)
    keys = {x["key"] for x in rows}
    rep_keys = {x["key"] for x in rows if x["repair_dead"] + x["repair_live"] > 0}
    first = {}
    for x in rows:
        first.setdefault(x["key"], x)
    by_gen = defaultdict(int)
    for x in rows:
        by_gen[x["gen"]] += x["steps"]
    props = {
        "champion within N (summary)": s["conns"] <= n,
        "train-best within N (summary)": s["trainbest_conns"] <= n,
        "champion.pkl within N, a MaxConnsGenome": (complexity(champ)[1] <= n
                                                    and type(champ).__name__ == "MaxConnsGenome"),
        "every evaluated genome within N": all(x["conns"] <= n for x in rows),
        "summary held-out acc = champion_net's": abs(
            champion_net(champ, cfg).accuracy(Xte, yte) - s["heldout_acc"]) < 1e-12,
        "max_conns recorded": s["max_conns"] == n and s["args"]["max_conns"] == n,
        "genomes_created = distinct genomes logged": s["genomes_created"] == len(keys),
        "genomes_repaired = distinct genomes logged with a repair":
            s["genomes_repaired"] == len(rep_keys),
        "repair_disabled_* = sums over distinct genomes":
            s["repair_disabled_dead"] == sum(x["repair_dead"] for x in first.values())
            and s["repair_disabled_live"] == sum(x["repair_live"] for x in first.values()),
        "generation 0: 100 new genomes, none repaired; not every genome repaired":
            int(hist[0]["n_new_genomes"]) == 100 and int(hist[0]["n_repaired"]) == 0
            and s["genomes_repaired"] < s["genomes_created"],
        "history n_new_genomes / n_repaired sum to the summary":
            sum(int(h["n_new_genomes"]) for h in hist) == s["genomes_created"]
            and sum(int(h["n_repaired"]) for h in hist) == s["genomes_repaired"],
        "budget: spent = generations x refined x steps, within budget":
            s["budget_spent"] == gens * k * steps and s["generations_run"] == gens
            and s["within_budget"] and s["refine_calls"] == gens * k
            and s["genomes_refined"] == gens * k,
        "every generation spends exactly refined x steps":
            all(v == k * steps for v in by_gen.values()) and len(by_gen) == gens,
        f"{strategy}: inheritance as before (genome changed iff refined)":
            all((x["hash_before"] != x["hash_after"]) == (x["steps"] > 0) for x in rows)
            if strategy == "lamarckian" else
            all(x["hash_before"] == x["hash_after"] for x in rows),
    }
    fails = [name for name, ok in props.items() if not ok]
    return fails, s["genomes_repaired"]


def test_runs(tmp):
    print("main.py --max-conns: short real runs, checked from their own logs")
    gens = 8
    cells, meta = {}, {}
    for arm, (strategy, steps, frac) in ARMS.items():
        for ds in ("xor", "spiral", "spiral3"):
            for seed in (SEED0, SEED0 + 1):
                for cname, path, n in (("std", "neat_config.ini", 6),
                                       ("sine", "neat_config_sine.ini", 6),
                                       ("std3", "neat_config.ini", 3)):
                    if cname == "std3" and seed != SEED0:
                        continue
                    rid = f"{arm}_{cname}_{ds}_s{seed}"
                    cells[rid] = ["--dataset", ds, "--seed", str(seed), "--lr", "0.3",
                                  "--max-conns", str(n), "--steps", str(steps),
                                  "--refine-fraction", str(frac), "--strategy", strategy,
                                  "--config", path, "--generations", str(gens)]
                    meta[rid] = (n, strategy, steps, frac)
    results = run_cells(tmp, cells)
    failed, repairs, champ_sizes = {}, 0, Counter()
    for rid, res in results.items():
        n, strategy, steps, frac = meta[rid]
        fails, rep = check_cell(rid, res, n, gens, strategy, steps, frac)
        if fails:
            failed[rid] = fails
        repairs += rep
        if res[1]:
            champ_sizes[(n, res[1]["conns"])] += 1
    check(f"{len(cells)} runs (4 allocations/strategies x 3 datasets x 2 seeds x "
          "{std, sine} at N=6, plus N=3): every property holds",
          not failed, f"failures: {failed}" if failed else
          f"{repairs} genomes repaired in total; champion sizes (N, conns): "
          f"{dict(sorted(champ_sizes.items()))}")
    check("repair was exercised (not a vacuous pass)", repairs > 0, repairs)


def test_budget(tmp):
    print("budget with --max-conns: full default budget, and an overrun")
    cells = {
        "full_lam": ["--dataset", "spiral3", "--seed", str(SEED0 + 4), "--lr", "0.3", "--max-conns", "6",
                     "--steps", "10", "--refine-fraction", "1.0", "--strategy", "lamarckian"],
        "full_bald": ["--dataset", "spiral", "--seed", str(SEED0 + 5), "--lr", "0.3", "--max-conns", "6",
                      "--steps", "100", "--refine-fraction", "0.1", "--strategy", "baldwinian",
                      "--config", "neat_config_sine.ini"],
    }
    results = run_cells(tmp, cells)
    for rid, (r, s, rows, hist, out) in results.items():
        ok = (r.returncode == 0 and s and s["generations_planned"] == 30
              and s["generations_run"] == 30 and s["budget_spent"] == 30000
              and s["within_budget"] and s["conns"] <= 6 and s["trainbest_conns"] <= 6
              and all(x["conns"] <= 6 for x in rows))
        check(f"{rid}: plan() gives 30 generations, 30,000/30,000 steps, everything within 6",
              # no accuracy printed: predictions for the smallnet grid are
              # written before its results are seen
              ok, s and f"{s['budget']}; champion {s['conns']} conns; "
                        f"{s['genomes_repaired']}/{s['genomes_created']} genomes repaired")
    r, s, rows, _, _ = run_main(tmp, "halving", "--dataset", "spiral", "--seed", str(SEED0 + 6),
                                "--allocation", "halving", "--generations", "6", "--strategy",
                                "lamarckian", "--lr", "0.3", "--max-conns", "6")
    check("successive halving with --max-conns: 984 steps per generation, everything within 6",
          r.returncode == 0 and s and s["budget_spent"] == 6 * 984 and s["conns"] <= 6
          and s["trainbest_conns"] <= 6 and all(x["conns"] <= 6 for x in rows),
          s and f"{s['budget']}; {s['genomes_repaired']} repaired" or r.stderr[-200:])
    r, s, _, _, out = run_main(tmp, "over", "--dataset", "xor", "--steps", "10", "--budget", "1000",
                               "--generations", "3", "--strategy", "baldwinian", "--lr", "0.3",
                               "--max-conns", "6")
    check("overrun with --max-conns: exits 4, logs kept, marked invalid, champion within N",
          r.returncode == 4 and s and not s["within_budget"] and s["max_conns"] == 6
          and s["conns"] <= 6 and (out / "fitness_history.csv").exists(), s and s["budget"])
    for bad in ("0", "-2", "x"):
        r = subprocess.run([PY, str(HERE / "main.py"), "--quiet", "--run-id", f"{tmp}/bad",
                            "--steps", "0", "--generations", "1", "--max-conns", bad],
                           capture_output=True, text=True)
        check(f"--max-conns {bad} is a usage error (exit 2)", r.returncode == 2,
              r.stderr.strip()[-80:])


# ---------------------------------------------------------- 4. determinism
def comparable(s):
    s = copy.deepcopy(s)
    for k in ("seconds",):
        s.pop(k, None)
    s["args"].pop("run_id", None)
    return s


def test_determinism(tmp):
    print("determinism with --max-conns")
    seed = str(SEED0 + 7)
    args = ["--dataset", "spiral", "--seed", seed, "--lr", "0.3", "--max-conns", "6",
            "--steps", "10", "--strategy", "lamarckian", "--generations", "12",
            "--config", "neat_config_sine.ini"]
    res = run_cells(tmp, {"det_a": args, "det_b": args,
                          "det_c": [a if a != seed else str(SEED0 + 8) for a in args]})
    a, b, c = res["det_a"], res["det_b"], res["det_c"]
    if not all(x[0].returncode == 0 and x[1] for x in (a, b, c)):
        check("determinism runs ran", False,
              " | ".join(x[0].stderr.strip()[-150:] for x in (a, b, c)))
        return
    same = ((a[4] / "fitness_history.csv").read_bytes() == (b[4] / "fitness_history.csv").read_bytes()
            and (a[4] / "genomes.jsonl").read_bytes() == (b[4] / "genomes.jsonl").read_bytes()
            and (a[4] / "champion.pkl").read_bytes() == (b[4] / "champion.pkl").read_bytes()
            and comparable(a[1]) == comparable(b[1]))
    check("same args twice: identical fitness_history.csv, genomes.jsonl, champion.pkl, "
          "summary.json (but seconds and run id)", same and a[1]["genomes_repaired"] > 0,
          f"{a[1]['genomes_repaired']} repairs")
    check("...and a different seed gives a different run",
          (a[4] / "genomes.jsonl").read_bytes() != (c[4] / "genomes.jsonl").read_bytes())


# ---------------------------------------- 5. default behaviour bit-identical
def test_default_identical(tmp):
    print(f"without --max-conns: bit-identical to the code at {BASELINE}")
    base = Path(tmp) / "baseline_code"
    base.mkdir()
    for f in BASELINE_FILES:
        r = subprocess.run(["git", "show", f"{BASELINE}:q2_backprop_neat/{f}"], cwd=HERE,
                           capture_output=True, text=True)
        if r.returncode != 0:
            skip("baseline comparison", f"cannot read {f} at {BASELINE}: {r.stderr.strip()[:80]}")
            return
        (base / f).write_text(r.stdout)
    r = subprocess.run(["git", "show", f"{BASELINE}:q2_backprop_neat/main.py"], cwd=HERE,
                       capture_output=True, text=True)
    check("the baseline main.py has no --max-conns (it is the code before the flag)",
          "--max-conns" not in r.stdout and "--max-conns" in (HERE / "main.py").read_text())

    configs = {
        "lam_xor": ["--dataset", "xor", "--seed", "1", "--steps", "10", "--generations", "4",
                    "--strategy", "lamarckian", "--lr", "0.3"],
        "bald_spiral_deep": ["--dataset", "spiral", "--seed", "2", "--steps", "100",
                             "--refine-fraction", "0.1", "--generations", "4",
                             "--strategy", "baldwinian", "--lr", "0.3"],
        "halving_spiral3": ["--dataset", "spiral3", "--allocation", "halving", "--generations",
                            "2", "--strategy", "lamarckian", "--lr", "0.3"],
        "penalty_xor": ["--dataset", "xor", "--steps", "5", "--generations", "3",
                        "--strategy", "baldwinian", "--lr", "0.3", "--penalty", "0.01",
                        "--penalty-form", "sqrt"],
        "plain_neat": ["--dataset", "spiral", "--steps", "0", "--generations", "6"],
        "full_lam_spiral3": ["--dataset", "spiral3", "--seed", "3", "--steps", "10",
                             "--strategy", "lamarckian", "--lr", "0.3"],
    }
    with ThreadPoolExecutor(WORKERS) as ex:
        futs = {(name, which): ex.submit(run_main, tmp, f"{which}_{name}", *args, main=main)
                for name, args in configs.items()
                for which, main in (("new", HERE / "main.py"), ("old", base / "main.py"))}
        res = {k: f.result() for k, f in futs.items()}
    for name in configs:
        new, old = res[(name, "new")], res[(name, "old")]
        if new[1] is None or old[1] is None:
            check(f"{name}: both ran", False, (new[0].stderr + old[0].stderr).strip()[-300:])
            continue
        files_same = all((new[4] / f).read_bytes() == (old[4] / f).read_bytes()
                         for f in ("fitness_history.csv", "genomes.jsonl"))
        sn, so = comparable(new[1]), comparable(old[1])
        for s in (sn, so):
            s.pop("git_commit")
            s["args"].pop("config")
        extra = sn["args"].pop("max_conns", "missing")
        cn = pickle.loads((new[4] / "champion.pkl").read_bytes())
        co = pickle.loads((old[4] / "champion.pkl").read_bytes())
        rn, ro = getattr(cn, "refined_params", None), getattr(co, "refined_params", None)
        champ_same = (state(cn) == state(co) and enabled(cn) == enabled(co)
                      and type(cn).__name__ == type(co).__name__ == "SigmoidOutputGenome"
                      and ((rn is None and ro is None)
                           or (rn is not None and ro is not None and np.array_equal(rn, ro))))
        stdout_same = (new[0].stdout.replace(str(new[4]), "<run>")
                       == old[0].stdout.replace(str(old[4]), "<run>"))
        check(f"{name}: fitness_history.csv and genomes.jsonl byte-identical, summary.json "
              "equal (args gain only max_conns: null), same champion and stdout",
              files_same and sn == so and extra is None and champ_same
              and new[0].returncode == old[0].returncode == 0
              and stdout_same,
              f"{new[1]['generations_run']} generations, held-out {new[1]['heldout_acc']:.3f}")


# ------------------------------------------------------------- 6. the grid
def test_grid(tmp):
    print("experiments/grids/smallnet.json")
    import gridlib
    grid = gridlib.load_grid(HERE / "experiments" / "grids" / "smallnet.json")
    cells = grid.cells(Path(tmp) / "gridruns")
    arms = dict(grid.arms)
    check("4 arms x 3 datasets x 10 seeds = 120 cells, every one with --max-conns 6 and lr 0.3",
          len(cells) == 120 and set(arms) == {"lam_std", "bald_std", "lam_sine", "bald_sine"}
          and all(a["--max-conns"] == 6 and a["--lr"] == 0.3 for a in arms.values())
          and sorted(grid.seeds) == list(range(10)) and not grid.warnings, grid.warnings)
    want = {"lam_std": ("lamarckian", 10, 1.0, "neat_config.ini"),
            "bald_std": ("baldwinian", 100, 0.1, "neat_config.ini"),
            "lam_sine": ("lamarckian", 10, 1.0, "neat_config_sine.ini"),
            "bald_sine": ("baldwinian", 100, 0.1, "neat_config_sine.ini")}
    check("arms are the settled allocations, std and sine configs",
          all((a["--strategy"], a["--steps"], a["--refine-fraction"], a["--config"]) == want[n]
              for n, a in arms.items()))
    # One cell per arm, run as the harness runs it (its argv, cwd main.py's
    # directory) but for 2 generations and on seed SEED0 + 9 instead of 9 (no
    # graded cell is run): the recorded args must match what it asked for.
    picked = []
    for c in cells:
        if c.dataset == "spiral3" and c.seed == 9:
            seed = SEED0 + 9
            req = tuple((f, seed if f == "--seed" else v) for f, v in c.requested)
            picked.append(dataclasses.replace(c, seed=seed, requested=req,
                                              dir=c.dir.parent / f"{c.dataset}_s{seed}"))
    with ThreadPoolExecutor(WORKERS) as ex:
        procs = list(ex.map(lambda c: subprocess.run(
            c.argv(PY, HERE / "main.py") + ["--generations", "2"], cwd=HERE,
            capture_output=True, text=True), picked))
    bad = {}
    for c, p in zip(picked, procs):
        s, err = gridlib.read_json(c.dir / gridlib.SUMMARY_FILE)
        diffs = gridlib.compare_args(c, s["args"]) if s else [f"no summary: {err}"]
        verdict = gridlib.judge_summary(s)[0] if s else "crashed"
        if p.returncode or diffs or verdict != "ok" or s["conns"] > 6:
            bad[c.arm] = (p.returncode, diffs, verdict, p.stderr.strip()[-200:])
    check("a cell of each arm runs, records exactly the requested args, is judged ok",
          len(picked) == 4 and not bad, bad)


def evolved_repair():
    test_repair_evolved(collect_genomes(seeds=(0,), config_path=CONFIG_PATH)
                        + collect_genomes(seeds=(0,), config_path=SINE_PATH))


GROUPS = {  # name: (function, needs a temp dir)
    "unit": (lambda: (test_repair_handbuilt(), test_entry_points()), False),
    "evolved": (evolved_repair, False),
    "populations": (test_populations, False),
    "runs": (test_runs, True),
    "budget": (test_budget, True),
    "determinism": (test_determinism, True),
    "default": (test_default_identical, True),
    "grid": (test_grid, True),
}


def main(argv) -> int:
    names = argv or list(GROUPS)
    unknown = set(names) - set(GROUPS)
    if unknown:
        print(f"unknown groups {sorted(unknown)}; choose from {list(GROUPS)}")
        return 2
    with tempfile.TemporaryDirectory() as tmp:
        for name in names:
            fn, needs_tmp = GROUPS[name]
            fn(tmp) if needs_tmp else fn()
    tail = f" ({check.skipped} skipped)" if check.skipped else ""
    print(f"\n{'all checks passed' if not check.failed else f'{check.failed} FAILED'}{tail}")
    return 1 if check.failed else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

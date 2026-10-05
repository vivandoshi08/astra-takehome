"""Checks for the fixes made to the scaffold after audit_scaffold.py.

    python test_scaffold_fixes.py

Each check reproduces the shipped failure and asserts it no longer happens.
Run dirs go to a temp directory; nothing under runs/ is touched.
"""

from __future__ import annotations

import copy
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import neat
import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

import main as driver  # noqa: E402
from datasets import get_dataset  # noqa: E402
from network import GenomeNetwork, champion_net  # noqa: E402

PY = sys.executable
CONFIG = driver.load_config(str(HERE / "neat_config.ini"))
X, Y, _, _ = get_dataset("spiral")


def check(name, ok, detail=""):
    print(f"  {'OK  ' if ok else 'FAIL'} {name}{' — ' + detail if detail else ''}")
    if not ok:
        check.failed += 1


check.failed = 0


def run_pop(evaluate, generations=40, seed=0):
    pop = neat.Population(CONFIG, seed=seed)
    pop.run(evaluate, generations)


def test_output_pinned():
    print("output node stays sigmoid")
    out_acts, hidden_acts = set(), set()

    def ev(genomes, cfg):
        for _, g in genomes:
            out_acts.add(g.nodes[0].activation)
            hidden_acts.update(n.activation for k, n in g.nodes.items() if k != 0)
            g.fitness = GenomeNetwork(g, cfg).accuracy(X, Y)

    run_pop(ev)
    check("every output node is sigmoid over 40 generations", out_acts == {"sigmoid"},
          str(sorted(out_acts)))
    check("hidden nodes still mutate activation", len(hidden_acts) > 1,
          str(sorted(hidden_acts)))


def test_write_back_clamps():
    print("write_back clamps to the config range")
    g = CONFIG.genome_type(0)
    g.configure_new(CONFIG.genome_config)
    net = GenomeNetwork(g, CONFIG)
    theta = net.get_params()
    theta[0], theta[-1] = 50.0, -45.0
    net.set_params(theta)
    n = net.write_back(g)
    rebuilt = GenomeNetwork(g, CONFIG).get_params()
    check("weight 50 stored as 30, bias -45 as -30",
          rebuilt[0] == 30.0 and rebuilt[-1] == -30.0, f"{rebuilt[0]}, {rebuilt[-1]}")
    check("write_back reports 2 clamped values", n == 2, str(n))
    check("lower/upper match the parameter vector",
          net.lower.shape == net.upper.shape == theta.shape)


def test_refined_params():
    print("refined parameters reach the champion, and only the genome that earned them")
    g = CONFIG.genome_type(0)
    g.configure_new(CONFIG.genome_config)
    g.refined_params = GenomeNetwork(g, CONFIG).get_params() + 1.0
    check("champion_net loads refined_params",
          np.array_equal(champion_net(g, CONFIG).get_params(), g.refined_params))
    check("GenomeNetwork alone does not",
          not np.array_equal(GenomeNetwork(g, CONFIG).get_params(), g.refined_params))
    driver.evaluate_genome(g, CONFIG, X, Y)
    check("evaluate_genome clears a stale vector", g.refined_params is None)

    # Every evaluation stores a vector; at the next evaluation only elites
    # (the same objects, carried over) may still hold one.
    leaked, elites, prev_keys = 0, 0, set()

    def ev(genomes, cfg):
        nonlocal leaked, elites, prev_keys
        for gid, g in genomes:
            if getattr(g, "refined_params", None) is not None:
                if gid in prev_keys:
                    elites += 1
                else:
                    leaked += 1
            g.refined_params = np.zeros(1)
            g.fitness = GenomeNetwork(g, cfg).accuracy(X, Y)
        prev_keys = {gid for gid, _ in genomes}

    run_pop(ev, generations=20)
    check("no offspring inherits a refined_params vector", leaked == 0,
          f"{leaked} leaked, {elites} held by carried-over elites")


def cli(*args):
    return subprocess.run([PY, str(HERE / "main.py"), "--quiet", *args],
                          capture_output=True, text=True)


def test_cli(tmp):
    print("main.py run bookkeeping")
    r = cli("--dataset", "xor", "--steps", "0", "--run-id", f"{tmp}/a")
    check("--steps 0 without --generations is refused, not a ZeroDivisionError",
          r.returncode == 2 and "needs --generations" in r.stderr)

    r = cli("--dataset", "xor", "--steps", "0", "--generations", "5", "--run-id", f"{tmp}/a")
    s = json.loads(Path(f"{tmp}/a/summary.json").read_text())
    check("summary.json written with budget and champion fields",
          r.returncode == 0 and s["within_budget"] and "budget" in s
          and "heldout_acc" in s and "git_commit" in s, s.get("budget", ""))

    r = cli("--dataset", "xor", "--steps", "0", "--generations", "5", "--run-id", f"{tmp}/a")
    check("re-using a run id is refused", r.returncode == 2 and "already holds" in r.stderr)
    r = cli("--dataset", "xor", "--steps", "0", "--generations", "5", "--run-id",
            f"{tmp}/a", "--overwrite")
    check("--overwrite allows it", r.returncode == 0)

    # Overrun: patch evaluate_genome to charge, then ask for more generations
    # than a 1,000-step budget affords. The run must still write its logs.
    code = (
        "import sys; sys.argv = ['main.py', '--quiet', '--dataset', 'xor', "
        "'--steps', '10', '--budget', '1000', '--generations', '3', "
        f"'--run-id', '{tmp}/b']\n"
        f"sys.path.insert(0, {str(HERE)!r})\n"
        "import main\n"
        "orig = main.evaluate_genome\n"
        "def charging(genome, config, X, y, budget=None, steps=0):\n"
        "    budget.charge(steps)\n"
        "    return orig(genome, config, X, y, budget, steps)\n"
        "main.evaluate_genome = charging\n"
        "raise SystemExit(main.main())\n"
    )
    r = subprocess.run([PY, "-c", code], capture_output=True, text=True)
    sb = json.loads(Path(f"{tmp}/b/summary.json").read_text())
    check("a budget overrun exits 1, keeps its logs, and is marked invalid",
          r.returncode == 1 and not sb["within_budget"] and sb["budget_error"]
          and Path(f"{tmp}/b/fitness_history.csv").exists()
          and "WARNING" in r.stdout, sb["budget"])


def test_viz(tmp):
    print("viz.py output naming")
    out = Path(tmp) / "xor.png"
    r = subprocess.run([PY, str(HERE / "viz.py"), "--dataset", "xor",
                        "--champion", f"{tmp}/a/champion.pkl", "--out", str(out)],
                       capture_output=True, text=True)
    fit = Path(tmp) / "xor_fitness.png"
    check("--out without 'boundary' keeps the boundary plot and writes a separate "
          "fitness plot",
          r.returncode == 0 and out.exists() and fit.exists()
          and out.read_bytes() != fit.read_bytes(), r.stderr.strip()[-200:])


def main() -> int:
    test_output_pinned()
    test_write_back_clamps()
    test_refined_params()
    with tempfile.TemporaryDirectory() as tmp:
        test_cli(tmp)
        test_viz(tmp)
    print(f"\n{'all checks passed' if not check.failed else f'{check.failed} FAILED'}")
    return 1 if check.failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

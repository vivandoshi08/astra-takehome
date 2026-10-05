"""End-to-end checks of main.py's refinement, allocation and bookkeeping.

    python test_main.py

Runs short real runs into a temp directory and checks them from their own
logs (summary.json, genomes.jsonl, champion.pkl). Nothing under runs/.
"""

from __future__ import annotations

import json
import pickle
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

import main as driver  # noqa: E402
from datasets import get_dataset  # noqa: E402
from network import GenomeNetwork, champion_net, load_config  # noqa: E402

PY = sys.executable
CONFIG = load_config(HERE / "neat_config.ini")
LR = "0.05"


def check(name, ok, detail=""):
    print(f"  {'OK  ' if ok else 'FAIL'} {name}{' — ' + str(detail) if detail != '' else ''}")
    if not ok:
        check.failed += 1


check.failed = 0


def run(tmp, rid, *args):
    out = Path(tmp) / rid
    r = subprocess.run([PY, str(HERE / "main.py"), "--quiet", "--log-genomes",
                        "--run-id", str(out), *args], capture_output=True, text=True)
    summary = json.loads((out / "summary.json").read_text()) if (out / "summary.json").exists() else None
    rows = ([json.loads(line) for line in (out / "genomes.jsonl").open()]
            if (out / "genomes.jsonl").exists() else [])
    return r, summary, rows, out


def by_gen(rows):
    g = defaultdict(list)
    for row in rows:
        g[row["gen"]].append(row)
    return g


def test_split():
    print("validation split")
    for d in ("xor", "spiral", "spiral3"):
        Xtr, ytr, _, _ = get_dataset(d)
        a = driver.split_validation(Xtr, ytr, 0.2)
        b = driver.split_validation(Xtr, ytr, 0.2)
        fit = {tuple(x) for x in a[0]}
        val = {tuple(x) for x in a[2]}
        check(f"{d}: fit/val disjoint, cover train, deterministic",
              not fit & val and fit | val == {tuple(x) for x in Xtr}
              and all(np.array_equal(p, q) for p, q in zip(a, b)),
              f"{len(fit)} / {len(val)}")


def test_fixed(tmp):
    print("fixed allocation, all refined: budget and inheritance")
    for strat in ("lamarckian", "baldwinian"):
        r, s, rows, _ = run(tmp, f"fix_{strat}", "--dataset", "xor", "--steps", "10",
                            "--generations", "3", "--strategy", strat, "--lr", LR)
        check(f"{strat}: run ok", r.returncode == 0, r.stderr.strip()[-200:])
        if s is None:
            continue
        check(f"{strat}: spent 3 gens x 100 genomes x 10 steps",
              s["budget_spent"] == 3000 and s["refine_calls"] == 300
              and s["budget_refinements"] == 300, s["budget"])
        check(f"{strat}: every genome refined 10 steps", all(x["steps"] == 10 for x in rows),
              Counter(x["steps"] for x in rows))
        changed = sum(x["hash_before"] != x["hash_after"] for x in rows)
        if strat == "lamarckian":
            check("lamarckian: refined weights written into the genome",
                  changed == len(rows), f"{changed}/{len(rows)} genomes changed")
        else:
            check("baldwinian: genome weights never change", changed == 0,
                  f"{changed}/{len(rows)} genomes changed")
        check(f"{strat}: nothing clamped", all(x["clamped"] == 0 for x in rows))

        # Elites carried into the next generation: under Lamarckian they
        # start from the refined weights (pre == last post); under Baldwinian
        # they start from the same weights and repeat the same refinement.
        g = by_gen(rows)
        pairs = []
        for t in range(2):
            prev = {x["key"]: x for x in g[t]}
            pairs += [(prev[x["key"]], x) for x in g[t + 1] if x["key"] in prev]
        if strat == "lamarckian":
            ok = all(b["hash_before"] == a["hash_after"] and b["pre_loss"] == a["loss"]
                     for a, b in pairs)
            check("lamarckian: a carried-over elite starts where its refinement ended",
                  ok and pairs, f"{len(pairs)} elites")
        else:
            ok = all(b["hash_before"] == a["hash_before"] and b["loss"] == a["loss"]
                     and b["acc"] == a["acc"] for a, b in pairs)
            check("baldwinian: a carried-over elite repeats an identical refinement",
                  ok and pairs, f"{len(pairs)} elites")


def test_partial(tmp):
    print("fixed allocation, top 25% refined")
    r, s, rows, _ = run(tmp, "partial", "--dataset", "spiral", "--steps", "40",
                        "--refine-fraction", "0.25", "--generations", "3",
                        "--strategy", "baldwinian", "--lr", LR)
    check("run ok", r.returncode == 0, r.stderr.strip()[-200:])
    if s is None:
        return
    check("spent 3 x 25 x 40", s["budget_spent"] == 3000, s["budget"])
    ok = True
    for gen, gr in by_gen(rows).items():
        ranked = sorted(gr, key=lambda x: (-x["pre_acc"], x["pre_loss"], x["key"]))
        top = {x["key"] for x in ranked[:25]}
        refined = {x["key"] for x in gr if x["steps"] > 0}
        ok &= top == refined
    check("the refined 25 are the top 25 by pre-refinement fitness", ok)
    unrefined = [x for x in rows if x["steps"] == 0]
    check("unrefined genomes keep their pre-refinement fitness",
          all(x["fitness"] == x["pre_acc"] for x in unrefined), f"{len(unrefined)} rows")


def test_halving(tmp):
    print("successive halving")
    r, s, rows, _ = run(tmp, "halving", "--dataset", "spiral", "--allocation", "halving",
                        "--generations", "2", "--strategy", "lamarckian", "--lr", LR)
    check("run ok", r.returncode == 0, r.stderr.strip()[-200:])
    if s is None:
        return
    check("984 steps per generation", s["budget_spent"] == 2 * 984 and
          s["steps_per_generation"] == 984, s["budget"])
    for gen, gr in by_gen(rows).items():
        dist = Counter(x["steps"] for x in gr)
        check(f"gen {gen}: 50 x 2, 25 x 6, 13 x 14, 6 x 30, 6 x 62 steps",
              dist == Counter({2: 50, 6: 25, 14: 13, 30: 6, 62: 6}), dict(dist))
    r2, s2, _, _ = run(tmp, "halving_full", "--dataset", "spiral", "--allocation",
                       "halving", "--strategy", "baldwinian", "--lr", LR)
    check("default budget: 30 generations, 29,520 steps, within budget",
          s2 and s2["generations_run"] == 30 and s2["budget_spent"] == 29520
          and s2["within_budget"], s2 and s2["budget"])


def test_champion(tmp):
    print("champion: chosen on validation, evaluated with its refined parameters")
    r, s, rows, out = run(tmp, "champ", "--dataset", "spiral", "--steps", "20",
                          "--generations", "5", "--strategy", "baldwinian", "--lr", LR)
    if s is None:
        check("run ok", False, r.stderr.strip()[-200:])
        return
    champ = pickle.loads((out / "champion.pkl").read_bytes())
    Xtr, ytr, Xte, yte = get_dataset("spiral")
    Xf, yf, Xv, yv = driver.split_validation(Xtr, ytr, 0.2)
    net = champion_net(champ, CONFIG)
    raw = GenomeNetwork(champ, CONFIG)
    check("champion carries refined_params", getattr(champ, "refined_params", None) is not None)
    check("summary val/held-out acc = champion_net's", abs(net.accuracy(Xv, yv) - s["val_acc"]) < 1e-12
          and abs(net.accuracy(Xte, yte) - s["heldout_acc"]) < 1e-12)
    check("baldwinian champion's genome weights differ from its refined ones",
          not np.allclose(raw.get_params(), net.get_params()))
    gen_best = {}
    for x in rows:
        b = gen_best.get(x["gen"])
        if b is None or (x["fitness"], -x["loss"], -x["key"]) > (b["fitness"], -b["loss"], -b["key"]):
            gen_best[x["gen"]] = x
    best_val = max(b["val_acc"] for b in gen_best.values())
    check("champion = the generation-best with the highest validation accuracy",
          abs(s["val_acc"] - best_val) < 1e-12 and
          gen_best[s["champion_generation"]]["val_acc"] == best_val,
          f"val {s['val_acc']:.3f}, generation {s['champion_generation']}")


def test_penalty(tmp):
    print("complexity penalty")
    r, s, rows, _ = run(tmp, "pen", "--dataset", "xor", "--steps", "5", "--generations", "2",
                        "--strategy", "baldwinian", "--lr", LR, "--penalty", "0.01")
    check("fitness = acc - 0.01 * conns",
          rows and all(abs(x["fitness"] - (x["acc"] - 0.01 * x["conns"])) < 1e-12 for x in rows))
    _, s2, rows2, _ = run(tmp, "pen_champ", "--dataset", "spiral", "--steps", "5",
                          "--generations", "6", "--strategy", "baldwinian", "--lr", LR,
                          "--penalty", "0.02")
    gen_best = {}
    for x in rows2:
        b_ = gen_best.get(x["gen"])
        if b_ is None or (x["fitness"], -x["loss"], -x["key"]) > (b_["fitness"], -b_["loss"], -b_["key"]):
            gen_best[x["gen"]] = x
    keyf = lambda x: (x["val_acc"] - 0.02 * x["conns"], x["fitness"])
    want = max(sorted(gen_best), key=lambda g: (keyf(gen_best[g]), -g))
    check("with a penalty, the champion maximises penalised validation accuracy",
          s2 and s2["champion_generation"] == want, f"generation {s2 and s2['champion_generation']}, expected {want}")
    check("penalty forms", abs(driver.penalised(0.9, 4, 0.1, "sqrt") - 0.7) < 1e-12
          and abs(driver.penalised(0.9, 4, 0.1, "mult") - (1 - 0.1 * 1.2)) < 1e-12)


def test_budget_and_determinism(tmp):
    print("budget overrun and determinism")
    r, s, _, out = run(tmp, "over", "--dataset", "xor", "--steps", "10", "--budget", "1000",
                       "--generations", "3", "--strategy", "baldwinian", "--lr", LR)
    check("overrun exits 4, keeps logs, marked invalid",
          r.returncode == driver.BUDGET_EXIT and s and not s["within_budget"]
          and (out / "fitness_history.csv").exists() and "WARNING" in r.stdout,
          s and s["budget"])
    a = run(tmp, "det_a", "--dataset", "spiral3", "--steps", "10", "--generations", "4",
            "--strategy", "lamarckian", "--lr", LR)
    b = run(tmp, "det_b", "--dataset", "spiral3", "--steps", "10", "--generations", "4",
            "--strategy", "lamarckian", "--lr", LR)
    same = ((a[3] / "fitness_history.csv").read_text() == (b[3] / "fitness_history.csv").read_text()
            and a[1]["heldout_acc"] == b[1]["heldout_acc"])
    check("same args twice -> identical history and result", same)


def main() -> int:
    test_split()
    with tempfile.TemporaryDirectory() as tmp:
        test_fixed(tmp)
        test_partial(tmp)
        test_halving(tmp)
        test_champion(tmp)
        test_penalty(tmp)
        test_budget_and_determinism(tmp)
    print(f"\n{'all checks passed' if not check.failed else f'{check.failed} FAILED'}")
    return 1 if check.failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

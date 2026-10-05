"""Checks that the harness's bookkeeping failures are loud, using fake_main.py.

    python experiments/tests/test_harness.py [--runs-root DIR]

Runs in seconds. Prints PASS/FAIL per check; exits 1 if any check fails.
DIR must not already hold these test grids (default: a fresh temp dir).
"""

import argparse
import csv
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
sys.path.insert(0, str(EXP))
from summarize import select_by_val  # noqa: E402

FAKE = HERE / "fake_main.py"
failures = []


def check(name, cond, detail=""):
    print(f"{'PASS' if cond else 'FAIL'}  {name}" + (f"  -- {detail}" if detail and not cond else ""))
    if not cond:
        failures.append(name)


def run_grid(spec, root, *extra, env=None):
    path = Path(root) / f"{spec['name']}.grid.json"
    path.write_text(json.dumps(spec))
    p = subprocess.run([sys.executable, str(EXP / "run_grid.py"), str(path), "--runs-root",
                        str(root), "--main", str(FAKE), "--workers", "4", *extra],
                       capture_output=True, text=True, env={**os.environ, **(env or {})})
    return p.returncode, p.stdout + p.stderr


def results(root, name):
    with (Path(root) / name / "results.csv").open() as fh:
        return list(csv.DictReader(fh))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-root", default=None)
    root = Path(ap.parse_args().runs_root or tempfile.mkdtemp(prefix="harness_test_"))
    root.mkdir(parents=True, exist_ok=True)
    print(f"runs root: {root}\n")

    # --- 1. every outcome, and selection by validation only ----------------
    mixed = {"name": "t_mixed", "datasets": ["xor"], "seeds": [0, 1, 2], "timeout": 3,
             "arms": [
                 {"name": "val_best", "args": {"--val": 0.90, "--heldout": 0.60}},
                 {"name": "heldout_best", "args": {"--val": 0.80, "--heldout": 0.95}},
                 {"name": "extra", "args": {"--val": 0.85, "--extra": 42}},
                 {"name": "over", "args": {"--mode": "over-budget", "--val": 0.99}},
                 {"name": "boom", "args": {"--mode": "traceback", "--val": 0.99}},
                 {"name": "hang", "args": {"--mode": "hang", "--val": 0.99}},
                 {"name": "nochamp", "args": {"--mode": "no-champion"}},
                 {"name": "nan", "args": {"--mode": "nan", "--val": 0.5}},
                 {"name": "badflag", "args": {"--no-such-flag": 1}}]}
    rc, out = run_grid(mixed, root)
    print(out)
    check("mixed grid exits 1 (not all ok)", rc == 1, f"rc={rc}")
    check("final count is complete",
          "FINAL COUNT: ok 9 | invalid-budget 3 | crashed 12 | timeout 3 | missing 0 | "
          "args-mismatch 0" in out)
    rows = results(root, "t_mixed")
    check("results.csv has a row for every cell, failed ones included", len(rows) == 27,
          f"{len(rows)} rows")
    by_arm = {}
    for r in rows:
        by_arm.setdefault(r["arm"], set()).add(r["outcome"])
    check("exit 1 + traceback is crashed, not invalid-budget", by_arm["boom"] == {"crashed"})
    check("exit 4 + within_budget=false is invalid-budget", by_arm["over"] == {"invalid-budget"})
    check("hung run is a timeout", by_arm["hang"] == {"timeout"})
    check("exit 0 without a champion is crashed", by_arm["nochamp"] == {"crashed"})
    check("NaN val_acc is crashed, not silently ranked", by_arm["nan"] == {"crashed"})
    check("unknown flag (exit 2) is crashed", by_arm["badflag"] == {"crashed"})
    check("extra numeric field passes through to results.csv",
          all(r.get("my_extra_metric") == "42.0" for r in rows if r["arm"] == "extra"))
    md = (root / "t_mixed" / "summary.md").read_text()
    check("selected arm is the val-best, not the held-out-best (and not the over-budget arm)",
          "| xor | val_best | 0.910 |" in md and "selected arm for xor (by val mean only): val_best" in out)
    check("summary.md names the not-counted cells", "t_mixed/hang/xor_s0 | timeout" in md)
    check("my_extra_metric aggregated in summary.md", "my_extra_metric" in md)

    # --- 2. resume ----------------------------------------------------------
    before = (root / "t_mixed/val_best/xor_s0/summary.json").stat().st_mtime_ns
    rc, out = run_grid(mixed, root)
    check("resume runs nothing new; failed cells not retried without the flag",
          "to run: 0 new + 0 rerun | already done: 9 ok, 3 invalid-budget | "
          "failed, not rerun: 15" in out, out[:400])
    check("resume did not touch completed cells",
          (root / "t_mixed/val_best/xor_s0/summary.json").stat().st_mtime_ns == before)

    # --- 3. changed args refused --------------------------------------------
    changed = json.loads(json.dumps(mixed))
    changed["arms"][0]["args"]["--val"] = 0.91
    rc, out = run_grid(changed, root)
    check("changed args for existing cells are refused (exit 3)",
          rc == 3 and "REFUSED: 3 cell(s)" in out and "--val: requested 0.91, recorded 0.9" in out,
          out[-600:])
    check("refusal ran nothing",
          (root / "t_mixed/val_best/xor_s0/summary.json").stat().st_mtime_ns == before)

    # --- 4. --rerun-failed archives the old attempt ---------------------------
    rc, out = run_grid(mixed, root, "--rerun-failed")
    check("--rerun-failed reruns only crashed/timeout cells",
          "to run: 0 new + 15 rerun" in out, out[:300])
    check("old attempt archived in _attempts/1",
          (root / "t_mixed/boom/xor_s0/_attempts/1/stdout.log").exists())

    # --- 5. no val_acc -> no selection ----------------------------------------
    noval = {"name": "t_noval", "datasets": ["xor", "spiral"], "seeds": [0, 1],
             "arms": [{"name": "a", "args": {"--heldout": 0.6}},
                      {"name": "b", "args": {"--heldout": 0.9}}]}
    rc, out = run_grid(noval, root)
    check("without val_acc the selection says so (does not fall back to held-out)",
          rc == 0 and "selected arm for xor (by val mean only): —  [n/a — no val_acc]" in out,
          out[-500:])

    # --- 6. determinism check catches nondeterminism ----------------------------
    det = {"name": "t_det", "datasets": ["xor"], "seeds": [0, 1],
           "arms": [{"name": "steady", "args": {"--val": 0.8}},
                    {"name": "noisy", "args": {"--mode": "random", "--val": 0.8}}]}
    rc, out = run_grid(det, root, "--verify-determinism", "4")
    check("determinism: nondeterministic cells flagged, exit 5",
          rc == 5 and "MISMATCH  t_det/noisy/xor_s0" in out and "identical t_det/steady/xor_s0" in out,
          out[-800:])

    # --- 7. a main.py default or the code changing mid-grid is flagged ----------
    drift = {"name": "t_drift", "datasets": ["xor"], "seeds": [0], "arms": [{"name": "a", "args": {}}]}
    run_grid(drift, root)
    probe = HERE / "_fingerprint_probe.py"
    probe.write_text("# temporary: makes the code fingerprint change\n")
    try:
        drift["seeds"] = [0, 1]
        rc, out = run_grid(drift, root, env={"FAKE_DEFAULT_STEPS": "20"})
    finally:
        probe.unlink()
    check("a default that changed between run and resume is flagged",
          "arm 'a': cells recorded 2 different argument sets (differing: steps)" in out, out[-800:])
    check("cells from different code versions are flagged, naming the file",
          "different versions of the code" in out and "_fingerprint_probe.py" in out, out[-800:])

    # --- 8. a second run_grid on the same grid is refused ------------------------
    lock = root / "t_noval" / ".lock"
    lock.write_text(json.dumps({"pid": os.getpid(), "hostname": os.uname().nodename}))
    try:
        rc, out = run_grid(noval, root)
    finally:
        lock.unlink()
    check("grid lock held by a live process -> refused", rc == 3 and "another run_grid" in out,
          out[-300:])

    # --- 9. select_by_val in isolation ------------------------------------------
    check("select_by_val: highest mean wins", select_by_val({"a": [0.8, 0.95], "b": [0.86]})[0] == "a")
    check("select_by_val: ties are not broken silently",
          select_by_val({"a": [0.5], "b": [0.5]})[0] is None)
    check("select_by_val: an arm without val blocks selection",
          select_by_val({"a": [0.9], "b": None})[0] is None)

    print(f"\n{'ALL PASSED' if not failures else f'{len(failures)} FAILED: {failures}'}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

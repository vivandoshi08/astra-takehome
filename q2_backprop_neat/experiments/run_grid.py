"""Run a grid of main.py cells in parallel, resumably, and summarize it.

    python experiments/run_grid.py experiments/grids/demo.json
    python experiments/run_grid.py GRID.json --workers 4 --runs-root /tmp/runs
    python experiments/run_grid.py GRID.json --dry-run           # plan only
    python experiments/run_grid.py GRID.json --rerun-failed      # retry crashed/timeout cells
    python experiments/run_grid.py GRID.json --verify-determinism 2

Each cell is <grid>/<arm>/<dataset>_s<seed> under --runs-root. A cell is
skipped only if its summary.json records exactly the arguments the grid asks
for; if it records different ones, the whole invocation is refused before
anything runs. Crashed and timed-out cells are not retried without
--rerun-failed (the old files are then moved to <cell>/_attempts/<n>/).

Exit status: 0 every cell ok; 1 some cell not ok; 3 refused (grid error,
args mismatch, grid locked); 5 determinism mismatch; 130 interrupted.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import signal
import socket
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from gridlib import (ATTEMPTS_DIR, DEFAULT_MAIN, DEFAULT_RUNS_ROOT,  # noqa: E402
                     DEFAULT_TIMEOUT, HARNESS_FILE, HISTORY_FILE, LOG_FILE,
                     RUN_FILES, SUMMARY_FILE, Cell, Grid, GridError, Status,
                     classify, code_fingerprint, compare_args, files_that_differ,
                     is_number, judge_run, load_grid, now_iso, pid_alive,
                     read_json, write_json_atomic)
import summarize  # noqa: E402

# One BLAS thread per run: N parallel runs x M BLAS threads would oversubscribe.
CHILD_ENV = {"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
             "MKL_NUM_THREADS": "1", "VECLIB_MAXIMUM_THREADS": "1"}


class GridLock:
    """<grid dir>/.lock: one run_grid per grid at a time, so no two processes
    ever write the same run directory."""

    def __init__(self, grid_dir: Path):
        self.path = grid_dir / ".lock"
        self.held = False

    def acquire(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        me = {"pid": os.getpid(), "hostname": socket.gethostname(), "since": now_iso()}
        for _ in range(3):
            try:
                fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            except FileExistsError:
                other, _ = read_json(self.path)
                if (other and other.get("hostname") == socket.gethostname()
                        and not pid_alive(other.get("pid"))):
                    print(f"note: removing stale lock {self.path} (pid {other.get('pid')} is gone)")
                    self.path.unlink(missing_ok=True)
                    continue
                raise GridError(f"{self.path} is held by {other}: another run_grid is "
                                "working on this grid. Wait for it, or remove the lock if "
                                "you are sure it is dead.")
            with os.fdopen(fd, "w") as fh:
                json.dump(me, fh)
            self.held = True
            return
        raise GridError(f"could not take {self.path}")

    def release(self) -> None:
        if self.held:
            other, _ = read_json(self.path)
            if other and other.get("pid") == os.getpid():
                self.path.unlink(missing_ok=True)
            self.held = False


class Runner:
    """Launches cells; tracks live processes so Ctrl-C can kill them all."""

    def __init__(self, python: str, main: Path, timeout: float, fp_root: Path, fp_skip: tuple):
        self.python, self.main, self.timeout = python, main, timeout
        self.fp_root, self.fp_skip = fp_root, fp_skip
        self.env = {**os.environ, **CHILD_ENV}
        self.lock = threading.Lock()
        self.in_flight: set[Path] = set()
        self.procs: dict[Path, subprocess.Popen] = {}
        self.stopping = threading.Event()

    def kill_all(self) -> None:
        self.stopping.set()
        with self.lock:
            for p in self.procs.values():
                try:
                    os.killpg(p.pid, signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    pass

    def run(self, cell: Cell, run_dir: Path, archive: bool) -> Status:
        run_dir = run_dir.resolve()
        with self.lock:
            if run_dir in self.in_flight:
                raise RuntimeError(f"BUG: {run_dir} is already being run")
            self.in_flight.add(run_dir)
        try:
            return self._run(cell, run_dir, archive)
        finally:
            with self.lock:
                self.in_flight.discard(run_dir)

    def _run(self, cell: Cell, run_dir: Path, archive: bool) -> Status:
        run_dir.mkdir(parents=True, exist_ok=True)
        attempt = archive_previous(run_dir) if archive else n_attempts(run_dir) + 1
        leftovers = [f for f in RUN_FILES if (run_dir / f).exists()]
        if leftovers:   # never launch onto existing outputs: they could be read as this run's
            raise RuntimeError(f"BUG: {run_dir} still holds {leftovers} before launch")

        argv = cell.argv(self.python, self.main, run_dir)
        fp_before, files_before = code_fingerprint(self.fp_root, self.fp_skip)
        record = {
            "cell": cell.cell_id, "outcome": "running", "reason": "",
            "argv": argv, "cwd": str(self.main.parent), "env_overrides": CHILD_ENV,
            "attempt": attempt, "timeout_s": self.timeout,
            "harness_pid": os.getpid(), "hostname": socket.gethostname(),
            "started_at": now_iso(), "code_hash_before": fp_before,
            "code_files": files_before,
        }
        write_json_atomic(run_dir / HARNESS_FILE, record)

        t0 = time.monotonic()
        timed_out = False
        with (run_dir / LOG_FILE).open("w") as log:
            proc = subprocess.Popen(argv, cwd=self.main.parent, env=self.env,
                                    stdin=subprocess.DEVNULL, stdout=log,
                                    stderr=subprocess.STDOUT, start_new_session=True)
            with self.lock:
                self.procs[run_dir] = proc
            record["child_pid"] = proc.pid
            write_json_atomic(run_dir / HARNESS_FILE, record)
            if self.stopping.is_set():
                os.killpg(proc.pid, signal.SIGKILL)
            try:
                exit_code = proc.wait(timeout=self.timeout)
            except subprocess.TimeoutExpired:
                timed_out = True
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                exit_code = proc.wait()
            finally:
                with self.lock:
                    self.procs.pop(run_dir, None)
        duration = time.monotonic() - t0

        log_text = (run_dir / LOG_FILE).read_text(errors="replace")
        summary, serr = read_json(run_dir / SUMMARY_FILE)
        outcome, reason = judge_run(exit_code, timed_out, log_text, summary, serr, self.timeout)
        if self.stopping.is_set() and outcome != "ok":
            outcome, reason = "crashed", f"interrupted (Ctrl-C); {reason}"
        if summary is not None and outcome in ("ok", "invalid-budget"):
            diffs = compare_args(cell, summary.get("args")) if run_dir == cell.dir.resolve() else []
            if diffs:   # main.py did not record what it was given
                outcome, reason = "args-mismatch", "; ".join(diffs)
        fp_after, files_after = code_fingerprint(self.fp_root, self.fp_skip)
        record.update({
            "outcome": outcome, "reason": reason, "exit_code": exit_code,
            "timed_out": timed_out, "duration_s": round(duration, 2),
            "finished_at": now_iso(), "code_hash_after": fp_after,
        })
        if fp_after != fp_before:
            record["code_changed_during_run"] = files_that_differ(files_before, files_after)
        write_json_atomic(run_dir / HARNESS_FILE, record)
        return Status(cell, outcome, reason, summary, record)


def n_attempts(run_dir: Path) -> int:
    a = run_dir / ATTEMPTS_DIR
    return sum(1 for p in a.iterdir() if p.is_dir()) if a.is_dir() else 0


def archive_previous(run_dir: Path) -> int:
    """Move a failed attempt's files to <run dir>/_attempts/<n>/; return the
    number of the attempt about to start."""
    entries = [p for p in run_dir.iterdir() if p.name != ATTEMPTS_DIR]
    k = n_attempts(run_dir)
    if entries:
        k += 1
        dest = run_dir / ATTEMPTS_DIR / str(k)
        dest.mkdir(parents=True)
        for p in entries:
            p.rename(dest / p.name)
    return k + 1


# ----------------------------------------------------------------------------
# Planning
# ----------------------------------------------------------------------------

def plan(cells: list[Cell], rerun_failed: bool):
    """Decide per cell: run, rerun (archive first), skip, or refuse."""
    actions, refusals = [], []
    for cell in cells:
        st = classify(cell)
        if st.outcome == "args-mismatch":
            refusals.append(st)
            act = "refuse"
        elif st.outcome == "missing" and st.reason.startswith("in progress"):
            refusals.append(st)
            act = "refuse"
        elif st.outcome == "missing":
            act = "run"
        elif st.outcome in ("ok", "invalid-budget"):
            act = "skip"
        else:   # crashed / timeout
            act = "rerun" if rerun_failed else "keep-failed"
        actions.append((cell, st, act))
    return actions, refusals


def print_plan(grid: Grid, actions, grid_dir: Path, rerun_failed: bool) -> None:
    n = len(actions)
    counts = {a: sum(1 for _, _, x in actions if x == a)
              for a in ("run", "rerun", "skip", "keep-failed", "refuse")}
    print(f"grid {grid.name!r}: {len(grid.arms)} arms x {len(grid.datasets)} datasets x "
          f"{len(grid.seeds)} seeds = {n} cells in {grid_dir}")
    done_ok = sum(1 for _, s, a in actions if a == "skip" and s.outcome == "ok")
    print(f"  to run: {counts['run']} new + {counts['rerun']} rerun | already done: "
          f"{done_ok} ok, {counts['skip'] - done_ok} invalid-budget | "
          f"failed, not rerun: {counts['keep-failed']} | refused: {counts['refuse']}")
    failed = [(c, s) for c, s, a in actions if a == "keep-failed"]
    if failed:
        print(f"  {len(failed)} cells failed earlier and will NOT be rerun "
              "(pass --rerun-failed to retry them):")
        for c, s in failed:
            print(f"    {s.outcome:<8} {c.cell_id}: {s.reason}")
    if rerun_failed:
        for c, s, a in actions:
            if a == "rerun":
                print(f"    rerun    {c.cell_id} (was {s.outcome}: {s.reason})")


def explain_refusals(refusals: list[Status]) -> None:
    mism = [s for s in refusals if s.outcome == "args-mismatch"]
    busy = [s for s in refusals if s.outcome != "args-mismatch"]
    if mism:
        print(f"\nREFUSED: {len(mism)} cell(s) already hold results produced with different "
              "arguments than this grid asks for:", file=sys.stderr)
        for s in mism:
            print(f"  {s.cell.cell_id}: {s.reason}", file=sys.stderr)
        print("A cell's results must come from one configuration. Give the changed "
              "configuration a new arm name or grid name; or, if the old results are "
              "wrong, move those run directories away yourself. Nothing was run.",
              file=sys.stderr)
    for s in busy:
        print(f"REFUSED: {s.cell.cell_id} is {s.reason}", file=sys.stderr)


# ----------------------------------------------------------------------------
# Determinism
# ----------------------------------------------------------------------------

IGNORED_FOR_DETERMINISM = {"seconds"}


def diff_summaries(a: dict, b: dict) -> list[str]:
    diffs = []
    for key in sorted(set(a) | set(b)):
        if key in IGNORED_FOR_DETERMINISM:
            continue
        va, vb = a.get(key, "<absent>"), b.get(key, "<absent>")
        if key == "args" and isinstance(va, dict) and isinstance(vb, dict):
            va = {k: v for k, v in va.items() if k != "run_id"}
            vb = {k: v for k, v in vb.items() if k != "run_id"}
        if va != vb:
            extra = f" (|diff| {abs(va - vb):.3g})" if is_number(va) and is_number(vb) else ""
            diffs.append(f"{key}: {va!r} vs {vb!r}{extra}")
    return diffs


def verify_determinism(statuses: list[Status], k: int, runner: Runner, workers: int,
                       grid_dir: Path, seed: int | None) -> int:
    ok = [s for s in statuses if s.outcome == "ok"]
    if seed is None:
        seed = random.SystemRandom().randrange(10**6)
    chosen = random.Random(seed).sample(ok, min(k, len(ok)))
    print(f"\n=== determinism: re-running {len(chosen)} of {len(ok)} ok cells "
          f"(choice seed {seed}) ===")
    if len(chosen) < k:
        print(f"WARNING: asked for {k} cells but only {len(ok)} are ok")
    base = grid_dir / "_determinism" / datetime.now().strftime("%Y%m%d-%H%M%S")
    jobs = {s.cell.cell_id: (s, base / s.cell.arm / s.cell.dir.name) for s in chosen}
    results, mismatches = [], 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(runner.run, s.cell, d, False): cid for cid, (s, d) in jobs.items()}
        for fut in as_completed(futs):
            cid = futs[fut]
            orig, rdir = jobs[cid]
            rerun = fut.result()
            entry = {"cell": cid, "rerun_dir": str(rdir), "rerun_outcome": rerun.outcome,
                     "code_hash_original": (orig.harness or {}).get("code_hash_before"),
                     "code_hash_rerun": rerun.harness.get("code_hash_before")}
            if rerun.outcome != "ok" or rerun.summary is None:
                entry["diffs"] = [f"rerun outcome {rerun.outcome}: {rerun.reason}"]
            else:
                entry["diffs"] = diff_summaries(orig.summary, rerun.summary)
                h0 = (orig.cell.dir / HISTORY_FILE).read_bytes()
                h1 = (rdir / HISTORY_FILE).read_bytes()
                entry["fitness_history_identical"] = h0 == h1
                if h0 != h1:
                    entry["diffs"].append(f"{HISTORY_FILE} differs")
            same_code = entry["code_hash_original"] == entry["code_hash_rerun"]
            entry["same_code"] = same_code
            mismatches += bool(entry["diffs"])
            tag = "MISMATCH" if entry["diffs"] else "identical"
            print(f"  {tag:<9} {cid}" + ("" if same_code else
                  "  (NOTE: code changed since the original run; a mismatch is not "
                  "attributable to nondeterminism alone)"))
            for d in entry["diffs"]:
                print(f"      {d}")
            results.append(entry)
    report = {"time": now_iso(), "choice_seed": seed, "k": k, "checked": len(results),
              "mismatches": mismatches, "ignored_fields": sorted(IGNORED_FOR_DETERMINISM),
              "cells": sorted(results, key=lambda e: e["cell"])}
    write_json_atomic(grid_dir / "determinism.json", report)
    verdict = ("all identical (every summary field except " +
               ", ".join(sorted(IGNORED_FOR_DETERMINISM)) + ", and fitness_history.csv)"
               if not mismatches else f"{mismatches} MISMATCH(ES)")
    print(f"determinism: {len(results)} cells re-run -> {verdict}; reruns kept in {base}")
    return mismatches


# ----------------------------------------------------------------------------

def _raise_interrupt(signum, frame):
    raise KeyboardInterrupt(f"signal {signum}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("grid", help="grid spec (JSON)")
    ap.add_argument("--workers", type=int, default=min(6, os.cpu_count() or 1),
                    help="parallel runs (default: min(6, cores))")
    ap.add_argument("--runs-root", default=str(DEFAULT_RUNS_ROOT),
                    help=f"where cells go (default {DEFAULT_RUNS_ROOT})")
    ap.add_argument("--dry-run", action="store_true", help="print the plan; run nothing")
    ap.add_argument("--rerun-failed", action="store_true",
                    help="rerun crashed/timed-out cells (old files kept in _attempts/)")
    ap.add_argument("--verify-determinism", type=int, default=0, metavar="K",
                    help="afterwards, re-run K random ok cells into a temp dir and diff")
    ap.add_argument("--determinism-seed", type=int, default=None,
                    help="seed for choosing the K cells (default: random, printed)")
    ap.add_argument("--timeout", type=float, default=None,
                    help=f"seconds per run (default: the grid's 'timeout', else {DEFAULT_TIMEOUT:g})")
    ap.add_argument("--main", default=str(DEFAULT_MAIN),
                    help="driver script (testing hook; default main.py)")
    args = ap.parse_args(argv)
    signal.signal(signal.SIGTERM, _raise_interrupt)

    try:
        grid = load_grid(args.grid)
    except GridError as e:
        print(f"grid error: {e}", file=sys.stderr)
        return 3
    if args.workers < 1:
        ap.error("--workers must be >= 1")
    main_path = Path(args.main).resolve()
    if not main_path.is_file():
        ap.error(f"no driver at {main_path}")
    runs_root = Path(args.runs_root).resolve()
    grid_dir = grid.grid_dir(runs_root)
    timeout = args.timeout or grid.timeout or DEFAULT_TIMEOUT
    if main_path != DEFAULT_MAIN.resolve():
        print(f"NOTE: driver is {main_path}, not main.py")
    for w in grid.warnings:
        print(f"WARNING: {w}")

    cells = grid.cells(runs_root)
    actions, refusals = plan(cells, args.rerun_failed)
    print_plan(grid, actions, grid_dir, args.rerun_failed)
    if refusals:
        explain_refusals(refusals)
        return 3
    todo = [(c, a == "rerun") for c, _, a in actions if a in ("run", "rerun")]
    if args.dry_run:
        if todo:
            print("first command:\n  " + " ".join(
                repr(x) if " " in x else x
                for x in todo[0][0].argv(sys.executable, main_path)))
        print(f"dry run: nothing executed (timeout {timeout:g} s, {args.workers} workers)")
        return 0

    lock = GridLock(grid_dir)
    try:
        lock.acquire()
    except GridError as e:
        print(f"REFUSED: {e}", file=sys.stderr)
        return 3
    runner = Runner(sys.executable, main_path, timeout, main_path.parent, (runs_root,))
    interrupted = False
    try:
        # Planning happened before the lock: re-plan under it so nothing
        # changed in between.
        actions2, refusals2 = plan(cells, args.rerun_failed)
        if refusals2 or [a for *_, a in actions2] != [a for *_, a in actions]:
            print("REFUSED: the grid's state changed between planning and locking; rerun.",
                  file=sys.stderr)
            return 3
        stored = grid_dir / "grid.json"
        old, _ = read_json(stored)
        if old is not None and old != grid.spec:
            write_json_atomic(grid_dir / "grid.prev.json", old)
            print("NOTE: the grid spec differs from the one stored in this grid directory; "
                  "stored copy updated (previous one in grid.prev.json)")
        write_json_atomic(stored, grid.spec)

        print(f"running {len(todo)} cells with {args.workers} workers, timeout {timeout:g} s")
        t0 = time.monotonic()
        width = len(str(len(todo)))
        # Not a `with` block: on Ctrl-C the children (in their own sessions, so
        # they do not see the SIGINT) must be killed *before* waiting on them.
        ex = ThreadPoolExecutor(max_workers=args.workers)
        try:
            futs = {ex.submit(runner.run, c, c.dir, rerun): c for c, rerun in todo}
            for i, fut in enumerate(as_completed(futs), 1):
                st = fut.result()
                s = st.summary or {}
                acc = (f"heldout {s['heldout_acc']:.3f}" if is_number(s.get("heldout_acc"))
                       else "")
                if is_number(s.get("val_acc")):
                    acc += f" val {s['val_acc']:.3f}"
                print(f"[{i:>{width}}/{len(todo)}] {st.outcome:<14} {st.cell.cell_id:<40} "
                      f"{st.harness.get('duration_s', 0):6.1f}s  "
                      + (acc if st.outcome == "ok" else st.reason[:160]), flush=True)
        except BaseException as e:
            interrupted = isinstance(e, KeyboardInterrupt)
            print(f"\n{'interrupted' if interrupted else 'harness error'}: killing running "
                  "cells ...", file=sys.stderr)
            runner.kill_all()
            if not interrupted:
                raise
        finally:
            ex.shutdown(wait=True, cancel_futures=True)
        print(f"finished in {time.monotonic() - t0:.1f} s")
        if interrupted:
            return 130

        report = summarize.summarize(grid, runs_root)
        status = 0 if report["all_ok"] else 1
        if args.verify_determinism > 0:
            statuses = [classify(c) for c in cells]
            if verify_determinism(statuses, args.verify_determinism, runner, args.workers,
                                  grid_dir, args.determinism_seed):
                status = 5
            summarize.summarize(grid, runs_root, echo=False)  # adds determinism.json to summary.md
        return status
    finally:
        lock.release()


if __name__ == "__main__":
    raise SystemExit(main())

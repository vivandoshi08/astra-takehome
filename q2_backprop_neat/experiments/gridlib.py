"""Shared core of the Q2 experiment harness: grid specs, cells, run-dir state.

Everything that decides whether a cell counts lives here, so run_grid.py and
summarize.py cannot disagree about it. The rules:

* A cell is one main.py run: (grid, arm, dataset, seed). Its run directory is
  <runs-root>/<grid>/<arm>/<dataset>_s<seed>/.
* A cell's results are accepted only if summary.json records exactly the
  arguments the grid asks for. Anything else is an args-mismatch and is
  never silently mixed in.
* Outcomes: ok, invalid-budget (ran, but over budget: does not count),
  crashed, timeout, missing (never run), args-mismatch. Only `ok` cells enter
  the statistics; every other cell is counted and listed.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import socket
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

EXP_DIR = Path(__file__).resolve().parent
Q2_DIR = EXP_DIR.parent
DEFAULT_MAIN = Q2_DIR / "main.py"
DEFAULT_RUNS_ROOT = Q2_DIR / "runs"
DEFAULT_TIMEOUT = 600.0

OUTCOMES = ("ok", "invalid-budget", "crashed", "timeout", "missing", "args-mismatch")
BUDGET_EXIT = 4  # main.py's exit code for an over-budget run (main.BUDGET_EXIT)

HARNESS_FILE = "harness.json"      # written by run_grid: what was launched, how it ended
SUMMARY_FILE = "summary.json"      # written by main.py
HISTORY_FILE = "fitness_history.csv"
LOG_FILE = "stdout.log"            # main.py's stdout + stderr
ATTEMPTS_DIR = "_attempts"         # earlier failed attempts, moved aside by --rerun-failed
RUN_FILES = (SUMMARY_FILE, HISTORY_FILE, "champion.pkl", LOG_FILE, HARNESS_FILE)

# Flags the harness sets itself; a grid may not set them.
HARNESS_FLAGS = {"--dataset", "--seed", "--run-id", "--quiet", "--overwrite"}
# Recorded args that legitimately differ between cells of one arm.
PER_CELL_ARGS = {"dataset", "seed", "run_id", "quiet", "overwrite"}

NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
FLAG_RE = re.compile(r"^--[a-z0-9][a-z0-9-]*$")
GRID_KEYS = {"name", "datasets", "seeds", "arms", "common_args", "timeout", "description"}
ARM_KEYS = {"name", "args", "description"}


class GridError(ValueError):
    """The grid spec, or the state on disk, does not allow a safe run."""


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def is_number(v) -> bool:
    """A finite int/float that is not a bool (bool is an int subclass)."""
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def fmt_value(v) -> str:
    """How a grid value is written on the command line."""
    if isinstance(v, bool):
        raise TypeError("bools are flags, not values")
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        return repr(v)          # shortest string that round-trips exactly
    return str(v)


def flag_dest(flag: str) -> str:
    """argparse's dest for a long flag: --refine-fraction -> refine_fraction."""
    return flag[2:].replace("-", "_")


# ----------------------------------------------------------------------------
# Grid spec
# ----------------------------------------------------------------------------

@dataclass(frozen=True)
class Cell:
    grid: str
    arm: str
    dataset: str
    seed: int
    requested: tuple          # ((flag, value), ...): exactly what this cell asks main.py for
    dir: Path

    @property
    def cell_id(self) -> str:
        return f"{self.grid}/{self.arm}/{self.dataset}_s{self.seed}"

    def argv(self, python: str, main: Path, run_dir: Path | None = None) -> list[str]:
        out = [python, str(main), "--quiet", "--run-id", str(run_dir or self.dir)]
        for flag, value in self.requested:
            if value is True:
                out.append(flag)
            elif value is False:
                continue
            else:
                out += [flag, fmt_value(value)]
        return out


@dataclass
class Grid:
    name: str
    datasets: list
    seeds: list
    arms: list                # [(arm name, {flag: value})], common_args already merged in
    timeout: float | None
    spec: dict                # the validated JSON, as loaded
    path: Path | None = None
    warnings: list = field(default_factory=list)

    def grid_dir(self, runs_root: Path) -> Path:
        return Path(runs_root).resolve() / self.name

    def cells(self, runs_root: Path) -> list[Cell]:
        gdir = self.grid_dir(runs_root)
        out = []
        for arm, args in self.arms:
            for ds in self.datasets:
                for seed in self.seeds:
                    req = (("--dataset", ds), ("--seed", seed)) + tuple(args.items())
                    out.append(Cell(self.name, arm, ds, seed, req,
                                    gdir / arm / f"{ds}_s{seed}"))
        ids = [c.cell_id for c in out]
        assert len(ids) == len(set(ids)), "duplicate cell ids (validation should prevent this)"
        return out


def _check_args(where: str, args) -> dict:
    if not isinstance(args, dict):
        raise GridError(f"{where}: must be an object of {{\"--flag\": value}}")
    dests = {}
    for flag, value in args.items():
        if not FLAG_RE.match(flag):
            raise GridError(f"{where}: {flag!r} is not a long flag like \"--steps\"")
        if flag in HARNESS_FLAGS:
            raise GridError(f"{where}: {flag} is set by the harness, not the grid")
        dest = flag_dest(flag)
        if dest in dests:
            raise GridError(f"{where}: {flag} and {dests[dest]} are the same argument")
        dests[dest] = flag
        if isinstance(value, float) and not math.isfinite(value):
            raise GridError(f"{where}: {flag} = {value} is not finite")
        if not isinstance(value, (bool, int, float, str)):
            raise GridError(f"{where}: {flag} = {value!r}: values must be a number, "
                            "a string, or true/false for a bare flag")
    return dict(args)


def parse_grid(spec: dict, path: Path | None = None) -> Grid:
    if not isinstance(spec, dict):
        raise GridError("grid must be a JSON object")
    unknown = set(spec) - GRID_KEYS
    if unknown:
        raise GridError(f"unknown grid keys {sorted(unknown)} (allowed: {sorted(GRID_KEYS)})")
    for key in ("name", "datasets", "seeds", "arms"):
        if key not in spec:
            raise GridError(f"grid is missing {key!r}")
    name = spec["name"]
    if not isinstance(name, str) or not NAME_RE.match(name):
        raise GridError(f"grid name {name!r}: use letters, digits, '_', '-', '.'")

    datasets = spec["datasets"]
    if (not isinstance(datasets, list) or not datasets
            or not all(isinstance(d, str) and NAME_RE.match(d) for d in datasets)):
        raise GridError("datasets must be a non-empty list of names")
    if len(set(datasets)) != len(datasets):
        raise GridError(f"datasets has duplicates: {datasets}")

    seeds = spec["seeds"]
    if (not isinstance(seeds, list) or not seeds
            or not all(isinstance(s, int) and not isinstance(s, bool) for s in seeds)):
        raise GridError("seeds must be a non-empty list of integers")
    if len(set(seeds)) != len(seeds):
        raise GridError(f"seeds has duplicates: {seeds}")

    common = _check_args("common_args", spec.get("common_args", {}))

    timeout = spec.get("timeout")
    if timeout is not None and (not is_number(timeout) or timeout <= 0):
        raise GridError(f"timeout must be a positive number of seconds, got {timeout!r}")

    arms_spec = spec["arms"]
    if not isinstance(arms_spec, list) or not arms_spec:
        raise GridError("arms must be a non-empty list")
    arms, warnings, seen = [], [], {}
    for i, arm in enumerate(arms_spec):
        if not isinstance(arm, dict):
            raise GridError(f"arms[{i}] must be an object")
        bad = set(arm) - ARM_KEYS
        if bad:
            raise GridError(f"arms[{i}]: unknown keys {sorted(bad)} (allowed: {sorted(ARM_KEYS)})")
        aname = arm.get("name")
        if not isinstance(aname, str) or not NAME_RE.match(aname):
            # names starting with "_" are reserved for harness directories
            raise GridError(f"arms[{i}]: name {aname!r}: use letters, digits, '_', '-', '.'"
                            " and do not start with '_'")
        if aname in seen:
            raise GridError(f"two arms are named {aname!r}")
        own = _check_args(f"arm {aname!r} args", arm.get("args", {}))
        clash = sorted(set(own) & set(common))
        if clash:
            raise GridError(f"arm {aname!r} sets {clash}, which common_args also sets; "
                            "set it in one place (per arm if it varies)")
        merged = {**common, **own}
        merged_dests = {flag_dest(f): f for f in merged}
        if len(merged_dests) != len(merged):
            raise GridError(f"arm {aname!r}: common_args and args spell the same flag twice")
        key = tuple(sorted((k, repr(v)) for k, v in merged.items()))
        if key in seen.values():
            other = next(n for n, k in seen.items() if k == key)
            warnings.append(f"arms {other!r} and {aname!r} have identical arguments "
                            "(copy-paste slip?)")
        seen[aname] = key
        arms.append((aname, merged))

    return Grid(name, list(datasets), list(seeds), arms, timeout, spec,
                Path(path) if path else None, warnings)


def load_grid(path) -> Grid:
    path = Path(path)
    try:
        spec = json.loads(path.read_text())
    except (OSError, ValueError) as e:
        raise GridError(f"cannot read grid {path}: {e}") from e
    return parse_grid(spec, path)


# ----------------------------------------------------------------------------
# Comparing what was asked for with what main.py recorded
# ----------------------------------------------------------------------------

def same_value(want, got) -> bool:
    """Does main.py's recorded value equal the value the grid passed?

    The grid value went through the command line as a string and argparse's
    type conversion, so convert it the same way before comparing.
    """
    if isinstance(want, bool):
        return got is want or (want is False and got is None)
    if isinstance(got, bool) or got is None:
        return False
    if isinstance(got, (int, float)):
        try:
            return type(got)(fmt_value(want)) == got
        except (TypeError, ValueError):
            return False
    if isinstance(got, str):
        return fmt_value(want) == got
    return False


def compare_args(cell: Cell, recorded) -> list[str]:
    """Differences between what `cell` asks for and summary.json's `args`."""
    if not isinstance(recorded, dict):
        return ["summary.json has no 'args' object"]
    diffs = []
    for flag, want in cell.requested:
        dest = flag_dest(flag)
        if dest not in recorded:
            diffs.append(f"{flag}: requested {want!r}, but main.py recorded no {dest!r}")
        elif not same_value(want, recorded[dest]):
            diffs.append(f"{flag}: requested {want!r}, recorded {recorded[dest]!r}")
    rid = recorded.get("run_id")
    if rid is None or Path(str(rid)).parts[-3:] != Path(cell.cell_id).parts:
        diffs.append(f"--run-id: recorded {rid!r}, expected one ending in {cell.cell_id}")
    return diffs


# ----------------------------------------------------------------------------
# Deciding a cell's outcome
# ----------------------------------------------------------------------------

def read_json(path: Path):
    """(object, None) | (None, None) if absent | (None, error) if unreadable."""
    try:
        return json.loads(Path(path).read_text()), None
    except FileNotFoundError:
        return None, None
    except (OSError, ValueError) as e:
        return None, f"{Path(path).name} unreadable: {e}"


def write_json_atomic(path: Path, obj) -> None:
    path = Path(path)
    tmp = path.with_name(f".{path.name}.tmp{os.getpid()}")
    tmp.write_text(json.dumps(obj, indent=2, default=str) + "\n")
    os.replace(tmp, path)


def pid_alive(pid) -> bool:
    if not isinstance(pid, int) or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def judge_summary(summary: dict) -> tuple[str, str]:
    """ok / invalid-budget / crashed, from summary.json alone."""
    wb = summary.get("within_budget")
    if wb is False:
        return "invalid-budget", str(summary.get("budget_error") or summary.get("budget") or "")
    if wb is not True:
        return "crashed", f"summary.json: within_budget is {wb!r}, not true/false"
    spent, limit = summary.get("budget_spent"), summary.get("budget_limit")
    if is_number(spent) and is_number(limit) and spent > limit:
        return "invalid-budget", (f"budget_spent {spent} > budget_limit {limit} although "
                                  "within_budget is true (main.py inconsistency)")
    # val_acc may be absent or null (no validation split); if given it must be sane.
    keys = ["heldout_acc"] + (["val_acc"] if summary.get("val_acc") is not None else [])
    for key in keys:
        v = summary.get(key)
        if not is_number(v) or not 0.0 <= v <= 1.0:
            return "crashed", f"summary.json: {key} = {v!r} is not a finite accuracy in [0, 1]"
    return "ok", ""


def judge_run(exit_code: int, timed_out: bool, log_text: str, summary, summary_err,
              timeout: float) -> tuple[str, str]:
    """Outcome of a run that just finished. main.py exits BUDGET_EXIT (4) for
    an over-budget run; that also needs a summary.json saying
    within_budget=false and no traceback. Exit 1 is Python's uncaught
    exception, so it is always a crash."""
    if timed_out:
        return "timeout", f"killed after {timeout:g} s"
    lines = [ln for ln in log_text.splitlines() if ln.strip()]
    tail = lines[-1].strip()[:300] if lines else "(no output)"
    if summary_err:
        return "crashed", f"exit {exit_code}; {summary_err}"
    if exit_code == 2:
        return "crashed", f"exit 2 (usage error): {tail}"
    if exit_code not in (0, BUDGET_EXIT):
        return "crashed", f"exit {exit_code}: {tail}"
    if "Traceback (most recent call last)" in log_text:
        return "crashed", f"exit {exit_code} with a traceback: {tail}"
    if summary is None:
        return "crashed", f"exit {exit_code} but no summary.json: {tail}"
    outcome, reason = judge_summary(summary)
    if exit_code == BUDGET_EXIT and outcome != "invalid-budget":
        return "crashed", (f"exit {BUDGET_EXIT} but summary.json reads as "
                           f"{outcome!r}: {tail}")
    if exit_code == 0 and outcome == "invalid-budget":
        reason += " (main.py exited 0 for an over-budget run)"
    return outcome, reason


@dataclass
class Status:
    cell: Cell
    outcome: str
    reason: str = ""
    summary: dict | None = None
    harness: dict | None = None


def classify(cell: Cell) -> Status:
    """The cell's outcome as recorded on disk."""
    d = cell.dir
    harness, herr = read_json(d / HARNESS_FILE)
    summary, serr = read_json(d / SUMMARY_FILE)
    if not (d.is_dir() and any((d / f).exists() for f in RUN_FILES)):
        return Status(cell, "missing", "not run")
    if herr or serr:
        return Status(cell, "crashed", herr or serr, summary, harness)
    if summary is not None:
        diffs = compare_args(cell, summary.get("args"))
        if diffs:
            return Status(cell, "args-mismatch", "; ".join(diffs), summary, harness)
    hout = harness.get("outcome") if harness else None
    if hout == "running":
        pid, child = harness.get("harness_pid"), harness.get("child_pid")
        here = harness.get("hostname") == socket.gethostname()
        if here and pid_alive(pid):
            return Status(cell, "missing", f"in progress (run_grid pid {pid})", summary, harness)
        if here and pid_alive(child):
            return Status(cell, "missing", f"in progress: run_grid died but its main.py "
                          f"(pid {child}) is still running", summary, harness)
        return Status(cell, "crashed", f"incomplete: run_grid (pid {pid}) stopped while this "
                      "cell was running", summary, harness)
    if hout in ("crashed", "timeout", "args-mismatch"):
        return Status(cell, hout, harness.get("reason", ""), summary, harness)
    if summary is None:
        return Status(cell, "crashed", "incomplete: files present but no summary.json"
                      + ("" if harness else " and no harness.json"), None, harness)
    outcome, reason = judge_summary(summary)
    if harness is not None and hout != outcome:
        return Status(cell, "crashed", f"harness.json says {hout!r} but summary.json now reads "
                      f"as {outcome!r} (edited after the run?)", summary, harness)
    if harness is None:
        reason = (reason + "; " if reason else "") + "no harness.json (not run by run_grid)"
    return Status(cell, outcome, reason, summary, harness)


# ----------------------------------------------------------------------------
# Which code ran
# ----------------------------------------------------------------------------

CODE_SUFFIXES = {".py", ".ini", ".cfg", ".toml", ".yaml", ".yml"}
SKIP_DIRS = {"runs", "experiments", "__pycache__"}


def code_fingerprint(root: Path, skip: tuple = ()) -> tuple[str, dict]:
    """sha256 over the code files under `root` (the directory main.py lives in).

    git_commit's "-dirty" says only that *something* is uncommitted; two runs
    marked "abc123-dirty" may have run different code. This hash tells them
    apart, and the per-file hashes say which file differed.
    """
    root = Path(root).resolve()
    skip = {Path(s).resolve() for s in skip}
    files = {}
    for dirpath, dirnames, filenames in os.walk(root):
        dp = Path(dirpath)
        dirnames[:] = sorted(d for d in dirnames
                             if d not in SKIP_DIRS and not d.startswith(".")
                             and (dp / d).resolve() not in skip)
        for fn in sorted(filenames):
            p = dp / fn
            if p.suffix in CODE_SUFFIXES:
                try:
                    files[str(p.relative_to(root))] = hashlib.sha256(p.read_bytes()).hexdigest()[:12]
                except OSError:
                    files[str(p.relative_to(root))] = "unreadable"
    total = hashlib.sha256("".join(f"{k}:{v}\n" for k, v in sorted(files.items())).encode())
    return total.hexdigest()[:12], files


def files_that_differ(a: dict, b: dict) -> list[str]:
    return sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))

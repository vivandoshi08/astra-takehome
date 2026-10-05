"""Summarize a grid: <grid dir>/results.csv (one row per cell) and summary.md.

    python experiments/summarize.py runs/demo                 # a grid directory
    python experiments/summarize.py experiments/grids/demo.json [--runs-root DIR]

Every cell the grid defines gets a row in results.csv, including cells that
crashed, timed out, were never run, or ran with other arguments; only `ok`
cells enter the statistics, and summary.md lists the rest by name.

Held-out accuracy is reported, never used to choose. The "selected arm" for a
dataset is the arm with the highest mean validation accuracy (`val_acc`);
`select_by_val` is only ever given validation numbers. Without `val_acc` the
selection is reported as not possible.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from gridlib import (DEFAULT_RUNS_ROOT, OUTCOMES, PER_CELL_ARGS,  # noqa: E402
                     RUN_FILES, Grid, GridError, Status, classify,
                     files_that_differ, is_number, load_grid, now_iso, read_json)

QUESTION_BUDGET = 30_000
MAIN_FIELDS = ("heldout_acc", "val_acc", "train_acc", "conns", "budget_spent")
HARNESS_COLS = ("exit_code", "duration_s", "attempt", "code_hash_before", "started_at")
NO_VAL = "n/a — no val_acc"


def stats(values):
    vals = [v for v in values if is_number(v)]
    n = len(vals)
    return {
        "n": n,
        "mean": statistics.fmean(vals) if n else None,
        "sd": statistics.stdev(vals) if n >= 2 else None,   # sample sd (n-1), not SE
        "min": min(vals) if n else None,
        "max": max(vals) if n else None,
    }


def select_by_val(val_by_arm: dict) -> tuple[str | None, str]:
    """Pick the arm with the highest mean validation accuracy.

    `val_by_arm` maps arm -> list of val_acc values (or None if that arm has
    no val_acc). Held-out numbers are never passed in, so they cannot leak
    into the choice.
    """
    if not val_by_arm:
        return None, "n/a — no complete arm"
    if all(v is None for v in val_by_arm.values()):
        return None, NO_VAL
    lacking = sorted(a for a, v in val_by_arm.items() if v is None)
    if lacking:
        return None, f"n/a — no val_acc for arm(s) {', '.join(lacking)}"
    means = {a: statistics.fmean(v) for a, v in val_by_arm.items()}
    best = max(means.values())
    top = sorted(a for a, m in means.items() if m == best)
    if len(top) > 1:
        return None, f"n/a — tie on val mean {best:.4f}: {', '.join(top)}"
    return top[0], ""


# ----------------------------------------------------------------------------

def flatten(st: Status) -> dict:
    c = st.cell
    row = {"grid": c.grid, "arm": c.arm, "dataset": c.dataset, "seed": c.seed,
           "cell": c.cell_id, "outcome": st.outcome, "reason": st.reason}
    for k, v in (st.summary or {}).items():
        if k == "args":
            continue
        row[k] = json.dumps(v) if isinstance(v, (dict, list)) else v
    for k, v in ((st.summary or {}).get("args") or {}).items():
        row[f"args.{k}"] = json.dumps(v) if isinstance(v, (dict, list)) else v
    for k in HARNESS_COLS:
        if st.harness and k in st.harness:
            row[f"harness.{k}"] = st.harness[k]
    return row


def unexpected_dirs(grid: Grid, grid_dir: Path) -> list[str]:
    """Run directories on disk that this grid does not define (an arm renamed
    or removed, a seed dropped): never aggregated, always reported."""
    out = []
    if not grid_dir.is_dir():
        return out
    arms = {a for a, _ in grid.arms}
    expected = {f"{d}_s{s}" for d in grid.datasets for s in grid.seeds}
    for arm_dir in sorted(grid_dir.iterdir()):
        if not arm_dir.is_dir() or arm_dir.name.startswith(("_", ".")):
            continue
        for cell_dir in sorted(p for p in arm_dir.iterdir() if p.is_dir()):
            if arm_dir.name in arms and cell_dir.name in expected:
                continue
            if any((cell_dir / f).exists() for f in RUN_FILES):
                out.append(f"{grid.name}/{arm_dir.name}/{cell_dir.name}")
    return out


def find_warnings(grid: Grid, statuses: list[Status], unexpected: list[str]) -> list[str]:
    w = list(grid.warnings)
    ran = [s for s in statuses if s.summary is not None]

    commits = Counter(str(s.summary.get("git_commit")) for s in ran)
    if len(commits) > 1:
        w.append(f"cells come from {len(commits)} different git commits: "
                 + ", ".join(f"{c} ({n} cells)" for c, n in commits.most_common()))
    dirty = sum(n for c, n in commits.items() if c.endswith("-dirty"))
    if dirty:
        w.append(f"{dirty}/{len(ran)} cells ran from uncommitted code (git_commit ends "
                 "in -dirty), so the commit does not identify the code they ran")

    hashed = [s for s in statuses if s.harness and s.harness.get("code_hash_before")]
    by_hash = defaultdict(list)
    for s in hashed:
        by_hash[s.harness["code_hash_before"]].append(s)
    if len(by_hash) > 1:
        groups = sorted(by_hash.values(), key=len, reverse=True)
        ref = groups[0][0].harness.get("code_files", {})
        parts = []
        for g in groups[1:]:
            diff = files_that_differ(ref, g[0].harness.get("code_files", {}))
            parts.append(f"{len(g)} cells (e.g. {g[0].cell.cell_id}) differ from the "
                         f"{len(groups[0])}-cell majority in: {', '.join(diff) or '?'}")
        w.append(f"cells ran {len(by_hash)} different versions of the code: " + "; ".join(parts))
    for s in hashed:
        changed = s.harness.get("code_changed_during_run")
        if changed:
            w.append(f"{s.cell.cell_id}: code changed while it was running ({', '.join(changed)})")
    no_h = [s for s in ran if not s.harness]
    if no_h:
        w.append(f"{len(no_h)} cells have summary.json but no harness.json (not run by "
                 "run_grid; code version unverified), e.g. " + no_h[0].cell.cell_id)

    # Within an arm, everything but the per-cell args must be identical: this
    # catches a main.py default that changed between a run and its resume.
    by_arm = defaultdict(list)
    for s in ran:
        by_arm[s.cell.arm].append(s)
    signatures = {}
    for arm, sts in by_arm.items():
        sigs = defaultdict(list)
        for s in sts:
            a = {k: v for k, v in (s.summary.get("args") or {}).items() if k not in PER_CELL_ARGS}
            sigs[json.dumps(a, sort_keys=True, default=str)].append(s.cell.cell_id)
        if len(sigs) > 1:
            dicts = [json.loads(k) for k in sigs]
            keys = sorted(k for k in set().union(*dicts) if len({json.dumps(d.get(k)) for d in dicts}) > 1)
            w.append(f"arm {arm!r}: cells recorded {len(sigs)} different argument sets "
                     f"(differing: {', '.join(keys)}); its cells are not one configuration")
        else:
            signatures.setdefault(next(iter(sigs)), []).append(arm)
    for arms in signatures.values():
        if len(arms) > 1:
            w.append(f"arms {', '.join(sorted(arms))} recorded identical full arguments: "
                     "they are the same configuration")

    # Refinement asked for but not paid for (or the reverse) means the
    # allocation the args describe is not what ran.
    unpaid = [s for s in ran if is_number((s.summary.get("args") or {}).get("steps"))
              and s.summary["args"]["steps"] > 0
              and (s.summary["args"].get("refine_fraction") or 0) > 0
              and s.summary.get("budget_spent") == 0]
    if unpaid:
        w.append(f"{len(unpaid)} cells asked for refinement (--steps > 0) but spent 0 gradient "
                 "steps: refinement is not wired in or never ran (e.g. "
                 f"{unpaid[0].cell.cell_id})")
    free = [s for s in ran if (s.summary.get("args") or {}).get("steps") == 0
            and is_number(s.summary.get("budget_spent")) and s.summary["budget_spent"] > 0]
    if free:
        w.append(f"{len(free)} cells have --steps 0 but spent gradient steps (e.g. "
                 f"{free[0].cell.cell_id})")

    limits = Counter(s.summary.get("budget_limit") for s in ran)
    off = {k: n for k, n in limits.items() if k != QUESTION_BUDGET}
    if off:
        w.append("cells ran with budget_limit other than the question's 30,000: "
                 + ", ".join(f"{k} ({n} cells)" for k, n in off.items()))
    if unexpected:
        w.append(f"{len(unexpected)} run directories in the grid dir are not cells of this "
                 f"grid and are ignored: {', '.join(unexpected[:6])}"
                 + (" ..." if len(unexpected) > 6 else ""))
    return w


def aggregate(grid: Grid, statuses: list[Status]):
    by = defaultdict(list)
    for s in statuses:
        by[(s.cell.dataset, s.cell.arm)].append(s)
    rows, selection = [], {}
    for ds in grid.datasets:
        val_by_arm, excluded, ds_rows = {}, [], []
        for arm, _ in grid.arms:
            sts = by[(ds, arm)]
            ok = [s for s in sts if s.outcome == "ok"]
            cnt = Counter(s.outcome for s in sts)
            r = {"dataset": ds, "arm": arm, "expected": len(sts), "n": len(ok),
                 "n_invalid": cnt["invalid-budget"],
                 "n_failed": len(sts) - len(ok) - cnt["invalid-budget"]}
            for f in MAIN_FIELDS:
                r[f] = stats([s.summary.get(f) for s in ok])
            has_val = [is_number(s.summary.get("val_acc")) for s in ok]
            r["val_state"] = ("all" if ok and all(has_val) else
                              "none" if not any(has_val) else "partial")
            ds_rows.append(r)
            if len(ok) != len(sts):
                excluded.append(f"{arm} ({len(ok)}/{len(sts)} ok)")
            elif r["val_state"] == "all":
                val_by_arm[arm] = [s.summary["val_acc"] for s in ok]
            else:
                val_by_arm[arm] = None
        if excluded and all(r["val_state"] == "none" for r in ds_rows):
            chosen, note = None, NO_VAL
        else:
            chosen, note = select_by_val(val_by_arm)
        if excluded:
            note = (note + "; " if note else "") + "not eligible (not all cells ok): " + ", ".join(excluded)
        selection[ds] = (chosen, note)
        for r in ds_rows:
            r["selected"] = r["arm"] == chosen
        rows += ds_rows
    return rows, selection


def other_numeric_fields(statuses: list[Status]) -> list[str]:
    seen = []
    for s in statuses:
        if s.outcome != "ok":
            continue
        for k, v in s.summary.items():
            if k not in MAIN_FIELDS and k not in seen and is_number(v):
                seen.append(k)
    return seen


# ----------------------------------------------------------------------------

def _f(x, fmt=".3f"):
    return "—" if x is None else format(x, fmt)


def _g(x):
    """Compact number for the extra-fields table."""
    if x is None:
        return "—"
    if abs(x) >= 1000 and float(x).is_integer():
        return f"{x:,.0f}"
    return f"{x:.4g}"


def _pm(st):
    if st["n"] == 0:
        return "—"
    return f"{st['mean']:.3f} ± {_f(st['sd'])}"


def render_md(grid: Grid, grid_dir: Path, statuses, rows, selection, warnings,
              extra_fields, determinism) -> str:
    counts = Counter(s.outcome for s in statuses)
    L = [f"# Grid `{grid.name}`", ""]
    if grid.spec.get("description"):
        L += [str(grid.spec["description"]), ""]
    L += [f"Generated {now_iso()} by `experiments/summarize.py` from `{grid_dir}`.",
          f"{len(statuses)} cells = {len(grid.arms)} arms × {len(grid.datasets)} datasets "
          f"× {len(grid.seeds)} seeds (seeds {grid.seeds}).", "",
          "**Outcomes:** " + " · ".join(f"{o} {counts.get(o, 0)}" for o in OUTCOMES), ""]
    if warnings:
        L += ["## Warnings", ""] + [f"- **{x}**" for x in warnings] + [""]
    bad = [s for s in statuses if s.outcome != "ok"]
    if bad:
        L += ["## Cells not counted in the statistics", "",
              "| cell | outcome | reason |", "|---|---|---|"]
        L += [f"| {s.cell.cell_id} | {s.outcome} | {s.reason.replace('|', '/')[:200]} |"
              for s in bad]
        L += [""]

    L += ["## Results by dataset and arm", "",
          "Held-out accuracy is the reported number and is never used to choose. "
          "Statistics are over `ok` cells only (n of expected); sd is the sample "
          "standard deviation across seeds (n−1), not a standard error. "
          "`fail` counts crashed, timed-out, missing and args-mismatch cells.", "",
          "| dataset | arm | n | invalid | fail | held-out mean ± sd | min | max "
          "| val mean ± sd | train mean | conns mean | budget spent mean | selected |",
          "|---|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|:-:|"]
    for r in rows:
        h = r["heldout_acc"]
        val = (_pm(r["val_acc"]) if r["val_state"] == "all" else
               "no val_acc" if r["val_state"] == "none" else "partial!")
        L.append(f"| {r['dataset']} | {r['arm']} | {r['n']}/{r['expected']} | {r['n_invalid']} "
                 f"| {r['n_failed']} | {_pm(h)} | {_f(h['min'])} | {_f(h['max'])} | {val} "
                 f"| {_f(r['train_acc']['mean'])} | {_f(r['conns']['mean'], '.1f')} "
                 f"| {_f(r['budget_spent']['mean'], ',.0f')} | {'**yes**' if r['selected'] else ''} |")
    L += ["", "## Selected arm per dataset", "",
          "Chosen by the highest mean validation accuracy (`val_acc`) among arms whose "
          "cells all completed `ok`. Held-out accuracy is shown for reporting only; the "
          "selection code is never given it.", "",
          "| dataset | selected arm | val mean | held-out mean ± sd (reported, not used) | note |",
          "|---|---|--:|--:|---|"]
    for ds in grid.datasets:
        chosen, note = selection[ds]
        r = next((r for r in rows if r["dataset"] == ds and r["arm"] == chosen), None)
        L.append(f"| {ds} | {chosen or '—'} | {_f(r['val_acc']['mean']) if r else '—'} "
                 f"| {_pm(r['heldout_acc']) if r else '—'} | {note} |")
    if extra_fields:
        by = defaultdict(list)
        for s in statuses:
            if s.outcome == "ok":
                by[(s.cell.dataset, s.cell.arm)].append(s.summary)
        L += ["", "## Other numeric summary fields (mean over ok cells)", "",
              "| dataset | arm | " + " | ".join(extra_fields) + " |",
              "|---|---|" + "--:|" * len(extra_fields)]
        for r in rows:
            sums = by[(r["dataset"], r["arm"])]
            cells = [_g(stats([x.get(f) for x in sums])["mean"]) for f in extra_fields]
            L.append(f"| {r['dataset']} | {r['arm']} | " + " | ".join(cells) + " |")
    if determinism:
        L += ["", "## Determinism check", "",
              f"{determinism.get('time')}: re-ran {determinism.get('checked')} ok cells "
              f"(choice seed {determinism.get('choice_seed')}); "
              f"**{determinism.get('mismatches')} mismatch(es)** in summary fields "
              f"(ignoring {', '.join(determinism.get('ignored_fields', []))}) and fitness_history.csv.", ""]
        for e in determinism.get("cells", []):
            L.append(f"- {e['cell']}: " + ("identical" if not e["diffs"] else
                                          "MISMATCH: " + "; ".join(e["diffs"])[:300]))
    L += ["", "Per-cell data: `results.csv`. Outcomes: ok = counted; invalid-budget = "
          "exceeded the gradient-step budget, does not count; crashed / timeout = no "
          "usable result; missing = never run; args-mismatch = the run on disk used "
          "other arguments than the grid asks for.", ""]
    return "\n".join(L)


def summarize(grid: Grid, runs_root, write: bool = True, echo: bool = True) -> dict:
    grid_dir = grid.grid_dir(runs_root)
    statuses = [classify(c) for c in grid.cells(runs_root)]
    unexpected = unexpected_dirs(grid, grid_dir)
    warnings = find_warnings(grid, statuses, unexpected)
    rows, selection = aggregate(grid, statuses)
    extra = other_numeric_fields(statuses)
    determinism, _ = read_json(grid_dir / "determinism.json")

    flat = [flatten(s) for s in statuses]
    cols = []
    for r in flat:
        cols += [k for k in r if k not in cols]
    md = render_md(grid, grid_dir, statuses, rows, selection, warnings, extra, determinism)
    if write:
        grid_dir.mkdir(parents=True, exist_ok=True)
        with (grid_dir / "results.csv").open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=cols)
            w.writeheader()
            w.writerows(flat)
        (grid_dir / "summary.md").write_text(md)

    counts = Counter(s.outcome for s in statuses)
    if echo:
        print(f"\n=== summary of grid {grid.name!r} ({len(statuses)} cells) ===")
        print("FINAL COUNT: " + " | ".join(f"{o} {counts.get(o, 0)}" for o in OUTCOMES))
        for s in statuses:
            if s.outcome != "ok":
                print(f"  NOT COUNTED  {s.outcome:<14} {s.cell.cell_id}: {s.reason[:200]}")
        for x in warnings:
            print(f"WARNING: {x}")
        for ds in grid.datasets:
            chosen, note = selection[ds]
            print(f"selected arm for {ds} (by val mean only): {chosen or '—'}"
                  + (f"  [{note}]" if note else ""))
        if write:
            print(f"wrote {grid_dir / 'results.csv'} ({len(flat)} rows) and "
                  f"{grid_dir / 'summary.md'}")
    return {"statuses": statuses, "rows": rows, "selection": selection,
            "warnings": warnings, "counts": dict(counts),
            "all_ok": counts.get("ok", 0) == len(statuses)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("target", help="a grid directory, or a grid spec JSON")
    ap.add_argument("--runs-root", default=None,
                    help=f"with a grid spec: where its cells are (default {DEFAULT_RUNS_ROOT})")
    ap.add_argument("--print-md", action="store_true", help="also print summary.md")
    args = ap.parse_args(argv)
    target = Path(args.target).resolve()
    try:
        if target.is_dir():
            if args.runs_root:
                ap.error("--runs-root goes with a grid spec, not a grid directory")
            grid = load_grid(target / "grid.json")
            if grid.name != target.name:
                raise GridError(f"{target}/grid.json names grid {grid.name!r}, "
                                f"but the directory is {target.name!r}")
            runs_root = target.parent
        else:
            grid = load_grid(target)
            runs_root = Path(args.runs_root or DEFAULT_RUNS_ROOT)
    except GridError as e:
        print(f"error: {e}", file=sys.stderr)
        return 3
    report = summarize(grid, runs_root)
    if args.print_md:
        print((grid.grid_dir(runs_root) / "summary.md").read_text())
    return 0 if report["all_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

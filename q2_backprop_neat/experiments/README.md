# Experiment harness

Runs grids of `main.py` cells in parallel, resumably, and summarizes them so
that bookkeeping mistakes are loud: no cell is counted twice, dropped
silently, mixed across configurations, or chosen on held-out accuracy.

## Define a grid

```json
{
  "name": "alloc",
  "datasets": ["xor", "spiral", "spiral3"],
  "seeds": [0, 1, 2],
  "common_args": {"--refine-fraction": 1.0},
  "arms": [
    {"name": "s10", "args": {"--steps": 10}},
    {"name": "s40", "args": {"--steps": 40}}
  ],
  "timeout": 600
}
```

A cell is one `(arm, dataset, seed)`; its run directory is
`<runs-root>/<name>/<arm>/<dataset>_s<seed>/`. Values are numbers or strings;
`true` passes a bare flag. The harness sets `--dataset --seed --run-id --quiet`
itself. A flag may be in `common_args` or an arm, not both. Unknown keys,
duplicate arms/seeds/datasets and identical arms are rejected or warned about.

## Run, resume, rerun

```bash
python experiments/run_grid.py experiments/grids/demo.json             # run (6 workers)
python experiments/run_grid.py GRID.json --dry-run                     # show the plan only
python experiments/run_grid.py GRID.json                               # again = resume
python experiments/run_grid.py GRID.json --rerun-failed                # retry crashed/timeout
python experiments/run_grid.py GRID.json --verify-determinism 2        # re-run 2 cells, diff
python experiments/run_grid.py GRID.json --workers 4 --runs-root DIR --timeout 300
```

- **Resume** skips a cell only if its `summary.json` records exactly the
  arguments the grid asks for. If any cell records different ones, nothing
  runs and the differences are listed. Rename the arm (or grid) for a new
  configuration; never edit an arm in place.
- **Crashed / timed-out** cells are not retried unless you pass
  `--rerun-failed`. The old files then move to `<cell>/_attempts/<n>/`.
- One `run_grid` per grid at a time (`<grid dir>/.lock`). Ctrl-C kills the
  running cells and marks them crashed.
- `--verify-determinism K` re-runs K random ok cells into
  `<grid dir>/_determinism/<time>/` and diffs every `summary.json` field
  except `seconds`, plus `fitness_history.csv`. The result is
  `determinism.json`, and summary.md reports it.

Exit status: 0 all cells ok · 1 some cell not ok · 3 refused · 5 determinism mismatch · 130 interrupted.

## Summarize

Runs automatically at the end of `run_grid`, or on its own:

```bash
python experiments/summarize.py runs/alloc                  # a grid directory
python experiments/summarize.py GRID.json --runs-root DIR   # or a spec
```

It writes `<grid dir>/results.csv` (one row per expected cell, failed ones
included, every summary field flattened plus `args.*` and `harness.*`) and
`<grid dir>/summary.md`. The summary has outcome counts, warnings, the cells
not counted, and per (dataset, arm): n, invalid, failed, held-out mean ± sample
sd, min and max, then val mean ± sd, train, conns and budget-spent means. It
also shows the **selected arm** per dataset.

**Selection uses validation accuracy only.** The selected arm is the one
with the highest mean `val_acc` among arms whose cells are all ok. The
selection function is never given held-out numbers. With no `val_acc` it
reports `n/a — no val_acc`. A tie, or an arm that lacks `val_acc`, also
gives n/a.

## Outcomes

| outcome | meaning | counted? |
|---|---|---|
| `ok` | exit 0, `within_budget` true, finite `heldout_acc` (and `val_acc` if given) | yes |
| `invalid-budget` | ran but exceeded the gradient-step budget (exit 4 + `within_budget: false`) | no, listed |
| `crashed` | non-zero exit, a traceback (an uncaught exception exits 1), no/corrupt `summary.json`, a non-finite accuracy, or Ctrl-C | no, listed |
| `timeout` | killed after the per-run timeout | no, listed |
| `missing` | never run | no, listed |
| `args-mismatch` | the run on disk used other arguments than the grid asks for | no, listed; `run_grid` refuses |

The summary also **warns** about the following:

- cells from different `git_commit`s, or any `-dirty` commit;
- cells that ran different code. Each run's code fingerprint is in
  `harness.json`, and the warning names the files that differ;
- code that changed while a cell ran;
- an arm whose cells recorded different full arguments, as when a `main.py`
  default changes between a run and its resume;
- arms that are the same configuration;
- `--steps > 0` with 0 gradient steps spent;
- a `budget_limit` other than 30,000;
- run directories that are not cells of the grid.

## Files

- `run_grid.py`: scheduling, resume, locking, timeouts, the determinism check.
- `summarize.py`: `results.csv`, `summary.md`, selection.
- `gridlib.py`: the grid spec, cells, the args comparison and outcome rules
  (shared, so the two scripts cannot disagree).
- `grids/demo.json`: 2 arms × 3 datasets × 2 seeds of plain NEAT, as a smoke test.
- `tests/test_harness.py`: 27 checks of the failure paths against
  `tests/fake_main.py`, a stand-in driver that can crash, hang, overspend or
  be nondeterministic on request. Run it with
  `python experiments/tests/test_harness.py`.

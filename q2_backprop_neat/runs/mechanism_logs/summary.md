# Grid `mechanism_logs`

Generated 2026-10-04T20:16:20-07:00 by `experiments/summarize.py` from `/Users/vivandoshi/Documents/astra-takehome 3/q2_backprop_neat/runs/mechanism_logs`.
12 cells = 2 arms × 2 datasets × 3 seeds (seeds [0, 1, 2]).

**Outcomes:** ok 12 · invalid-budget 0 · crashed 0 · timeout 0 · missing 0 · args-mismatch 0

## Results by dataset and arm

Held-out accuracy is the reported number and is never used to choose. Statistics are over `ok` cells only (n of expected); sd is the sample standard deviation across seeds (n−1), not a standard error. `fail` counts crashed, timed-out, missing and args-mismatch cells.

| dataset | arm | n | invalid | fail | held-out mean ± sd | min | max | val mean ± sd | train mean | conns mean | budget spent mean | selected |
|---|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|:-:|
| spiral | lamarckian_log | 3/3 | 0 | 0 | 0.856 ± 0.096 | 0.800 | 0.967 | 0.876 ± 0.095 | 0.887 | 7.0 | 30,000 | **yes** |
| spiral | baldwinian_log | 3/3 | 0 | 0 | 0.791 ± 0.017 | 0.773 | 0.807 | 0.790 ± 0.036 | 0.786 | 4.0 | 30,000 |  |
| spiral3 | lamarckian_log | 3/3 | 0 | 0 | 0.793 ± 0.029 | 0.773 | 0.827 | 0.843 ± 0.029 | 0.796 | 9.0 | 30,000 | **yes** |
| spiral3 | baldwinian_log | 3/3 | 0 | 0 | 0.733 ± 0.020 | 0.713 | 0.753 | 0.743 ± 0.029 | 0.746 | 9.7 | 30,000 |  |

## Selected arm per dataset

Chosen by the highest mean validation accuracy (`val_acc`) among arms whose cells all completed `ok`. Held-out accuracy is shown for reporting only; the selection code is never given it.

| dataset | selected arm | val mean | held-out mean ± sd (reported, not used) | note |
|---|---|--:|--:|---|
| spiral | lamarckian_log | 0.876 | 0.856 ± 0.096 |  |
| spiral3 | lamarckian_log | 0.843 | 0.793 ± 0.029 |  |

## Other numeric summary fields (mean over ok cells)

| dataset | arm | seconds | generations_planned | generations_run | steps_per_generation | budget_limit | budget_refinements | refine_calls | genomes_refined | n_fit | n_val | heldout_majority_class | champion_generation | champion_fitness | fit_acc | nodes | max_abs_param | trainbest_heldout_acc | trainbest_conns |
|---|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| spiral | lamarckian_log | 3.653 | 30 | 30 | 1,000 | 30,000 | 3,000 | 3,000 | 3,000 | 280 | 70 | 0.52 | 22 | 0.8893 | 0.8893 | 3.333 | 11.95 | 0.8556 | 7 |
| spiral | baldwinian_log | 2.467 | 30 | 30 | 1,000 | 30,000 | 3,000 | 3,000 | 3,000 | 280 | 70 | 0.52 | 21.33 | 0.7845 | 0.7845 | 2.333 | 3.861 | 0.7711 | 4.333 |
| spiral3 | lamarckian_log | 3.273 | 30 | 30 | 1,000 | 30,000 | 3,000 | 3,000 | 3,000 | 280 | 70 | 0.5667 | 28.33 | 0.7845 | 0.7845 | 5 | 10.56 | 0.7733 | 8.333 |
| spiral3 | baldwinian_log | 2.603 | 30 | 30 | 1,000 | 30,000 | 3,000 | 3,000 | 3,000 | 280 | 70 | 0.5667 | 27.67 | 0.7464 | 0.7464 | 5 | 3.649 | 0.7244 | 9.333 |

Per-cell data: `results.csv`. Outcomes: ok = counted; invalid-budget = exceeded the gradient-step budget, does not count; crashed / timeout = no usable result; missing = never run; args-mismatch = the run on disk used other arguments than the grid asks for.

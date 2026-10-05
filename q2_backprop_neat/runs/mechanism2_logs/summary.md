# Grid `mechanism2_logs`

Generated 2026-10-04T20:21:26-07:00 by `experiments/summarize.py` from `/Users/vivandoshi/Documents/astra-takehome 3/q2_backprop_neat/runs/mechanism2_logs`.
12 cells = 2 arms × 2 datasets × 3 seeds (seeds [0, 1, 2]).

**Outcomes:** ok 12 · invalid-budget 0 · crashed 0 · timeout 0 · missing 0 · args-mismatch 0

## Results by dataset and arm

Held-out accuracy is the reported number and is never used to choose. Statistics are over `ok` cells only (n of expected); sd is the sample standard deviation across seeds (n−1), not a standard error. `fail` counts crashed, timed-out, missing and args-mismatch cells.

| dataset | arm | n | invalid | fail | held-out mean ± sd | min | max | val mean ± sd | train mean | conns mean | budget spent mean | selected |
|---|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|:-:|
| spiral | lamarckian_lowmut_log | 3/3 | 0 | 0 | 0.864 ± 0.115 | 0.740 | 0.967 | 0.919 ± 0.070 | 0.913 | 10.0 | 30,000 | **yes** |
| spiral | baldwinian_lowmut_log | 3/3 | 0 | 0 | 0.820 ± 0.050 | 0.773 | 0.873 | 0.800 ± 0.065 | 0.822 | 6.0 | 30,000 |  |
| spiral3 | lamarckian_lowmut_log | 3/3 | 0 | 0 | 0.813 ± 0.052 | 0.753 | 0.847 | 0.805 ± 0.030 | 0.803 | 10.7 | 30,000 | **yes** |
| spiral3 | baldwinian_lowmut_log | 3/3 | 0 | 0 | 0.607 ± 0.007 | 0.600 | 0.613 | 0.729 ± 0.000 | 0.685 | 2.0 | 30,000 |  |

## Selected arm per dataset

Chosen by the highest mean validation accuracy (`val_acc`) among arms whose cells all completed `ok`. Held-out accuracy is shown for reporting only; the selection code is never given it.

| dataset | selected arm | val mean | held-out mean ± sd (reported, not used) | note |
|---|---|--:|--:|---|
| spiral | lamarckian_lowmut_log | 0.919 | 0.864 ± 0.115 |  |
| spiral3 | lamarckian_lowmut_log | 0.805 | 0.813 ± 0.052 |  |

## Other numeric summary fields (mean over ok cells)

| dataset | arm | seconds | generations_planned | generations_run | steps_per_generation | budget_limit | budget_refinements | refine_calls | genomes_refined | n_fit | n_val | heldout_majority_class | champion_generation | champion_fitness | fit_acc | nodes | max_abs_param | trainbest_heldout_acc | trainbest_conns |
|---|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| spiral | lamarckian_lowmut_log | 3.88 | 30 | 30 | 1,000 | 30,000 | 3,000 | 3,000 | 3,000 | 280 | 70 | 0.52 | 23.33 | 0.9119 | 0.9119 | 5 | 14.09 | 0.8956 | 10 |
| spiral | baldwinian_lowmut_log | 2.193 | 30 | 30 | 1,000 | 30,000 | 3,000 | 3,000 | 3,000 | 280 | 70 | 0.52 | 27 | 0.8274 | 0.8274 | 3.333 | 3.982 | 0.82 | 5.333 |
| spiral3 | lamarckian_lowmut_log | 3.277 | 30 | 30 | 1,000 | 30,000 | 3,000 | 3,000 | 3,000 | 280 | 70 | 0.5667 | 24.67 | 0.8024 | 0.8024 | 5.667 | 10.59 | 0.8133 | 10.67 |
| spiral3 | baldwinian_lowmut_log | 1.773 | 30 | 30 | 1,000 | 30,000 | 3,000 | 3,000 | 3,000 | 280 | 70 | 0.5667 | 6.667 | 0.6738 | 0.6738 | 1 | 1.165 | 0.6044 | 2 |

Per-cell data: `results.csv`. Outcomes: ok = counted; invalid-budget = exceeded the gradient-step budget, does not count; crashed / timeout = no usable result; missing = never run; args-mismatch = the run on disk used other arguments than the grid asks for.

# Grid `mechanism_final_logs`

Generated 2026-10-04T20:25:28-07:00 by `experiments/summarize.py` from `/Users/vivandoshi/Documents/astra-takehome 3/q2_backprop_neat/runs/mechanism_final_logs`.
12 cells = 2 arms × 2 datasets × 3 seeds (seeds [20, 21, 22]).

**Outcomes:** ok 12 · invalid-budget 0 · crashed 0 · timeout 0 · missing 0 · args-mismatch 0

## Results by dataset and arm

Held-out accuracy is the reported number and is never used to choose. Statistics are over `ok` cells only (n of expected); sd is the sample standard deviation across seeds (n−1), not a standard error. `fail` counts crashed, timed-out, missing and args-mismatch cells.

| dataset | arm | n | invalid | fail | held-out mean ± sd | min | max | val mean ± sd | train mean | conns mean | budget spent mean | selected |
|---|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|:-:|
| spiral | lam_default_log | 3/3 | 0 | 0 | 0.898 ± 0.025 | 0.880 | 0.927 | 0.910 ± 0.054 | 0.934 | 9.0 | 30,000 |  |
| spiral | lam_norep_log | 3/3 | 0 | 0 | 0.904 ± 0.031 | 0.887 | 0.940 | 0.919 ± 0.046 | 0.938 | 14.3 | 30,000 | **yes** |
| spiral3 | lam_default_log | 3/3 | 0 | 0 | 0.798 ± 0.034 | 0.767 | 0.833 | 0.829 ± 0.038 | 0.806 | 10.3 | 30,000 |  |
| spiral3 | lam_norep_log | 3/3 | 0 | 0 | 0.853 ± 0.058 | 0.787 | 0.893 | 0.895 ± 0.058 | 0.868 | 17.0 | 30,000 | **yes** |

## Selected arm per dataset

Chosen by the highest mean validation accuracy (`val_acc`) among arms whose cells all completed `ok`. Held-out accuracy is shown for reporting only; the selection code is never given it.

| dataset | selected arm | val mean | held-out mean ± sd (reported, not used) | note |
|---|---|--:|--:|---|
| spiral | lam_norep_log | 0.919 | 0.904 ± 0.031 |  |
| spiral3 | lam_norep_log | 0.895 | 0.853 ± 0.058 |  |

## Other numeric summary fields (mean over ok cells)

| dataset | arm | seconds | generations_planned | generations_run | steps_per_generation | budget_limit | budget_refinements | refine_calls | genomes_refined | n_fit | n_val | heldout_majority_class | champion_generation | champion_fitness | fit_acc | nodes | max_abs_param | trainbest_heldout_acc | trainbest_conns |
|---|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| spiral | lam_default_log | 3.917 | 30 | 30 | 1,000 | 30,000 | 3,000 | 3,000 | 3,000 | 280 | 70 | 0.52 | 23.33 | 0.9405 | 0.9405 | 4.667 | 13.86 | 0.8978 | 9 |
| spiral | lam_norep_log | 3.813 | 30 | 30 | 1,000 | 30,000 | 3,000 | 3,000 | 3,000 | 280 | 70 | 0.52 | 25 | 0.9429 | 0.9429 | 8.333 | 12.61 | 0.9044 | 13.67 |
| spiral3 | lam_default_log | 3.603 | 30 | 30 | 1,000 | 30,000 | 3,000 | 3,000 | 3,000 | 280 | 70 | 0.5667 | 27.67 | 0.8 | 0.8 | 5.333 | 11.63 | 0.7978 | 10.33 |
| spiral3 | lam_norep_log | 4.033 | 30 | 30 | 1,000 | 30,000 | 3,000 | 3,000 | 3,000 | 280 | 70 | 0.5667 | 25.67 | 0.8607 | 0.8607 | 9.333 | 14.27 | 0.8622 | 17.33 |

Per-cell data: `results.csv`. Outcomes: ok = counted; invalid-budget = exceeded the gradient-step budget, does not count; crashed / timeout = no usable result; missing = never run; args-mismatch = the run on disk used other arguments than the grid asks for.

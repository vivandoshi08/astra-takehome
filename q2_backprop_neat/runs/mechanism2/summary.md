# Grid `mechanism2`

Generated 2026-10-04T20:21:19-07:00 by `experiments/summarize.py` from `/Users/vivandoshi/Documents/astra-takehome 3/q2_backprop_neat/runs/mechanism2`.
60 cells = 2 arms × 3 datasets × 10 seeds (seeds [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]).

**Outcomes:** ok 60 · invalid-budget 0 · crashed 0 · timeout 0 · missing 0 · args-mismatch 0

## Results by dataset and arm

Held-out accuracy is the reported number and is never used to choose. Statistics are over `ok` cells only (n of expected); sd is the sample standard deviation across seeds (n−1), not a standard error. `fail` counts crashed, timed-out, missing and args-mismatch cells.

| dataset | arm | n | invalid | fail | held-out mean ± sd | min | max | val mean ± sd | train mean | conns mean | budget spent mean | selected |
|---|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|:-:|
| xor | lam_lowmut | 10/10 | 0 | 0 | 0.967 ± 0.007 | 0.960 | 0.987 | 0.996 ± 0.007 | 0.993 | 10.8 | 30,000 | **yes** |
| xor | bald_lowmut | 10/10 | 0 | 0 | 0.920 ± 0.109 | 0.707 | 0.993 | 0.924 ± 0.116 | 0.923 | 8.5 | 30,000 |  |
| spiral | lam_lowmut | 10/10 | 0 | 0 | 0.908 ± 0.077 | 0.740 | 0.987 | 0.931 ± 0.059 | 0.937 | 11.6 | 30,000 | **yes** |
| spiral | bald_lowmut | 10/10 | 0 | 0 | 0.821 ± 0.048 | 0.753 | 0.887 | 0.809 ± 0.046 | 0.828 | 7.4 | 30,000 |  |
| spiral3 | lam_lowmut | 10/10 | 0 | 0 | 0.819 ± 0.038 | 0.753 | 0.873 | 0.823 ± 0.047 | 0.814 | 12.0 | 30,000 | **yes** |
| spiral3 | bald_lowmut | 10/10 | 0 | 0 | 0.645 ± 0.050 | 0.600 | 0.760 | 0.741 ± 0.026 | 0.703 | 3.6 | 30,000 |  |

## Selected arm per dataset

Chosen by the highest mean validation accuracy (`val_acc`) among arms whose cells all completed `ok`. Held-out accuracy is shown for reporting only; the selection code is never given it.

| dataset | selected arm | val mean | held-out mean ± sd (reported, not used) | note |
|---|---|--:|--:|---|
| xor | lam_lowmut | 0.996 | 0.967 ± 0.007 |  |
| spiral | lam_lowmut | 0.931 | 0.908 ± 0.077 |  |
| spiral3 | lam_lowmut | 0.823 | 0.819 ± 0.038 |  |

## Other numeric summary fields (mean over ok cells)

| dataset | arm | seconds | generations_planned | generations_run | steps_per_generation | budget_limit | budget_refinements | refine_calls | genomes_refined | n_fit | n_val | heldout_majority_class | champion_generation | champion_fitness | fit_acc | nodes | max_abs_param | trainbest_heldout_acc | trainbest_conns |
|---|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| xor | lam_lowmut | 3.683 | 30 | 30 | 1,000 | 30,000 | 3,000 | 3,000 | 3,000 | 280 | 70 | 0.5133 | 20.8 | 0.9918 | 0.9918 | 5.2 | 22.17 | 0.968 | 11.3 |
| xor | bald_lowmut | 2.447 | 30 | 30 | 1,000 | 30,000 | 3,000 | 3,000 | 3,000 | 280 | 70 | 0.5133 | 20.2 | 0.9225 | 0.9225 | 4 | 3.334 | 0.9233 | 8.2 |
| spiral | lam_lowmut | 3.675 | 30 | 30 | 1,000 | 30,000 | 3,000 | 3,000 | 3,000 | 280 | 70 | 0.52 | 24.8 | 0.9379 | 0.9379 | 6.1 | 14.23 | 0.916 | 11.6 |
| spiral | bald_lowmut | 2.3 | 30 | 30 | 1,000 | 30,000 | 3,000 | 3,000 | 3,000 | 280 | 70 | 0.52 | 25 | 0.8325 | 0.8325 | 4.3 | 3.847 | 0.8187 | 7.3 |
| spiral3 | lam_lowmut | 3.581 | 30 | 30 | 1,000 | 30,000 | 3,000 | 3,000 | 3,000 | 280 | 70 | 0.5667 | 26.1 | 0.8121 | 0.8121 | 6.8 | 11.51 | 0.818 | 11.9 |
| spiral3 | bald_lowmut | 2.019 | 30 | 30 | 1,000 | 30,000 | 3,000 | 3,000 | 3,000 | 280 | 70 | 0.5667 | 13.6 | 0.6929 | 0.6929 | 1.9 | 2.295 | 0.6473 | 3.8 |

Per-cell data: `results.csv`. Outcomes: ok = counted; invalid-budget = exceeded the gradient-step budget, does not count; crashed / timeout = no usable result; missing = never run; args-mismatch = the run on disk used other arguments than the grid asks for.

# Grid `mechanism_final`

Generated 2026-10-04T20:25:19-07:00 by `experiments/summarize.py` from `/Users/vivandoshi/Documents/astra-takehome 3/q2_backprop_neat/runs/mechanism_final`.
120 cells = 4 arms × 3 datasets × 10 seeds (seeds [20, 21, 22, 23, 24, 25, 26, 27, 28, 29]).

**Outcomes:** ok 120 · invalid-budget 0 · crashed 0 · timeout 0 · missing 0 · args-mismatch 0

## Results by dataset and arm

Held-out accuracy is the reported number and is never used to choose. Statistics are over `ok` cells only (n of expected); sd is the sample standard deviation across seeds (n−1), not a standard error. `fail` counts crashed, timed-out, missing and args-mismatch cells.

| dataset | arm | n | invalid | fail | held-out mean ± sd | min | max | val mean ± sd | train mean | conns mean | budget spent mean | selected |
|---|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|:-:|
| xor | lam_default | 10/10 | 0 | 0 | 0.968 ± 0.008 | 0.953 | 0.980 | 0.996 ± 0.007 | 0.993 | 11.7 | 30,000 |  |
| xor | bald_default | 10/10 | 0 | 0 | 0.956 ± 0.036 | 0.860 | 0.987 | 0.976 ± 0.042 | 0.966 | 10.3 | 30,000 |  |
| xor | lam_norep | 10/10 | 0 | 0 | 0.969 ± 0.009 | 0.947 | 0.980 | 0.996 ± 0.007 | 0.993 | 12.7 | 30,000 |  |
| xor | bald_norep | 10/10 | 0 | 0 | 0.955 ± 0.083 | 0.720 | 0.993 | 0.960 ± 0.086 | 0.955 | 11.0 | 30,000 |  |
| spiral | lam_default | 10/10 | 0 | 0 | 0.879 ± 0.047 | 0.793 | 0.927 | 0.904 ± 0.053 | 0.918 | 10.4 | 30,000 |  |
| spiral | bald_default | 10/10 | 0 | 0 | 0.813 ± 0.032 | 0.767 | 0.853 | 0.803 ± 0.027 | 0.826 | 7.0 | 30,000 |  |
| spiral | lam_norep | 10/10 | 0 | 0 | 0.889 ± 0.059 | 0.780 | 0.940 | 0.919 ± 0.059 | 0.927 | 13.2 | 30,000 | **yes** |
| spiral | bald_norep | 10/10 | 0 | 0 | 0.794 ± 0.025 | 0.753 | 0.847 | 0.794 ± 0.034 | 0.800 | 6.9 | 30,000 |  |
| spiral3 | lam_default | 10/10 | 0 | 0 | 0.807 ± 0.043 | 0.753 | 0.887 | 0.849 ± 0.050 | 0.825 | 12.6 | 30,000 |  |
| spiral3 | bald_default | 10/10 | 0 | 0 | 0.655 ± 0.050 | 0.600 | 0.740 | 0.731 ± 0.030 | 0.702 | 4.5 | 30,000 |  |
| spiral3 | lam_norep | 10/10 | 0 | 0 | 0.828 ± 0.060 | 0.727 | 0.913 | 0.866 ± 0.051 | 0.848 | 15.1 | 30,000 | **yes** |
| spiral3 | bald_norep | 10/10 | 0 | 0 | 0.701 ± 0.057 | 0.607 | 0.773 | 0.769 ± 0.030 | 0.740 | 7.0 | 30,000 |  |

## Selected arm per dataset

Chosen by the highest mean validation accuracy (`val_acc`) among arms whose cells all completed `ok`. Held-out accuracy is shown for reporting only; the selection code is never given it.

| dataset | selected arm | val mean | held-out mean ± sd (reported, not used) | note |
|---|---|--:|--:|---|
| xor | — | — | — | n/a — tie on val mean 0.9957: lam_default, lam_norep |
| spiral | lam_norep | 0.919 | 0.889 ± 0.059 |  |
| spiral3 | lam_norep | 0.866 | 0.828 ± 0.060 |  |

## Other numeric summary fields (mean over ok cells)

| dataset | arm | seconds | generations_planned | generations_run | steps_per_generation | budget_limit | budget_refinements | refine_calls | genomes_refined | n_fit | n_val | heldout_majority_class | champion_generation | champion_fitness | fit_acc | nodes | max_abs_param | trainbest_heldout_acc | trainbest_conns |
|---|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| xor | lam_default | 3.786 | 30 | 30 | 1,000 | 30,000 | 3,000 | 3,000 | 3,000 | 280 | 70 | 0.5133 | 22.8 | 0.9921 | 0.9921 | 5.7 | 23.09 | 0.9787 | 11.2 |
| xor | bald_default | 3.226 | 30 | 30 | 1,000 | 30,000 | 3,000 | 3,000 | 3,000 | 280 | 70 | 0.5133 | 23 | 0.9632 | 0.9632 | 4.9 | 4.714 | 0.962 | 10.4 |
| xor | lam_norep | 3.813 | 30 | 30 | 1,000 | 30,000 | 3,000 | 3,000 | 3,000 | 280 | 70 | 0.5133 | 19.7 | 0.9921 | 0.9921 | 5.9 | 23.08 | 0.974 | 11.9 |
| xor | bald_norep | 3.009 | 30 | 30 | 1,000 | 30,000 | 3,000 | 3,000 | 3,000 | 280 | 70 | 0.5133 | 23.1 | 0.9532 | 0.9532 | 5.7 | 5.147 | 0.9547 | 10.9 |
| spiral | lam_default | 3.52 | 30 | 30 | 1,000 | 30,000 | 3,000 | 3,000 | 3,000 | 280 | 70 | 0.52 | 25.6 | 0.9211 | 0.9211 | 5.6 | 14.39 | 0.8813 | 10.1 |
| spiral | bald_default | 2.629 | 30 | 30 | 1,000 | 30,000 | 3,000 | 3,000 | 3,000 | 280 | 70 | 0.52 | 22.1 | 0.8321 | 0.8321 | 3.8 | 4.452 | 0.8087 | 7.6 |
| spiral | lam_norep | 3.709 | 30 | 30 | 1,000 | 30,000 | 3,000 | 3,000 | 3,000 | 280 | 70 | 0.52 | 25.3 | 0.9296 | 0.9296 | 7.5 | 14.41 | 0.8873 | 12.5 |
| spiral | bald_norep | 2.457 | 30 | 30 | 1,000 | 30,000 | 3,000 | 3,000 | 3,000 | 280 | 70 | 0.52 | 24.1 | 0.8014 | 0.8014 | 4.1 | 4.743 | 0.798 | 6.8 |
| spiral3 | lam_default | 3.422 | 30 | 30 | 1,000 | 30,000 | 3,000 | 3,000 | 3,000 | 280 | 70 | 0.5667 | 26.7 | 0.8189 | 0.8189 | 6.8 | 13.4 | 0.81 | 11.3 |
| spiral3 | bald_default | 2.412 | 30 | 30 | 1,000 | 30,000 | 3,000 | 3,000 | 3,000 | 280 | 70 | 0.5667 | 18.4 | 0.6946 | 0.6946 | 2.8 | 2.785 | 0.6747 | 5.7 |
| spiral3 | lam_norep | 3.94 | 30 | 30 | 1,000 | 30,000 | 3,000 | 3,000 | 3,000 | 280 | 70 | 0.5667 | 26 | 0.8436 | 0.8436 | 8.1 | 15.53 | 0.834 | 16.1 |
| spiral3 | bald_norep | 2.93 | 30 | 30 | 1,000 | 30,000 | 3,000 | 3,000 | 3,000 | 280 | 70 | 0.5667 | 20.6 | 0.7325 | 0.7325 | 4 | 3.733 | 0.7073 | 8 |

Per-cell data: `results.csv`. Outcomes: ok = counted; invalid-budget = exceeded the gradient-step budget, does not count; crashed / timeout = no usable result; missing = never run; args-mismatch = the run on disk used other arguments than the grid asks for.

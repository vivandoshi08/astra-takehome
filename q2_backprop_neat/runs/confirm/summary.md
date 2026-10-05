# Grid `confirm`

Generated 2026-10-04T20:09:16-07:00 by `experiments/summarize.py` from `/Users/vivandoshi/Documents/astra-takehome 3/q2_backprop_neat/runs/confirm`.
36 cells = 4 arms × 3 datasets × 3 seeds (seeds [10, 11, 12]).

**Outcomes:** ok 36 · invalid-budget 0 · crashed 0 · timeout 0 · missing 0 · args-mismatch 0

## Results by dataset and arm

Held-out accuracy is the reported number and is never used to choose. Statistics are over `ok` cells only (n of expected); sd is the sample standard deviation across seeds (n−1), not a standard error. `fail` counts crashed, timed-out, missing and args-mismatch cells.

| dataset | arm | n | invalid | fail | held-out mean ± sd | min | max | val mean ± sd | train mean | conns mean | budget spent mean | selected |
|---|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|:-:|
| xor | A_lamarckian | 3/3 | 0 | 0 | 0.967 ± 0.007 | 0.960 | 0.973 | 1.000 ± 0.000 | 0.993 | 11.3 | 30,000 | **yes** |
| xor | A_baldwinian | 3/3 | 0 | 0 | 0.978 ± 0.004 | 0.973 | 0.980 | 0.981 ± 0.008 | 0.980 | 9.3 | 30,000 |  |
| xor | B_lamarckian | 3/3 | 0 | 0 | 0.978 ± 0.010 | 0.967 | 0.987 | 0.981 ± 0.008 | 0.988 | 9.0 | 30,000 |  |
| xor | B_baldwinian | 3/3 | 0 | 0 | 0.971 ± 0.010 | 0.960 | 0.980 | 0.990 ± 0.008 | 0.990 | 12.0 | 30,000 |  |
| spiral | A_lamarckian | 3/3 | 0 | 0 | 0.889 ± 0.004 | 0.887 | 0.893 | 0.886 ± 0.000 | 0.917 | 11.3 | 30,000 |  |
| spiral | A_baldwinian | 3/3 | 0 | 0 | 0.778 ± 0.014 | 0.767 | 0.793 | 0.790 ± 0.030 | 0.781 | 8.7 | 30,000 |  |
| spiral | B_lamarckian | 3/3 | 0 | 0 | 0.913 ± 0.040 | 0.873 | 0.953 | 0.929 ± 0.062 | 0.943 | 11.3 | 30,000 | **yes** |
| spiral | B_baldwinian | 3/3 | 0 | 0 | 0.833 ± 0.064 | 0.793 | 0.907 | 0.843 ± 0.062 | 0.866 | 7.3 | 30,000 |  |
| spiral3 | A_lamarckian | 3/3 | 0 | 0 | 0.787 ± 0.037 | 0.753 | 0.827 | 0.829 ± 0.038 | 0.804 | 12.0 | 30,000 |  |
| spiral3 | A_baldwinian | 3/3 | 0 | 0 | 0.647 ± 0.041 | 0.600 | 0.673 | 0.724 ± 0.036 | 0.694 | 6.3 | 30,000 |  |
| spiral3 | B_lamarckian | 3/3 | 0 | 0 | 0.787 ± 0.100 | 0.687 | 0.887 | 0.876 ± 0.066 | 0.829 | 9.7 | 30,000 | **yes** |
| spiral3 | B_baldwinian | 3/3 | 0 | 0 | 0.682 ± 0.020 | 0.660 | 0.700 | 0.757 ± 0.025 | 0.714 | 5.7 | 30,000 |  |

## Selected arm per dataset

Chosen by the highest mean validation accuracy (`val_acc`) among arms whose cells all completed `ok`. Held-out accuracy is shown for reporting only; the selection code is never given it.

| dataset | selected arm | val mean | held-out mean ± sd (reported, not used) | note |
|---|---|--:|--:|---|
| xor | A_lamarckian | 1.000 | 0.967 ± 0.007 |  |
| spiral | B_lamarckian | 0.929 | 0.913 ± 0.040 |  |
| spiral3 | B_lamarckian | 0.876 | 0.787 ± 0.100 |  |

## Other numeric summary fields (mean over ok cells)

| dataset | arm | seconds | generations_planned | generations_run | steps_per_generation | budget_limit | budget_refinements | refine_calls | genomes_refined | n_fit | n_val | heldout_majority_class | champion_generation | champion_fitness | fit_acc | nodes | max_abs_param | trainbest_heldout_acc | trainbest_conns |
|---|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| xor | A_lamarckian | 4.857 | 30 | 30 | 1,000 | 30,000 | 3,000 | 3,000 | 3,000 | 280 | 70 | 0.5133 | 21.33 | 0.9917 | 0.9917 | 6 | 27.43 | 0.9667 | 11.67 |
| xor | A_baldwinian | 4.27 | 30 | 30 | 1,000 | 30,000 | 3,000 | 3,000 | 3,000 | 280 | 70 | 0.5133 | 19.33 | 0.9798 | 0.9798 | 4.333 | 4.128 | 0.98 | 10 |
| xor | B_lamarckian | 2.997 | 30 | 30 | 1,000 | 30,000 | 300 | 300 | 300 | 280 | 70 | 0.5133 | 18 | 0.9893 | 0.9893 | 4 | 28.69 | 0.9667 | 8.333 |
| xor | B_baldwinian | 3.257 | 30 | 30 | 1,000 | 30,000 | 300 | 300 | 300 | 280 | 70 | 0.5133 | 23 | 0.9893 | 0.9893 | 6 | 11.12 | 0.9733 | 11.67 |
| spiral | A_lamarckian | 4.607 | 30 | 30 | 1,000 | 30,000 | 3,000 | 3,000 | 3,000 | 280 | 70 | 0.52 | 23.67 | 0.925 | 0.925 | 6.333 | 9.83 | 0.8889 | 11.33 |
| spiral | A_baldwinian | 3.447 | 30 | 30 | 1,000 | 30,000 | 3,000 | 3,000 | 3,000 | 280 | 70 | 0.52 | 20.33 | 0.7786 | 0.7786 | 6 | 3.859 | 0.7889 | 9.667 |
| spiral | B_lamarckian | 3.37 | 30 | 30 | 1,000 | 30,000 | 300 | 300 | 300 | 280 | 70 | 0.52 | 20.33 | 0.9464 | 0.9464 | 5.333 | 29.93 | 0.9133 | 11.33 |
| spiral | B_baldwinian | 2.543 | 30 | 30 | 1,000 | 30,000 | 300 | 300 | 300 | 280 | 70 | 0.52 | 11 | 0.8714 | 0.8714 | 4 | 20.13 | 0.8289 | 9.333 |
| spiral3 | A_lamarckian | 4.347 | 30 | 30 | 1,000 | 30,000 | 3,000 | 3,000 | 3,000 | 280 | 70 | 0.5667 | 25.33 | 0.7976 | 0.7976 | 6 | 13.14 | 0.7933 | 13 |
| spiral3 | A_baldwinian | 2.737 | 30 | 30 | 1,000 | 30,000 | 3,000 | 3,000 | 3,000 | 280 | 70 | 0.5667 | 16.67 | 0.6869 | 0.6869 | 4 | 2.843 | 0.6644 | 8 |
| spiral3 | B_lamarckian | 3.667 | 30 | 30 | 1,000 | 30,000 | 300 | 300 | 300 | 280 | 70 | 0.5667 | 19 | 0.8167 | 0.8167 | 4.667 | 29.59 | 0.8156 | 11 |
| spiral3 | B_baldwinian | 2.237 | 30 | 30 | 1,000 | 30,000 | 300 | 300 | 300 | 280 | 70 | 0.5667 | 8.333 | 0.7036 | 0.7036 | 3.333 | 13.66 | 0.7022 | 6.667 |

Per-cell data: `results.csv`. Outcomes: ok = counted; invalid-budget = exceeded the gradient-step budget, does not count; crashed / timeout = no usable result; missing = never run; args-mismatch = the run on disk used other arguments than the grid asks for.

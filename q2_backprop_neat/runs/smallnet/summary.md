# Grid `smallnet`

Generated 2026-10-04T21:04:39-07:00 by `experiments/summarize.py` from `/Users/vivandoshi/Documents/astra-takehome 3/q2_backprop_neat/runs/smallnet`.
120 cells = 4 arms × 3 datasets × 10 seeds (seeds [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]).

**Outcomes:** ok 120 · invalid-budget 0 · crashed 0 · timeout 0 · missing 0 · args-mismatch 0

## Results by dataset and arm

Held-out accuracy is the reported number and is never used to choose. Statistics are over `ok` cells only (n of expected); sd is the sample standard deviation across seeds (n−1), not a standard error. `fail` counts crashed, timed-out, missing and args-mismatch cells.

| dataset | arm | n | invalid | fail | held-out mean ± sd | min | max | val mean ± sd | train mean | conns mean | budget spent mean | selected |
|---|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|:-:|
| xor | lam_std | 10/10 | 0 | 0 | 0.967 ± 0.019 | 0.920 | 0.987 | 0.989 ± 0.009 | 0.977 | 6.0 | 30,000 |  |
| xor | bald_std | 10/10 | 0 | 0 | 0.956 ± 0.020 | 0.927 | 0.980 | 0.963 ± 0.017 | 0.963 | 6.0 | 30,000 |  |
| xor | lam_sine | 10/10 | 0 | 0 | 0.969 ± 0.004 | 0.960 | 0.973 | 0.997 ± 0.006 | 0.991 | 5.7 | 30,000 | **yes** |
| xor | bald_sine | 10/10 | 0 | 0 | 0.976 ± 0.006 | 0.967 | 0.980 | 0.989 ± 0.009 | 0.983 | 5.9 | 30,000 |  |
| spiral | lam_std | 10/10 | 0 | 0 | 0.826 ± 0.043 | 0.787 | 0.900 | 0.834 ± 0.054 | 0.836 | 5.5 | 30,000 |  |
| spiral | bald_std | 10/10 | 0 | 0 | 0.826 ± 0.071 | 0.700 | 0.913 | 0.851 ± 0.050 | 0.857 | 5.4 | 30,000 |  |
| spiral | lam_sine | 10/10 | 0 | 0 | 0.906 ± 0.034 | 0.860 | 0.987 | 0.943 ± 0.030 | 0.947 | 6.0 | 30,000 | **yes** |
| spiral | bald_sine | 10/10 | 0 | 0 | 0.849 ± 0.048 | 0.740 | 0.913 | 0.861 ± 0.053 | 0.887 | 5.8 | 30,000 |  |
| spiral3 | lam_std | 10/10 | 0 | 0 | 0.779 ± 0.030 | 0.753 | 0.847 | 0.794 ± 0.031 | 0.791 | 6.0 | 30,000 |  |
| spiral3 | bald_std | 10/10 | 0 | 0 | 0.711 ± 0.033 | 0.673 | 0.787 | 0.750 ± 0.046 | 0.716 | 5.7 | 30,000 |  |
| spiral3 | lam_sine | 10/10 | 0 | 0 | 0.863 ± 0.049 | 0.773 | 0.940 | 0.933 ± 0.049 | 0.893 | 6.0 | 30,000 | **yes** |
| spiral3 | bald_sine | 10/10 | 0 | 0 | 0.816 ± 0.056 | 0.720 | 0.887 | 0.861 ± 0.060 | 0.825 | 5.9 | 30,000 |  |

## Selected arm per dataset

Chosen by the highest mean validation accuracy (`val_acc`) among arms whose cells all completed `ok`. Held-out accuracy is shown for reporting only; the selection code is never given it.

| dataset | selected arm | val mean | held-out mean ± sd (reported, not used) | note |
|---|---|--:|--:|---|
| xor | lam_sine | 0.997 | 0.969 ± 0.004 |  |
| spiral | lam_sine | 0.943 | 0.906 ± 0.034 |  |
| spiral3 | lam_sine | 0.933 | 0.863 ± 0.049 |  |

## Other numeric summary fields (mean over ok cells)

| dataset | arm | seconds | generations_planned | generations_run | steps_per_generation | budget_limit | budget_refinements | refine_calls | genomes_refined | n_fit | n_val | heldout_majority_class | max_conns | genomes_created | genomes_repaired | repair_disabled_dead | repair_disabled_live | genomes_over_limit_evaluated | champion_generation | champion_fitness | fit_acc | nodes | max_abs_param | trainbest_heldout_acc | trainbest_conns |
|---|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| xor | lam_std | 3.964 | 30 | 30 | 1,000 | 30,000 | 3,000 | 3,000 | 3,000 | 280 | 70 | 0.5133 | 6 | 2,550 | 719.4 | 204.2 | 666 | 0 | 21 | 0.9746 | 0.9746 | 3.8 | 17.3 | 0.964 | 6 |
| xor | bald_std | 2.996 | 30 | 30 | 1,000 | 30,000 | 300 | 300 | 300 | 280 | 70 | 0.5133 | 6 | 2880 | 388.6 | 188.7 | 277.9 | 0 | 14.5 | 0.9632 | 0.9632 | 3.1 | 12.3 | 0.9453 | 6 |
| xor | lam_sine | 2.9 | 30 | 30 | 1,000 | 30,000 | 3,000 | 3,000 | 3,000 | 280 | 70 | 0.5133 | 6 | 2708 | 769.2 | 315.5 | 649.3 | 0 | 16.8 | 0.9893 | 0.9893 | 5 | 21.73 | 0.9713 | 5.6 |
| xor | bald_sine | 2.269 | 30 | 30 | 1,000 | 30,000 | 300 | 300 | 300 | 280 | 70 | 0.5133 | 6 | 2840 | 429.4 | 221 | 307.2 | 0 | 20.2 | 0.9814 | 0.9814 | 3.9 | 12.64 | 0.976 | 5.9 |
| spiral | lam_std | 3.536 | 30 | 30 | 1,000 | 30,000 | 3,000 | 3,000 | 3,000 | 280 | 70 | 0.52 | 6 | 2746 | 763.8 | 314.7 | 642.6 | 0 | 18.4 | 0.8368 | 0.8368 | 4.5 | 11.1 | 0.8267 | 5.5 |
| spiral | bald_std | 2.197 | 30 | 30 | 1,000 | 30,000 | 300 | 300 | 300 | 280 | 70 | 0.52 | 6 | 2896 | 298 | 146.4 | 209 | 0 | 16.5 | 0.8586 | 0.8586 | 3.1 | 16.38 | 0.8447 | 5.2 |
| spiral | lam_sine | 2.955 | 30 | 30 | 1,000 | 30,000 | 3,000 | 3,000 | 3,000 | 280 | 70 | 0.52 | 6 | 2809 | 790.9 | 300.8 | 684.1 | 0 | 24.6 | 0.9486 | 0.9486 | 4.8 | 10.49 | 0.9107 | 6 |
| spiral | bald_sine | 2.359 | 30 | 30 | 1,000 | 30,000 | 300 | 300 | 300 | 280 | 70 | 0.52 | 6 | 2875 | 416.4 | 279.3 | 235.5 | 0 | 21.3 | 0.8936 | 0.8936 | 4.8 | 11.29 | 0.8593 | 5.7 |
| spiral3 | lam_std | 3.393 | 30 | 30 | 1,000 | 30,000 | 3,000 | 3,000 | 3,000 | 280 | 70 | 0.5667 | 6 | 2,821 | 627.8 | 309 | 477.1 | 0 | 24.1 | 0.7904 | 0.7904 | 5 | 12.69 | 0.7733 | 6 |
| spiral3 | bald_std | 2.073 | 30 | 30 | 1,000 | 30,000 | 300 | 300 | 300 | 280 | 70 | 0.5667 | 6 | 2860 | 281.4 | 175.1 | 161.3 | 0 | 14.5 | 0.7071 | 0.7071 | 3.5 | 12.97 | 0.712 | 5.7 |
| spiral3 | lam_sine | 2.786 | 30 | 30 | 1,000 | 30,000 | 3,000 | 3,000 | 3,000 | 280 | 70 | 0.5667 | 6 | 2870 | 720.1 | 431.5 | 487.8 | 0 | 24.4 | 0.8825 | 0.8825 | 5.1 | 9.292 | 0.87 | 6 |
| spiral3 | bald_sine | 2.374 | 30 | 30 | 1,000 | 30,000 | 300 | 300 | 300 | 280 | 70 | 0.5667 | 6 | 2874 | 335 | 214.2 | 194.9 | 0 | 21 | 0.8164 | 0.8164 | 4.3 | 8.06 | 0.824 | 6 |

Per-cell data: `results.csv`. Outcomes: ok = counted; invalid-budget = exceeded the gradient-step budget, does not count; crashed / timeout = no usable result; missing = never run; args-mismatch = the run on disk used other arguments than the grid asks for.

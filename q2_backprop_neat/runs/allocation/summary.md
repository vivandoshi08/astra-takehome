# Grid `allocation`

Generated 2026-10-04T20:00:55-07:00 by `experiments/summarize.py` from `/Users/vivandoshi/Documents/astra-takehome 3/q2_backprop_neat/runs/allocation`.
90 cells = 10 arms × 3 datasets × 3 seeds (seeds [0, 1, 2]).

**Outcomes:** ok 86 · invalid-budget 0 · crashed 4 · timeout 0 · missing 0 · args-mismatch 0

## Cells not counted in the statistics

| cell | outcome | reason |
|---|---|---|
| allocation/D_lamarckian/xor_s1 | crashed | exit 1: RuntimeError: Configuration conflict: population size 100 is less than num_species * min_species_size 102 (51 * 2). Cannot satisfy per-species minima. |
| allocation/D_lamarckian/spiral_s0 | crashed | exit 1: RuntimeError: Configuration conflict: population size 100 is less than num_species * min_species_size 102 (51 * 2). Cannot satisfy per-species minima. |
| allocation/D_lamarckian/spiral3_s0 | crashed | exit 1: RuntimeError: Configuration conflict: population size 100 is less than num_species * min_species_size 102 (51 * 2). Cannot satisfy per-species minima. |
| allocation/D_lamarckian/spiral3_s1 | crashed | exit 1: RuntimeError: Configuration conflict: population size 100 is less than num_species * min_species_size 102 (51 * 2). Cannot satisfy per-species minima. |

## Results by dataset and arm

Held-out accuracy is the reported number and is never used to choose. Statistics are over `ok` cells only (n of expected); sd is the sample standard deviation across seeds (n−1), not a standard error. `fail` counts crashed, timed-out, missing and args-mismatch cells.

| dataset | arm | n | invalid | fail | held-out mean ± sd | min | max | val mean ± sd | train mean | conns mean | budget spent mean | selected |
|---|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|:-:|
| xor | A_lamarckian | 3/3 | 0 | 0 | 0.971 ± 0.010 | 0.960 | 0.980 | 0.995 ± 0.008 | 0.992 | 12.3 | 30,000 |  |
| xor | A_baldwinian | 3/3 | 0 | 0 | 0.969 ± 0.010 | 0.960 | 0.980 | 0.990 ± 0.008 | 0.981 | 9.0 | 30,000 |  |
| xor | B_lamarckian | 3/3 | 0 | 0 | 0.951 ± 0.021 | 0.927 | 0.967 | 0.967 ± 0.036 | 0.971 | 6.7 | 30,000 |  |
| xor | B_baldwinian | 3/3 | 0 | 0 | 0.962 ± 0.025 | 0.933 | 0.980 | 0.981 ± 0.022 | 0.977 | 8.7 | 30,000 |  |
| xor | C_lamarckian | 3/3 | 0 | 0 | 0.960 ± 0.012 | 0.947 | 0.967 | 0.962 ± 0.030 | 0.971 | 6.7 | 30,000 |  |
| xor | C_baldwinian | 3/3 | 0 | 0 | 0.876 ± 0.148 | 0.707 | 0.980 | 0.886 ± 0.161 | 0.877 | 5.0 | 30,000 |  |
| xor | D_lamarckian | 2/3 | 0 | 1 | 0.957 ± 0.014 | 0.947 | 0.967 | 1.000 ± 0.000 | 0.994 | 8.0 | 29,520 |  |
| xor | D_baldwinian | 3/3 | 0 | 0 | 0.976 ± 0.010 | 0.967 | 0.987 | 0.986 ± 0.014 | 0.987 | 10.7 | 29,520 |  |
| xor | E_lamarckian | 3/3 | 0 | 0 | 0.967 ± 0.007 | 0.960 | 0.973 | 1.000 ± 0.000 | 0.982 | 13.7 | 30,000 | **yes** |
| xor | E_baldwinian | 3/3 | 0 | 0 | 0.978 ± 0.015 | 0.960 | 0.987 | 0.986 ± 0.014 | 0.975 | 13.0 | 30,000 |  |
| spiral | A_lamarckian | 3/3 | 0 | 0 | 0.856 ± 0.096 | 0.800 | 0.967 | 0.876 ± 0.095 | 0.887 | 7.0 | 30,000 |  |
| spiral | A_baldwinian | 3/3 | 0 | 0 | 0.791 ± 0.017 | 0.773 | 0.807 | 0.790 ± 0.036 | 0.786 | 4.0 | 30,000 |  |
| spiral | B_lamarckian | 3/3 | 0 | 0 | 0.840 ± 0.041 | 0.793 | 0.867 | 0.881 ± 0.008 | 0.882 | 9.3 | 30,000 |  |
| spiral | B_baldwinian | 3/3 | 0 | 0 | 0.818 ± 0.023 | 0.793 | 0.840 | 0.838 ± 0.046 | 0.849 | 7.0 | 30,000 |  |
| spiral | C_lamarckian | 3/3 | 0 | 0 | 0.804 ± 0.020 | 0.787 | 0.827 | 0.829 ± 0.014 | 0.829 | 6.0 | 30,000 |  |
| spiral | C_baldwinian | 3/3 | 0 | 0 | 0.860 ± 0.024 | 0.833 | 0.880 | 0.852 ± 0.030 | 0.870 | 5.7 | 30,000 |  |
| spiral | D_lamarckian | 2/3 | 0 | 1 | 0.900 ± 0.019 | 0.887 | 0.913 | 0.914 ± 0.040 | 0.934 | 13.0 | 29,520 |  |
| spiral | D_baldwinian | 3/3 | 0 | 0 | 0.927 ± 0.037 | 0.887 | 0.960 | 0.948 ± 0.054 | 0.960 | 9.0 | 29,520 |  |
| spiral | E_lamarckian | 3/3 | 0 | 0 | 0.936 ± 0.004 | 0.933 | 0.940 | 0.971 ± 0.000 | 0.980 | 11.0 | 30,000 | **yes** |
| spiral | E_baldwinian | 3/3 | 0 | 0 | 0.838 ± 0.032 | 0.813 | 0.873 | 0.833 ± 0.022 | 0.853 | 7.7 | 30,000 |  |
| spiral3 | A_lamarckian | 3/3 | 0 | 0 | 0.793 ± 0.029 | 0.773 | 0.827 | 0.843 ± 0.029 | 0.796 | 9.0 | 30,000 |  |
| spiral3 | A_baldwinian | 3/3 | 0 | 0 | 0.733 ± 0.020 | 0.713 | 0.753 | 0.743 ± 0.029 | 0.746 | 9.7 | 30,000 |  |
| spiral3 | B_lamarckian | 3/3 | 0 | 0 | 0.800 ± 0.107 | 0.713 | 0.920 | 0.871 ± 0.065 | 0.843 | 11.0 | 30,000 | **yes** |
| spiral3 | B_baldwinian | 3/3 | 0 | 0 | 0.749 ± 0.091 | 0.667 | 0.847 | 0.795 ± 0.044 | 0.771 | 11.3 | 30,000 |  |
| spiral3 | C_lamarckian | 3/3 | 0 | 0 | 0.749 ± 0.034 | 0.720 | 0.787 | 0.800 ± 0.038 | 0.767 | 6.3 | 30,000 |  |
| spiral3 | C_baldwinian | 3/3 | 0 | 0 | 0.664 ± 0.032 | 0.640 | 0.700 | 0.710 ± 0.016 | 0.682 | 4.7 | 30,000 |  |
| spiral3 | D_lamarckian | 1/3 | 0 | 2 | 0.867 ± — | 0.867 | 0.867 | 0.957 ± — | 0.903 | 18.0 | 29,520 |  |
| spiral3 | D_baldwinian | 3/3 | 0 | 0 | 0.680 ± 0.029 | 0.660 | 0.713 | 0.738 ± 0.016 | 0.683 | 6.0 | 29,520 |  |
| spiral3 | E_lamarckian | 3/3 | 0 | 0 | 0.776 ± 0.020 | 0.753 | 0.793 | 0.814 ± 0.062 | 0.789 | 12.0 | 30,000 |  |
| spiral3 | E_baldwinian | 3/3 | 0 | 0 | 0.664 ± 0.052 | 0.607 | 0.707 | 0.748 ± 0.022 | 0.720 | 3.7 | 30,000 |  |

## Selected arm per dataset

Chosen by the highest mean validation accuracy (`val_acc`) among arms whose cells all completed `ok`. Held-out accuracy is shown for reporting only; the selection code is never given it.

| dataset | selected arm | val mean | held-out mean ± sd (reported, not used) | note |
|---|---|--:|--:|---|
| xor | E_lamarckian | 1.000 | 0.967 ± 0.007 | not eligible (not all cells ok): D_lamarckian (2/3 ok) |
| spiral | E_lamarckian | 0.971 | 0.936 ± 0.004 | not eligible (not all cells ok): D_lamarckian (2/3 ok) |
| spiral3 | B_lamarckian | 0.871 | 0.800 ± 0.107 | not eligible (not all cells ok): D_lamarckian (1/3 ok) |

## Other numeric summary fields (mean over ok cells)

| dataset | arm | seconds | generations_planned | generations_run | steps_per_generation | budget_limit | budget_refinements | refine_calls | genomes_refined | n_fit | n_val | heldout_majority_class | champion_generation | champion_fitness | fit_acc | nodes | max_abs_param | trainbest_heldout_acc | trainbest_conns |
|---|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| xor | A_lamarckian | 5.043 | 30 | 30 | 1,000 | 30,000 | 3,000 | 3,000 | 3,000 | 280 | 70 | 0.5133 | 21.33 | 0.9917 | 0.9917 | 6.333 | 23.73 | 0.9778 | 12.67 |
| xor | A_baldwinian | 3.14 | 30 | 30 | 1,000 | 30,000 | 3,000 | 3,000 | 3,000 | 280 | 70 | 0.5133 | 23 | 0.9786 | 0.9786 | 4.333 | 5.411 | 0.9778 | 10 |
| xor | B_lamarckian | 3.19 | 30 | 30 | 1,000 | 30,000 | 300 | 300 | 300 | 280 | 70 | 0.5133 | 14 | 0.9726 | 0.9726 | 3 | 24.08 | 0.9578 | 7 |
| xor | B_baldwinian | 2.803 | 30 | 30 | 1,000 | 30,000 | 300 | 300 | 300 | 280 | 70 | 0.5133 | 18.33 | 0.9762 | 0.9762 | 4.667 | 13.6 | 0.9622 | 9.333 |
| xor | C_lamarckian | 1.83 | 6 | 6 | 5,000 | 30,000 | 600 | 600 | 600 | 280 | 70 | 0.5133 | 5 | 0.9738 | 0.9738 | 3.667 | 15.98 | 0.96 | 6.667 |
| xor | C_baldwinian | 1.447 | 6 | 6 | 5,000 | 30,000 | 600 | 600 | 600 | 280 | 70 | 0.5133 | 4.333 | 0.875 | 0.875 | 2.667 | 7.476 | 0.8711 | 5 |
| xor | D_lamarckian | 4.045 | 30 | 30 | 984 | 30,000 | 5,790 | 5,790 | 3,000 | 280 | 70 | 0.5133 | 14 | 0.9929 | 0.9929 | 4 | 20.64 | 0.9667 | 9 |
| xor | D_baldwinian | 3.613 | 30 | 30 | 984 | 30,000 | 5,790 | 5,790 | 3,000 | 280 | 70 | 0.5133 | 19 | 0.9869 | 0.9869 | 5.333 | 11.21 | 0.9733 | 11 |
| xor | E_lamarckian | 5.52 | 60 | 60 | 500 | 30,000 | 6,000 | 6,000 | 6,000 | 280 | 70 | 0.5133 | 36.67 | 0.9774 | 0.9774 | 7 | 13.98 | 0.9711 | 12.67 |
| xor | E_baldwinian | 4.907 | 60 | 60 | 500 | 30,000 | 6,000 | 6,000 | 6,000 | 280 | 70 | 0.5133 | 52 | 0.9726 | 0.9726 | 9.667 | 7.189 | 0.9756 | 13.33 |
| spiral | A_lamarckian | 4.48 | 30 | 30 | 1,000 | 30,000 | 3,000 | 3,000 | 3,000 | 280 | 70 | 0.52 | 22 | 0.8893 | 0.8893 | 3.333 | 11.95 | 0.8556 | 7 |
| spiral | A_baldwinian | 3.28 | 30 | 30 | 1,000 | 30,000 | 3,000 | 3,000 | 3,000 | 280 | 70 | 0.52 | 21.33 | 0.7845 | 0.7845 | 2.333 | 3.861 | 0.7711 | 4.333 |
| spiral | B_lamarckian | 3.217 | 30 | 30 | 1,000 | 30,000 | 300 | 300 | 300 | 280 | 70 | 0.52 | 21.33 | 0.8821 | 0.8821 | 4.333 | 30 | 0.84 | 9.333 |
| spiral | B_baldwinian | 3.34 | 30 | 30 | 1,000 | 30,000 | 300 | 300 | 300 | 280 | 70 | 0.52 | 18 | 0.8512 | 0.8512 | 3.667 | 9.722 | 0.8044 | 7 |
| spiral | C_lamarckian | 2.207 | 6 | 6 | 5,000 | 30,000 | 600 | 600 | 600 | 280 | 70 | 0.52 | 5 | 0.8286 | 0.8286 | 4 | 15.23 | 0.8067 | 5.333 |
| spiral | C_baldwinian | 1.86 | 6 | 6 | 5,000 | 30,000 | 600 | 600 | 600 | 280 | 70 | 0.52 | 4.333 | 0.8738 | 0.8738 | 3.333 | 11.44 | 0.8689 | 6 |
| spiral | D_lamarckian | 4.18 | 30 | 30 | 984 | 30,000 | 5,790 | 5,790 | 3,000 | 280 | 70 | 0.52 | 19.5 | 0.9393 | 0.9393 | 6.5 | 29.55 | 0.8967 | 13 |
| spiral | D_baldwinian | 3.303 | 30 | 30 | 984 | 30,000 | 5,790 | 5,790 | 3,000 | 280 | 70 | 0.52 | 19.33 | 0.9631 | 0.9631 | 4.333 | 11.58 | 0.9267 | 9 |
| spiral | E_lamarckian | 5.287 | 60 | 60 | 500 | 30,000 | 6,000 | 6,000 | 6,000 | 280 | 70 | 0.52 | 38.33 | 0.9821 | 0.9821 | 5.667 | 16.34 | 0.9378 | 11 |
| spiral | E_baldwinian | 3.903 | 60 | 60 | 500 | 30,000 | 6,000 | 6,000 | 6,000 | 280 | 70 | 0.52 | 35.67 | 0.8583 | 0.8583 | 3.667 | 4.216 | 0.8444 | 6.333 |
| spiral3 | A_lamarckian | 3.303 | 30 | 30 | 1,000 | 30,000 | 3,000 | 3,000 | 3,000 | 280 | 70 | 0.5667 | 28.33 | 0.7845 | 0.7845 | 5 | 10.56 | 0.7733 | 8.333 |
| spiral3 | A_baldwinian | 4 | 30 | 30 | 1,000 | 30,000 | 3,000 | 3,000 | 3,000 | 280 | 70 | 0.5667 | 27.67 | 0.7464 | 0.7464 | 5 | 3.649 | 0.7244 | 9.333 |
| spiral3 | B_lamarckian | 3.073 | 30 | 30 | 1,000 | 30,000 | 300 | 300 | 300 | 280 | 70 | 0.5667 | 22.33 | 0.8357 | 0.8357 | 5 | 30 | 0.8133 | 10.67 |
| spiral3 | B_baldwinian | 3.24 | 30 | 30 | 1,000 | 30,000 | 300 | 300 | 300 | 280 | 70 | 0.5667 | 23.33 | 0.7655 | 0.7655 | 7 | 17.59 | 0.7489 | 11.33 |
| spiral3 | C_lamarckian | 2.347 | 6 | 6 | 5,000 | 30,000 | 600 | 600 | 600 | 280 | 70 | 0.5667 | 4.333 | 0.7583 | 0.7583 | 3.333 | 21.79 | 0.7467 | 6.667 |
| spiral3 | C_baldwinian | 1.66 | 6 | 6 | 5,000 | 30,000 | 600 | 600 | 600 | 280 | 70 | 0.5667 | 3.333 | 0.675 | 0.675 | 2.667 | 4.466 | 0.6844 | 4.667 |
| spiral3 | D_lamarckian | 3.81 | 30 | 30 | 984 | 30,000 | 5,790 | 5,790 | 3,000 | 280 | 70 | 0.5667 | 29 | 0.8893 | 0.8893 | 8 | 30 | 0.8667 | 18 |
| spiral3 | D_baldwinian | 2.307 | 30 | 30 | 984 | 30,000 | 5,790 | 5,790 | 3,000 | 280 | 70 | 0.5667 | 19.33 | 0.669 | 0.669 | 4 | 9.059 | 0.6733 | 7.667 |
| spiral3 | E_lamarckian | 5.18 | 60 | 60 | 500 | 30,000 | 6,000 | 6,000 | 6,000 | 280 | 70 | 0.5667 | 36.67 | 0.7821 | 0.7821 | 6.333 | 8.721 | 0.7911 | 11 |
| spiral3 | E_baldwinian | 3.053 | 60 | 60 | 500 | 30,000 | 6,000 | 6,000 | 6,000 | 280 | 70 | 0.5667 | 33.67 | 0.7131 | 0.7131 | 2 | 3.369 | 0.6844 | 4 |

Per-cell data: `results.csv`. Outcomes: ok = counted; invalid-budget = exceeded the gradient-step budget, does not count; crashed / timeout = no usable result; missing = never run; args-mismatch = the run on disk used other arguments than the grid asks for.

# Grid `allocation`

Generated 2026-10-04T22:07:49-07:00 by `experiments/summarize.py` from `/Users/vivandoshi/Documents/astra-takehome 3/.claude/worktrees/q2-audit-fixes/q2_backprop_neat/runs/allocation`.
300 cells = 10 arms × 3 datasets × 10 seeds (seeds [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]).

**Outcomes:** ok 295 · invalid-budget 0 · crashed 5 · timeout 0 · missing 0 · args-mismatch 0

## Warnings

- **cells come from 3 different git commits: 34dc34c (207 cells), dfa1495 (86 cells), 6b4440e (7 cells)**
- **cells ran 2 different versions of the code: 7 cells (e.g. allocation/B_lamarckian/spiral_s4) differ from the 293-cell majority in: main.py, neat_config_lowmut.ini, neat_config_noreplace.ini, neat_config_sine.ini, neat_config_topo.ini, network.py, test_gradients.py, test_main.py, test_smallnet.py**
- **arm 'B_lamarckian': cells recorded 2 different argument sets (differing: config); its cells are not one configuration**
- **arm 'D_lamarckian': cells recorded 2 different argument sets (differing: config); its cells are not one configuration**

## Cells not counted in the statistics

| cell | outcome | reason |
|---|---|---|
| allocation/B_lamarckian/spiral_s4 | crashed | exit 5: wrote /Users/vivandoshi/Documents/astra-takehome 3/.claude/worktrees/q2-audit-fixes/q2_backprop_neat/runs/allocation/B_lamarckian/spiral_s4/ |
| allocation/D_lamarckian/xor_s1 | crashed | exit 5: wrote /Users/vivandoshi/Documents/astra-takehome 3/.claude/worktrees/q2-audit-fixes/q2_backprop_neat/runs/allocation/D_lamarckian/xor_s1/ |
| allocation/D_lamarckian/spiral_s0 | crashed | exit 5: wrote /Users/vivandoshi/Documents/astra-takehome 3/.claude/worktrees/q2-audit-fixes/q2_backprop_neat/runs/allocation/D_lamarckian/spiral_s0/ |
| allocation/D_lamarckian/spiral_s6 | crashed | exit 5: wrote /Users/vivandoshi/Documents/astra-takehome 3/.claude/worktrees/q2-audit-fixes/q2_backprop_neat/runs/allocation/D_lamarckian/spiral_s6/ |
| allocation/D_lamarckian/spiral_s9 | crashed | exit 5: wrote /Users/vivandoshi/Documents/astra-takehome 3/.claude/worktrees/q2-audit-fixes/q2_backprop_neat/runs/allocation/D_lamarckian/spiral_s9/ |

## Results by dataset and arm

Held-out accuracy is the reported number and is never used to choose. Statistics are over `ok` cells only (n of expected); sd is the sample standard deviation across seeds (n−1), not a standard error. `fail` counts crashed, timed-out, missing and args-mismatch cells.

| dataset | arm | n | invalid | fail | held-out mean ± sd | min | max | val mean ± sd | train mean | conns mean | budget spent mean | selected |
|---|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|:-:|
| xor | A_lamarckian | 10/10 | 0 | 0 | 0.967 ± 0.007 | 0.960 | 0.980 | 0.999 ± 0.005 | 0.993 | 12.3 | 30,000 |  |
| xor | A_baldwinian | 10/10 | 0 | 0 | 0.973 ± 0.009 | 0.960 | 0.987 | 0.986 ± 0.007 | 0.978 | 9.7 | 30,000 |  |
| xor | B_lamarckian | 10/10 | 0 | 0 | 0.967 ± 0.016 | 0.927 | 0.980 | 0.984 ± 0.022 | 0.985 | 8.7 | 30,000 |  |
| xor | B_baldwinian | 10/10 | 0 | 0 | 0.964 ± 0.018 | 0.933 | 0.980 | 0.984 ± 0.016 | 0.980 | 8.7 | 30,000 |  |
| xor | C_lamarckian | 10/10 | 0 | 0 | 0.963 ± 0.017 | 0.933 | 0.987 | 0.967 ± 0.026 | 0.972 | 6.8 | 30,000 |  |
| xor | C_baldwinian | 10/10 | 0 | 0 | 0.928 ± 0.082 | 0.707 | 0.993 | 0.939 ± 0.087 | 0.939 | 6.0 | 30,000 |  |
| xor | D_lamarckian | 9/10 | 0 | 1 | 0.966 ± 0.013 | 0.947 | 0.987 | 0.995 ± 0.007 | 0.991 | 9.0 | 29,520 |  |
| xor | D_baldwinian | 10/10 | 0 | 0 | 0.976 ± 0.006 | 0.967 | 0.987 | 0.987 ± 0.008 | 0.989 | 10.6 | 29,520 |  |
| xor | E_lamarckian | 10/10 | 0 | 0 | 0.968 ± 0.004 | 0.960 | 0.973 | 0.999 ± 0.005 | 0.988 | 13.5 | 30,000 |  |
| xor | E_baldwinian | 10/10 | 0 | 0 | 0.966 ± 0.022 | 0.913 | 0.987 | 0.986 ± 0.013 | 0.974 | 11.9 | 30,000 |  |
| spiral | A_lamarckian | 10/10 | 0 | 0 | 0.903 ± 0.069 | 0.800 | 0.980 | 0.929 ± 0.069 | 0.936 | 10.5 | 30,000 | **yes** |
| spiral | A_baldwinian | 10/10 | 0 | 0 | 0.815 ± 0.035 | 0.773 | 0.873 | 0.817 ± 0.031 | 0.820 | 8.1 | 30,000 |  |
| spiral | B_lamarckian | 9/10 | 0 | 1 | 0.877 ± 0.068 | 0.793 | 0.993 | 0.897 ± 0.054 | 0.904 | 9.0 | 30,000 |  |
| spiral | B_baldwinian | 10/10 | 0 | 0 | 0.869 ± 0.057 | 0.793 | 0.980 | 0.881 ± 0.050 | 0.897 | 9.1 | 30,000 |  |
| spiral | C_lamarckian | 10/10 | 0 | 0 | 0.808 ± 0.038 | 0.753 | 0.900 | 0.821 ± 0.056 | 0.823 | 5.6 | 30,000 |  |
| spiral | C_baldwinian | 10/10 | 0 | 0 | 0.845 ± 0.024 | 0.807 | 0.880 | 0.844 ± 0.026 | 0.858 | 6.0 | 30,000 |  |
| spiral | D_lamarckian | 7/10 | 0 | 3 | 0.903 ± 0.050 | 0.813 | 0.980 | 0.924 ± 0.065 | 0.941 | 11.4 | 29,520 |  |
| spiral | D_baldwinian | 10/10 | 0 | 0 | 0.879 ± 0.059 | 0.793 | 0.960 | 0.906 ± 0.061 | 0.921 | 10.0 | 29,520 |  |
| spiral | E_lamarckian | 10/10 | 0 | 0 | 0.877 ± 0.061 | 0.787 | 0.940 | 0.899 ± 0.071 | 0.913 | 11.5 | 30,000 |  |
| spiral | E_baldwinian | 10/10 | 0 | 0 | 0.851 ± 0.037 | 0.800 | 0.920 | 0.847 ± 0.048 | 0.869 | 8.4 | 30,000 |  |
| spiral3 | A_lamarckian | 10/10 | 0 | 0 | 0.825 ± 0.055 | 0.773 | 0.953 | 0.854 ± 0.067 | 0.832 | 10.2 | 30,000 |  |
| spiral3 | A_baldwinian | 10/10 | 0 | 0 | 0.677 ± 0.061 | 0.600 | 0.753 | 0.747 ± 0.039 | 0.723 | 6.4 | 30,000 |  |
| spiral3 | B_lamarckian | 10/10 | 0 | 0 | 0.809 ± 0.061 | 0.713 | 0.920 | 0.866 ± 0.063 | 0.831 | 10.0 | 30,000 |  |
| spiral3 | B_baldwinian | 10/10 | 0 | 0 | 0.740 ± 0.060 | 0.667 | 0.847 | 0.797 ± 0.054 | 0.769 | 8.4 | 30,000 |  |
| spiral3 | C_lamarckian | 10/10 | 0 | 0 | 0.736 ± 0.024 | 0.707 | 0.787 | 0.767 ± 0.033 | 0.751 | 6.1 | 30,000 |  |
| spiral3 | C_baldwinian | 10/10 | 0 | 0 | 0.695 ± 0.033 | 0.640 | 0.740 | 0.740 ± 0.031 | 0.705 | 5.3 | 30,000 |  |
| spiral3 | D_lamarckian | 10/10 | 0 | 0 | 0.834 ± 0.040 | 0.787 | 0.880 | 0.893 ± 0.044 | 0.862 | 11.9 | 29,520 | **yes** |
| spiral3 | D_baldwinian | 10/10 | 0 | 0 | 0.683 ± 0.027 | 0.660 | 0.733 | 0.730 ± 0.011 | 0.685 | 5.4 | 29,520 |  |
| spiral3 | E_lamarckian | 10/10 | 0 | 0 | 0.793 ± 0.045 | 0.727 | 0.847 | 0.823 ± 0.033 | 0.802 | 12.5 | 30,000 |  |
| spiral3 | E_baldwinian | 10/10 | 0 | 0 | 0.673 ± 0.064 | 0.600 | 0.760 | 0.757 ± 0.033 | 0.722 | 4.6 | 30,000 |  |

## Selected arm per dataset

Chosen by the highest mean validation accuracy (`val_acc`) among arms whose cells all completed `ok`. Held-out accuracy is shown for reporting only; the selection code is never given it.

| dataset | selected arm | val mean | held-out mean ± sd (reported, not used) | note |
|---|---|--:|--:|---|
| xor | — | — | — | n/a — tie on val mean 0.9986: A_lamarckian, E_lamarckian; not eligible (not all cells ok): D_lamarckian (9/10 ok) |
| spiral | A_lamarckian | 0.929 | 0.903 ± 0.069 | not eligible (not all cells ok): B_lamarckian (9/10 ok), D_lamarckian (7/10 ok) |
| spiral3 | D_lamarckian | 0.893 | 0.834 ± 0.040 |  |

## Other numeric summary fields (mean over ok cells)

| dataset | arm | seconds | generations_planned | generations_run | steps_per_generation | budget_limit | budget_refinements | refine_calls | genomes_refined | n_fit | n_val | heldout_majority_class | champion_generation | champion_fitness | fit_acc | nodes | max_abs_param | trainbest_heldout_acc | trainbest_conns |
|---|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| xor | A_lamarckian | 6.953 | 30 | 30 | 1,000 | 30,000 | 3,000 | 3,000 | 3,000 | 280 | 70 | 0.5133 | 19.4 | 0.9918 | 0.9918 | 5.8 | 22.65 | 0.9733 | 12.7 |
| xor | A_baldwinian | 3.053 | 30 | 30 | 1,000 | 30,000 | 3,000 | 3,000 | 3,000 | 280 | 70 | 0.5133 | 22.7 | 0.9761 | 0.9761 | 4.6 | 4.988 | 0.9747 | 10 |
| xor | B_lamarckian | 3.07 | 30 | 30 | 1,000 | 30,000 | 300 | 300 | 300 | 280 | 70 | 0.5133 | 18.5 | 0.9857 | 0.9857 | 4 | 27.44 | 0.9653 | 8.9 |
| xor | B_baldwinian | 2.523 | 30 | 30 | 1,000 | 30,000 | 300 | 300 | 300 | 280 | 70 | 0.5133 | 20.1 | 0.9786 | 0.9786 | 4.8 | 12 | 0.9633 | 8.9 |
| xor | C_lamarckian | 1.639 | 6 | 6 | 5,000 | 30,000 | 600 | 600 | 600 | 280 | 70 | 0.5133 | 4.6 | 0.9736 | 0.9736 | 3.5 | 17.53 | 0.9627 | 6.7 |
| xor | C_baldwinian | 1.568 | 6 | 6 | 5,000 | 30,000 | 600 | 600 | 600 | 280 | 70 | 0.5133 | 4.6 | 0.9389 | 0.9389 | 3 | 9.42 | 0.9267 | 6 |
| xor | D_lamarckian | 3.82 | 30 | 30 | 984 | 30,000 | 5,790 | 5,790 | 3,000 | 280 | 70 | 0.5133 | 13.67 | 0.9905 | 0.9905 | 4.444 | 24.35 | 0.9659 | 10 |
| xor | D_baldwinian | 3.572 | 30 | 30 | 984 | 30,000 | 5,790 | 5,790 | 3,000 | 280 | 70 | 0.5133 | 21.1 | 0.9893 | 0.9893 | 5.5 | 10.33 | 0.9747 | 11 |
| xor | E_lamarckian | 5.428 | 60 | 60 | 500 | 30,000 | 6,000 | 6,000 | 6,000 | 280 | 70 | 0.5133 | 41.7 | 0.9857 | 0.9857 | 7 | 18.84 | 0.9707 | 13.2 |
| xor | E_baldwinian | 4.342 | 60 | 60 | 500 | 30,000 | 6,000 | 6,000 | 6,000 | 280 | 70 | 0.5133 | 49.9 | 0.9714 | 0.9714 | 7.7 | 5.602 | 0.966 | 12.3 |
| spiral | A_lamarckian | 4.528 | 30 | 30 | 1,000 | 30,000 | 3,000 | 3,000 | 3,000 | 280 | 70 | 0.52 | 25.1 | 0.9382 | 0.9382 | 5.3 | 13.13 | 0.9047 | 10.5 |
| spiral | A_baldwinian | 2.79 | 30 | 30 | 1,000 | 30,000 | 3,000 | 3,000 | 3,000 | 280 | 70 | 0.52 | 22.8 | 0.8207 | 0.8207 | 4.7 | 4.333 | 0.8133 | 7.8 |
| spiral | B_lamarckian | 3.088 | 30 | 30 | 1,000 | 30,000 | 300 | 300 | 300 | 280 | 70 | 0.52 | 19.44 | 0.9063 | 0.9063 | 4.333 | 28.08 | 0.877 | 8.556 |
| spiral | B_baldwinian | 2.859 | 30 | 30 | 1,000 | 30,000 | 300 | 300 | 300 | 280 | 70 | 0.52 | 19.5 | 0.9004 | 0.9004 | 5 | 13.99 | 0.8647 | 9.1 |
| spiral | C_lamarckian | 1.868 | 6 | 6 | 5,000 | 30,000 | 600 | 600 | 600 | 280 | 70 | 0.52 | 4.4 | 0.8239 | 0.8239 | 3.3 | 14.82 | 0.8113 | 5.7 |
| spiral | C_baldwinian | 1.739 | 6 | 6 | 5,000 | 30,000 | 600 | 600 | 600 | 280 | 70 | 0.52 | 4.3 | 0.8618 | 0.8618 | 3.5 | 10.63 | 0.8427 | 6.3 |
| spiral | D_lamarckian | 4.041 | 30 | 30 | 984 | 30,000 | 5,790 | 5,790 | 3,000 | 280 | 70 | 0.52 | 23.14 | 0.9454 | 0.9454 | 5.571 | 27.67 | 0.901 | 11.43 |
| spiral | D_baldwinian | 3.171 | 30 | 30 | 984 | 30,000 | 5,790 | 5,790 | 3,000 | 280 | 70 | 0.52 | 22.4 | 0.925 | 0.925 | 5.8 | 11.19 | 0.8813 | 9.9 |
| spiral | E_lamarckian | 5.135 | 60 | 60 | 500 | 30,000 | 6,000 | 6,000 | 6,000 | 280 | 70 | 0.52 | 43 | 0.9161 | 0.9161 | 6.4 | 12.71 | 0.8787 | 11.9 |
| spiral | E_baldwinian | 3.797 | 60 | 60 | 500 | 30,000 | 6,000 | 6,000 | 6,000 | 280 | 70 | 0.52 | 42.6 | 0.8746 | 0.8746 | 4.9 | 4.4 | 0.848 | 7.9 |
| spiral3 | A_lamarckian | 3.751 | 30 | 30 | 1,000 | 30,000 | 3,000 | 3,000 | 3,000 | 280 | 70 | 0.5667 | 25.7 | 0.8261 | 0.8261 | 5.2 | 11.95 | 0.8193 | 9.7 |
| spiral3 | A_baldwinian | 3.363 | 30 | 30 | 1,000 | 30,000 | 3,000 | 3,000 | 3,000 | 280 | 70 | 0.5667 | 19.1 | 0.7164 | 0.7164 | 3.3 | 3.348 | 0.684 | 6.9 |
| spiral3 | B_lamarckian | 3.188 | 30 | 30 | 1,000 | 30,000 | 300 | 300 | 300 | 280 | 70 | 0.5667 | 21.7 | 0.8218 | 0.8218 | 4.6 | 29.49 | 0.812 | 9.6 |
| spiral3 | B_baldwinian | 2.698 | 30 | 30 | 1,000 | 30,000 | 300 | 300 | 300 | 280 | 70 | 0.5667 | 17.7 | 0.7621 | 0.7621 | 4.8 | 15.13 | 0.7467 | 8.5 |
| spiral3 | C_lamarckian | 1.878 | 6 | 6 | 5,000 | 30,000 | 600 | 600 | 600 | 280 | 70 | 0.5667 | 4.4 | 0.7468 | 0.7468 | 3.4 | 19.42 | 0.7387 | 6.4 |
| spiral3 | C_baldwinian | 1.525 | 6 | 6 | 5,000 | 30,000 | 600 | 600 | 600 | 280 | 70 | 0.5667 | 4.1 | 0.6961 | 0.6961 | 2.7 | 5.967 | 0.702 | 5.6 |
| spiral3 | D_lamarckian | 3.995 | 30 | 30 | 984 | 30,000 | 5,790 | 5,790 | 3,000 | 280 | 70 | 0.5667 | 24.4 | 0.8543 | 0.8543 | 5.7 | 25.65 | 0.8373 | 12.5 |
| spiral3 | D_baldwinian | 2.382 | 30 | 30 | 984 | 30,000 | 5,790 | 5,790 | 3,000 | 280 | 70 | 0.5667 | 14 | 0.6743 | 0.6743 | 3.7 | 12.02 | 0.6873 | 6.2 |
| spiral3 | E_lamarckian | 4.974 | 60 | 60 | 500 | 30,000 | 6,000 | 6,000 | 6,000 | 280 | 70 | 0.5667 | 43.8 | 0.7964 | 0.7964 | 7.7 | 10.37 | 0.8067 | 13.7 |
| spiral3 | E_baldwinian | 3.042 | 60 | 60 | 500 | 30,000 | 6,000 | 6,000 | 6,000 | 280 | 70 | 0.5667 | 28.4 | 0.7129 | 0.7129 | 2.5 | 3.037 | 0.6993 | 4.9 |

Per-cell data: `results.csv`. Outcomes: ok = counted; invalid-budget = exceeded the gradient-step budget, does not count; crashed / timeout = no usable result; missing = never run; args-mismatch = the run on disk used other arguments than the grid asks for.

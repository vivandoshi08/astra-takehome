# Grid `first_result`

Generated 2026-10-04T19:37:53-07:00 by `experiments/summarize.py` from `/Users/vivandoshi/Documents/astra-takehome 3/q2_backprop_neat/runs/first_result`.
6 cells = 2 arms × 3 datasets × 1 seeds (seeds [0]).

**Outcomes:** ok 6 · invalid-budget 0 · crashed 0 · timeout 0 · missing 0 · args-mismatch 0

## Results by dataset and arm

Held-out accuracy is the reported number and is never used to choose. Statistics are over `ok` cells only (n of expected); sd is the sample standard deviation across seeds (n−1), not a standard error. `fail` counts crashed, timed-out, missing and args-mismatch cells.

| dataset | arm | n | invalid | fail | held-out mean ± sd | min | max | val mean ± sd | train mean | conns mean | budget spent mean | selected |
|---|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|:-:|
| xor | lamarckian | 1/1 | 0 | 0 | 0.973 ± — | 0.973 | 0.973 | 1.000 ± — | 0.991 | 12.0 | 30,000 | **yes** |
| xor | baldwinian | 1/1 | 0 | 0 | 0.980 ± — | 0.980 | 0.980 | 0.986 ± — | 0.980 | 8.0 | 30,000 |  |
| spiral | lamarckian | 1/1 | 0 | 0 | 0.800 ± — | 0.800 | 0.800 | 0.829 ± — | 0.851 | 8.0 | 30,000 | **yes** |
| spiral | baldwinian | 1/1 | 0 | 0 | 0.793 ± — | 0.793 | 0.793 | 0.786 ± — | 0.791 | 4.0 | 30,000 |  |
| spiral3 | lamarckian | 1/1 | 0 | 0 | 0.780 ± — | 0.780 | 0.780 | 0.871 ± — | 0.820 | 6.0 | 30,000 | **yes** |
| spiral3 | baldwinian | 1/1 | 0 | 0 | 0.713 ± — | 0.713 | 0.713 | 0.714 ± — | 0.717 | 8.0 | 30,000 |  |

## Selected arm per dataset

Chosen by the highest mean validation accuracy (`val_acc`) among arms whose cells all completed `ok`. Held-out accuracy is shown for reporting only; the selection code is never given it.

| dataset | selected arm | val mean | held-out mean ± sd (reported, not used) | note |
|---|---|--:|--:|---|
| xor | lamarckian | 1.000 | 0.973 ± — |  |
| spiral | lamarckian | 0.829 | 0.800 ± — |  |
| spiral3 | lamarckian | 0.871 | 0.780 ± — |  |

## Other numeric summary fields (mean over ok cells)

| dataset | arm | seconds | generations_planned | generations_run | steps_per_generation | budget_limit | budget_refinements | refine_calls | genomes_refined | n_fit | n_val | heldout_majority_class | champion_generation | champion_fitness | fit_acc | nodes | max_abs_param | trainbest_heldout_acc | trainbest_conns |
|---|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| xor | lamarckian | 5.17 | 30 | 30 | 1,000 | 30,000 | 3,000 | 3,000 | 3,000 | 280 | 70 | 0.5133 | 21 | 0.9893 | 0.9893 | 7 | 30 | 0.98 | 12 |
| xor | baldwinian | 4.48 | 30 | 30 | 1,000 | 30,000 | 3,000 | 3,000 | 3,000 | 280 | 70 | 0.5133 | 28 | 0.9786 | 0.9786 | 4 | 4.915 | 0.98 | 8 |
| spiral | lamarckian | 4.37 | 30 | 30 | 1,000 | 30,000 | 3,000 | 3,000 | 3,000 | 280 | 70 | 0.52 | 29 | 0.8571 | 0.8571 | 4 | 7.251 | 0.8 | 8 |
| spiral | baldwinian | 3.72 | 30 | 30 | 1,000 | 30,000 | 3,000 | 3,000 | 3,000 | 280 | 70 | 0.52 | 27 | 0.7929 | 0.7929 | 2 | 5.083 | 0.7933 | 4 |
| spiral3 | lamarckian | 4.25 | 30 | 30 | 1,000 | 30,000 | 3,000 | 3,000 | 3,000 | 280 | 70 | 0.5667 | 28 | 0.8071 | 0.8071 | 4 | 14.08 | 0.78 | 6 |
| spiral3 | baldwinian | 3.83 | 30 | 30 | 1,000 | 30,000 | 3,000 | 3,000 | 3,000 | 280 | 70 | 0.5667 | 26 | 0.7179 | 0.7179 | 5 | 4.193 | 0.7133 | 8 |

Per-cell data: `results.csv`. Outcomes: ok = counted; invalid-budget = exceeded the gradient-step budget, does not count; crashed / timeout = no usable result; missing = never run; args-mismatch = the run on disk used other arguments than the grid asks for.

# Allocation study — predictions, committed before the grid runs

Grid: `experiments/grids/allocation.json`. Scored by
`experiments/score_predictions.py` from `runs/allocation/results.csv`.

## Arms (all lr 0.3, 30,000 steps, seeds 0–2, xor / spiral / spiral3, both strategies)

| arm | allocation | generations |
|---|---|---|
| A | 10 steps x 100% refined | 30 |
| B | 100 steps x top 10% by pre-refinement fitness | 30 |
| C | 50 steps x 100% refined | 6 |
| D | successive halving (100x2, 50x4, 25x8, 12x16, 6x32) | 30 |
| E | 5 steps x 100% refined | 60 |

## What the pilot measured (runs/pilot, seed 100, Baldwinian genomes)

1. Pre-refinement fitness barely predicts post-refinement fitness: Spearman
   rho vs the 300-step ranking 0.07 / 0.07 / 0.09; the top 10% by untrained
   fitness holds 0–2 of the eventual top 10.
2. The ranking after 10 steps is near-random on the spirals (rho 0.02 /
   0.27; xor 0.71), although 10 steps give 89–99% of the median gain.
3. Successive halving's cuts would eliminate 4–6 of the 300-step top 6.
4. (first_result) Evolution is still improving at generation 29 on the
   spirals: champions found at generations 26–29, networks growing from 2
   to 6–9 connections.

The pilot's mechanical rule (steps where rho >= 0.9) predicted 1–3
generations; it is not used, because it gives generations no value (4).

## Predictions

- **P1** — Baldwinian, spirals: **B < A**. B's elite is chosen by
  pre-refinement fitness, which is nearly uninformative (1).
  Under Lamarckian the gap **B − A is smaller (less negative) than under
  Baldwinian**: inherited refined weights make pre-refinement fitness
  informative.
- **P2** — both strategies, spirals: **C < A**. Six generations cannot grow
  networks the way thirty do (4), whatever the gain in ranking fidelity (2).
- **P3** — Baldwinian, spirals: **D < A**, because of late bloomers (3).
- **E vs A**: no prediction.
- **xor**: no arm differs from A by more than the seed spread, except
  possibly C.

## Scoring (fixed now)

Per dataset and strategy, on held-out accuracy over seeds 0–2:
diff = mean(X) − mean(A); spread = sqrt((sd(X)² + sd(A)²) / 2).
- **supported**: diff has the predicted sign and |diff| > spread
- **contradicted**: diff has the opposite sign and |diff| > spread
- **inconclusive**: otherwise

For P1's Lamarckian part: supported if (B − A)_Lamarckian > (B − A)_Baldwinian
by more than the larger of the two spreads; contradicted if it is smaller
by that much; else inconclusive.

Three seeds is little; "inconclusive" is an expected outcome and will be
reported as such. Any allocation chosen after this grid is confirmed on
seeds 3–5, which no decision has used.

# Mechanism final test — fresh seeds 20–29, committed before running

## The narrowed mechanism (from rounds 1–3)

Lamarckian beats Baldwinian because refinement carries forward through
inherited weights (round 1: children start nearer their parents' refined
accuracy; Baldwinian closes much of the gap only by refining deeper).
The carry-over is lossy. Not because of crossover (round 2, Test A) or the
perturbation rate (round 2, Test B), but mainly because **weight/bias
replacement** resets refined parameters to fresh N(0, 1) draws (round 3:
~60% of the damage; structural mutation ~35%; activation ~10%).

## Test (grid `mechanism_final.json`, seeds 20–29 — never used by any decision)

10 x 100%, 30 generations, lr 0.3, both strategies, with
`neat_config.ini` (default) and `neat_config_noreplace.ini` (weight and
bias replace rates 0.1 -> 0, nothing else). Genome logs for Lamarckian,
both configs, spiral/spiral3, seeds 20–22 (`mechanism_final_logs.json`).

## Predictions (spirals)

- **F1 (mechanistic)** without replacement Lamarckian children keep more:
  median child loss (pre-refinement − parents' refined fit accuracy) is
  at least 0.03 less negative than with the default config.
- **F2 (consequence)** removing replacement helps Lamarckian more than
  Baldwinian: (lam_norep − lam_default) − (bald_norep − bald_default) > 0
  beyond spread (sqrt of the mean of the four seed variances).
  Contradicted if < 0 beyond spread; else inconclusive.
- **F3 (replication)** the main effect holds on fresh seeds:
  lam_default − bald_default > 0 beyond spread.

## Refutation

The replacement account is refuted if F1 fails (removing replacement does
not make children keep more). If F1 holds and F2 does not, replacement
explains the lossy carry-over but the lost refinement does not matter for
final accuracy at this budget. This is the last mechanism test.

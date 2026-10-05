# Mechanism round 2: why is Lamarckian inheritance lossy? — committed before running

## What round 1 left unexplained

Under Lamarckian inheritance a child's pre-refinement fit accuracy is a
median 0.129 below its parents' refined accuracy (round 1 predicted > −0.05;
contradicted). Baldwinian children are 0.179 below. Lamarckian inheritance
carries refinement forward, but most of it is lost each generation.

## Competing hypotheses

- **H3 — parameter mutation erodes it.** Every child has 80% of its
  weights and 70% of its biases perturbed (N(0, 0.5)), which undoes refined
  values.
- **H4 — crossover erodes it.** A child of two different parents takes
  each matching gene from either one; weights refined in two different
  networks do not fit together.

## Test A — observational, existing logs (runs/mechanism_logs, Lamarckian, seeds 0–2)

neat-python records a child's parents; when both are the same genome the
child is a clone (crossover of a genome with itself), otherwise a cross.
Statistic: median over children (generation >= 1, not carried-over elites)
of loss = child pre-refinement fit accuracy − mean of its parents'
post-refinement fit accuracy. d = median loss(clones) − median loss(crosses).

- H4 predicts **d >= +0.05** (clones keep clearly more).
- H3 predicts **|d| < 0.03** (clones lose about as much as crosses), and
  clones still lose >= 0.08.
- In between: neither clearly; both contribute.

Also reported: clones with unchanged node and connection counts (only
parameter mutation can have changed them) — H3 predicts these still lose
>= 0.08.

## Test B — manipulation (grid `mechanism2.json`, seeds 0–9; logs `mechanism2_logs.json`, seeds 0–2)

`neat_config_lowmut.ini`: weight_mutate_rate 0.8 -> 0.2 and bias_mutate_rate
0.7 -> 0.2, nothing else. Both strategies, 10 x 100%, 30 generations.
Compared with `lam_d10` / `bald_d10` from runs/mechanism (same seeds,
same allocation, default config).

- **B1 (H3)** Lamarckian children keep more: median loss > −0.08 under low
  mutation (default config: −0.129).
- **B2 (H3)** low mutation helps Lamarckian more than Baldwinian, on the
  spirals: interaction = (lam_lowmut − lam_d10) − (bald_lowmut − bald_d10)
  > 0 beyond spread, spread = sqrt(mean of the four seed variances).
  Contradicted if < 0 beyond spread; else inconclusive. (The interaction
  separates "inheritance is kept" from "less random search", which affects
  both strategies.)
- **B3 (control)** Baldwinian children still lose most: median loss < −0.10
  (they do not inherit refinement whatever the mutation rate).

## What would refute what

- H3 is refuted if Test A gives d >= +0.05 and clones lose < 0.08, or if
  B1 fails (lower mutation does not make Lamarckian children keep more).
- H4 is refuted if Test A gives |d| < 0.03.
- If B1 holds but B2 does not, keeping inheritance does not translate into
  accuracy, and the accuracy story needs something beyond H3.

## After this round

The narrowed claim is confirmed once on fresh seeds 20–29 (never used
by any decision), then the mechanism work stops.

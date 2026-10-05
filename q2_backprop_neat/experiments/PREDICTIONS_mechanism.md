# Lamarckian vs Baldwinian: mechanism — predictions committed before running

## What is being explained

Allocation grid (seeds 0–9) and confirmation (seeds 10–12): with the same
allocation (10 steps x 100%, 30 generations) Lamarckian beats Baldwinian by
+0.09–0.11 (spiral) and +0.14–0.15 (spiral3) held-out. The two strategies
prefer different allocations: Baldwinian gains from fewer, deeper
refinements (100 x top 10%: +0.04–0.06); Lamarckian does not. Lamarckian
runs also fragment into 6–49 species (Baldwinian 2–5) and 7/150 Lamarckian
runs crashed on neat-python's species limit.

## Hypothesis H1 — accumulation

Under Lamarckian inheritance a child starts from its parents' refined
weights, so refinement accumulates along a lineage: 10 steps per generation
add up over 30 generations. Under Baldwinian inheritance every evaluation
restarts from unrefined genome weights, so a genome's fitness reflects only
10 steps — which the pilot found ranks spiral genomes almost at random
(Spearman 0.02–0.27 vs 300 steps). Lamarckian wins because its effective
refinement depth is far larger; Baldwinian is held back by shallow,
noisy fitness.

## Hypothesis H2 — fragmentation is a side effect

Lamarckian weights drift (to the +-30 bound); NEAT's compatibility distance
includes weight differences, so species fragment. H2: this fragmentation is
a by-product of H1, not a cause of the accuracy difference.

## Experiments (grid `experiments/grids/mechanism.json`, seeds 0–9, all 30 generations, lr 0.3)

- **E1 — dose-response (tests H1).** Baldwinian with increasing depth per
  refined genome at fixed generations: `bald_d10` (10 x 100%), `bald_d30`
  (30 x 33%), `bald_d100` (100 x 10%), `bald_d300` (300 x 3%; spends 27,000
  of 30,000). Reference `lam_d10` (Lamarckian 10 x 100%).
- **E2 — speciation by topology only (tests H2).** `lam_topo`, `bald_topo`:
  10 x 100% with `neat_config_topo.ini` (compatibility_weight_coefficient
  0.5 -> 0; the only change).
- **E3 — head start (observational, H1).** `experiments/grids/mechanism_logs.json`:
  10 x 100%, both strategies, spiral/spiral3, seeds 0–2, `--log-genomes`.

## Predictions (spiral and spiral3; held-out unless stated)

spread = sqrt((sd_x² + sd_y²)/2) over seeds; "beyond spread" = |diff| > spread.

- **M1 (H1)** deeper refinement helps Baldwinian: bald_d300 − bald_d10 > 0
  beyond spread. Supported / contradicted (negative beyond spread) /
  inconclusive. Also reported: adjacent reversals beyond spread along
  d10 -> d30 -> d100 -> d300 (H1 predicts none).
- **M2 (H1)** depth closes the gap: gap(d) = lam_d10 − bald_d. Supported if
  min over d in {30, 100, 300} of gap(d) <= 0.5 x gap(d10); contradicted if
  every gap(d) >= gap(d10); otherwise partial.
- **M3 (H2)** under topology-only speciation, Lamarckian no longer
  fragments: no crashed lam_topo cell, and median (over runs) of the
  per-run maximum species count <= 5.
- **M4 (H2)** the Lamarckian advantage survives without fragmentation:
  lam_topo − bald_topo > 0 beyond spread. Also reported: its size relative
  to lam_d10 − bald_d10 (H2 predicts 0.5x–1.5x).
- **M5 (H1, E3)** children inherit refinement under Lamarckian only: over
  non-elite children in generations >= 1, median of (child pre-refinement
  fit accuracy − mean of its parents' post-refinement fit accuracy) is
  > −0.05 for Lamarckian and < −0.10 for Baldwinian; and the median
  refinement gain (post − pre) in generations 20–29 is smaller for
  Lamarckian than for Baldwinian.
- Consistency: `bald_d10`, `bald_d100`, `lam_d10` repeat allocation-grid
  configurations (A/B), so their per-seed results must equal those runs.

## What would refute what

- H1 is refuted if M1 is contradicted, or M2 is contradicted (depth does
  not close any of the gap), or M5 fails for Lamarckian (children do not
  start near their parents).
- H2 is refuted if M4 fails (the advantage disappears when speciation
  ignores weights): fragmentation would then be part of the cause.

These run once. Whatever comes out is reported; at most one labelled
follow-up if a result is ambiguous.

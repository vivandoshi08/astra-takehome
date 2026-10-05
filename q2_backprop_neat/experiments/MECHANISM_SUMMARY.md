# Lamarckian vs Baldwinian — what the mechanism work established

Each round's predictions were committed before it ran (PREDICTIONS_*.md);
verdicts below are by those pre-committed rules. Held-out accuracy,
spirals; xor is at its ceiling under every condition.

## The effect (replicated three times on disjoint seeds)

Same allocation (10 steps x 100%, 30 generations), Lamarckian − Baldwinian:
| seeds | spiral | spiral3 |
|---|--:|--:|
| 0–9 (allocation grid) | +0.088 | +0.148 |
| 10–12 (confirmation) | +0.111 | +0.140 |
| 20–29 (final test) | +0.065 | +0.153 |

## Round 1 — accumulation (H1) and fragmentation (H2)

- Baldwinian refined deeper at fixed generations closes the gap on spiral
  (0.815 -> 0.897 at 30 steps x 33%, vs Lamarckian 0.903) but only ~40% on
  spiral3; the dose-response is an inverted U (too few genomes refined at
  300 steps x 3%). H1 partly supported; depth trades against breadth.
- Lamarckian children start 0.129 below their parents' refined accuracy
  (predicted > −0.05: contradicted); Baldwinian 0.179. Carry-over exists
  but is lossy.
- Speciation by topology only removes Lamarckian fragmentation (median 1
  species, no crashes) and the advantage survives on spiral3 (+0.109) but
  halves on spiral (+0.047, inconclusive); Lamarckian loses ~0.04 without
  weight-aware speciation, Baldwinian nothing. "Pure side effect" (H2)
  partly refuted.

## Round 2 — why is the carry-over lossy?

- Crossover (H4) refuted: clones lose as much as crosses (0.125 vs 0.129).
- Perturbation rate (H3 as stated) refuted: weight/bias perturbation
  0.8/0.7 -> 0.2 leaves the loss at 0.125.

## Round 3 — which mutation operator?

Mutating refined genomes one operator at a time: replacement does 60% of
the damage, structural 32–38%, perturbation 28–34%, activation 8–10%
(overlapping). Replacement (H5) supported; activation (H6) not.

## Final test (fresh seeds 20–29)

Replacement switched off: children keep 0.025 more of their parents'
refinement (5 of 6 runs; predicted >= 0.03: contradicted, direction as
predicted). Final accuracy changes are within noise for both strategies
(F2 inconclusive). The main effect replicates (F3 supported).

## Where this leaves the explanation

- Supported: Lamarckian's advantage is robust; inherited refinement gives
  children a partial head start (~0.05 over Baldwinian); Baldwinian's
  handicap is shallow, noisy fitness, and it recovers part of the gap by
  refining fewer genomes deeper.
- The carry-over is eroded by mutation as a whole — no single operator
  accounts for it (replacement is the largest in isolation but removing
  it recovers only ~0.025), and recovering it does not measurably change
  final accuracy at this budget.
- Not explained: why Lamarckian benefits from weight-aware speciation, and
  why depth closes the gap on spiral but not spiral3.

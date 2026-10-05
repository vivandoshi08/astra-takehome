# Complexity frontier — predictions committed before running

## Penalty (what is implemented, and why)

Fitness = penalised(fit accuracy after refinement, C), C = enabled
connections (`network.complexity`, the count the question uses). The
champion is the generation-best with the highest *penalised* validation
accuracy (same lambda and form), so selection does not undo the penalty.

- **linear** (main sweep): acc − λ·C. Every connection costs λ accuracy
  whatever the network's size or accuracy; λ reads directly as "accuracy a
  connection must buy to stay". The same pressure on every dataset.
- **mult** (second sweep): 1 − (1 − acc)·(1 + λ·√C) — one reading of the
  backprop-NEAT write-up ("multiply the fitness score by a factor that
  grows in proportion to the square root of the number of connections",
  applied to the error so that larger means worse). The pressure scales
  with the error: an accurate network pays almost nothing.

## Design (grid `experiments/grids/frontier.json`)

Settled allocations: Lamarckian 10 steps x 100%; Baldwinian 100 steps x
top 10%; lr 0.3; 30 generations; seeds 0–9; xor / spiral / spiral3.
linear λ = 0, 0.001, 0.003, 0.01, 0.03; mult λ = 0.1, 0.3, 1.0.
At λ = 0 champions have 8–12 connections (allocation grid), so linear
λ = 0.03 costs ~0.3 of fitness at today's size, λ = 0.001 ~0.01.
λ = 0 repeats the allocation grid's A_lamarckian / B_baldwinian cells and
must reproduce them exactly.

## Predictions (spread = sqrt((sd_x² + sd_y²)/2) over seeds)

- **FP1** the penalty shrinks networks: mean champion connections at
  linear λ = 0.03 < at λ = 0 beyond spread, in all 6 dataset x strategy
  cells.
- **FP2** monotone: along linear λ, no step to a larger λ increases mean
  connections beyond spread.
- **FP3** the price differs by dataset: at linear λ = 0.01, held-out vs
  λ = 0 is within spread on xor and drops beyond spread on both spirals,
  for both strategies (the spirals need more connections; capacity check
  in the session: a plain 2-H-1 net needs H >= 4 for spiral, 8 for spiral3).
- **FP4** no free lunch: no penalised arm beats λ = 0 on held-out beyond
  spread in any dataset x strategy cell (the networks are already small;
  the penalty buys size, not accuracy).
- **FP5** the mult form shrinks accurate networks less: the relative
  reduction in connections at mult λ = 1.0 vs λ = 0 is smaller on xor than
  on spiral3, for both strategies.
- **FP6** the Lamarckian advantage persists along the linear sweep:
  Lamarckian held-out mean >= Baldwinian at every λ on both spirals
  (count of the 10 dataset x λ cells reported).

# Small-network track (<= 6 enabled connections) — predictions committed before running

## Design (grid `experiments/grids/smallnet.json`, written by the agent that built --max-conns)

`--max-conns 6`, lr 0.3, seeds 0–9, xor / spiral / spiral3, the settled
allocations: `lam_std` (Lamarckian 10 x 100%) and `bald_std` (Baldwinian
100 x top 10%) with `neat_config.ini`; `lam_sine` / `bald_sine` the same with
`neat_config_sine.ini` (hidden activations sigmoid tanh relu **sin gauss**).
Every genome is repaired at creation to <= 6 enabled connections
(MaxConnsGenome), so every champion is within the limit.
The unconstrained reference is the allocation grid's A_lamarckian /
B_baldwinian cells (same allocations, same seeds).

The sine arms are a separate, labelled result: adding activations changes
NEAT's toolbox, so they are not comparable with the unconstrained numbers
as "the same method, smaller".

## Disclosure

Three graded cells (lam_std spiral s0, lam_std spiral3 s4, bald_sine spiral
s5) were run by the building agent during development, before these
predictions. It did not report their numbers; these predictions were
written without them. Every verdict is also reported with those three cells
excluded.

## What the predictions rest on

Frontier (`runs/frontier`): Lamarckian champions near 5–6 connections under
a penalty scored xor 0.928 (5.0 conns), spiral 0.855 (5.8), spiral3 0.726
(4.2) – 0.779 (6.8). A fully connected 2-H-1 tanh network with H = 2 (6
connections) trained to convergence reached validation 0.91 / 0.81 / 0.71
(session capacity check, quoted in PREDICTIONS_frontier.md).

## Predictions (held-out; spread = sqrt((sd_x² + sd_y²)/2) over seeds)

- **S1a** xor: the cap costs little — for both strategies, std(<=6) vs
  unconstrained differs by no more than max(spread, 0.02).
- **S1b** spirals: the cap costs accuracy — std(<=6) < unconstrained beyond
  spread, both strategies, both spirals.
- **S2** sine helps on the spirals under the cap: sine − std > 0 beyond
  spread, both strategies, both spirals. (A sin unit draws repeated bands,
  so a few connections can cut through several arm crossings.)
- **S3** sine does not help xor: sine − std within spread, both strategies.
- **S4** the Lamarckian advantage persists under the cap: lam_std >
  bald_std on both spirals (sign).

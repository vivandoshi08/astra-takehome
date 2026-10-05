# Confirmation on untouched seeds 10–12 — claims committed before running

Allocations settled on from the allocation grid (seeds 0–9), chosen by
validation accuracy among arms with no crashed cells:
- Lamarckian: **A** (10 steps x 100% refined, 30 generations)
- Baldwinian: **B** (100 steps x top 10% refined, 30 generations)

Grid `experiments/grids/confirm.json`: A and B x both strategies x
xor/spiral/spiral3 x seeds 10–12 (no decision has used these seeds).
Scored by `experiments/score_confirm.py` on held-out accuracy.

Claims from seeds 0–9, each checked on both spirals by the sign of the
seed-10–12 mean difference (three seeds: a sign check, not a significance
test; the size is reported next to the seeds-0–9 value):
- **C1** Lamarckian A > Baldwinian A (seeds 0–9: +0.088 spiral, +0.148 spiral3)
- **C2** Baldwinian B > Baldwinian A (+0.053, +0.063)
- **C3** Lamarckian B < Lamarckian A (−0.026, −0.017; small, may flip)
- **C4** the chosen pair: Lamarckian A > Baldwinian B (+0.034, +0.085)

Expected held-out ranges for the chosen allocations (seeds 0–9 mean ± sd):
Lamarckian A 0.903 ± 0.069 / 0.825 ± 0.055; Baldwinian B 0.869 ± 0.057 /
0.740 ± 0.060 (spiral / spiral3). xor: all four arms 0.96–0.98.

# Q3 spending plan

Written and committed before any loop run. The $10 OpenRouter key is the
binding constraint; this file says how it is divided, why, when to stop, and
what we expect each group to show. Results are compared against it in
`report.md`.

## What a run costs

From `runs/probe-01` (two real calls with the loop's real prompt): a call is
~1,200 input tokens at generation 0 (more later, as the prompt carries three
programs) and ~1,000 output tokens, $0.006–0.008. A default run is
12 generations × 4 candidates = 48 calls, so **~$0.40**, as the README
says. Reasoning effort "low" works on Haiku 4.5 and adds ~400 output tokens
per call, ~25% per run. `spend.py` matches OpenRouter's own per-call cost
exactly.

## The division

| | group | purpose | runs | est. $ | cumulative |
|---|---|---|---|---|---|
| 0 | probe (done) | API works; reasoning works; cost per call | 2 calls | 0.01 | 0.01 |
| 1 | **A: shipped loop** | how much the loop varies by itself (deliverable 3, arm 1) | 5 | 2.00 | 2.01 |
| 2 | **B: one change** | does the change beat that variation? (arm 2) | 5 | 2.00–2.50 | ~4.50 |
| 3 | C: finding | only if the finding needs new runs to test it | 0–4 | 0–1.60 | ~6.10 |
| 4 | extension | if A vs B is borderline, 3 more of each; otherwise more runs where the evidence is weakest | 3–6 | 1.20–2.40 | ~8.50 |
| 5 | margin | crashed runs, the provider meter's lag | — | ~1.00 | 10.00 |

Why this shape: the A-vs-B comparison is the one result graded on its
spread, so it gets nearly half the money up front, and the extension is
held back to strengthen whichever conclusion is weakest at the end.
Five runs per arm rather than the minimum three: three values barely show
a spread, and with five against five a rank test can show a clean
separation (chance ≈ 1 in 126 two-sided); the extra four runs cost ~$1.60.
Model: Haiku 4.5 throughout unless group A shows it plateauing well short
of the targets — a Sonnet arm would cost ~$4 and most of the extension.

## Stop rules

1. **After the first run (a01) only**: check that every file is written,
   the cost is ≤ $0.60, and `spend.py` agrees with the provider. Fix before
   spending more. a02–a05 start only after this.
2. After every group: `spend.py` and the key's meter, then commit the runs.
3. Any run costing more than $0.60: pause and find out why.
4. Never start a run that could take the total past $9.50 (the provider's
   meter settles late).

## Predictions for group A (shipped loop, 5 runs)

Written before a01. The probe's first answer (−0.1405, worse than the SEED)
lowered my expectations for early generations; the loop keeps the SEED in
the population, so its best can only improve.

- Champion score on PUBLIC: median about **−0.013**, range −0.010 to −0.017.
- Spread between best and worst run: **at least 0.002** — one run would
  prove nothing.
- Each champion on HELD_OUT within **±0.0005** of its PUBLIC score
  (programs are generic; little to overfit).
- At least **85%** of answers valid; under **5%** cut off at 2,048 tokens.
- **$0.35–0.45** per run.
- No champion beats the honest optimum (−0.0096) on HELD_OUT; any that does
  is investigated as a leak before anything else.

Group B's change is chosen after group A, from its logs, with its own
prediction committed here before its first run.

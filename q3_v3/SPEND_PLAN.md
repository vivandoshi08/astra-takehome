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

## After group A: groups B, C1, C2 (written before they run)

Group A (shipped loop × 5, $1.96): median −0.0173, spread 0.0098; no
duplicates in the prompt, no cut-off answers, little hidden progress behind
the 4-decimal display — so deduplication, more decimals and a longer answer
limit have nothing to fix. What the logs do show: almost all of every
champion's error is in its encoder (`decompose_champions.py`), and 0 of 173
logged encoders ever draws an ellipse or searches. Encode *can* call decode
(both are compiled from the same source); a ten-line search added to the
SEED alone scores −0.0156, better than A's median.

| group | the one change vs group A | runs | est. $ | question |
|---|---|---|---|---|
| **B** | `--prompt-hint`: one sentence saying encode can call decode, e.g. to adjust its six numbers until decode(z) matches | 5 | ~2.25 | deliverable 3's change: is the loop stuck because Haiku never *thinks of* fitting? |
| C1 | `EVO_REASONING=low` | 5 | ~2.50 | alternative: does thinking first find it unprompted? |
| C2 | `EVO_MODEL=anthropic/claude-sonnet-5` | 3 | ~2.40 | alternative: does a stronger model find it unprompted? |

All 13 run in parallel. Expected total after them ≈ $9.10. `spend_guard.py
--limit 9.50` stops any still-running run cleanly if the real spend gets
there (stop rule 4).

**Predictions.** "Searching encoder" = an encoder that calls decode or a
helper that draws, or loops over candidate parameter values comparing errors.

- **B (hint):** at least a third of answers from generation 1 on have a
  searching encoder. Champion median **≤ −0.0120**; all five better than A's
  median (−0.0173) and at least four better than A's best (−0.0146), so the
  difference survives the spread (rank test p ≲ 0.03). HELD_OUT within
  ±0.0005 of PUBLIC. Some answers time out (< 10%). $0.40–0.55 per run.
- **C1 (reasoning):** searching encoders under 5%. Median about −0.016;
  ranges overlap A's; the difference does not survive the spread.
  $0.45–0.55 per run.
- **C2 (Sonnet):** some searching encoders (10–40%). Median about −0.013,
  better than A's; with 3 runs against 5 no rank test can be decisive
  (smallest two-sided p = 2/56 ≈ 0.036). $0.70–0.95 per run.

## C2 revised: GPT-6 Luna instead of Sonnet (written before the runs)

Sonnet 5 was stopped after 7 calls a run (every answer cut off by its own
reasoning; DEVLOG). The user chose an OpenAI model instead: `openai/gpt-6-luna`
($0.10 / $0.50 per Mtok). Probes (`runs/probe-gpt6luna`, `runs/probe-gpt41mini`):

- Luna with shipped settings has Sonnet's problem: 1,996 reasoning tokens,
  cut off, no code.
- Luna with `EVO_REASONING=low`: 701 reasoning tokens, complete answer, and
  on its **first** try an encoder that renders the ellipse and searches its
  parameters (coordinate descent) — −0.00977 PUBLIC, −0.00983 HELD_OUT, no
  suspicious imports. Unprompted.

So C2 runs Luna with reasoning "low". Its clean one-change comparison is
**C1** (Haiku, reasoning "low"): only the model differs. ~$0.001 a call, so
five runs cost ~$0.50.

| group | change | runs | est. $ |
|---|---|---|---|
| C2 | `EVO_MODEL=openai/gpt-6-luna EVO_REASONING=low` | 5 | ~0.50 |

**Predictions (C2):** most answers (> 50%) from generation 1 on have a
searching encoder. Champion median ≤ −0.0100; all five better than all five
C1 champions (rank test p ≈ 0.008). HELD_OUT within ±0.0003 of PUBLIC.
5–20% of answers cut off (reasoning length varies). $0.05–0.15 per run.

# Q3 — Race the optimizer loop

## Summary

- **Our model** fits the generator's six numbers to each image. It scores
  **−0.0096 on PUBLIC and −0.0097 on HELD_OUT**: Bonus tier, at the noise
  floor.
- **The shipped loop** (Claude Haiku 4.5, 12 generations, run five times)
  has a champion median of **−0.0173**, about PCA's level. Champions from
  identical settings span 0.0098.
- **The one change** is one sentence in the prompt saying encode may call
  decode. It moves the median to **−0.0102** over seven runs (exact rank
  test p = 0.010). Reasoning effort "low" changes nothing (p = 1.0).
- **The incumbent VAE's −0.0051** comes from memorising PUBLIC. It is not a
  six-number model.
- **The finding:** Haiku never thinks of fitting. None of its 480 encoders
  without the hint search. GPT-6 Luna searches unprompted and sits at the
  floor.
- **Spend:** $8.98 of $10 over 25 loop runs and 3 probes.

Everything was committed as it happened:
- `DEVLOG.md` is the running log.
- `AUDIT.md` is the pre-flight audit.
- `SPEND_PLAN.md` holds the plan and the predictions, each written before
  the money it covers was spent.

---

## 1. Our model and the race (deliverable 1)

**How it works.** `student.py` is about 140 lines of numpy, half of them comments. Every image is
`data.render(cx, cy, a, b, θ, A)` plus N(0, 0.1²) noise, so the model works
by analysis by synthesis: draw a guess, compare it with the image, adjust.

- **encode** starts from a first guess built from the bright pixels:
  - the centre is their mean position;
  - the axes are 2√λ of their covariance, because a filled ellipse with
    semi-axes a, b has variances a²/4 and b²/4;
  - the tilt comes from the same covariance;
  - the brightness is fitted by least squares given that shape.
- It then refines all six numbers by Levenberg–Marquardt through
  `data.render`. It starts from three tilts, because the tilt of a
  near-circle is barely visible.
- **decode** is `data.render`. The README allows numpy and `data.render`.
- **Development:** the few settings were chosen on fresh `train_images`
  (seeds 7 and 2024). PUBLIC and HELD_OUT were used only to score it.

**The race.** Scored by `race.py` with the task's own evaluator. Tables are
in `figures/race_public.txt` and `figures/race_held_out.txt`, and the
reconstruction plots in `figures/race_*.png`. The race covers the best
champion of each group; every run's champion is scored on both sets in its
own `run_meta.json`, by the same evaluator.

| entry | PUBLIC | HELD_OUT |
|---|---|---|
| **our model** | **−0.0096** | **−0.0097** |
| vae (k=6), incumbent | −0.0051 | −0.0346 |
| pca (k=6) | −0.0173 | −0.0195 |
| mean | −0.0454 | −0.0444 |
| best champion, shipped loop (a02) | −0.0146 | −0.0146 |
| best champion, + reasoning (r01) | −0.0131 | −0.0118 |
| best champion, + hint (b07) | −0.0097 | −0.0097 |
| best champion, GPT-6 Luna (l03) | −0.0096 | −0.0097 |

**What both actually do** (`compare_models.py`; `figures/compare_models.png`
shows eight PUBLIC images through each model, with error maps on one colour
scale). Our model, the best hint champion and the best Luna champion draw
crisp ellipses of the right size and tilt, and their error maps are noise.
The best shipped champion (a02) draws soft, blurred ellipses. It makes a
small ellipse too large, which leaves a bright ring of error around it, and
it tilts a round one. Measured against the *clean* image, so that only each
model's own mistakes remain:

| model | error per pixel: interior | rim | background | share of its error on the rim |
|---|---|---|---|---|
| our model | 0.0007 | 0.0027 | 0.00001 | 81% |
| shipped best (a02) | **0.0187** | **0.0231** | **0.0025** | 35% |
| hint best (b07) | 0.0008 | 0.0038 | 0.00000 | 85% |
| Luna best (l03) | 0.0007 | 0.0024 | 0.00001 | 79% |

The shipped champion is wrong everywhere: its interior error is 28× ours,
and its blur spills light into the background. The three models that search
have little left except at the rim, the hardest pixels.

The blur is not an accident. Given a02's own encoder, the generator's sharp
decoder scores worse than a02's blurred one (−0.0157 against −0.0146), and
the same holds for a04 (−0.0250 against −0.0178) and a05 (−0.0174 against
−0.0165). A soft rim hedges against an encoder that misjudges size. Haiku
improved the drawing to hide the measuring, the same story the finding
tells.

**Where its errors come from.** Almost entirely the pixel noise.

- **The floor.** A six-number fit can also soak up the noise along the
  directions it moves in. So the floor is about −σ²(1 − 6/256) = −0.0098 in
  expectation, and −0.0096 on PUBLIC, whose noise happens to be a little
  low. It is an approximate floor, not a bound: a nonlinear fit absorbs
  roughly 6.5 directions' worth.
- **Wrong valleys.** Fits that stop in a worse local minimum cost about
  5 × 10⁻⁶ against a best-of-many-starts fit.
- **Swapped axes.** In 15–20% of images the best fit has |b| > |a|. The
  generator never draws that shape, but it matches the noisy image slightly
  better.

**Independent check.** A separate agent tried to break the model:

- Its output is bit-identical with every dataset array overwritten with
  garbage.
- It scores at the floor on 10 unused seeds and on flipped, rotated and
  shifted copies of PUBLIC.
- Rounding its latents to float32 changes nothing, so nothing is hidden in
  the low digits.
- It caught three wrong claims of mine about the floor (section 7, item 2).

Scripts and outputs are in `audit/student/`.

## 2. Runs, and how the money was spent (deliverable 2)

Spend is from `python spend.py`, which prices each call by the model named
in its own log tag; mock runs are free. The provider's own meter agrees:
$8.95 against $8.98, the gap being calls that had not settled.

| group | what it was for | runs | $ | best / median champion (PUBLIC) |
|---|---|---|---|---|
| probes | key works, real cost per call, reasoning on Haiku, OpenAI candidates | 3 × 2 calls | 0.02 | — |
| **A** shipped loop | how much the loop varies by itself | 5 | 1.96 | −0.0146 / −0.0173 |
| **B** + prompt hint | deliverable 3's one change, plus a 2-run extension | 7 | 3.73 | −0.0097 / −0.0102 |
| **C1** + reasoning "low" | does thinking alone find fitting? | 5 | 2.62 | −0.0131 / −0.0175 |
| C2a Sonnet 5 | does a stronger model find it? | 3, stopped | 0.46 | nothing valid |
| **C2** GPT-6 Luna + reasoning "low" | does another vendor's model find it? | 5, stopped early | 0.19 | −0.0096 / −0.0097 |
| | | | **8.98** | |

**How I decided.** The plan in `SPEND_PLAN.md` was committed before the
first loop run.

- A probe came first ($0.014), to learn what a call really costs: about
  $0.40 a run.
- Nearly half the money went to the A-vs-B comparison, because deliverable
  3 is graded on its spread. Five runs an arm rather than three, because
  with three there is barely a spread to see.
- An extension was held back for whichever comparison ended up borderline,
  plus about $1 of margin.
- A guard (`spend_guard.py`) would stop every run cleanly at $9.50.
- B's change was chosen only after A, from A's logs:
  - no duplicate programs ever appeared in the prompt;
  - no answer was ever cut off;
  - the 4-decimal display hid real progress only 5 times;
  - but every champion's error sat in its encoder.

  Its prediction was written down first.

**What it bought.**
- **A** measured the variance (0.0098 between runs), so one run proves
  nothing.
- **B** found the effect.
- **C1** ruled out "more thinking".
- **C2** showed the behaviour is specific to Haiku, not to LLM optimisers.
- **The extension** made the borderline A-vs-B comparison clearer: p went
  from 0.032 to 0.010.

**The runs that went nowhere.** Sonnet 5 is the upgrade the README names.
With the shipped client, all 20 of its answers were cut off. Sonnet reasons
by default, and its hidden reasoning (1,062–2,048 tokens) shares
`MAX_TOKENS = 2048` with the answer. I stopped it after 6–7 calls a run, at
$0.46. Fixing it would have been a second change (reasoning off, or a
larger limit), so it was dropped.

**Changing model.** At the user's choice the slot went to an OpenAI model.
OpenAI's reasoning models have the same trap, but GPT-6 Luna works with
reasoning "low", at about $0.001 a call. Its clean comparison is C1, where
only the model differs.

The Luna runs were stopped after 5–9 of their 12 generations, to finish in time. Every
one had sat at −0.0096 or −0.0097 since generation 1 or 2, and a
champion's score can only improve, so the stop costs no conclusion. Two of
their one-off HELD_OUT scorings timed out under machine load; I re-scored
those champions with a longer limit (`rescore_held_out.py`, kept beside the
untouched logs).

**What is left: $1.02, on purpose.** A placebo arm, the one control I would
buy next, costs about $2.50. One more run of anything would not change a
conclusion. The guard's margin covers calls that settle late.

## 3. Run-to-run variance, and one setting (deliverable 3)

These are champion scores on PUBLIC. HELD_OUT gives the same picture and
the same p-values. `compare_groups.py` runs an exact two-sided Mann–Whitney
test over every way of splitting the runs, so with 5 against 5 the smallest
possible p is 0.008.

| group | champions | median | spread |
|---|---|---|---|
| A shipped | −0.0146, −0.0165, −0.0173, −0.0178, −0.0244 | −0.0173 | 0.0098 |
| B + hint | −0.0097, −0.0098, −0.0099, −0.0102, −0.0114, −0.0118, −0.0167 | −0.0102 | 0.0070 |
| C1 + reasoning | −0.0131, −0.0151, −0.0175, −0.0213, −0.0226 | −0.0175 | 0.0095 |
| C2 Luna + reasoning | −0.0096, −0.0097, −0.0097, −0.0097, −0.0097 | −0.0097 | 0.0001 |

- **The shipped loop's own spread is large.** Identical settings produced
  anything from "beats PCA by 15%" to "40% worse than PCA". A single run
  per setting cannot tell two settings apart.
- **B vs A** (the one change):
  - A B champion beats an A champion in 33 of 35 pairings, **p = 0.010**.
  - The ranges still overlap, through one run: b05, whose encoders rarely
    searched (6 of 48 answers).
  - So the effect survives the spread when measured by rank. It is not a
    clean separation run for run. The b05 exception is itself evidence for
    the mechanism (section 5).
- **C1 vs A:** p = 1.00. Not separated.
- **C2 vs C1** (only the model differs): every Luna champion beats every
  Haiku one, **p = 0.008, no overlap**.

**Predictions against results** (from `SPEND_PLAN.md`, written before each
group ran):

| | prediction | result |
|---|---|---|
| A | median about −0.013 | ✗ −0.0173 (too optimistic) |
| A | spread at least 0.002 | ✓ 0.0098 |
| A | HELD_OUT within ±0.0005 of PUBLIC | ✓ in 4 of 5 runs |
| A | at least 85% of answers valid | ✓ 100% |
| A | $0.35–0.45 a run | ✓ |
| B | at least a third of encoders search | ✓ 72% |
| B | median at most −0.0120 | ✓ −0.0102 |
| B | at least 4 of 5 beat A's best | ✓ 4 of 5, then 6 of 7 |
| B | p ≲ 0.03 | ✓ 0.032 (5 runs), 0.010 (7 runs) |
| B | $0.40–0.55 a run | ✗ up to $0.66: searching programs are longer |
| extension | both runs search and beat A's best | ✓ (b06, b07) |
| C1 | under 5% of encoders search, no separation | ✓ 0% and p = 1.0 |
| C2 | more than half of encoders search | ✓ 94% |
| C2 | median at most −0.0100, all five beat C1 | ✓ −0.0097, p = 0.008 |
| C2 | 5–20% of answers cut off | ✓ 12% |
| C2 | $0.05–0.15 a run | ✗ $0.03–0.05, but the runs were stopped early |

## 4. Verdict on the incumbent (deliverable 4)

The VAE's −0.0051 is not a plausible score for a six-number model on this
data. It is memorisation of the 128 PUBLIC images it was trained on:

- It scores past the noise floor.
- It reproduces 59% of each PUBLIC image's own noise, against 2.4% on
  HELD_OUT.
- Retrained the same way on 128 *other* images, it scores −0.0049 on those
  and −0.0345 on PUBLIC.

`finding.md` gives the argument and the control. What follows from it:
every result here is judged against the honest floor and reported on
HELD_OUT as well.

## 5. The finding (deliverable 5)

See `finding.md`. In one line: the loop's ceiling is a missing idea, not
missing ability.

- Haiku never writes an encoder that searches through its own decoder:
  0 of 480 without the hint, with or without reasoning.
- Given one sentence saying it may, it does so in 72% of answers and
  reaches the floor.
- Another vendor's model does it unprompted.
- Splitting each champion's error into encoder and decoder parts puts
  almost all of the gap in the encoder: it loses 0.0020–0.0144 for A and C1
  (the decoder 0.0001–0.0040), and 0.0000–0.0002 for the B runs that
  search.

## 6. Before spending anything: the audit

Two independent passes over the shipped code, plus my own re-runs of the
headline checks. The full list is in `AUDIT.md`, reproducible with
`bash audit/run_all.sh`. What mattered:

- **"The six numbers are the only channel" was false nine ways.** Shared
  builtins, module attributes, `sys.modules`, frame inspection and
  `import data` each scored a perfect 0.0000. Candidate code could also
  patch the evaluator for the rest of the run.
- **A failed API call lost calls that had already been billed.** They went
  missing from `token_log.jsonl`, and the run died without writing
  `best.py`. The likeliest trigger is a 402 when the key runs dry.
- **`spend.py` priced every run at one model's rate**, and counted mock runs
  as spend.

I fixed only what cannot change what the loop searches:

- Candidates are scored in fresh subprocesses (`evo/sandbox.py`).
- The model is called, and charged, one call at a time.
- Every answer and every setting is logged.

The evidence that these fixes are neutral:

- An honest candidate scores bit-identically to the shipped path, checked
  on 7 programs and both image sets.
- `sanity_check.sh` still passes.
- Each fix has a test in `tests/test_loop_fixes.py` (14/14 pass).

The prompt, the population rule and the code regex were left as shipped, so
group A is the shipped loop. The hint is a flag (`--prompt-hint`); with it
off, the prompt is byte-identical to the shipped one.

## 7. What went wrong, and how I found it

1. **Two of my own claims about the floor were wrong.**
   - I said −0.0100 was the best possible score. The audit showed a fit
     beats the clean truth (−0.009597 against −0.009833).
   - I said six numbers cannot carry the noise. A bit-packing encoder that
     hides it in the float64 digits scores −0.0043 on PUBLIC *and*
     HELD_OUT.

   I rebuilt the VAE verdict on better evidence: the noise-copying fraction
   and the retraining control.
2. **Three more claims of mine, caught by the independent check of
   `student.py`.**
   - I wrote the formula value as −0.0096; it is −0.0098.
   - I called it a bound; it is an approximate floor.
   - I wrote that the model "beats a fit started from the true parameters".
     Those wins came from swapped-axis fits, not escaped local minima.

   The check reproduced every number; it was my reading of them that was
   wrong.
3. **A shell bug launched two runs instead of four.** zsh does not split an
   unquoted `$pair`, so `"a02 a03"` became a single run id. I found it when
   `summarize_runs.py` printed run names with spaces and a per-run cost
   check could not find `runs/a02/`. Nothing was lost; the directories were
   renamed and the rename logged. The same trap later broke a `race.py`
   call. Launches now go through an explicit script, checked with `ps`.
4. **Sonnet's answers were all cut off.** The sign was three runs at
   `valid 0/4` with identical token counts. Their `usage` showed the
   reasoning tokens. I stopped it at $0.46. I first wrote 21 calls and
   $0.49; the logs say 20 and $0.4646, and the DEVLOG is corrected.
5. **My search classifier missed renderers nested inside `encode`.** A Luna
   run sat at −0.0096 with "0 searching encoders", which is impossible for
   an encoder that only measures. I read its code, fixed the classifier,
   and checked that Haiku's 0 of 480 did not change.
6. **Two load and scheduling slips.**
   - I restarted the spend guard before adding Luna's price, so it would
     have priced Luna about 100 times too high and stopped every run.
     Caught before launch.
   - I twice ran a CPU-heavy analysis alongside paid runs. The sandbox's
     limit is wall-clock time, so the load contributed to Luna's timeouts
     (26 of 145 answers; its searching programs are slow anyway) and made
     two champions' HELD_OUT scoring time out. Both champions were
     re-scored.
7. **My prediction for the shipped loop was too optimistic** (−0.013
   against −0.0173). Haiku's very first answer, at −0.14, had the right
   method and two wrong constants. That should have warned me.
8. **The audit agents wrote into the repo twice**: a VAE cache file and
   some empty folders. Both were removed. The shipped cache is
   byte-identical by sha1.

## 8. Time and limits

**Time:** about 3.5 hours against the suggested 5. The audit took the first
hour.

**Limits:**
- Five to seven runs an arm detects large effects only.
- The hint arm has no placebo control (`finding.md`).
- The sandbox's limit is wall-clock, so timeouts depend on machine load.
  Searching programs are slow, so this mostly hit Luna.
- The sandbox keeps the key and the repo's modules out of a candidate's
  reach, but does not jail files or the network.

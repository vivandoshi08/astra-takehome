# Q3 — Race the optimizer loop

**Summary.**
- **Our model** fits the generator's six numbers to each image. It scores
  **−0.0096 on PUBLIC and −0.0097 on HELD_OUT**: the Bonus tier, and at the
  noise floor.
- **The shipped loop** (Claude Haiku 4.5, 12 generations), run five times:
  champion median **−0.0173**, about PCA's level, with a spread of 0.0098
  between runs.
- **The one change:** one sentence in the prompt saying encode may call
  decode. It moves the median to [[PENDING:B_MEDIAN]] (exact rank test vs
  the shipped loop [[PENDING:B_P]]). Reasoning effort "low" changes nothing
  (p = 1.0).
- **The incumbent VAE's −0.0051** is memorisation of PUBLIC, not a
  six-number model.
- **The finding:** Haiku never thinks of fitting. All 480 of its encoders
  without the hint only measure. GPT-6 Luna fits unprompted.
- **Spend:** [[PENDING:SPEND]] of $10 over [[PENDING:RUNS]].

Everything is committed as it happened. `DEVLOG.md` is the running log,
`AUDIT.md` the pre-flight audit, and `SPEND_PLAN.md` the plan and
predictions, each written before the money they cover was spent.

---

## 1. Our model and the race (deliverable 1)

**How it works** (`student.py`, ~100 lines, numpy only). Every image is
`data.render(cx, cy, a, b, θ, A)` plus N(0, 0.1²) noise, so the model is
analysis by synthesis.

- **encode** makes a first guess from the bright pixels. The centre is their
  mean position. The axes are 2√λ of their covariance, because a filled
  ellipse with semi-axes a, b has variances a²/4 and b²/4. The tilt comes
  from the covariance, and the brightness from least squares given that
  shape.
- It then refines all six numbers by Levenberg–Marquardt through
  `data.render`, starting from three tilts, because a near-circle's tilt is
  barely visible.
- **decode** is `data.render`. The README allows numpy and `data.render`.
- **Development.** The few settings were chosen on fresh `train_images`
  (seeds 7 and 2024). PUBLIC and HELD_OUT were used only to score it.

**The race** (`race.py`, the task's own evaluator; full tables in
`figures/race_public.txt` and `figures/race_held_out.txt`, reconstruction
plots in `figures/race_*.png`):

| entry | PUBLIC | HELD_OUT |
|---|---|---|
| **our model** | **−0.0096** | **−0.0097** |
| vae (k=6), incumbent | −0.0051 | −0.0346 |
| pca (k=6) | −0.0173 | −0.0195 |
| mean | −0.0454 | −0.0444 |
| best champion, shipped loop (a02) | −0.0146 | −0.0146 |
| best champion, + hint (b04) | −0.0098 | −0.0099 |
| best champion, + reasoning (r01) | −0.0131 | −0.0118 |
| best champion, GPT-6 Luna (C2) | [[PENDING:L_BEST]] | [[PENDING:L_BEST_H]] |

**Where its errors come from.** Almost entirely the pixel noise.

- **The floor.** A fit of six numbers absorbs the noise along the directions
  it can move in, so the floor is about −σ²(1 − 6/256) = −0.0098 in
  expectation. On PUBLIC, whose noise happens to be a little low, it is
  −0.0096. This is an approximate floor, not a bound: a nonlinear fit
  absorbs roughly 6.5 directions' worth.
- **Wrong valleys.** Fits that stop in a worse local minimum cost about
  5 × 10⁻⁶ against a best-of-many-starts fit.
- **Swapped axes.** In 15–20% of images the best fit has |b| > |a|. The
  generator never draws that, but it matches the noisy image slightly
  better.

**Checked independently.** A separate agent tried to break the model.
- Its output is bit-identical when every dataset array is overwritten with
  garbage.
- It scores at the floor on 10 unused seeds and on flipped, rotated and
  shifted copies of PUBLIC.
- Rounding its latents to float32 changes nothing, so nothing is hidden in
  the low digits.
- It caught three wrong claims of mine about the floor (section 7).

Scripts and outputs are in `audit/student/`.

## 2. Runs, and how the money was spent (deliverable 2)

`python spend.py` (each call priced by the model in its own tag; mock runs
free). [[PENDING:SPEND_TABLE]]

| group | what it was for | runs | $ |
|---|---|---|---|
| probes | key works; real cost per call; reasoning on Haiku; OpenAI candidates | 3 × 2 calls | 0.02 |
| **A** shipped loop | how much the loop varies by itself | 5 | 1.97 |
| **B** + prompt hint | deliverable 3's one change | 5 + 2 | [[PENDING:B_COST]] |
| **C1** + reasoning "low" | does thinking alone find fitting? | 5 | 2.62 |
| C2a Sonnet 5 | does a stronger model find it? | 3, stopped | 0.46 |
| **C2** GPT-6 Luna + reasoning "low" | does another vendor's model find it? | 5 | [[PENDING:L_COST]] |

**How I decided.** The plan (`SPEND_PLAN.md`) was committed before the
first loop run.

- A probe came first ($0.014), to learn what a call really costs: about
  $0.40 a run.
- Nearly half the money went to the A-vs-B comparison, because deliverable
  3 is graded on its spread. Five runs per arm, not three: with three,
  there is barely a spread to see.
- An extension was held back for whichever comparison ended up borderline,
  plus a ~$1 margin. A guard (`spend_guard.py`) would stop all runs
  cleanly at $9.50.
- Group B's change was chosen only after group A, from A's logs, with its
  prediction written first.

**What it bought.**
- **A** measured the variance: 0.0098 from best to worst run, so one run
  proves nothing.
- **B** found the effect.
- **C1** ruled out "more thinking".
- **C2** showed the behaviour is model-specific.
- The extension went to B, the borderline comparison.

**The runs that went nowhere.** Sonnet 5 was the README's suggested
upgrade. With the shipped client, every one of its 20 answers was cut off:
Sonnet reasons by default, and its hidden reasoning (1,062–2,048 tokens)
shares `MAX_TOKENS = 2048` with the answer. I stopped it after 6–7 calls a
run ($0.46). The fix would have been a second change (reasoning off, or a
larger limit), so it was dropped.

**Model change.** At the user's choice the slot went to an OpenAI model.
OpenAI's reasoning models have the same trap. GPT-6 Luna works with
reasoning "low", at about $0.001 a call. Its clean comparison is C1, where
only the model differs.

## 3. Run-to-run variance, and one setting (deliverable 3)

Champion score on PUBLIC for every run (`compare_groups.py`, an exact
two-sided Mann–Whitney test over all splits; with 5 vs 5 the smallest
possible p is 0.008):

| group | runs | median | range |
|---|---|---|---|
| A shipped | −0.0146 −0.0165 −0.0173 −0.0178 −0.0244 | −0.0173 | 0.0098 |
| B + hint | −0.0098 −0.0099 −0.0102 −0.0114 −0.0167 [[PENDING:B67]] | [[PENDING:B_MEDIAN]] | [[PENDING:B_RANGE]] |
| C1 + reasoning | −0.0131 −0.0151 −0.0175 −0.0213 −0.0226 | −0.0175 | 0.0095 |

- **B vs A:** [[PENDING:B_VERDICT]]
- **C1 vs A:** p = 1.00. Not separated.
- **C2 (Luna) vs C1:** [[PENDING:L_VERDICT]]

**Predictions** (from `SPEND_PLAN.md`, written before each group ran):

- **Group A.**
  - Median about −0.013: **wrong** (−0.0173).
  - Spread at least 0.002: right.
  - HELD_OUT within ±0.0005 of PUBLIC: right in 4 of 5 runs.
  - At least 85% valid answers: right (100%).
  - $0.35–0.45 a run: right.
- **Group B.**
  - At least a third of encoders search: right.
  - Median at most −0.0120: right.
  - At least four of five beat A's best: right.
  - p ≲ 0.03: [[PENDING]].
  - $0.40–0.55 a run: **wrong**. Searching programs are longer, so runs
    cost up to $0.66.
- **Group C1.** Under 5% of encoders search, and no separation from A:
  right.
- **Group C2.** [[PENDING:L_PRED]]

## 4. Verdict on the incumbent (deliverable 4)

The VAE's −0.0051 is not a plausible score for a six-number model on this
data. It is memorisation of the 128 PUBLIC images it was trained on.

- It scores past the noise floor.
- It reproduces 59% of each PUBLIC image's own noise, against 2.4% on
  HELD_OUT.
- Retrained identically on 128 *other* images, it scores −0.0049 on those
  and −0.0345 on PUBLIC.

The argument and the control are in `finding.md`.

What follows: every result here is judged against the honest floor and
reported on HELD_OUT as well.

## 5. The finding (deliverable 5)

See `finding.md`. In one line: the loop's ceiling is a missing idea, not
missing ability. Haiku never writes an encoder that searches through its
own decoder. One sentence pointing out that it can moves it to the floor,
while more reasoning does not. Another vendor's model does it unprompted.

## 6. Before spending anything: the audit

Two independent passes over the shipped code, plus my own re-runs of the
headline checks (`AUDIT.md`, reproducible with `bash audit/run_all.sh`).
The issues that mattered:

- **"The six numbers are the only channel" was false nine ways.** Shared
  builtins, module attributes, `sys.modules`, frame inspection, and
  `import data` all scored a perfect 0.0000. Candidate code could also
  patch the evaluator for the rest of the run.
- **A failed API call lost already-billed calls** from `token_log.jsonl`
  and killed the run without `best.py`. The likeliest trigger is a 402
  when the key runs dry.
- **`spend.py` priced every run at one model's rate**, and counted mock
  runs as spend.

I fixed only what cannot change what the loop searches:
- Candidates are scored in fresh subprocesses (`evo/sandbox.py`).
- Model calls are made and charged one at a time.
- Every answer and every setting is logged.

An honest candidate scores bit-identically to the shipped path (checked on
7 programs, both image sets). `sanity_check.sh` still passes, and each fix
has a test (`tests/test_loop_fixes.py`, 14/14). The prompt, the population
rule and the code regex were left as shipped, so group A is the shipped
loop.

## 7. What went wrong, and how I found it

1. **Two of my own claims about the floor were wrong.** I said −0.0100 was
   the best possible, and that six numbers cannot carry noise. The audit
   showed a fit beats the clean truth (−0.009597 vs −0.009833), and that a
   bit-packing encoder hides noise in the float64 digits (−0.0043 on PUBLIC
   *and* HELD_OUT). The VAE verdict was rebuilt on better evidence: the
   noise-copying fraction and the retraining control.
2. **Three more claims, caught by the independent check of `student.py`.**
   I wrote the formula value as −0.0096; it is −0.0098. I called it a bound;
   it is an approximate floor. And "beats the fit started from the true
   parameters" came from swapped-axis fits, not escaped local minima. The
   check reproduced every number, but my interpretation was wrong.
3. **A shell bug launched two runs instead of four.** zsh does not
   word-split an unquoted `$pair`, so `"a02 a03"` became one run id. I found
   it when `summarize_runs.py` printed run names with spaces and a per-run
   cost check could not find `runs/a02/`. Nothing was lost; the directories
   were renamed and logged. The same trap hit a `race.py` call later.
   Launches now use an explicit script, checked with `ps`.
4. **Sonnet's answers were all cut off.** I spotted it from three runs at
   `valid 0/4` with identical token counts, and confirmed it from `usage`
   (reasoning tokens). Stopped it at $0.46.
5. **My search classifier missed renderers nested inside `encode`.** A Luna
   run sat at −0.0096 with "0 searching encoders", which a measuring
   encoder cannot reach. I read its code, fixed the classifier, and checked
   that Haiku's 0/480 did not change.
6. **The spend guard was restarted before Luna's price was added**, so it
   would have priced Luna about 100× too high and stopped every run. Caught
   before launch.
7. **My prediction for the shipped loop was too optimistic** (−0.013 vs
   −0.0173). Haiku's very first answer (−0.14) had the right method and two
   wrong constants, which should have warned me.
8. **The audit agents wrote into the repo twice**: a VAE cache file and
   empty folders. Both were removed, and I confirmed the shipped cache is
   byte-identical (sha1).

## 8. Time and limits

**Time:** about [[PENDING:TIME]] against the suggested 5 hours. The audit
took the first hour.

**Limits:**
- Five runs per arm detects large effects only.
- The hint arm has no placebo control. A neutral sentence of the same
  length is the next run I would buy (`finding.md`).
- The sandbox removes the key and the repo's modules from a candidate's
  reach, but does not jail files or the network.

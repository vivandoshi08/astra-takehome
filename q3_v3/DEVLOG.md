# Q3 dev log

## 2026-10-04 — reading the scaffold, first free checks

Read every file in `q3_v3/`. What the task is, in one line: squeeze a 16x16
noisy ellipse image into 6 numbers and rebuild it; score is minus the mean
squared error against the **noisy** image.

**The noise floor.** `data.py` adds N(0, 0.1²) noise to every pixel. A model
that rebuilds the true clean ellipse perfectly is still wrong by the noise,
so it scores about −σ² = −0.0100. Six numbers cannot carry 256 independent
noise values, so no honest model should beat that on images it has not
memorised. The graded Bonus tier (≥ −0.0105) sits just under the floor.

**The incumbent fails that test.** `baselines.vae_model()` trains on
`data.PUBLIC` (128 images, 4000 epochs, decoder 6→64→256, ~17K weights) and is
then scored on `data.PUBLIC`. `python audit_baselines.py`:

|             | PUBLIC  | HELD_OUT |
|---|---|---|
| clean truth | −0.0098 | −0.0099 |
| mean        | −0.0454 | −0.0444 |
| pca (k=6)   | −0.0173 | −0.0195 |
| vae (k=6)   | **−0.0051** | **−0.0346** |

The VAE "beats" the perfect model on PUBLIC and falls behind PCA on HELD_OUT.
It has memorised PUBLIC's noise. Working hypothesis for deliverable 4; still
to do: a proper argument (lower bound for a 6-D code) and a VAE retrained on
fresh `data.train_images()` as a control.

**Plumbing, from a free mock run** (`--client mock --generations 3 --run-id
mock-smoke`): runs fine, best −0.0162 from a canned "centroid + size" program
— already past PCA. But `spend.py` priced the mock run at $0.02, though its
footer says "mock runs are free and log nothing". `evo/loop.py` always writes
`token_log.jsonl` and the mock client reports fake token counts; `spend.py`
ignores the `mock|` tag. Left the run directory out of git for now.

**Other things noticed, not yet checked:**
- No timeout on candidate evaluation: an infinite loop in a model's program
  would hang the run.
- `MAX_TOKENS = 2048`: a long answer cut off before its closing fence has no
  code block, and is dropped silently (counted only as "valid 3/4").
- The seed decoder is exactly `data.render`, so the generator's own formula
  is in the prompt from generation 0.

## 2026-10-04 — scaffold audit (AUDIT.md)

Ran two independent audit passes on the shipped code before building
anything: maths/data/models and loop/accounting/plumbing. No network, fake
clients only. Re-ran the headline checks myself before believing them.
Scripts in `audit/`, results in `audit/results/`, summary in `AUDIT.md`.

**Two of my claims above were wrong.**
1. *"Reconstructing the clean ellipse scores about −0.0100, the best any
   model can do."* The honest optimum is −σ²(1 − 6/256) ≈ **−0.0096**: a
   per-image least-squares fit also soaks up the noise along the 6
   directions it can move in, so it beats the clean truth (fit −0.009597 vs
   clean −0.009833 on PUBLIC, matching theory −0.009599).
2. *"Six numbers cannot carry 256 independent noise values."* False for
   float64: each latent carries 53 exact integer bits. A bit-packing encoder
   scores −0.0043 on PUBLIC **and** HELD_OUT. The argument only holds for a
   smooth decoder (like the VAE's tanh MLP) or a finite-precision code. The
   VAE verdict survives on better evidence: it reproduces 58.8% of PUBLIC's
   own noise vs 2.4% on HELD_OUT (honest ≈ 6/256 = 2.3%), and the same
   network trained on fresh images scores −0.0136 / −0.0135.

**Biggest new findings.** The "6 numbers are the only channel" isolation
is broken nine ways (each scores 0.0000; `import data` + true parameters
scores −0.00983, which looks honest). A failed API call — e.g. a 402 when
the key runs dry — loses already-billed calls from `token_log.jsonl` and
kills the run without `best.py`. `spend.py` prices all runs at one model's
rate.

**Process slips during the audit, both caught and undone.** One check
patched `baselines.VAE_BETA` at runtime, which made `vae_model()` retrain and
write a new file into `baseline_cache/` (itself finding E2); moved out within
a minute, shipped `vae_k6.npz` byte-identical (sha1 f532a4ae). A driver call
with a relative path created empty dirs under `q3_v3/out/`; removed. The
ported scripts now write scratch files to `$TMPDIR/q3_audit` only. I also
first ran `check_floor.py` from the wrong directory (ModuleNotFoundError) —
the math checks need the CWD to be `q3_v3/`; `audit/run_all.sh` handles it.

## 2026-10-04 — fixing the plumbing before spending anything

Decision: apply only the fixes that cannot change what the loop searches,
and keep the three that do (code regex, duplicates in the prompt, 4-decimal
scores) as shipped, so deliverable 3's default arm is still the shipped
loop. Each fix has a check in `tests/test_loop_fixes.py` (14/14).

- **Scoring in subprocesses** (`evo/sandbox.py`). The point that needed
  proving was "an honest candidate scores the same". Checked bit-for-bit on
  7 programs (SEED, the 4 mock programs, two realistic ones) on PUBLIC and
  HELD_OUT before wiring it in — identical to the last digit, 0.1–0.7 s per
  candidate. All of the audit's exploits now fail; the SEED still scores
  its true −0.02719 after each of them.
- **One paid call at a time.** Same prompt for all four candidates of a
  generation, so the search is unchanged; a failure now loses nothing.
  `sanity_check.sh` requires `run()` to raise `BudgetExhausted` at the cap,
  so a stop still raises — with the best-so-far attached for `main` to
  write.
- **New per-run files**: `responses.jsonl` (every call, including failures,
  with `finish_reason`) and `run_meta.json` (settings, git commit, stop
  reason, PUBLIC and HELD_OUT score of the champion, suspicious imports).
  `run_log.jsonl` and `token_log.jsonl` are exactly as before.
- **Residual risk, written down:** the sandbox removes the key and the
  repo's modules from a candidate's reach but does not jail files or the
  network.
- `runs/mock-smoke` is kept and committed: a run made with the shipped loop
  and shipped `spend.py`, which priced it at $0.02; the fixed `spend.py`
  prices it at $0.00.

## 2026-10-04 — pre-flight: key, prices, probe (predictions first)

**Free checks done before any spend.** The key's own meter (OpenRouter
`/api/v1/key`, free) reads limit $10, usage $0.00, expires 2026-10-21 — a
second meter to check `spend.py` against. OpenRouter's model list (free)
confirms `spend.py`'s rates for Haiku 4.5 ($1/$5 per Mtok), Sonnet 5, Opus 5
and gpt-4o-mini, and lists **reasoning as supported for Haiku 4.5**, against
the README's "Haiku 4.5 does not" (AUDIT.md F1). Not yet tested by a call.

**Probe** (`probe_api.py`, mock-tested in a temp copy): two real calls with
the loop's real generation-0 prompt, logged through `TokenBudget` so
`spend.py` counts them. Predictions, written before running:

- *default*: ~1,000–1,200 input tokens; 600–1,500 output; `finish_reason`
  "stop"; a code block that scores better than the SEED (−0.0272), likely
  −0.012 to −0.020; ≈ $0.004–0.009.
- *reasoning "low"*: accepted. Reasoning tokens > 0 and billed as output.
  OpenRouter turns effort into a thinking budget inside `max_tokens`
  (min ~1,024 of our 2,048), so the answer has ~1,000 tokens left: about
  even odds it is cut off (`finish_reason` "length", no code).
- The provider's meter, once settled, matches `spend.py` to within $0.001.

**Probe results** (`runs/probe-01`, $0.0139):

| | predicted | got |
|---|---|---|
| default: tokens in / out | 1,000–1,200 / 600–1,500 | 1,209 / 1,019 ✓ |
| default: finish | stop | stop ✓ |
| default: score | better than SEED (−0.0272) | **−0.1405 ✗** — worse than the mean image |
| reasoning "low": accepted | yes | yes ✓ — 402 reasoning tokens, billed as output (1,279 total) |
| reasoning "low": cut off | about even odds | no ✗ — the thinking budget was smaller than I assumed |
| meters agree | within $0.001 | `spend.py` $0.013937 = the provider's per-call `cost` field exactly; the key meter showed one of two calls settled at first |

Why the default answer scored −0.14: Haiku had the right idea (second
moments of the bright pixels give axes and tilt) and two wrong constants:
axes = 2.4·√λ where a filled ellipse needs 2·√λ (20% too large), and
brightness = the brightest pixel, which with σ = 0.1 noise sits well above
the true level. The reasoning answer made the same choices (2.4·√λ, max
pixel) plus a tanh rim instead of the known linear one: −0.0630. So a
first attempt can be worse than the SEED; the loop keeps the SEED in the
population, which is what protects it.

AUDIT.md F1 settled: Haiku 4.5 does accept `EVO_REASONING` on OpenRouter,
against the README. Real cost per call is ~$0.006–0.008 at generation 0,
so ~$0.40 per 12-generation run holds; reasoning "low" adds ~25%.

## 2026-10-04 — student.py: per-image fit of the generator's six numbers

Analysis by synthesis: a moments first guess (axes = 2·√λ, the right
constant for a filled ellipse; brightness by least squares given the shape),
then Levenberg-Marquardt on all six with `data.render` as the decoder, from
three starting tilts. numpy only (scipy is not in requirements.txt).
Developed on fresh `train_images` (seeds 7 and 2024) with `student_dev.py`;
scored on PUBLIC and HELD_OUT once, after.

| set | student | theory −σ²(1−6/256) on that set | clean truth |
|---|---|---|---|
| fresh, seed 7 | −0.009738 | −0.009766 (expected) | −0.009995 |
| fresh, seed 2024 | −0.009619 | −0.009766 (expected) | −0.009866 |
| PUBLIC | **−0.009600** | −0.009599 | −0.009833 |
| HELD_OUT | **−0.009680** | −0.009699 | −0.009933 |

Prediction was −0.0096 to −0.0098 on both: met. Bonus tier (≥ −0.0105).
Against an "oracle-start" fit (same optimiser started from the true
parameters) it is better on average (−1.3e-5 per image): 12/256 images
better by >1e-4, 1/256 worse (img 185, a small ellipse, a/b 1.44). So
almost nothing is left to local minima; the remaining error is the noise.
4 ms per image; deterministic.

## 2026-10-04 — VAE verdict control: predictions (before running)

`vae_verdict.py` retrains the incumbent's exact recipe (hidden 64, β 0.01,
seed 0, 4000 epochs) on fresh sets of 128, 512 and 2048 images, and scores
each on its own training set, PUBLIC and HELD_OUT. Predictions:

- n=128 fresh: past the honest optimum on its **own** training images
  (around −0.005, like the shipped one on PUBLIC), but PUBLIC ≈ HELD_OUT ≈
  −0.035. Noise fraction on its training set ~0.6, on HELD_OUT ~0.02.
- As n grows, own-train and PUBLIC/HELD_OUT scores converge, towards the
  −0.0135 measured for 2048 images in the audit; noise fraction on the
  training set falls toward 0.02.
- No model beats −0.0096 on images it was not trained on.

**VAE control results** (`vae_verdict.py`, 66 s):

| same recipe, trained on | own train | PUBLIC | HELD_OUT | noise copied: train / HELD_OUT |
|---|---|---|---|---|
| PUBLIC (shipped) | −0.0051 | −0.0051 | −0.0346 | 0.588 / 0.024 |
| fresh 128 | −0.0049 | −0.0345 | −0.0344 | 0.595 / 0.022 |
| fresh 512 | −0.0103 | −0.0174 | −0.0171 | 0.171 / 0.025 |
| fresh 2048 | −0.0113 | −0.0144 | −0.0143 | 0.059 / 0.023 |

Predictions met: the recipe scores ~−0.005 on whichever 128 images it
trained on and ~−0.035 on everything else; it copies ~60% of its training
images' noise and ~2% (the honest 6/256) of anything else's; with more
training images the gap closes. Not predicted: at 2048 images, 4000 epochs
(−0.0144 on PUBLIC) is worse than the audit's 1000 epochs (−0.0136) —
longer training still overfits. No model beats the honest optimum on images
it did not train on. (The script prints the *expected* optimum, −0.00977;
PUBLIC's own noise happens to be a little lower, hence −0.0096 there.)

## 2026-10-04 — group A, first three runs; a launch bug of mine

a01 passed stop rule 1 ($0.399, all files, meters agree), so I launched the
rest "two at a time" with `for pair in "a02 a03" "a04 a05"; do for r in
$pair; ...`. The shell here is zsh, which does not word-split an unquoted
`$pair`, so each "pair" started **one** run whose id was the whole string:
`runs/a02 a03` and `runs/a04 a05`. Found it when `summarize_runs.py`
printed run names with spaces and the per-run cost check could not find
`runs/a02/`. Nothing was lost or double-billed — two runs instead of four.
Renamed the directories to `a02` and `a03` (their `run_meta.json` still
says `"run_id": "a02 a03"` / `"a04 a05"`; logs left as written) and
launched a04 and a05 as two explicit commands. Each run's console output is
kept as `console.txt`.

## 2026-10-04 — group A complete (shipped loop × 5): results vs predictions

`python summarize_runs.py --group a` — $1.96 for five runs ($0.37–0.42
each); key meter $1.956 vs `spend.py` $1.98 a few minutes later (settling).

| run | PUBLIC | HELD_OUT | $ |
|---|---|---|---|
| a01 | −0.01726 | −0.01702 | 0.40 |
| a02 | −0.01461 | −0.01457 | 0.37 |
| a03 | −0.02438 | −0.02541 | 0.40 |
| a04 | −0.01780 | −0.01770 | 0.42 |
| a05 | −0.01647 | −0.01642 | 0.38 |
| median / spread | **−0.0173 / 0.0098** | −0.0170 / 0.0108 | |

| prediction | result |
|---|---|
| median about −0.013 | ✗ −0.0173 — the loop sits at PCA's level (−0.0173) |
| range −0.010 to −0.017 | ✗ −0.0146 to −0.0244 |
| spread ≥ 0.002 | ✓ 0.0098 — one run could land anywhere from "beats PCA" to "worse than PCA by 40%" |
| HELD_OUT within ±0.0005 of PUBLIC | ✓ 4/5; a03 ✗ (−0.0010) |
| ≥ 85% valid, < 5% cut off | ✓ 240/240 valid, 0 cut off |
| $0.35–0.45 per run | ✓ |
| none past −0.0096 on HELD_OUT | ✓ |

Signals for choosing group B's change:
- **Duplicates in the prompt: none.** `dup3` = 1/12 in every run is only
  generation 0, when the population holds just the SEED (a counting quirk in
  my metric). Deduplication — my pre-run lean — has nothing to fix here.
- **Answers cut off: none.** Raising MAX_TOKENS has nothing to fix.
- **4-decimal display:** the printed best stood still in 29 generations;
  in only 5 of them had the real best improved underneath. Mostly real
  stagnation, not hidden progress.
- **Where the error is** (`decompose_champions.py`, all five): encoder loss
  0.0034–0.0133, decoder loss 0.0003–0.0027. Through its own decoder, an
  encoder that searched would reach −0.0099 to −0.0123. And 0 of 173 logged
  encoders draws an ellipse or iterates — every one measures once.

## 2026-10-05 — group C2 (Sonnet 5) stopped after 7 calls each: every answer cut off

All 13 runs of B, C1, C2 started in parallel (explicit launcher script,
checked with `ps`: 13 processes, right env vars, guard running). After
generation 0 the three Sonnet runs showed `valid 0/4` with identical token
counts (13,692) — four answers each of exactly 2,048 output tokens.
`responses.jsonl`: 21/21 answers `finish_reason = length`, no code; usage
shows **1,062–2,048 reasoning tokens per answer**. Sonnet 5 reasons by
default on OpenRouter even with `EVO_REASONING` unset, and its hidden
reasoning shares the shipped `MAX_TOKENS = 2048` with the answer, so some
answers come back empty. Stopped s01–s03 with SIGTERM (clean: logs,
best.py = the SEED, `stop_reason: interrupted`). Cost $0.49 (21 × $0.0232).

So the README's "Sonnet 5 is about $0.80 a run" does not hold with the
shipped client: as shipped, a Sonnet run produces no programs at all.
Running it would need reasoning turned off or a larger MAX_TOKENS — a
decision for the user, not taken here. B and C1 continue.

**Replacing Sonnet** (user's call: an OpenAI model, then specifically GPT
Luna). Checked OpenRouter's list first (free): OpenAI's reasoning models
would hit the same cut-off. Probed `gpt-4.1-mini` (non-reasoning, $0.003:
fine, −0.056/−0.022) before the user picked Luna; kept as a record. Probed
`gpt-6-luna` ($0.002): as shipped, cut off by 1,996 reasoning tokens; with
reasoning "low", a complete answer that **searches** — moments start, then
coordinate descent through its own renderer, least-squares brightness —
scoring −0.00977 PUBLIC / −0.00983 HELD_OUT on its first attempt, no hint.
Haiku's 173 encoders never did this. A guard slip of mine on the way: I
restarted `spend_guard.py` before adding Luna's price to `spend.py`, so it
would have priced Luna at the "unknown model" fallback (~100× too dear) and
stopped every run; caught before launching, restarted after the price was
in. The guard and `spend.py` both read $5.86.

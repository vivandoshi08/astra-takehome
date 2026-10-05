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

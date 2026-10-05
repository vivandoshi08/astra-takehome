# Q3 scaffold audit — before any model or loop run

Done on the shipped code (commit `d23a4bd`) before writing `student.py` or
spending any credit, so every later result rests on checked assumptions. Two
independent passes: maths/data/models, and loop/accounting/plumbing. No
network calls; fake clients only. Reproduce everything:

    PY=python bash audit/run_all.sh     # ~4 min; text results -> audit/results/

Scratch files (npz, png, fake run dirs) go to `$TMPDIR/q3_audit`, never the
repo. Each finding below names the check that shows it.

`audit/results/` is the output against the **shipped** code (commit
`05e2b64`). The fixes have their own checks, each reproducing the shipped
failure and asserting it is gone:

    python tests/test_loop_fixes.py     # 14 checks, ~20 s

None of the fixes changes what the loop searches: an honest candidate scores
bit-identically to the shipped in-process path (7 programs, PUBLIC and
HELD_OUT), `run_log.jsonl` keeps its schema, and the prompt, population rule
and code regex are untouched. `sanity_check.sh` still passes.

Status: **fixed** = changed, with the commit · **fix** = proposed change
before real runs · **doc** = documented, not
changed · **decide** = changes what the loop does, so it is an experimental
choice, not a bug fix · **verify** = needs one cheap real call.

## A. Can the scores be trusted?

| # | finding | evidence | status |
|---|---|---|---|
| A1 | The VAE incumbent is trained on `PUBLIC`, the set it is scored on. It reproduces **58.8%** of PUBLIC's own pixel noise; an honest 6-number fit reproduces about 6/256 = 2.3% (it does 2.4% on HELD_OUT). | PUBLIC −0.0051 / HELD_OUT −0.0346. Same network trained on 2,048 fresh images: −0.0136 / −0.0135. `math/check_baselines --vae-control`, `math/check_misc` | doc — deliverable 4 |
| A2 | The graded tiers are scored on PUBLIC, so a student model fit to PUBLIC clears Bonus by memorising. | `student = vae_fns()` scores −0.0051 (Bonus); a β=0 VAE −0.0046 PUBLIC / −0.0505 HELD_OUT. `math/check_misc`, `math/check_vae_clip_practice` | doc — never fit on PUBLIC; always report HELD_OUT next to PUBLIC |
| A3 | **The honest floor is about −σ²(1 − 6/256) = −0.00977 in expectation, not −σ² = −0.0100; on PUBLIC, whose noise is a little low, about −0.0096.** A per-image least-squares fit also absorbs the noise along the directions it can move in (6 for a linear fit, ~6.5 for this nonlinear one), so it beats the clean truth by ≈ 6σ²/256. An approximate floor, not a bound (corrected after an independent check of `student.py`; the per-set figures below are clean-truth MSE − 6σ²/256). | Moments + Levenberg–Marquardt fit with `data.render`: PUBLIC −0.009597 (theory −0.009599), HELD_OUT −0.009678 (−0.009699); clean truth −0.009833 / −0.009933. `math/check_floor` | doc — corrects DEVLOG |
| A4 | **"Six numbers cannot carry the noise" is false for float64 latents** (53 integer-exact bits each, ~318 bits). The −σ² argument needs a smooth decoder or a finite-precision code. | Packing 16-bit fit parameters plus 222 residual signs into the 6 floats scores **−0.0043 on PUBLIC and HELD_OUT** — beats the VAE and generalises. `math/check_floor` | doc — corrects DEVLOG; the evaluator stays as shipped (thresholds depend on it) |
| A5 | **The latent is not the only channel between encode and decode.** Nine independent tricks score a perfect 0.0000: shared `__builtins__` dict, `builtins` module, an attribute on `math`, `sys.modules`, `gc`, `sys._getframe(1)` reading the evaluator's target, `import data` + lookup. `import data` + `PUBLIC_PARAMS` scores −0.00983 — indistinguishable from an honest model. | `loop/check_isolation`, `math/check_evaluate` | **fixed** (`63fe1ef`): `evo/sandbox.py` — encode and decode in separate fresh `python -I` processes; all nine tricks now fail or score like an honest bad model; champions rescored on HELD_OUT and grepped by `loop.main` |
| A6 | Candidate code runs at compile time inside the loop's process, and its changes persist: patching `tasks.ellipses.evaluate` makes every later candidate score 0.0; zeroing `data.PUBLIC` makes the honest SEED score −0.0393 instead of −0.0272; no-op'ing `TokenBudget.charge` stops the cap and the log. | `loop/check_isolation`, `loop/check_population` D | **fixed** (`63fe1ef`): candidate code never runs in the scorer's process; SEED scores its true −0.02719 after every exploit |
| A7 | Determinism is required by the rules but not enforced. One random encoder, scored 5 times: −0.0254 to −0.0262; the loop keeps the luckiest draw. | `loop/check_isolation` | **fixed** (`63fe1ef`): each process re-runs the first 8 inputs and rejects a different answer |
| A8 | Mean and PCA are also fit on PUBLIC. PCA's in-sample edge is +0.0012, so "beats PCA" (T2) is that much harder than beating an honest PCA. | PCA fit on PUBLIC −0.01731 vs on fresh data −0.01853. `math/check_baselines` | doc |

## B. Is the money accounted for?

| # | finding | evidence | status |
|---|---|---|---|
| B1 | `LLMClient.complete(n)` makes all n paid calls before any is charged. If one raises — a **402 when the key runs dry**, a network error after the SDK's 2 retries, `usage=None`, `choices=None` — the earlier billed calls of that generation never reach `token_log.jsonl`, and the run dies (exit 1) with no `best.py`. | 6 calls billed, 4 logged. `loop/check_accounting` C–E, `loop/check_main` | **fixed** (`63fe1ef`): one call requested and charged at a time; an API error stops the run with every billed call logged and `best.py` written |
| B2 | If the cap trips on response k of n, responses k+1..n are billed but never logged, none of that generation is scored, and `main` exits 0 with no `best.py`. Unreachable at default settings (a run peaks near 270K < 500K). | 8 billed, 6 logged. `loop/check_accounting` A | **fixed** (`63fe1ef`): the response that crosses the cap is still scored; `best.py` written |
| B3 | `spend.py` prices every run at one `--model` rate and ignores the tag that records each run's model and marks mock runs. | Synthetic mock + Haiku + Sonnet runs: true $1.01, reported $0.76 or $1.51. `runs/mock-smoke` priced at $0.02. `loop/check_spend` | **fixed** (`f66f172`); `runs/mock-smoke` now $0.00 |
| B4 | Reusing a `--run-id` silently appends to the old logs; `spend.py` merges both runs into one row. | `loop/check_main` reuse | **fixed** (`63fe1ef`): `main` refuses a non-empty run dir |
| B5 | `finish_reason` is thrown away, so truncation at `MAX_TOKENS=2048` is invisible. | `loop/check_accounting` D | **fixed** (`63fe1ef`): `finish_reason` and raw usage in `responses.jsonl` |
| B6 | "Hard-stops at 500K", "logs every call", "the two agree exactly" (README, spend.py) are false under B1–B3. | B1–B3 | doc |
| B7 | `budget.py`: raises when the total *equals* the cap; `summary()` can exceed 100%; default log path is relative to the CWD. | `loop/check_accounting` B, `loop/check_misc` | doc — `budget.py` is never edited |
| B8 | The openai SDK retries twice on 408/409/429/5xx, not on 402/400; a hung call can stall a run ~30 min (600 s × 3). | installed SDK source | **fixed** (`63fe1ef`): `timeout=120`, `max_retries=2`, written down in `llm_client.py` |
| B9 | `spend.py` crashes on a cut-off last line; `--runs` is CWD-relative; `--in-rate` without `--out-rate` is silently ignored. | `loop/check_spend` | **fixed** (`f66f172`) |

## C. Can a run die or lose work?

| # | finding | evidence | status |
|---|---|---|---|
| C1 | `evaluate` catches only `TypeError/ValueError` in its float conversions, and `loop.py` does not guard the call: one candidate returning `10**400` (OverflowError) kills a paid run. `SystemExit`/`KeyboardInterrupt` from candidate code pass `except Exception`; `SystemExit(0)` ends the run silently, exit 0, no `best.py`. | `math/check_evaluate` §4, `loop/check_main` | **fixed** (`63fe1ef`): every failure inside a candidate is a status, not a crash |
| C2 | No timeout: one infinite loop hangs the run. | `loop/check_timeout` | **fixed** (`63fe1ef`): 180 s per scoring process (`--eval-timeout`) |
| C3 | LLM-written code runs unsandboxed, twice (once per namespace), with `OPENROUTER_API_KEY` in its environment, file and network access. | canary file written twice. `loop/check_isolation` | **partly fixed** (`63fe1ef`): no key in the child's environment, temp CWD, repo modules off its path, timeout. Not jailed: a candidate can still read files or open sockets |
| C4 | No record of failures: the text of answers that had no code block, failed to compile, or scored −inf is never logged. | `loop/check_population` C | **fixed** (`63fe1ef`): `runs/<id>/responses.jsonl`, one line per call |
| C5 | Run settings (model, `EVO_REASONING`, candidates, generations, cap, commit) are never logged; a reasoning run looks identical to a default one. | `loop/check_main` | **fixed** (`63fe1ef`): `runs/<id>/run_meta.json` |

## D. Loop behaviour — experimental choices, not bugs

These change what the search does, so changing them would make the
"shipped default" arm something else. Each is a candidate for the one
changed setting in deliverable 3.

| # | finding | evidence | status |
|---|---|---|---|
| D1 | The code regex drops 15 of 23 plausible answer formats (```` ```py ````, ```` ```Python ````, two blocks, truncated fence, unfenced, …). Realistic for Haiku: truncation and split blocks. | `loop/check_extract` | **kept as shipped** for the default arm; drops are now visible in `responses.jsonl` |
| D2 | Duplicates re-enter the population: from generation 4 of a 12-generation mock run all 3 programs in the prompt are identical. | `loop/check_population` A/B | **kept as shipped** for the default arm; candidate for the one changed setting |
| D3 | Scores are shown to 4 decimals: −0.01041 and −0.01044 both print −0.0104, so the model cannot see progress near the target band. | `loop/check_population` E | **kept as shipped** for the default arm |

## E. The VAE as an instrument

| # | finding | evidence | status |
|---|---|---|---|
| E1 | Backward pass ignores the ±10 clip on logvar. Harmless for the shipped model (logvar in [−7.5, −5.6]); wrong for β ≤ 1e-3 (clip active on 6–60% of entries). | finite differences 0 vs analytic 2753. `math/check_vae_math`, `math/check_vae_clip_practice` | doc — mask in a copy before any small-β sweep |
| E2 | Cache: `vae_k6.npz` loads without checking its config (stores no epochs or training set); defaults bind at definition but compare at call time; `hidden=[64]` misses the cache; non-default configs write into the repo's `baseline_cache/`. | `math/check_baselines` §2–3 (this is how the audit accidentally wrote a file there once — removed, shipped file byte-identical) | doc — use `cache=False` for our own models |
| E3 | "Identical weights" on retrain holds to 7e-15, not bitwise; "about a minute" is 3 s here. `load()` resets Adam state. | `math/check_vae_retrain`, `math/check_vae_math` | doc |

## F. Prompt, docs and tooling

| # | finding | evidence | status |
|---|---|---|---|
| F1 | README: "Haiku 4.5 does not support reasoning effort" — may be wrong. If OpenRouter turns effort into a thinking budget inside `max_tokens=2048`, answers truncate. | `runs/probe-01` | **verified, README wrong**: Haiku 4.5 accepts reasoning "low" (402 reasoning tokens, billed as output, answer not cut off) |
| F2 | Prompt says the rim fades "over roughly one pixel": exactly 1 px along the minor axis, a/b px (up to 3.1) along the major. Rim pixels are 57% of the lit pixels and 28% of signal energy. It omits a ≥ 2.2 and the centre range. | `math/check_data` | doc |
| F3 | Digit grids clip negatives to 0 (44% of pixels are negative) and ≥ 0.95 to 9. | `math/check_data` | doc |
| F4 | θ is weakly identified for near-circles (a/b < 1.05 in 11% of PUBLIC) and wraps at π; the decoder needs a ≥ b. | `math/check_data` §3 | doc — matters for any regression on θ |
| F5 | The mock never produces an invalid, truncated or failing answer (20,000/20,000 valid), ignores the prompt, and is fixed-seed: it cannot exercise B1–C4 or measure variance. | `loop/check_mock` | **fixed in tests**: `tests/test_loop_fixes.py` has scripted failing clients; `mock_client.py` unchanged |
| F6 | `package_submission.sh` copies no `.py`, so `runs/*/best.py` reach the tarball only via the git bundle; its `student.py` check greps a marker the instructions say to keep (false alarm); `finding.md` template satisfies its report check (false all-clear). | `loop/check_packaging` | doc — commit `runs/` |
| F7 | `race.py --images` accepts any attribute (`SIGMA` crashes) and writes PNGs to the CWD, overwriting PUBLIC plots on a HELD_OUT run. | `run_race_sandboxed.sh` | doc — use `--no-plots` or a temp dir |
| F8 | Minor: dead `except BudgetExhausted: raise`; seed record says `valid_this_gen=0`; loop.py is 229 lines, not "about 150"; `sanity_check.sh`'s Q3 check hides its errors; `plot_traversal` sweeps 2nd–98th percentile, not the "observed range". | `loop/check_misc`, reading | doc |

## Checked and correct

- `data.render` matches its formula and `PUBLIC_CLEAN` exactly; the SEED
  decoder equals `render` to 6e-16; pixel (i, j) is at x = j, y = i.
- Generator ranges as declared; b < a always; θ ∈ [0, π) is the right
  period; six numbers do determine the clean image. Border clipping: 0/128
  PUBLIC, 1/256 HELD_OUT.
- Noise std 0.0992 (PUBLIC), 0.0997 (HELD_OUT).
- `evaluate` is a true per-pixel mean; inputs are copied, so an encoder
  cannot alter the target through its argument; `CLIP` cannot be gamed.
- VAE: every gradient matches finite differences (≤ 6e-9 relative); Adam
  matches a reference to 1e-16; save/load round-trips; `encode` returns μ.
- Mean and PCA formulas are correct.
- The call that crosses the cap is logged before the raise. Token-log tags
  carry model and generation. Output paths match the README.
- Cost: offline estimate $0.24–0.66 per default run, typical $0.33–0.41 —
  the README's ~$0.40 is plausible. Input dominates tokens; **output
  dominates dollars** (45–84%), so verbose or truncated answers and any
  reasoning tokens drive the bill. `loop/check_cost`

## What follows for the work

- **Every score we report comes in pairs: PUBLIC and HELD_OUT.** HELD_OUT is
  the only set no entry was fit to.
- **The target band is −0.0096 (honest optimum) to −0.0105 (Bonus)**, a
  9% MSE margin. Anything above −0.0096 on HELD_OUT is not an honest
  6-number model and gets investigated, not celebrated.
- The behaviour-neutral fixes (A5–A7, B, C) are in. Section D stays at the
  shipped behaviour, so the default arm of deliverable 3 is still the
  shipped loop; one of D1–D3 can be the one changed setting.

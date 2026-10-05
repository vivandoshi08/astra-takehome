# Q2 dev log — what broke, how it was found, what was done

Raw material for the report's "what went wrong". Scaffold issues found before
any refinement code are in `AUDIT.md`; this file starts after that.

## Bugs

| # | what broke | how it was found | what was done |
|---|---|---|---|
| 1 | `neat.Config` reads the ini section named after the genome class, so passing `SigmoidOutputGenome` failed with `NoSectionError` | the first run of `test_scaffold_fixes.py` crashed | `network.load_config()` parses with `DefaultGenome`, then sets `config.genome_type` (reproduction builds genomes through it) |
| 2 | `git_commit()` marked every run `-dirty`, because the run's own output files are untracked | noticed while preparing the first commits | `runs/` excluded from the dirty check; commit recorded at run start |
| 3 | `main.py` exited 1 both for "over budget" and for any uncaught exception; `--steps 0 --generations 0` raised inside neat-python, exited 1, wrote no summary, and could have been filed as a budget overrun | harness agent, reasoning about exit codes, then reproducing the case | over budget now exits 4 (`BUDGET_EXIT`); 1 is always a crash; the harness requires exit 4 + `within_budget: false` + no traceback |
| 4 | `--generations 0` was silently ignored (`args.generations or planned` treats 0 as unset): the run reported 0 and ran 30 | harness agent | `is not None`; generations must be >= 1 |
| 5 | `-dirty` does not identify the code a run used: two runs both stamped `609374e-dirty` can have run different code | harness agent | the harness fingerprints the code files before and after every run and warns when cells differ; graded grids run from committed code only |
| 6 | the finite-difference reference for the gradient check was badly conditioned: error grew like 1/h on genomes with confidently wrong points, because `forward()` only gives p near 1 to absolute precision | engine agent, sweeping the FD step and matching worst cases to large output values on misclassified points | tolerance derived from the reference's own round-off; a mirrored reference (negated output node) for the noisiest cases; mutation test: 8/9 injected gradient bugs and 6/6 injected Adam bugs caught (the 9th makes no numerical difference) |
| 7 | relu kinks are hit exactly, not by rare chance: neat-python gives every new node bias 0.0, so a relu fed only by an inactive relu sits at z = 0 on every point | engine agent, FD failures concentrated on such genomes | gradient uses the left derivative (0) there; the FD check skips and counts such cases (3 of 600) |
| 8 | three wrong expectations in our own tests (a "past the +-60 clip" case that never got past it; a hand-counted 3 dead connections that were 4; a mean misread) | companion assertions in the same tests | tests fixed; code was right each time |
| 9 | `lr_probe.py` tuned the learning rate on the full training split, which includes the validation points used to pick the champion | review of the engine agent's report | probe restricted to the fit split; recommendation unchanged (lr 0.3: never raises the loss on > 5% of genomes, >= 95% of the best lr's reduction in every cell) |
| 10 | reusing one `neat.Config` object across runs breaks reproducibility: the node-key counter lives on the config, so two seed-0 runs diverge from generation 2 | engine agent | `main.py` builds one config per process (safe); any multi-run script must call `load_config()` per run |
| 11 | `budget.refinements` counts `charge()` calls: one per genome for the fixed allocation, one per round for successive halving | engine agent | `summary.json` also reports `genomes_refined` and `refine_calls`; the step total is unaffected |

## Surprises to investigate

- **Lamarckian runs fragment into many species; 4 of 9 Lamarckian
  successive-halving runs crashed.** neat-python raised `RuntimeError:
  population size 100 is less than num_species * min_species_size 102
  (51 * 2)`. Max species per run in the allocation grid: Baldwinian 2–5 in
  every arm; Lamarckian 6–23 (A), 16–49 (B), 23–41 (D, survivors). Lamarckian
  weights reach the +-30 bound (max |param| 30 in most runs; Baldwinian
  7–21). Likely chain: written-back refined weights drift apart between
  lineages -> NEAT's compatibility distance (weight term, coefficient 0.5)
  passes the 3.0 threshold -> one species per genome or two -> with
  elitism 2 per species the population cannot hold them. Found when the
  harness reported the crashes; the crashed cells are kept as results
  (Lamarckian + halving is infeasible under this config on those seeds),
  not re-run. Note: main.py only catches BudgetExhausted, so a crash
  writes no fitness_history.csv; the species trajectory up to a crash is
  not logged.

- **Refinement collapses most xor genomes to a constant.** 80% of hidden-node
  xor genomes from plain NEAT converge to p = 0.5 everywhere (loss ln 2,
  gradient ~8e-6 after 2,000 steps); training accuracy falls 0.71 -> 0.52.
  Every lr from 0.03 up does it, so it is not the lr. Plain-NEAT genomes start
  confidently wrong (median loss 4.7) and descent shrinks the outputs toward
  0.5. A topology split ruled out "only genomes without hidden nodes". Spiral
  collapses 0% of the time, spiral3 4%. Consequence: accuracy-after-refinement
  as fitness can punish refinement on xor, and Lamarckian inheritance would
  pass the collapsed weights on.

## Process

- A parallel Q1 session committed onto `q2-backprop-neat` through the shared
  working directory (reflog shows its checkout and commit). Harmless here —
  the commit is in `origin/main` — but Q1 work now runs in its own worktree.

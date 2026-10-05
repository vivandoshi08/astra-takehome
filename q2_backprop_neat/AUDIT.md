# Q2 scaffold audit — before any refinement code

Done before writing refinement, so every later result rests on checked
assumptions. Reproduce:

    python audit_scaffold.py        # the checks, against the shipped behaviour
    python test_scaffold_fixes.py   # each fix: reproduces the failure, asserts it is gone

## Findings

| # | finding | evidence | status |
|---|---|---|---|
| 1 | A Baldwinian champion would be scored with its unrefined genome weights: `main.py` and `viz.py` rebuilt the champion from its genome | read of `main.py` final eval / `viz.py` | **fixed**: evaluation stores `genome.refined_params`; `champion_net()` loads it; cleared at every evaluation; 0 leaks to offspring over 20 generations |
| 2 | The output node's activation mutates (tanh 43%, relu 10% of evaluated genomes) while `predict()` thresholds at 0.5; `forward()` was documented as "logits" but returns the activation | `audit_scaffold.py`: output ranges per activation | **fixed**: `SigmoidOutputGenome` pins outputs to sigmoid (hidden nodes still mutate); docstring corrected |
| 3 | Elites are the same objects carried over and re-evaluated every generation (mean 3.7 per generation): Lamarckian elites accumulate refinement, Baldwinian elites repeat an identical one | `audit_scaffold.py`: 144/144 carried-over genomes are the same object | **design decision** (below) |
| 4 | `budget.summary()` was only printed; a budget overrun crashed before any log was written; `runs/<dataset>` was overwritten by the next run | read of `main.py` | **fixed**: `summary.json` per run (budget, git commit, args, accuracies, majority-class floor); overrun caught, logs kept, run marked invalid, exit 1 (later changed to exit 4: DEVLOG #3); re-using a run id refused without `--overwrite` |
| 5 | Champion is chosen on training fitness; there is no validation split | read of `Recorder` | **design decision** (below) |
| 6 | `network.py` uses sigmoid(z) / tanh(z); neat-python's own network uses sigmoid(5z) / tanh(2.5z) | max output difference 1.69 with neat's activations, 1.8e-15 with unscaled ones (topology and order otherwise identical) | **documented, not changed**: nothing uses `neat.nn`; the config's mutation powers are effectively 5x gentler than neat-python intends |
| 7 | `write_back` stored values outside the config range (50 stored as 50); NEAT clamps only on mutation, so the value later jumps to 30 | `audit_scaffold.py` | **fixed**: `write_back` clamps and returns the number clamped; `net.lower`/`net.upper` expose the range for refinement to project onto |
| 8 | 35% of enabled connections and 43% of nodes never reach the output (72% of genomes have some); `complexity()` counts them, their gradient is 0 | `audit_scaffold.py` | **kept**: the question counts *enabled* connections; noted in the docstring |
| 9 | `plan(..., steps=0)` divides by zero; `plan` floors (7 steps/genome uses 29,400) | `audit_scaffold.py` | **fixed**: `--steps 0` asks for `--generations`; warns if `--generations` exceeds the plan |
| 10 | "Gradient step" is not defined anywhere: full batch and single example cost the same; line search, second-order methods or finite-difference gradients could hide work | read of `QUESTIONS.md`, `budget.py`, `report.md` | **definition** (below) |
| — | `viz.py`: `out.replace('boundary', 'fitness')` overwrote the boundary plot when `--out` lacked "boundary"; default champion path relative to the CWD | read + test | **fixed** |

## Facts about the frozen data

- The spirals are easier than their names: "2.6 pi turns" is 2.6 pi *radians*,
  1.30 revolutions (spiral3: 1.50). Arms are 10–12 noise sd apart.
- Reference held-out accuracy from k-NN: xor 0.960–0.993, spiral 0.987–0.993,
  spiral3 1.000. xor's Bayes-optimal accuracy is 0.984. The ceiling is ~98–100%.
- Majority-class floor on held-out: xor 0.513, spiral 0.520, **spiral3 0.567**
  (train is 0.471 class 1: the split is imbalanced).
- Held-out has 150 points: one point is 0.67%.
- The shipped plain-NEAT spiral champion (0.767) is a corner cut, not a spiral:
  points concentrate on the outer arm (radius ~ sqrt(uniform)), so crude
  boundaries score ~75%. Accuracy alone cannot say a spiral was learned.

## Decisions for the refinement code

- **One gradient step** = one gradient of the full-batch training loss plus one
  optimiser update. No line search, second-order steps, or finite-difference
  gradients inside runs. Charge the steps actually taken; forward passes
  (fitness) are free, as in the shipped scaffold.
- **Elites** (correction after the post-merge audit: true under allocation A; under B and D an elite is refined only if selected again — DEVLOG #19, #20) are refined every generation they are evaluated, under both
  strategies, so both see the same evaluation protocol. That Lamarckian elites
  accumulate refinement this way is the first candidate mechanism to test.
- **Champion selection** uses a validation split carved from the training
  set, never the held-out set.

## Reference: the backprop-NEAT write-up

hardmaru (2016) backprops each network 600 times, full batch, raw x/y inputs,
and "multiply the fitness score by a factor that grows in proportion to the
square root of the number of connections". It does not say whether trained
weights are inherited. 600 steps x 100 genomes is 60,000 steps for one
generation; our 30,000 must cover a whole run.

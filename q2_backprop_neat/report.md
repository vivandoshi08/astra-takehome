# Q2 — Backprop-NEAT under a compute budget

*Every number comes from the committed file named beside it; where I derived
a number while writing (a sum or mean over a committed `results.csv` or
`fitness_history.csv`) I say so. Detail tables are in the appendix; the
predictions behind each experiment are in `experiments/PREDICTIONS_*.md`,
committed before the run they predict; bugs and corrections are in
`AUDIT.md` and `DEVLOG.md`.*

**Summary**

- Refinement runs inside the NEAT loop. Settled allocations, held-out mean
  over seeds 0–9 (xor / spiral / spiral3): Lamarckian 0.967 / 0.903 / 0.825,
  Baldwinian 0.964 / 0.869 / 0.740; plain NEAT 0.704 / 0.749 / 0.620.
- At equal allocation Lamarckian beats Baldwinian on the spirals, replicated
  on three disjoint seed sets (+0.065 to +0.153). The mechanism is only
  partly explained: inherited refinement gives children a head start, but
  mutation erases most of it each generation.
- Most of my allocation predictions failed (5 of 10 contradicted at 10
  seeds); the two strategies want opposite allocations.
- A linear size penalty buys smaller networks, never better ones. The
  write-up's multiplicative form, as I read it, makes the empty network
  optimal.
- With 6 connections, xor loses nothing; adding sine activations lets 6
  connections match the unconstrained spiral scores — by drawing stripes,
  not spirals. The boundary plots show that high spiral scores are mostly
  fits to the dense outer turns, not spirals.

## 1. What I built

- **Refinement.** Hand-written reverse-mode backprop
  (`GenomeNetwork.loss_and_grad`), mean binary cross-entropy from the output
  pre-activation; full-batch Adam, lr 0.3 (chosen by `lr_probe.py` on the fit
  split only), parameters projected onto the config's ±30 range after every
  update. `test_gradients.py` checks the gradient against finite differences
  on 300 evolved and 300 refined genomes plus edge cases (all pass); a
  mutation test caught 8/9 injected gradient bugs and 6/6 Adam bugs
  (DEVLOG #6).
- **One gradient step** = one full-batch gradient of the fit-split loss + one
  Adam update; no line search, second-order or finite-difference steps;
  forward passes free, as in the scaffold; the budget is charged for steps
  actually taken. Every graded run reports its `budget.summary()` (section 2,
  Appendix A1); none exceeded 30,000.
- **Fitness** = fit-split accuracy after refinement (minus the penalty, if
  any). **Elites** are re-refined every generation under both strategies, so
  both see one protocol.
- **Inheritance.** Lamarckian: `net.write_back(genome)`. Baldwinian: the
  genome is untouched; the refined vector is kept only as
  `genome.refined_params` (cleared at every evaluation) so the champion is
  evaluated with the weights that earned its fitness.
- **Data.** The 350 training points are split once (stratified, fixed seed)
  into 280 fit points (refinement, fitness) and 70 validation points
  (choosing the champion and choosing between arms); held-out is only ever
  reported. (The split cost nothing measurable: over 293 allocation cells,
  the validation-chosen champion differs from the training-fitness one by a
  mean −0.0024 held-out, identical in 197 — derived from
  `runs/allocation/results.csv`.)
- **Allocation modes**: `fixed` (top `--refine-fraction` by pre-refinement
  fitness, `--steps` each, generations from `budget.plan()`) and successive
  `halving` (100×2, 50×4, 25×8, 12×16, 6×32 steps per generation).
- **Penalty** on C = enabled connections: linear acc − λC (main: λ reads as
  the accuracy a connection must buy, the same on every dataset); mult
  1 − (1 − acc)(1 + λ√C), my reading of the write-up's "multiply the fitness
  score by a factor that grows in proportion to the square root of the
  number of connections". With a penalty the champion is chosen on
  penalised validation accuracy, so selection cannot undo it.
- **Small networks**: `--max-conns N` repairs every genome at creation to ≤ N
  enabled connections, so the budget's meaning is unchanged.
- **Harness**: `experiments/run_grid.py` runs grids in parallel, classifies
  every cell, records each run's commit and code fingerprint, and picks arms
  on validation only.

Reproduce the settled runs (from `q2_backprop_neat/`):
`python experiments/run_grid.py experiments/grids/allocation.json`, or one
run: `python main.py --dataset spiral --seed 0 --lr 0.3 --strategy
lamarckian --steps 10 --refine-fraction 1.0` (Baldwinian: `--strategy
baldwinian --steps 100 --refine-fraction 0.1`).

## 2. Results (deliverable 1)

Held-out, mean ± sd (`runs/allocation/summary.md`; seeds 10–12 never used for
a decision, `runs/confirm/confirm_scored.md`):

| dataset | allocation | seeds 0–9 | seeds 10–12 | conns | `budget.summary()` |
|---|---|--:|--:|--:|---|
| xor | Lamarckian 10×100% | 0.967 ± 0.007 | 0.967 ± 0.007 | 12.3 | 30,000/30,000 gradient steps over 3,000 refinements |
| xor | Baldwinian 100×10% | 0.964 ± 0.018 | 0.971 ± 0.010 | 8.7 | 30,000/30,000 gradient steps over 300 refinements |
| spiral | Lamarckian 10×100% | 0.903 ± 0.069 | 0.889 ± 0.004 | 10.5 | 30,000/30,000 gradient steps over 3,000 refinements |
| spiral | Baldwinian 100×10% | 0.869 ± 0.057 | 0.833 ± 0.064 | 9.1 | 30,000/30,000 gradient steps over 300 refinements |
| spiral3 | Lamarckian 10×100% | 0.825 ± 0.055 | 0.787 ± 0.037 | 10.2 | 30,000/30,000 gradient steps over 3,000 refinements |
| spiral3 | Baldwinian 100×10% | 0.740 ± 0.060 | 0.682 ± 0.020 | 8.4 | 30,000/30,000 gradient steps over 300 refinements |

Context: majority class 0.513 / 0.520 / 0.567; plain NEAT (30 generations,
`0/30,000 gradient steps`, 3 seeds, before the validation split) 0.704 /
0.749 / 0.620; k-NN 0.96–0.99 / 0.99 / 1.00, xor Bayes 0.984 (AUDIT.md).
xor is at its ceiling; the spirals are far below theirs. Seed spread is
large (Lamarckian spiral 0.800–0.980; one held-out point is 0.67%).

## 3. Allocation study (deliverable 2)

**Pilot first.** I refined 100 genomes per dataset to 300 steps
(`pilot.py`, unused seed 100; Appendix A4): pre-refinement fitness barely
predicts post-refinement fitness (Spearman 0.07–0.09); 10-step rankings are
near-random on the spirals (0.02 / 0.27); successive halving would cut 4–6
of the eventual top 6. My pre-written rule turned this into "100–300 steps,
1–3 generations" — degenerate, because it valued ranking a fixed set of
genomes and gave generations no value, while evolution was still improving
at generation 29. I discarded the rule and kept the measurements.

**Predictions committed before the grid** (`PREDICTIONS_allocation.md`): P1
Baldwinian B (100 steps × top 10%) < A (10 × 100%) on the spirals; P1b the
gap is smaller under Lamarckian; P2 C (50 × 100%, 6 generations) < A; P3
Baldwinian successive halving < A. 5 allocations × 2 strategies × 3 datasets,
seeds 0–2, then — **after seeing those** (3 supported / 3 contradicted / 4
inconclusive) — seeds 3–9, rule unchanged, both reported.

| arm | allocation | Lam spiral | Lam spiral3 | Bald spiral | Bald spiral3 |
|---|---|--:|--:|--:|--:|
| A | 10 × 100%, 30 gens | **0.903** | **0.825** | 0.815 | 0.677 |
| B | 100 × top 10%, 30 gens | 0.877 (n=9) | 0.809 | **0.869** | **0.740** |
| C | 50 × 100%, 6 gens | 0.808 | 0.736 | 0.845 | 0.695 |
| D | successive halving, 30 gens | 0.903 (n=7) | 0.833 (n=8) | 0.879 | 0.683 |
| E | 5 × 100%, 60 gens | 0.877 | 0.793 | 0.851 | 0.673 |

At seeds 0–9: 2 supported, 5 contradicted, 3 inconclusive. **B helps
Baldwinian** (+0.053 / +0.063) and not Lamarckian; halving *won* on spiral
for Baldwinian; only "6 generations hurts Lamarckian" held (−0.095 / −0.089).
I had priced B's weakness (refining genomes chosen by an uninformative score)
but not A's matching weakness under Baldwinian: a 10-step fitness that is
itself near-random. 7 of 150 Lamarckian cells crashed (section 8, item 6).

**Settled:** by mean validation accuracy on the spiral cells, among arms
with no crashed cell — **Lamarckian A** (0.891; D was higher on its
survivors but failed 6 of 30 runs) and **Baldwinian B** (0.839). On untouched
seeds 10–12 three of four committed claims held (Lamarckian A > Baldwinian A
+0.111 / +0.140; Baldwinian B > A +0.056 / +0.036; Lamarckian A > Baldwinian
B); "Lamarckian B < A" did not, so for Lamarckian A and B are equivalent.

## 4. Lamarckian or Baldwinian (deliverable 3)

**Which it does, and the evidence.** Either, by `--strategy`; I checked each
does what I intended. `test_main.py`, from the runs' own logs: Lamarckian
"300/300 genomes changed" vs Baldwinian "0/300"; a Lamarckian elite carried
over starts exactly where its refinement ended (same weight hash, same
loss); a Baldwinian elite repeats an identical refinement; a Baldwinian
champion's genome weights differ from the refined ones it is scored with.
Across the graded allocation runs, Baldwinian refined 387,000 genome
evaluations and changed 0 genomes; Lamarckian refined and changed 368,700
(derived from `fitness_history.csv`). And the population shows it
(`figures/inheritance_signature.png`): at the same allocation the
Lamarckian population's *pre-refinement* accuracy climbs to 0.79 / 0.68 /
0.62, while the Baldwinian population's stays at 0.52 / 0.50 / 0.52 for 30
generations although its refined accuracy rises — its genomes never improve
on their own; only how well they learn is selected.

**The difference**: Lamarckian − Baldwinian at equal allocation, spiral /
spiral3: +0.088 / +0.148 (seeds 0–9), +0.111 / +0.140 (10–12), +0.065 /
+0.153 (20–29). Lamarckian runs also fragment into many species (up to
46–50 per run in A/B/D vs 1–7 Baldwinian).

**Mechanism: hypotheses tested in order, each prediction committed first**
(`PREDICTIONS_mechanism*.md`; Appendix A5–A8):

| round | hypothesis | key prediction → result | what it settled |
|---|---|---|---|
| 1 | **H1 accumulation**: refinement adds up along Lamarckian lineages; Baldwinian restarts from shallow, noisy 10-step fitness | refining Baldwinian deeper closes ≥ half the gap → **supported** on spiral (gap 0.087 → 0.006 at 30 steps × 33%), **partial** on spiral3 (0.148 → 0.085); inverted U beyond that | depth helps Baldwinian until too few genomes are refined |
| | | Lamarckian children start within 0.05 of parents' refined accuracy → **contradicted**: −0.129 (Baldwinian −0.179) | carry-over exists but is lossy |
| | **H2** fragmentation is a side effect | topology-only speciation removes it (**supported**: 1 species, 0 crashes) and the advantage survives → spiral3 **yes** (+0.109), spiral **inconclusive** (+0.047); Lamarckian loses 0.04 without it | partly refuted: weight-aware speciation helps Lamarckian |
| 2 | **H4** crossover vs **H3** perturbation erases the carry-over | clones lose as much as crosses (−0.125 vs −0.129): **H4 refuted**; cutting perturbation 0.8/0.7 → 0.2 leaves the loss at −0.125: **H3 refuted** | something else in mutation |
| 3 | **H5** weight replacement vs **H6** activation swaps | mutating refined genomes one operator at a time: replacement 60% of the damage (**H5 supported**), activation 8–10% (**not**), structural ~35%, perturbation ~30% (overlapping) | replacement is the largest single eraser |
| final, fresh seeds 20–29 | replacement causes the loss | removing it: children keep 0.025 more (predicted ≥ 0.03: **contradicted**, narrowly); final accuracy unchanged (inconclusive); main effect replicates (**supported**) | |

**Where it stands.** Supported: the advantage is robust; inherited
refinement gives a partial head start; Baldwinian's handicap is shallow,
noisy fitness, and deeper refinement of fewer genomes recovers part of the
gap. The carry-over is eroded by mutation as a whole, not one operator, and
recovering part of it did not change accuracy at this budget. Unexplained:
why weight-aware speciation helps Lamarckian; why depth closes the gap on
spiral but not spiral3. I stopped there by a stopping rule committed in
advance.

## 5. Complexity frontier (deliverable 4)

Settled allocations × linear λ ∈ {0, 0.001, 0.003, 0.01, 0.03} and mult
λ ∈ {0.1, 0.3, 1.0} × 3 datasets × 10 seeds (480 runs, `runs/frontier`);
λ = 0 reproduces the allocation grid exactly.

![frontier](figures/frontier.png)

| linear λ | Lam xor | Lam spiral | Lam spiral3 | Bald xor | Bald spiral | Bald spiral3 |
|---|--:|--:|--:|--:|--:|--:|
| 0 | 0.967 (12.3) | 0.903 (10.5) | 0.825 (10.2) | 0.964 (8.7) | 0.869 (9.1) | 0.740 (8.4) |
| 0.001 | 0.968 (7.3) | 0.910 (9.4) | 0.780 (8.3) | 0.973 (7.5) | 0.826 (6.7) | 0.693 (5.5) |
| 0.003 | 0.969 (7.2) | 0.861 (7.9) | 0.779 (6.8) | 0.967 (6.9) | 0.781 (4.9) | 0.676 (4.4) |
| 0.01 | 0.975 (6.6) | 0.855 (5.8) | 0.726 (4.2) | 0.855 (4.7) | 0.729 (3.4) | 0.641 (1.2) |
| 0.03 | 0.928 (5.0) | 0.685 (1.4) | 0.641 (1.2) | 0.585 (1.8) | 0.668 (1.0) | 0.633 (1.3) |

Held-out (mean champion connections), 10 seeds. **Shape:** xor has a free
region — λ = 0.001 removes 5 of 12 connections at no cost (they did nothing),
and accuracy holds to ~6.6 connections; the spirals have none — every
removed connection costs accuracy. Baldwinian sits below Lamarckian at every
λ (10/10) and collapses sooner. **No penalty improved held-out accuracy**
beyond spread anywhere (FP4): the penalty buys size, not accuracy. Five of
six predictions held; the sixth (FP5, mult shrinks xor less than spiral3)
failed revealingly: at mult λ = 1 *every* champion has 0 connections. The
empty network pays no penalty (√0 = 0) and scores its constant-guess
accuracy (~0.51 on fit); a spiral network with error 0.15 and 9 connections
scores 1 − 0.15 × 4 = 0.40. Its held-out (0.487 / 0.480 / 0.433) is below the
majority floor because the training majority is the held-out minority.

## 6. Small-network track (deliverable 5)

`--max-conns 6` (every genome repaired at creation; no champion over 6),
settled allocations, 10 seeds (`runs/smallnet/smallnet_scored.md`;
predictions `PREDICTIONS_smallnet.md`).

| dataset | ≤ 6, Lamarckian | ≤ 6, Baldwinian | ≤ 6 + sin/gauss, Lamarckian *(separate)* | ≤ 6 + sin/gauss, Baldwinian *(separate)* | unconstrained Lamarckian |
|---|--:|--:|--:|--:|--:|
| xor | 0.967 ± 0.019 | 0.956 ± 0.020 | 0.969 ± 0.004 | 0.976 ± 0.006 | 0.967 ± 0.007 (12.3 conns) |
| spiral | 0.826 ± 0.043 | 0.826 ± 0.071 | 0.906 ± 0.034 | 0.849 ± 0.048 | 0.903 ± 0.069 (10.5) |
| spiral3 | 0.779 ± 0.030 | 0.711 ± 0.033 | 0.863 ± 0.049 | 0.816 ± 0.056 | 0.825 ± 0.055 (10.2) |

**Best with ≤ 6 connections, standard activations: xor 0.967, spiral 0.826,
spiral3 0.779** (Lamarckian; differences vs unconstrained 0.000 / −0.077 /
−0.046). With sine and gaussian hidden units — reported separately because
they change NEAT's toolbox — 6 connections reach 0.969 / 0.906 / 0.863,
matching or beating the unconstrained standard networks; 9–10 of 10 sine
champions use them. **But they draw stripes, not spirals**
(`figures/smallnet_sine_representative.png`): diagonal bands on spiral,
vertical bands on spiral3, whose period happens to match the spacing of the
arms. That does improve the inner third (median 0.74 / 0.72 vs 0.60 / 0.54
with standard activations; computed from the champions), but it is aliasing
the arms, not modelling them. Predictions: S1a, S1b (Lamarckian), S2 (3 of 4)
supported; S3 contradicted for Baldwinian (sine helped xor by +0.020); S4
held on spiral3 only. Three graded cells were run during development before
the predictions (their numbers were withheld from me); excluding them flips
two verdicts and moves no mean by more than 0.004.

## 7. Decision boundaries (deliverable 6)

`figures/` (`python experiments/make_figures.py`; readings with seed lists in
`figures/README.md`; all 69 plotted champions' accuracies re-checked against
`summary.json`). Representative = lower-median seed; `seeds_*.png` show all
10 seeds unsorted.

![spiral](figures/representative_spiral.png)
![spiral3](figures/representative_spiral3.png)
![accuracy by radius](figures/radius_accuracy.png)

- **xor — right shape.** All 10 Lamarckian champions split the plane into
  quadrants along the axes; 3 of 10 Baldwinian cut one blob off diagonally
  (wrong orientation, still 0.93–0.95). Plain NEAT: one cut, 0.69–0.71.
- **spiral — partly a spiral.** 6 of 10 Lamarckian champions curl into the
  inner arm (inner-third accuracy 0.76–0.94); the other 4 are corners or
  boxes that score 0.80–0.87 while getting the inner third at or below its
  majority rate. Plain NEAT is one vertical cut (0.69–0.79).
- **spiral3 — mostly not a spiral.** 16 of 20 refined champions are at or
  below the majority rate on the inner third; their 0.67–0.87 comes from
  boxes and stripes fitting the outer turns, where most points are. One
  champion (Lamarckian s5, 0.953) winds with the inner arm.

What the shapes say: held-out accuracy on these spirals mostly rewards
fitting the dense outer turns; ~0.80 is reachable without learning the
spiral, and the gains from refinement and from Lamarckian inheritance are
mostly gains there. No champion changes label along a ray as often as the
data do (≤ 1.87 flips vs 2.43 for 5-NN).

## 8. What went wrong, and how I found it (deliverable 7)

1. **The Baldwinian champion would have been scored with unrefined
   weights.** The shipped driver and `viz.py` rebuilt the champion from its
   genome, which under Baldwinian never holds the refined weights — every
   Baldwinian number would have been understated, in exactly the comparison
   deliverable 3 asks about. Found by auditing the scaffold before writing
   refinement. Fixed with `refined_params` + `champion_net()`; tests check no
   offspring inherits the vector and reported accuracies use it.
2. **The output node's activation mutated** (tanh 43%, relu 10% of genomes)
   while `predict()` thresholds at 0.5 and `forward()` was documented as
   logits. Found by tabulating output ranges in the audit. Fixed by pinning
   the output to sigmoid; the first fix crashed (neat-python reads the ini
   section named after the genome class).
3. **Exit code 1 meant two things and `--generations 0` was silently
   ignored.** An over-budget run and a crash both exited 1; `0 or planned`
   ran 30 generations while reporting 0. Found by the harness agent
   reasoning about exit codes. Over budget now exits 4; tests force an
   overrun and check it is logged and marked invalid.
4. **My learning-rate probe used validation data.** Found reviewing the
   engine agent's report, not by a test. Re-run on the fit split: same lr.
5. **The pilot misled twice.** Its rule recommended 1–3 generations;
   discarded by comparing with runs still improving at generation 29. Then
   predictions built on its findings (uninformative pre-screen, late
   bloomers) failed: B and halving *won* for Baldwinian. How a fixed set of
   genomes ranks is not what decides an evolutionary run.
6. **Lamarckian runs fragmented until neat-python crashed** (7 of 150 cells:
   51 species × 2 > 100 genomes). Found when the harness reported the
   crashes. Kept as results; crashed arms excluded from the choice; became
   hypothesis H2.
7. **My own bookkeeping was wrong in places, and so were most of my
   predictions.** Cross-checking commit messages against result files while
   drafting this report found a miscounted verdict tally, a species range
   generalised from 3 seeds, a "matches" that held for one dataset of two,
   and a stale exit code; each was re-derived and corrected with an appended
   note (`ef2763e`). One commit message (`2b735e8`) described a report edit
   that had not been written; the next commit added it and says so. Of my
   committed predictions, 5/10 allocation, 4 mechanism, 1 frontier and 2
   small-network predictions were contradicted; each is reported as such.

Smaller items (DEVLOG): the finite-difference reference was ill-conditioned
on confidently wrong points; relu nodes sit exactly at z = 0 because new
nodes get bias 0; reusing one `neat.Config` breaks reproducibility; a
`-dirty` commit label does not identify code (145 frontier runs read dirty
while an agent wrote files; the code fingerprint was identical for all 480);
an agent's worktree started from a Q1 commit; the write-up's penalty
collapses networks (section 5); refinement can lower accuracy (it minimises
cross-entropy) — under Baldwinian B it does so in 110 / 83 / 227 of 300
seed-generation cells.

## 9. Time and process

Q2 ran from 18:09 (branch checkout) to ~21:15: about 3 hours of wall-clock
against a budget of about 5. Runs take seconds, so the time went into design
and checking. Timeline: audit and fixes 18:09–18:51; engine, driver and
harness to 19:36; first result 19:38 (`q2-first-result`); pilot to 19:57;
allocation study and confirmation 20:00–20:09; mechanism 20:14–20:25;
frontier 20:33–20:40; figures, small networks, report to ~21:15.

I kept the design, every prediction, and the reading of results, and
delegated build pieces to agents — the gradient engine, the harness, the
small-network machinery (in an isolated worktree), the figures, and the
first draft of this report — re-running each agent's tests and checking its
key outputs before committing. They found real bugs (DEVLOG #3–#7,
#10–#15); my reviews found the validation leak (item 4) and the drafting
agent found my bookkeeping errors (item 7). All five test suites pass on the
final tree.

**Next:** settle Lamarckian A vs B with more seeds; test why weight-aware
speciation helps Lamarckian; give the spirals a fair shot at learning the
inner turns (radius/angle-aware activations or longer runs) and measure that
with the inner-third metric rather than overall accuracy.

---

## Appendix

### A1. Allocation grid, held-out mean ± sd (n), seeds 0–9 (`runs/allocation/predictions_scored_seeds0-9.md`)

| arm | strategy | xor | spiral | spiral3 | `budget.summary()` |
|---|---|--:|--:|--:|---|
| A 10 × 100%, 30 gens | Lam | 0.967 ± 0.007 (10) | 0.903 ± 0.069 (10) | 0.825 ± 0.055 (10) | 30,000/30,000 gradient steps over 3,000 refinements |
| | Bald | 0.973 ± 0.009 (10) | 0.815 ± 0.035 (10) | 0.677 ± 0.061 (10) | same |
| B 100 × top 10%, 30 gens | Lam | 0.967 ± 0.016 (10) | 0.877 ± 0.068 (9) | 0.809 ± 0.061 (10) | 30,000/30,000 gradient steps over 300 refinements |
| | Bald | 0.964 ± 0.018 (10) | 0.869 ± 0.057 (10) | 0.740 ± 0.060 (10) | same |
| C 50 × 100%, 6 gens | Lam | 0.963 ± 0.017 (10) | 0.808 ± 0.038 (10) | 0.736 ± 0.024 (10) | 30,000/30,000 gradient steps over 600 refinements |
| | Bald | 0.928 ± 0.082 (10) | 0.845 ± 0.024 (10) | 0.695 ± 0.033 (10) | same |
| D halving, 30 gens | Lam | 0.966 ± 0.013 (9) | 0.903 ± 0.050 (7) | 0.833 ± 0.040 (8) | 29,520/30,000 gradient steps over 5,790 refinements |
| | Bald | 0.976 ± 0.006 (10) | 0.879 ± 0.059 (10) | 0.683 ± 0.027 (10) | same |
| E 5 × 100%, 60 gens | Lam | 0.968 ± 0.004 (10) | 0.877 ± 0.061 (10) | 0.793 ± 0.045 (10) | 30,000/30,000 gradient steps over 6,000 refinements |
| | Bald | 0.966 ± 0.022 (10) | 0.851 ± 0.037 (10) | 0.673 ± 0.064 (10) | same |

Crashed cells, all with neat-python's per-species-minimum error:
B_lamarckian spiral_s4; D_lamarckian xor_s1, spiral_s0, spiral_s6,
spiral_s9, spiral3_s0, spiral3_s1.

Mean validation accuracy, seeds 0–9 (`runs/allocation/summary.md`):

| arm | Lam xor / spiral / spiral3 | Bald xor / spiral / spiral3 |
|---|---|---|
| A | 0.999 / 0.929 / 0.854 | 0.986 / 0.817 / 0.747 |
| B | 0.984 / 0.897 / 0.866 | 0.984 / 0.881 / 0.797 |
| C | 0.967 / 0.821 / 0.767 | 0.939 / 0.844 / 0.740 |
| D | 0.995 / 0.924 / 0.889 (survivors) | 0.987 / 0.906 / 0.730 |
| E | 0.999 / 0.899 / 0.823 | 0.986 / 0.847 / 0.757 |

### A2. Allocation verdicts (`predictions_scored_seeds0-2.md`, `predictions_scored_seeds0-9.md`)

| prediction | seeds 0–2 | seeds 0–9 |
|---|---|---|
| P1 Bald B < A, spiral | contradicted (+0.027) | contradicted (+0.053) |
| P1 Bald B < A, spiral3 | inconclusive (+0.016) | contradicted (+0.063) |
| P1b smaller gap under Lam, spiral / spiral3 | inconclusive / inconclusive | contradicted / contradicted |
| P2 Lam C < A, spiral | inconclusive (−0.051) | supported (−0.095) |
| P2 Lam C < A, spiral3 | supported (−0.044) | supported (−0.089) |
| P2 Bald C < A, spiral | contradicted (+0.069) | inconclusive (+0.029) |
| P2 Bald C < A, spiral3 | supported (−0.069) | inconclusive (+0.018) |
| P3 Bald D < A, spiral | contradicted (+0.136) | contradicted (+0.064) |
| P3 Bald D < A, spiral3 | supported (−0.053) | inconclusive (+0.005) |
| xor: nothing beyond spread except C | Lam B and D beyond spread | all within spread |

### A3. Confirmation, seeds 10–12 (`runs/confirm/confirm_scored.md`, 36/36 ok)

| arm | xor | spiral | spiral3 |
|---|--:|--:|--:|
| A_lamarckian | 0.967 ± 0.007 | 0.889 ± 0.004 | 0.787 ± 0.037 |
| A_baldwinian | 0.978 ± 0.004 | 0.778 ± 0.014 | 0.647 ± 0.041 |
| B_lamarckian | 0.978 ± 0.010 | 0.913 ± 0.040 | 0.787 ± 0.100 |
| B_baldwinian | 0.971 ± 0.010 | 0.833 ± 0.064 | 0.682 ± 0.020 |

### A4. Pilot (`runs/pilot/pilot.json`, seed 100, 100 genomes per dataset)

| | xor | spiral | spiral3 |
|---|--:|--:|--:|
| Spearman ρ, pre-refinement vs 300-step ranking | 0.069 | 0.073 | 0.089 |
| Spearman ρ, 10-step vs 300-step ranking | 0.707 | 0.017 | 0.272 |
| median share of the 300-step gain at 10 steps | 0.72 | 0.99 | 0.89 |
| 300-step top 10 found in the top 10% by pre-refinement fitness | 0 | 1 | 2 |
| 300-step top 6 that halving would cut | 5 | 6 | 4 |
| halving keeps the 300-step best | no | no | yes |
| the mechanical rule's choice | 100 × 100%, 3 gens | 200 × 100%, 1 gen | 300 × 100%, 1 gen |

### A5. Mechanism round 1 (`runs/mechanism/mechanism_scored.md`, 210/210 ok)

| arm | steps spent | xor | spiral | spiral3 | median max species (spiral / spiral3) |
|---|---|--:|--:|--:|--:|
| bald_d10 (10 × 100%) | 30,000 | 0.973 ± 0.009 | 0.815 ± 0.035 | 0.677 ± 0.061 | 3 / 3 |
| bald_d30 (30 × 33%) | 29,700 | 0.975 ± 0.009 | 0.897 ± 0.038 | 0.695 ± 0.031 | 3 / 4 |
| bald_d100 (100 × 10%) | 30,000 | 0.964 ± 0.018 | 0.869 ± 0.057 | 0.740 ± 0.060 | 3 / 3 |
| bald_d300 (300 × 3%) | 27,000 | 0.912 ± 0.114 | 0.833 ± 0.067 | 0.727 ± 0.061 | 3 / 3 |
| lam_d10 (10 × 100%) | 30,000 | 0.967 ± 0.007 | 0.903 ± 0.069 | 0.825 ± 0.055 | 11 / 8.5 |
| lam_topo | 30,000 | 0.967 ± 0.009 | 0.867 ± 0.052 | 0.786 ± 0.041 | 1 / 1 |
| bald_topo | 30,000 | 0.970 ± 0.019 | 0.820 ± 0.055 | 0.677 ± 0.046 | 1 / 1 |

Median refinement gain in generations 20–29: Lamarckian +0.036, Baldwinian +0.125.

### A6. Mechanism round 2 (`runs/mechanism2/mechanism2_scored.md`)

| median child loss (pre-refinement − parents' refined fit accuracy) | value |
|---|--:|
| Lamarckian clones (n = 2,999) | −0.125 |
| Lamarckian crosses (n = 12,980) | −0.129 |
| clones with unchanged node and connection counts (n = 852) | −0.089 |
| Lamarckian, low perturbation (n = 16,278) | −0.125 |
| Baldwinian, low perturbation (n = 16,870) | −0.207 |

| arm | xor | spiral | spiral3 |
|---|--:|--:|--:|
| lam_lowmut | 0.967 ± 0.007 | 0.908 ± 0.077 | 0.819 ± 0.038 |
| bald_lowmut | 0.920 ± 0.109 | 0.821 ± 0.048 | 0.645 ± 0.050 |

### A7. Mechanism round 3: mean fit-accuracy change per mutant (`runs/mechanism3/ablation.json`; 300 refined Lamarckian genomes per dataset, 20 mutants each, 0 gradient steps)

| operator | spiral (share) | spiral3 (share) |
|---|--:|--:|
| replace | −0.055 (0.61) | −0.038 (0.60) |
| structural | −0.035 (0.38) | −0.021 (0.32) |
| perturb | −0.026 (0.28) | −0.022 (0.34) |
| activation | −0.007 (0.08) | −0.006 (0.10) |
| full default | −0.091 | −0.063 |
| full low-mutation | −0.083 | −0.055 |

### A8. Final test, seeds 20–29 (`runs/mechanism_final/final_scored.md`, 120/120 ok)

| arm | xor | spiral | spiral3 |
|---|--:|--:|--:|
| lam_default | 0.968 ± 0.008 | 0.879 ± 0.047 | 0.807 ± 0.043 |
| bald_default | 0.956 ± 0.036 | 0.813 ± 0.032 | 0.655 ± 0.050 |
| lam_norep | 0.969 ± 0.009 | 0.889 ± 0.059 | 0.828 ± 0.060 |
| bald_norep | 0.955 ± 0.083 | 0.794 ± 0.025 | 0.701 ± 0.057 |

Median Lamarckian child loss: default −0.132 (n = 16,006), without replacement −0.107 (n = 16,672).

### A9. Plain-NEAT baseline (`runs/baseline_plain/*/summary.json`, code `11e5994`)

| dataset | seed 0 | seed 1 | seed 2 | mean ± sd |
|---|--:|--:|--:|--:|
| xor | 0.713 | 0.687 | 0.713 | 0.704 ± 0.015 |
| spiral | 0.760 | 0.793 | 0.693 | 0.749 ± 0.051 |
| spiral3 | 0.600 | 0.607 | 0.653 | 0.620 ± 0.029 |

### A10. First result, seed 0 (`runs/first_result/summary.md`, tag `q2-first-result`; every run 30,000/30,000 gradient steps over 3,000 refinements)

| | xor | spiral | spiral3 |
|---|--:|--:|--:|
| Lamarckian 10 × 100% | 0.973 | 0.800 | 0.780 |
| Baldwinian 10 × 100% | 0.980 | 0.793 | 0.713 |

### A11. Test suites

| suite | checks | result |
|---|---|---|
| `test_gradients.py` | finite differences on 300 evolved and 300 refined genomes, edge cases, Adam, bounds, resumption | `all checks passed` |
| `test_scaffold_fixes.py` | each audit fix: reproduces the failure, asserts it is gone | `all checks passed` |
| `test_main.py` | split, budget, inheritance evidence, halving schedule, champion (incl. penalised), penalty, overrun, determinism | `all checks passed` |
| `experiments/tests/test_harness.py` | 27 failure-path checks against a fake driver | 27/27 per commit `3372a86` |

### A12. Small-network track (`runs/smallnet/smallnet_scored.md`, 120/120 ok)

| arm (≤ 6 connections) | xor | spiral | spiral3 |
|---|--:|--:|--:|
| lam_std | 0.967 ± 0.019 (6.0) | 0.826 ± 0.043 (5.5) | 0.779 ± 0.030 (6.0) |
| bald_std | 0.956 ± 0.020 (6.0) | 0.826 ± 0.071 (5.4) | 0.711 ± 0.033 (5.7) |
| lam_sine | 0.969 ± 0.004 (5.7) | 0.906 ± 0.034 (6.0) | 0.863 ± 0.049 (6.0) |
| bald_sine | 0.976 ± 0.006 (5.9) | 0.849 ± 0.048 (5.8) | 0.816 ± 0.056 (5.9) |

Excluding the three cells run during development: lam_std spiral 0.830
(n=9), spiral3 0.782 (n=9), bald_sine spiral 0.849 (n=9); verdicts S1b
Lamarckian spiral3 → inconclusive, S4 spiral → holds (+0.004).

### A13. Complexity frontier, mult form (`runs/frontier/frontier_scored.md`)

| mult λ | Lam xor | Lam spiral | Lam spiral3 | Bald xor | Bald spiral | Bald spiral3 |
|---|--:|--:|--:|--:|--:|--:|
| 0.1 | 0.967 (9.1) | 0.831 (6.6) | 0.700 (4.3) | 0.663 (3.3) | 0.729 (4.4) | 0.624 (1.6) |
| 0.3 | 0.967 (8.9) | 0.759 (4.0) | 0.641 (1.2) | 0.567 (1.2) | 0.667 (1.3) | 0.643 (1.1) |
| 1.0 | 0.487 (0.0) | 0.480 (0.0) | 0.433 (0.0) | 0.487 (0.0) | 0.480 (0.0) | 0.433 (0.0) |

# Backprop-NEAT Under a Fixed Gradient Budget: Allocation, Inheritance, and Network Size

*ASTRA take-home, Question 2. All numbers are traceable to committed files,
named in table captions. Predictions for each experiment were committed
before the experiment ran (`experiments/PREDICTIONS_*.md`). Bugs and
corrections are logged in `AUDIT.md` and `DEVLOG.md`.*

## Abstract

We combine NEAT, which evolves network topology, with gradient-based weight
refinement on three two-dimensional classification tasks (xor and two
spirals), under a fixed budget of 30,000 gradient steps per run. We ask how
to allocate that budget across genomes and generations, whether refined
weights should be inherited (Lamarckian) or used only to score fitness
(Baldwinian), how a size penalty trades accuracy for network size, and what
accuracy is reachable with at most six connections. Lamarckian inheritance
outperforms Baldwinian inheritance on both spirals by 7–15 percentage
points, a result replicated on three disjoint sets of random seeds. The two
strategies prefer opposite allocations: Lamarckian works best when every
genome is refined lightly, Baldwinian when a few genomes are refined deeply.
We trace the Lamarckian advantage to refinement carried forward through
inheritance, but find that mutation erases most of it each generation. A
linear size penalty reduces network size without improving accuracy.
Finally, decision-boundary analysis shows that most spiral champions score
well by fitting the densely sampled outer turns rather than by learning the
spiral. Several of our predictions failed; we report them alongside those
that held.

## 1. Introduction

NEAT searches over network topologies but adjusts weights only by random
mutation, which is slow. Gradient descent adjusts weights efficiently but
cannot change topology. Backprop-NEAT combines the two: each genome's
network is briefly trained by gradient descent ("refined") during
evaluation, and evolution acts on the result. Refinement costs compute, so
under a fixed budget every gradient step spent refining one genome is a step
not spent on another genome or another generation.

This report addresses four questions:

1. **Allocation.** How should 30,000 gradient steps be divided between depth
   (steps per genome), breadth (genomes refined per generation) and
   duration (generations)?
2. **Inheritance.** Should refined weights be written back into the genome
   so that offspring inherit them (Lamarckian), or discarded after setting
   fitness (Baldwinian)? Why does one outperform the other?
3. **Complexity.** How does a penalty on network size trade accuracy for
   size?
4. **Small networks.** What accuracy is reachable with at most six enabled
   connections?

**Summary of findings.**
- With the best allocation for each strategy, held-out accuracy is 0.967 /
  0.903 / 0.825 (Lamarckian) and 0.964 / 0.869 / 0.740 (Baldwinian) on xor /
  spiral / spiral3, against 0.704 / 0.749 / 0.620 for plain NEAT.
- Lamarckian beats Baldwinian at equal allocation on both spirals, on three
  disjoint seed sets.
- The best allocation depends on the inheritance strategy.
- A linear size penalty buys smaller networks, never more accurate ones.
- High spiral accuracy mostly reflects fitting the outer turns, not
  learning the spiral.

**Where each deliverable is answered.**

| Deliverable | Section |
|---|---|
| 1. Working refinement and held-out accuracy | 2, 3 |
| 2. Allocation study | 4 |
| 3. Lamarckian or Baldwinian: evidence, mechanism and test | 5 |
| 4. Complexity frontier | 6 |
| 5. Six-or-fewer-connection result | 7 |
| 6. Decision boundaries | 8 |
| 7. What went wrong | 10 |

## 2. Setup

### 2.1 Tasks and data

Each task has 500 labelled points in the plane, split by the provided code
into 350 training and 150 held-out points (frozen; not regenerated).

- **xor**: four Gaussian blobs; diagonally opposite blobs share a label.
  Because the blobs overlap slightly, the best achievable accuracy is 0.984.
- **spiral** and **spiral3**: two interleaved spiral arms, one per class,
  making 1.30 and 1.50 revolutions respectively. (The "turns" parameter in
  the provided code is in units of π radians, so spiral3 has 1.5 turns, not
  3.) The arms are well separated: a nearest-neighbour classifier reaches
  0.99 and 1.00 on the held-out points.

Two properties of the data matter for interpreting results. First, the
majority-class baseline differs by task: 0.513, 0.520 and 0.567 on the
held-out sets. Second, points are sampled more densely at larger radius, so
on the spirals a single straight cut through the outer arm already scores
about 0.75 (`AUDIT.md`). With 150 held-out points, one point is 0.67
percentage points.

### 2.2 Networks and refinement

Each genome is executed as a feed-forward network whose sigmoid output gives
P(class 1). We pinned the output node's activation to sigmoid; in the
provided configuration it could mutate to tanh or relu, which is
incompatible with a probability output (Section 10). Hidden nodes use
sigmoid, tanh or relu.

**Refinement** minimises mean binary cross-entropy on the training data
with full-batch Adam (learning rate 0.3, chosen on training data only). We
wrote the backward pass by hand and verified it against finite differences
on 300 evolved genomes, before and after refinement, and on hand-built edge
cases. After each update, parameters are clipped to NEAT's ±30 weight range,
so the weights written back to a genome are exactly the weights that were
evaluated.

**Budget accounting.** One gradient step is one full-batch gradient
evaluation followed by one Adam update. We use no line searches,
second-order steps or finite-difference gradients during runs; forward
passes are free, as in the provided code. The budget is charged for the
steps actually taken. Every reported run spent at most 30,000 steps.

### 2.3 Evolution and inheritance

NEAT uses a population of 100 with the provided configuration. A genome's
fitness is its training accuracy after refinement (minus a size penalty in
Section 6). Genomes that are not refined keep their unrefined accuracy.
Elites carried over to the next generation are refined again under both
strategies, so that both follow the same protocol.

- **Lamarckian**: refined weights are written back into the genome and are
  inherited by its offspring.
- **Baldwinian**: the genome keeps its original weights. The refined weights
  set its fitness and are stored separately so that the final champion is
  evaluated with them; offspring never see them.

### 2.4 Protocol

**Data use.** The 350 training points are split once (stratified, with a
fixed seed shared by all runs) into 280 *fit* points, used for refinement
and fitness, and 70 *validation* points, used only to choose the champion
and to compare configurations. The 150 held-out points are used only for
reporting. Each generation's best genome is a candidate; the run's
*champion* is the candidate with the highest validation accuracy.

**Seeds and spread.** Each configuration is run with 10 random seeds unless
stated. We report mean ± sample standard deviation over seeds.

**Predictions and verdicts.** Before each experiment we committed written
predictions. A predicted difference between two configurations is scored
*supported* if it has the predicted sign and its magnitude exceeds the
typical seed spread, √((sd₁² + sd₂²)/2); *contradicted* if it has the
opposite sign by the same margin; and *inconclusive* otherwise.

## 3. Main results

Table 1 reports the two configurations we settled on (Section 4).

*Table 1. Held-out accuracy of the settled configurations. Seeds 10–12 were
never used for any decision. Sources: `runs/allocation/summary.md`,
`runs/confirm/confirm_scored.md`.*

| Task | Configuration | Seeds 0–9 | Seeds 10–12 | Connections | Budget used |
|---|---|--:|--:|--:|---|
| xor | Lamarckian, 10 steps × all genomes | 0.967 ± 0.007 | 0.967 ± 0.007 | 12.3 | 30,000/30,000 steps over 3,000 refinements |
| xor | Baldwinian, 100 steps × top 10% | 0.964 ± 0.018 | 0.971 ± 0.010 | 8.7 | 30,000/30,000 steps over 300 refinements |
| spiral | Lamarckian, 10 steps × all genomes | 0.903 ± 0.069 | 0.889 ± 0.004 | 10.5 | 30,000/30,000 steps over 3,000 refinements |
| spiral | Baldwinian, 100 steps × top 10% | 0.869 ± 0.057 | 0.833 ± 0.064 | 9.1 | 30,000/30,000 steps over 300 refinements |
| spiral3 | Lamarckian, 10 steps × all genomes | 0.825 ± 0.055 | 0.787 ± 0.037 | 10.2 | 30,000/30,000 steps over 3,000 refinements |
| spiral3 | Baldwinian, 100 steps × top 10% | 0.740 ± 0.060 | 0.682 ± 0.020 | 8.4 | 30,000/30,000 steps over 300 refinements |

For reference, plain NEAT without refinement (30 generations, 3 seeds)
reaches 0.704 ± 0.015, 0.749 ± 0.051 and 0.620 ± 0.029. Refinement helps on
every task. xor reaches its ceiling; both spirals remain well below theirs.
Seed-to-seed variation is large (Lamarckian spiral ranges from 0.800 to
0.980).

## 4. Allocating the gradient budget

### 4.1 Design

We compared five allocations (Table 2), each with both inheritance
strategies. Where only some genomes are refined, they are chosen by their
unrefined fitness.

*Table 2. Allocations compared. All spend 30,000 steps except D (29,520).*

| | Steps per refined genome | Genomes refined per generation | Generations |
|---|--:|--:|--:|
| A | 10 | 100 (all) | 30 |
| B | 100 | 10 (top 10%) | 30 |
| C | 50 | 100 (all) | 6 |
| D | successive halving: every genome gets 2 steps, the best half 4 more, and so on for five rounds | 100 → 6 | 30 |
| E | 5 | 100 (all) | 60 |

### 4.2 Pilot and predictions

Before the comparison, a pilot refined 100 genomes per task for 300 steps
each and measured how informative short refinement is (`runs/pilot`, seed
100, not used elsewhere). Unrefined fitness barely predicted refined fitness
(Spearman correlation 0.07–0.09), and on the spirals the ranking after 10
steps was almost unrelated to the ranking after 300 (0.02 and 0.27).

A rule we wrote in advance converted these measurements into an allocation
and recommended 100–300 steps per genome with only 1–3 generations. We
rejected it: it valued accurate ranking of a fixed set of genomes but gave
no value to further generations, while our first runs were still improving
at generation 29. Instead, we predicted from the pilot that (P1) B would
lose to A under Baldwinian inheritance, because B selects whom to refine by
an uninformative score; (P2) C would lose to A, because six generations are
too few to grow networks; and (P3) D would lose to A under Baldwinian
inheritance, because successive halving discards genomes that improve late.

### 4.3 Results

*Table 3. Held-out accuracy by allocation, seeds 0–9 (xor in Appendix A1).
n < 10 where runs crashed (Section 5.2). Source: `runs/allocation/summary.md`.*

| | Lamarckian spiral | Lamarckian spiral3 | Baldwinian spiral | Baldwinian spiral3 |
|---|--:|--:|--:|--:|
| A | **0.903** ± 0.069 | **0.825** ± 0.055 | 0.815 ± 0.035 | 0.677 ± 0.061 |
| B | 0.877 ± 0.068 (n=9) | 0.809 ± 0.061 | **0.869** ± 0.057 | **0.740** ± 0.060 |
| C | 0.808 ± 0.038 | 0.736 ± 0.024 | 0.845 ± 0.024 | 0.695 ± 0.033 |
| D | 0.903 ± 0.050 (n=7) | 0.833 ± 0.040 (n=8) | 0.879 ± 0.059 | 0.683 ± 0.027 |
| E | 0.877 ± 0.061 | 0.793 ± 0.045 | 0.851 ± 0.037 | 0.673 ± 0.064 |

We first ran three seeds. Their verdicts were mixed (3 supported, 3
contradicted, 4 inconclusive), with spiral and spiral3 disagreeing in
direction. After seeing them, we added seeds 3–9 without changing the
predictions or the scoring rule. With ten seeds, 2 predictions were
supported, 5 contradicted and 3 inconclusive. Three of the six clear
verdicts at three seeds did not survive.

The main contradiction is informative. Deep refinement of a few genomes
(B) *helped* Baldwinian inheritance by 5–6 points and did not help
Lamarckian. We had accounted for B's weakness (it refines genomes selected
by an uninformative score) but not for A's weakness under Baldwinian
inheritance: A's 10-step fitness is itself nearly uninformative, as the
pilot showed. Only one prediction was supported on both spirals: reducing
evolution to six generations (C) hurts Lamarckian inheritance (−9.5 and
−8.9 points).

### 4.4 Choice and confirmation

We chose, for each strategy, the allocation with the highest mean
validation accuracy on the spiral tasks, among allocations in which no run
crashed: **A for Lamarckian** (0.891) and **B for Baldwinian** (0.839).
Lamarckian D had higher validation accuracy among its surviving runs but
crashed in 6 of 30, so we did not choose it.

We then tested four claims on seeds 10–12 (Table 4). Three held. The fourth
(B worse than A for Lamarckian) did not, so for Lamarckian inheritance A and
B are indistinguishable.

*Table 4. Confirmation on unused seeds 10–12: mean difference in held-out
accuracy. Source: `runs/confirm/confirm_scored.md`.*

| Claim | Spiral | Spiral3 | Holds |
|---|--:|--:|:-:|
| Lamarckian A > Baldwinian A | +0.111 | +0.140 | yes |
| Baldwinian B > Baldwinian A | +0.056 | +0.036 | yes |
| Lamarckian A > Baldwinian B | +0.056 | +0.104 | yes |
| Lamarckian B < Lamarckian A | +0.024 | +0.000 | no |

## 5. Lamarckian versus Baldwinian inheritance

### 5.1 Verifying each strategy

The implementation supports both strategies (`--strategy`). We checked that
each does what it is meant to do, from the runs' own logs. Under Lamarckian
inheritance, every refined genome's stored weights changed (300 of 300 in
the test suite; 368,700 of 368,700 refined evaluations across the
allocation study). Under Baldwinian inheritance, none did (0 of 300; 0 of
387,000). An elite carried into the next generation starts, under
Lamarckian inheritance, exactly where its refinement ended, and under
Baldwinian inheritance repeats an identical refinement.

Figure 1 shows the population-level signature. At the same allocation, the
Lamarckian population's *unrefined* accuracy rises over 30 generations to
0.79, 0.68 and 0.62, while the Baldwinian population's stays at 0.52, 0.50
and 0.52 even as its refined accuracy rises. Under Baldwinian inheritance,
evolution selects genomes that learn well, but their own weights never
improve.

![inheritance signature](figures/inheritance_signature.png)
*Figure 1. Population-mean training accuracy before (dashed) and after
(solid) refinement, per generation; thin lines are individual seeds.*

### 5.2 The effect

At equal allocation (A), Lamarckian minus Baldwinian held-out accuracy is
+0.088 and +0.148 on spiral and spiral3 (seeds 0–9), +0.111 and +0.140
(seeds 10–12), and +0.065 and +0.153 (seeds 20–29). On xor both are at the
ceiling.

Lamarckian runs also split into many more NEAT species: up to 46–50 per run
under allocations A, B and D, against 1–7 under Baldwinian inheritance. In 7
of 150 Lamarckian runs (allocations B and D) the species count exceeded what
a population of 100 can support, and neat-python stopped the run. These runs
are reported as crashes, not rerun.

### 5.3 Mechanism

We tested hypotheses in sequence, each with predictions committed before
the experiment (Table 5).

*Table 5. Mechanism experiments. Seeds 0–9 unless stated; details in
Appendix A5–A8.*

| Hypothesis | Test | Result | Conclusion |
|---|---|---|---|
| **H1.** Lamarckian refinement accumulates along lineages; Baldwinian restarts each generation from a short, noisy refinement. | Refine Baldwinian genomes more deeply at fixed generations. | The gap to Lamarckian shrinks from 8.7 to 0.6 points on spiral (30 steps × top 33%) but only from 14.8 to 8.5 on spiral3; deeper still is worse. | Partly supported. Depth helps Baldwinian until too few genomes are refined. |
| | Compare a child's unrefined accuracy with its parents' refined accuracy. | Lamarckian children start 12.9 points below their parents (predicted: within 5); Baldwinian children 17.9 below. | Inherited refinement gives a head start of about 5 points, much less than predicted. |
| **H2.** Species fragmentation is a harmless side effect. | Group species by topology only, ignoring weights. | Fragmentation and crashes disappear. The Lamarckian advantage remains on spiral3 (+10.9) but halves on spiral (+4.7, inconclusive); Lamarckian accuracy drops about 4 points, Baldwinian is unchanged. | Partly refuted: weight-based speciation helps Lamarckian runs. |
| **H3/H4.** The head start is lost to small weight perturbations (H3) or to crossover (H4). | Compare clones (both parents the same genome) with crosses; separately, reduce the perturbation rate from 0.8 to 0.2. | Clones lose as much as crosses (12.5 vs 12.9 points); reducing perturbation leaves the loss unchanged. | Both refuted. |
| **H5/H6.** The head start is lost to weight replacement (H5) or to activation changes (H6). | Mutate refined genomes with one mutation operator at a time. | Replacement causes about 60% of the accuracy damage, structural changes about 35%, perturbation about 30%, activation changes 8–10% (shares overlap). | Replacement is the largest single cause; H6 refuted. |
| Final test of H5 (fresh seeds 20–29) | Disable replacement. | Children keep 2.5 more points (predicted: at least 3); final accuracy unchanged within noise; the main effect replicates. | Replacement explains part of the loss, and recovering it does not change accuracy at this budget. |

**Interpretation.** The evidence supports a partial account. Lamarckian
inheritance passes refinement to offspring, giving them a head start.
Baldwinian inheritance must re-learn every generation from a refinement too
short to rank genomes reliably, and recovers part of the gap when fewer
genomes are refined more deeply. The inherited head start is mostly erased
by mutation as a whole rather than by any one operator. Two observations
remain unexplained: why weight-based speciation helps Lamarckian runs, and
why deeper Baldwinian refinement closes the gap on spiral but not on
spiral3. We stopped after the final test, as planned in advance.

## 6. Complexity penalty

### 6.1 Definition

Let C be the number of enabled connections and acc the training accuracy
after refinement. We implemented two penalties:

- **Linear**: fitness = acc − λC. Each connection must improve accuracy by
  at least λ to be worth keeping, regardless of task or network size. This
  is our main penalty because λ has this direct interpretation.
- **Multiplicative**: fitness = 1 − (1 − acc)(1 + λ√C). This is our reading
  of the backprop-NEAT write-up, which describes multiplying fitness by a
  factor growing with the square root of the connection count but gives no
  formula. Applying the factor to the error makes larger networks worse.

With a penalty, the champion is chosen by penalised validation accuracy, so
that selection does not undo the penalty.

### 6.2 Results

We swept linear λ ∈ {0, 0.001, 0.003, 0.01, 0.03} and multiplicative
λ ∈ {0.1, 0.3, 1.0} (480 runs; Figure 2, Table 6).

![frontier](figures/frontier.png)
*Figure 2. Held-out accuracy against champion size as the penalty strength
varies. Top row: linear penalty; bottom row: multiplicative. Error bars are
standard deviations over 10 seeds; the dashed line is the majority-class
baseline.*

*Table 6. Linear penalty: held-out accuracy (mean champion connections), 10
seeds. Source: `runs/frontier/frontier_scored.md`; multiplicative results in
Appendix A13.*

| λ | Lamarckian xor | Lamarckian spiral | Lamarckian spiral3 | Baldwinian xor | Baldwinian spiral | Baldwinian spiral3 |
|---|--:|--:|--:|--:|--:|--:|
| 0 | 0.967 (12.3) | 0.903 (10.5) | 0.825 (10.2) | 0.964 (8.7) | 0.869 (9.1) | 0.740 (8.4) |
| 0.001 | 0.968 (7.3) | 0.910 (9.4) | 0.780 (8.3) | 0.973 (7.5) | 0.826 (6.7) | 0.693 (5.5) |
| 0.003 | 0.969 (7.2) | 0.861 (7.9) | 0.779 (6.8) | 0.967 (6.9) | 0.781 (4.9) | 0.676 (4.4) |
| 0.01 | 0.975 (6.6) | 0.855 (5.8) | 0.726 (4.2) | 0.855 (4.7) | 0.729 (3.4) | 0.641 (1.2) |
| 0.03 | 0.928 (5.0) | 0.685 (1.4) | 0.641 (1.2) | 0.585 (1.8) | 0.668 (1.0) | 0.633 (1.3) |

The frontier differs by task. On xor, the smallest penalty removes 5 of 12
connections at no cost in accuracy, and accuracy holds down to about 6.6
connections. On the spirals there is no such free region: accuracy falls
with every connection removed. Baldwinian inheritance lies below
Lamarckian at every penalty strength and loses accuracy sooner. No
penalised configuration was more accurate than the unpenalised one: the
penalty buys smaller networks, not better ones. Five of our six predictions
held.

### 6.3 The multiplicative penalty collapses

The sixth prediction, that the multiplicative penalty would shrink xor
networks less than spiral3 networks, failed in an informative way. At
λ = 1, every champion in every run had zero connections and predicted a
single class. An empty network incurs no penalty (√0 = 0) and scores the
constant-guess accuracy, about 0.51; a spiral network with 15% error and 9
connections scores 1 − 0.15 × 4 = 0.40. Under this reading of the
write-up, the empty network is optimal once λ is large enough. Its held-out
accuracy (0.487, 0.480, 0.433) falls below the majority baseline because the
training majority class is the held-out minority.

## 7. Networks with at most six connections

Every genome is repaired when it is created so that it has at most six
enabled connections: connections that do not affect the output are disabled
first, then those with the smallest weights. No champion exceeds the limit,
and the budget means the same as before. Adding sine and Gaussian hidden
activations changes NEAT's set of building blocks, so we report that
configuration separately.

*Table 7. Held-out accuracy with at most six connections, 10 seeds. Source:
`runs/smallnet/smallnet_scored.md`.*

| Task | Lamarckian | Baldwinian | Lamarckian + sine/Gaussian (separate) | Baldwinian + sine/Gaussian (separate) | Unconstrained Lamarckian |
|---|--:|--:|--:|--:|--:|
| xor | 0.967 ± 0.019 | 0.956 ± 0.020 | 0.969 ± 0.004 | 0.976 ± 0.006 | 0.967 ± 0.007 (12.3 conns) |
| spiral | 0.826 ± 0.043 | 0.826 ± 0.071 | 0.906 ± 0.034 | 0.849 ± 0.048 | 0.903 ± 0.069 (10.5) |
| spiral3 | 0.779 ± 0.030 | 0.711 ± 0.033 | 0.863 ± 0.049 | 0.816 ± 0.056 | 0.825 ± 0.055 (10.2) |

**With the standard activations, the best results with six or fewer
connections are 0.967 (xor), 0.826 (spiral) and 0.779 (spiral3)**, against
0.967, 0.903 and 0.825 unconstrained: no loss on xor, and 7.7 and 4.6 points
lost on the spirals.

With sine and Gaussian units, six connections match or exceed the
unconstrained standard networks (0.906 and 0.863). Nine or ten of every ten
champions use these units. However, Figure 3 shows that they do not draw
spirals: they draw parallel stripes whose spacing happens to match the
spacing of the arms. This does improve accuracy near the centre (median
0.74 and 0.72 on the innermost third of points, against 0.60 and 0.54 with
standard activations), but the stripes match the arms only along one
direction.

![sine](figures/smallnet_sine_representative.png)
*Figure 3. Median-seed Lamarckian champions with sine/Gaussian units and at
most six connections. Left: spiral (0.900). Right: spiral3 (0.873).*

Most predictions held (Appendix A12). Two failed: sine units also helped
Baldwinian xor slightly (+2.0 points), and the Lamarckian advantage held on
spiral3 but not on spiral. Three of the graded runs were executed during
development, before the predictions were written; their results were not
seen when the predictions were written, and excluding them changes no mean
by more than 0.004.

## 8. Decision boundaries

Figure 4 shows the median-seed champion for each spiral task; all ten seeds
per configuration are shown in `figures/seeds_*.png`, and xor in
`figures/representative_xor.png`. To measure whether the inner turns are
learned, Figure 5 splits the held-out points into three equal-count rings
by distance from the centre.

![spiral](figures/representative_spiral.png)
![spiral3](figures/representative_spiral3.png)
*Figure 4. Median-seed champions: plain NEAT, Lamarckian (10 steps × all
genomes), Baldwinian (100 steps × top 10%). Top: spiral. Bottom: spiral3.*

![radius](figures/radius_accuracy.png)
*Figure 5. Held-out accuracy within the inner, middle and outer third of
points by radius; one dot per seed. Dotted line: majority class within the
ring. Dashed line: a 5-nearest-neighbour reference.*

- **xor.** All ten Lamarckian champions divide the plane into four
  quadrants along the axes, the correct shape. Three of ten Baldwinian
  champions instead cut off one blob with a diagonal line, the wrong shape,
  yet score 0.933–0.953. Plain NEAT draws a single cut (0.687–0.713).
- **spiral.** Six of ten Lamarckian champions curl inward along the inner
  arm (0.76–0.94 accuracy on the inner third). The other four are corners or
  boxes that score 0.800–0.867 overall while doing no better than the
  majority class on the inner third. Plain NEAT draws one vertical cut.
- **spiral3.** Sixteen of the twenty refined champions do no better than
  the majority class on the inner third. Their 0.67–0.87 overall accuracy
  comes from boxes and stripes fitted to the outer turns, where most points
  lie. One champion (0.953) follows the inner arm.

No champion changes its predicted class along a ray from the centre as
often as the data do (at most 1.87 changes, against 2.43 for the
nearest-neighbour reference). Overall accuracy on these spirals therefore
mainly measures the fit to the outer turns; about 0.80 is reachable without
learning the spiral. The accuracy gains from refinement and from Lamarckian
inheritance are largely gains on the outer turns.

## 9. Discussion and limitations

The clearest results are that refinement helps on every task, that
Lamarckian inheritance outperforms Baldwinian inheritance on the spirals,
and that the best allocation depends on the inheritance strategy. The
mechanism behind the Lamarckian advantage is only partly established, and
the boundary analysis shows that most champions do not learn the spirals:
the networks found within this budget (8–12 connections) fit the dense
outer turns but rarely the inner ones.

Several limitations apply. Ten seeds and 150 held-out points give wide
intervals; differences below about five points should be read with
caution. Seeds 0–9 were used for many decisions, so effects measured on
them may be slightly optimistic; the main effect was therefore re-tested on
seeds 10–12 and 20–29. The plain-NEAT baseline was run before the
validation split was introduced and with three seeds. Finally, the pilot's
measurements did not predict the allocation results well, which suggests
that how refinement ranks a fixed set of genomes is a poor guide to how an
evolutionary run will behave.

## 10. What went wrong, and how it was found

1. **Baldwinian champions would have been scored with unrefined weights.**
   The provided driver and plotting code rebuilt the champion from its
   genome, which under Baldwinian inheritance never holds the refined
   weights. Every Baldwinian result would have been understated, precisely
   in the comparison deliverable 3 asks about. *Found* by auditing the
   provided code before writing refinement. *Fixed* by storing the refined
   weights with the genome and using them for evaluation and plotting;
   tests confirm that offspring never inherit them.
2. **The output node's activation could mutate.** In 53% of evaluated
   genomes the output was tanh or relu, while prediction thresholds the
   output at 0.5 and the documentation called it a logit. *Found* by
   tabulating output ranges during the audit. *Fixed* by pinning the output
   to sigmoid; the first fix crashed because neat-python reads the
   configuration section named after the genome class.
3. **An exit code had two meanings, and a zero was silently ignored.** Our
   first version signalled a budget overrun with exit code 1, which Python
   also uses for any crash, and `--generations 0` ran 30 generations while
   reporting 0. *Found* by the experiment-harness agent while reasoning
   about exit codes. *Fixed* with a dedicated exit code, input validation,
   and a test that forces an overrun.
4. **The learning-rate probe used validation data.** It tuned on all 350
   training points, including the 70 used to choose champions. *Found*
   while reviewing the agent-written probe. *Fixed* by restricting it to the
   fit points; the chosen learning rate did not change.
5. **The pilot misled twice.** Its rule recommended 1–3 generations, which
   we rejected by comparing with runs still improving at generation 29. The
   predictions we then based on its findings were mostly contradicted
   (Section 4.3).
6. **Lamarckian runs fragmented until neat-python stopped them** (7 of 150
   runs). *Found* through the harness's crash report. Crashed runs were
   kept as results, excluded from the allocation choice, and led to
   hypothesis H2.
7. **Our own records contained errors.** Cross-checking commit messages
   against result files while drafting this report found a miscounted
   verdict tally, a species range generalised from three seeds, an
   overstated comparison, and an outdated exit code in `AUDIT.md`. Each was
   re-derived and corrected with an appended note rather than rewritten.
   One commit message described a report edit that had not been saved; the
   following commit made the edit and says so.

Smaller issues are listed in `DEVLOG.md`, including an ill-conditioned
reference in the gradient check, relu nodes created exactly at their kink,
a reproducibility bug from reusing configuration objects, and a false
warning in the provided `package_submission.sh`: it reports that Q2 has no
training runs because its file search fails under `pipefail` when there are
many files.

## 11. Reproducibility, time and process

All experiments run from `q2_backprop_neat/` with the experiment harness,
for example `python experiments/run_grid.py experiments/grids/allocation.json`.
A single settled run is
`python main.py --dataset spiral --seed 0 --lr 0.3 --strategy lamarckian --steps 10 --refine-fraction 1.0`
(Baldwinian: `--strategy baldwinian --steps 100 --refine-fraction 0.1`).
Five test suites cover the gradient, the fixes to the provided code, the
driver, the six-connection limit and the harness; all pass.

The work took about 3 hours 10 minutes of the suggested 5. Runs take
seconds, so most of the time went into design and verification. The author
set the questions, wrote every prediction and interpreted every result.
Implementation pieces were delegated to coding agents (the gradient engine,
the experiment harness, the six-connection machinery, the figures and a
first draft of this report); each agent's tests were re-run and its outputs
checked before committing. The agents found several real bugs (items 3 and
6 above, and `DEVLOG.md` #5–#15); reviewing their work found the validation
leak (item 4); the drafting agent found the record errors (item 7).

**Future work.** Resolve whether allocations A and B differ for
Lamarckian inheritance; test why weight-based speciation helps Lamarckian
runs; and evaluate the spirals by inner-ring accuracy rather than overall
accuracy, with configurations that can represent the inner turns.

---

## Appendix

### A1. Allocation study, all tasks (seeds 0–9)

*Held-out accuracy, mean ± sd (n). Source:
`runs/allocation/predictions_scored_seeds0-9.md`.*

| Allocation | Strategy | xor | spiral | spiral3 | Budget used |
|---|---|--:|--:|--:|---|
| A: 10 × 100%, 30 gens | Lamarckian | 0.967 ± 0.007 (10) | 0.903 ± 0.069 (10) | 0.825 ± 0.055 (10) | 30,000 over 3,000 refinements |
| | Baldwinian | 0.973 ± 0.009 (10) | 0.815 ± 0.035 (10) | 0.677 ± 0.061 (10) | same |
| B: 100 × top 10%, 30 gens | Lamarckian | 0.967 ± 0.016 (10) | 0.877 ± 0.068 (9) | 0.809 ± 0.061 (10) | 30,000 over 300 |
| | Baldwinian | 0.964 ± 0.018 (10) | 0.869 ± 0.057 (10) | 0.740 ± 0.060 (10) | same |
| C: 50 × 100%, 6 gens | Lamarckian | 0.963 ± 0.017 (10) | 0.808 ± 0.038 (10) | 0.736 ± 0.024 (10) | 30,000 over 600 |
| | Baldwinian | 0.928 ± 0.082 (10) | 0.845 ± 0.024 (10) | 0.695 ± 0.033 (10) | same |
| D: halving, 30 gens | Lamarckian | 0.966 ± 0.013 (9) | 0.903 ± 0.050 (7) | 0.833 ± 0.040 (8) | 29,520 over 5,790 |
| | Baldwinian | 0.976 ± 0.006 (10) | 0.879 ± 0.059 (10) | 0.683 ± 0.027 (10) | same |
| E: 5 × 100%, 60 gens | Lamarckian | 0.968 ± 0.004 (10) | 0.877 ± 0.061 (10) | 0.793 ± 0.045 (10) | 30,000 over 6,000 |
| | Baldwinian | 0.966 ± 0.022 (10) | 0.851 ± 0.037 (10) | 0.673 ± 0.064 (10) | same |

Crashed runs (neat-python's per-species minimum): Lamarckian B spiral seed
4; Lamarckian D xor seed 1, spiral seeds 0, 6, 9, and spiral3 seeds 0, 1.

*Mean validation accuracy (xor / spiral / spiral3), used for the choice in
Section 4.4.*

| Allocation | Lamarckian | Baldwinian |
|---|---|---|
| A | 0.999 / 0.929 / 0.854 | 0.986 / 0.817 / 0.747 |
| B | 0.984 / 0.897 / 0.866 | 0.984 / 0.881 / 0.797 |
| C | 0.967 / 0.821 / 0.767 | 0.939 / 0.844 / 0.740 |
| D | 0.995 / 0.924 / 0.889 (surviving runs) | 0.987 / 0.906 / 0.730 |
| E | 0.999 / 0.899 / 0.823 | 0.986 / 0.847 / 0.757 |

### A2. Allocation predictions and verdicts

*Difference from allocation A in held-out accuracy. Sources:
`runs/allocation/predictions_scored_seeds0-2.md`,
`runs/allocation/predictions_scored_seeds0-9.md`.*

| Prediction | Seeds 0–2 | Seeds 0–9 |
|---|---|---|
| P1 Baldwinian B < A, spiral | contradicted (+0.027) | contradicted (+0.053) |
| P1 Baldwinian B < A, spiral3 | inconclusive (+0.016) | contradicted (+0.063) |
| P1b B − A smaller under Lamarckian, spiral / spiral3 | inconclusive / inconclusive | contradicted / contradicted |
| P2 Lamarckian C < A, spiral | inconclusive (−0.051) | supported (−0.095) |
| P2 Lamarckian C < A, spiral3 | supported (−0.044) | supported (−0.089) |
| P2 Baldwinian C < A, spiral | contradicted (+0.069) | inconclusive (+0.029) |
| P2 Baldwinian C < A, spiral3 | supported (−0.069) | inconclusive (+0.018) |
| P3 Baldwinian D < A, spiral | contradicted (+0.136) | contradicted (+0.064) |
| P3 Baldwinian D < A, spiral3 | supported (−0.053) | inconclusive (+0.005) |

### A3. Confirmation (seeds 10–12)

*Held-out accuracy; 36 of 36 runs completed. Source:
`runs/confirm/confirm_scored.md`.*

| Configuration | xor | spiral | spiral3 |
|---|--:|--:|--:|
| Lamarckian A | 0.967 ± 0.007 | 0.889 ± 0.004 | 0.787 ± 0.037 |
| Baldwinian A | 0.978 ± 0.004 | 0.778 ± 0.014 | 0.647 ± 0.041 |
| Lamarckian B | 0.978 ± 0.010 | 0.913 ± 0.040 | 0.787 ± 0.100 |
| Baldwinian B | 0.971 ± 0.010 | 0.833 ± 0.064 | 0.682 ± 0.020 |

### A4. Pilot measurements

*100 genomes per task from a Baldwinian run (seed 100), each refined for 300
steps. Source: `runs/pilot/pilot.json`.*

| | xor | spiral | spiral3 |
|---|--:|--:|--:|
| Spearman correlation, unrefined vs 300-step ranking | 0.069 | 0.073 | 0.089 |
| Spearman correlation, 10-step vs 300-step ranking | 0.707 | 0.017 | 0.272 |
| Median share of the 300-step gain reached after 10 steps | 0.72 | 0.99 | 0.89 |
| Eventual top 10 found in the top 10% by unrefined fitness | 0 | 1 | 2 |
| Eventual top 6 that successive halving would discard | 5 | 6 | 4 |
| Allocation recommended by the (rejected) rule | 100 steps, 3 gens | 200 steps, 1 gen | 300 steps, 1 gen |

### A5. Mechanism, round 1

*Held-out accuracy, 10 seeds. Source: `runs/mechanism/mechanism_scored.md`.*

| Configuration | Steps used | xor | spiral | spiral3 | Median max. species (spiral / spiral3) |
|---|---|--:|--:|--:|--:|
| Baldwinian, 10 × 100% | 30,000 | 0.973 ± 0.009 | 0.815 ± 0.035 | 0.677 ± 0.061 | 3 / 3 |
| Baldwinian, 30 × 33% | 29,700 | 0.975 ± 0.009 | 0.897 ± 0.038 | 0.695 ± 0.031 | 3 / 4 |
| Baldwinian, 100 × 10% | 30,000 | 0.964 ± 0.018 | 0.869 ± 0.057 | 0.740 ± 0.060 | 3 / 3 |
| Baldwinian, 300 × 3% | 27,000 | 0.912 ± 0.114 | 0.833 ± 0.067 | 0.727 ± 0.061 | 3 / 3 |
| Lamarckian, 10 × 100% | 30,000 | 0.967 ± 0.007 | 0.903 ± 0.069 | 0.825 ± 0.055 | 11 / 8.5 |
| Lamarckian, topology-only speciation | 30,000 | 0.967 ± 0.009 | 0.867 ± 0.052 | 0.786 ± 0.041 | 1 / 1 |
| Baldwinian, topology-only speciation | 30,000 | 0.970 ± 0.019 | 0.820 ± 0.055 | 0.677 ± 0.046 | 1 / 1 |

Median accuracy gain from refinement in generations 20–29: Lamarckian
+0.036, Baldwinian +0.125.

### A6. Mechanism, round 2

*Median loss of a child's unrefined accuracy relative to its parents'
refined accuracy. Source: `runs/mechanism2/mechanism2_scored.md`.*

| Children | Loss |
|---|--:|
| Lamarckian clones (n = 2,999) | −0.125 |
| Lamarckian crosses (n = 12,980) | −0.129 |
| Clones with unchanged structure (n = 852) | −0.089 |
| Lamarckian, reduced perturbation (n = 16,278) | −0.125 |
| Baldwinian, reduced perturbation (n = 16,870) | −0.207 |

Held-out accuracy with reduced perturbation: Lamarckian 0.967 / 0.908 /
0.819, Baldwinian 0.920 / 0.821 / 0.645 (xor / spiral / spiral3).

### A7. Mechanism, round 3: damage by mutation operator

*Mean change in training accuracy per mutant (share of the full mutation's
damage); 300 refined Lamarckian genomes per task, 20 mutants each. Source:
`runs/mechanism3/ablation.json`.*

| Operator | spiral | spiral3 |
|---|--:|--:|
| Weight/bias replacement | −0.055 (0.61) | −0.038 (0.60) |
| Structural (add/remove nodes and connections) | −0.035 (0.38) | −0.021 (0.32) |
| Weight/bias perturbation | −0.026 (0.28) | −0.022 (0.34) |
| Activation change | −0.007 (0.08) | −0.006 (0.10) |
| Full default mutation | −0.091 | −0.063 |
| Full mutation with reduced perturbation | −0.083 | −0.055 |

### A8. Mechanism, final test (seeds 20–29)

*Held-out accuracy; 120 of 120 runs completed. Source:
`runs/mechanism_final/final_scored.md`.*

| Configuration | xor | spiral | spiral3 |
|---|--:|--:|--:|
| Lamarckian, default | 0.968 ± 0.008 | 0.879 ± 0.047 | 0.807 ± 0.043 |
| Baldwinian, default | 0.956 ± 0.036 | 0.813 ± 0.032 | 0.655 ± 0.050 |
| Lamarckian, no replacement | 0.969 ± 0.009 | 0.889 ± 0.059 | 0.828 ± 0.060 |
| Baldwinian, no replacement | 0.955 ± 0.083 | 0.794 ± 0.025 | 0.701 ± 0.057 |

Median child loss under Lamarckian inheritance: −0.132 (default), −0.107
(no replacement).

### A9. Plain NEAT baseline

*Held-out accuracy, 30 generations, no refinement. Source:
`runs/baseline_plain/*/summary.json`.*

| Task | Seed 0 | Seed 1 | Seed 2 | Mean ± sd |
|---|--:|--:|--:|--:|
| xor | 0.713 | 0.687 | 0.713 | 0.704 ± 0.015 |
| spiral | 0.760 | 0.793 | 0.693 | 0.749 ± 0.051 |
| spiral3 | 0.600 | 0.607 | 0.653 | 0.620 ± 0.029 |

### A10. First result (tag `q2-first-result`)

*Seed 0, 10 steps × all genomes, 30,000 steps over 3,000 refinements.*

| | xor | spiral | spiral3 |
|---|--:|--:|--:|
| Lamarckian | 0.973 | 0.800 | 0.780 |
| Baldwinian | 0.980 | 0.793 | 0.713 |

### A11. Test suites

| Suite | What it checks |
|---|---|
| `test_gradients.py` | Gradient against finite differences (evolved, refined and edge-case genomes; all activations), Adam, clipping, resumption |
| `test_scaffold_fixes.py` | Each fix to the provided code reproduces the original failure and confirms it is gone |
| `test_main.py` | Data split, budget accounting, inheritance evidence, halving schedule, champion selection, penalty, overrun, determinism |
| `test_smallnet.py` | Connection limit and repair; unchanged behaviour without the limit |
| `experiments/tests/test_harness.py` | 27 failure cases of the experiment harness |

### A12. Six-connection limit: verdicts and sensitivity

*Source: `runs/smallnet/smallnet_scored.md`; predictions in
`experiments/PREDICTIONS_smallnet.md`.*

| Prediction | Result |
|---|---|
| The limit costs xor little (both strategies) | supported |
| The limit costs the spirals accuracy | supported for Lamarckian; inconclusive for Baldwinian |
| Sine/Gaussian units help the spirals | supported in 3 of 4 cases (Baldwinian spiral inconclusive) |
| Sine/Gaussian units do not help xor | supported for Lamarckian; contradicted for Baldwinian (+0.020) |
| The Lamarckian advantage persists under the limit | holds on spiral3 (+0.069); not on spiral (+0.000) |

Excluding the three runs executed during development: Lamarckian spiral
0.830 (n = 9), Lamarckian spiral3 0.782 (n = 9), Baldwinian sine spiral
0.849 (n = 9). Two verdicts change: the spiral3 cost of the limit for
Lamarckian becomes inconclusive, and the Lamarckian advantage on spiral
becomes a negligible +0.004.

### A13. Multiplicative penalty

*Held-out accuracy (mean champion connections), 10 seeds. Source:
`runs/frontier/frontier_scored.md`.*

| λ | Lamarckian xor | Lamarckian spiral | Lamarckian spiral3 | Baldwinian xor | Baldwinian spiral | Baldwinian spiral3 |
|---|--:|--:|--:|--:|--:|--:|
| 0.1 | 0.967 (9.1) | 0.831 (6.6) | 0.700 (4.3) | 0.663 (3.3) | 0.729 (4.4) | 0.624 (1.6) |
| 0.3 | 0.967 (8.9) | 0.759 (4.0) | 0.641 (1.2) | 0.567 (1.2) | 0.667 (1.3) | 0.643 (1.1) |
| 1.0 | 0.487 (0.0) | 0.480 (0.0) | 0.433 (0.0) | 0.487 (0.0) | 0.480 (0.0) | 0.433 (0.0) |

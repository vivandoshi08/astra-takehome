# Q2 figures: decision boundaries and fitness curves

Every figure here except `frontier.png` comes from one command, run from `q2_backprop_neat/`:

    python experiments/make_figures.py

It takes about 20 s and trains nothing. It loads committed `champion.pkl`, `summary.json` and `fitness_history.csv` files from `runs/`, and writes everything below plus `champions.csv`, the per-champion table behind the numbers quoted here. (`frontier.png` comes from `experiments/score_frontier.py`.)

## Conventions

- **Data drawn**: the held-out split, 150 points; one point is 0.67%. The colours, shading and extent (data ± 0.25) match `viz.plot_boundary`. Class 0 is blue and class 1 is orange; the shading is `predict()`. The thin black line is `forward() = 0.5`, the threshold `predict()` uses.
- **Champions** are the validation-selected `champion.pkl`, rebuilt with `network.champion_net()`. Baldwinian champions are therefore drawn with the refined parameters that earned their fitness.
- **Panel titles** give the held-out accuracy, the enabled connections and the node count. Both counts come from `network.complexity()`: "nodes" counts the output and hidden nodes but not the inputs, and enabled connections include ones that never reach the output (AUDIT.md #8).
- **Runs used**:
  - `runs/allocation/A_lamarckian` (10 steps x 100% of genomes), seeds 0–9.
  - `runs/allocation/B_baldwinian` (100 steps x top 10%), seeds 0–9. These two are the settled allocations.
  - `runs/allocation/A_baldwinian`, seeds 0–9, used only in `inheritance_signature.png`.
  - `runs/baseline_plain`, plain NEAT with no refinement, seeds 0–2. These runs come from the older `main.py` at 11e5994: they have no validation split, so the champion was chosen on training accuracy, and they ran 30 generations with 0 gradient steps. Their pickles are `SigmoidOutputGenome`, not `DefaultGenome`, and load with `load_config()` without change.
- **Representative ("median") seed**: sort an arm's seeds by (held-out accuracy, seed) and take position (n−1)//2. That is the lower median, with ties in accuracy going to the lower seed. For 10 seeds this is the 5th of 10, and for plain NEAT's 3 seeds it is the true median.
  - The rule picks between two seeds with equal accuracy for A_lamarckian on spiral (s3, s4 at 0.913), A_lamarckian on xor (s4, s6 at 0.967) and B_baldwinian on xor (s0, s3 at 0.973).
  - In the other three cases it picks the lower of two different values, so the representative is, if anything, slightly pessimistic.
- **Accuracy cross-check**: the held-out accuracy recomputed for every plotted champion (69 of them) matches its `summary.json` exactly, 69/69. The script prints any mismatch and exits 1 if there is one.
- **Shape descriptions** ("curl", "box", "stripes") are my readings of the images. Each one lists the seeds it covers so you can check it against the panels. The numeric checks below do not depend on that reading.

## Numeric checks used below (spirals only)

- **Accuracy by radius third**: held-out accuracy within equal-count thirds of r = |x|, 50 points each, so one seed's value has a standard error of about ±0.07. On these spirals radius tracks position along the arm (r = 1.1·t / (turns·π)), so the inner third is the inner 0.71 revolutions on spiral and the inner 0.86 on spiral3. Point density grows with r, so most points sit on the outer arms; the inner third is where a boundary has to wind to score.
- **Label changes per ray**: the mean number of times the predicted label flips along 360 rays from the origin, for r in [0, 1.1].
  - A straight cut gives at most 1 per ray.
  - The generating spirals alternate about 2.6 (spiral) and 3.0 (spiral3) times.
  - A 5-NN classifier on the training split gives 2.43 and 2.66.
  - Caveat: stripes also produce flips, so this count does not tell winding from stripes on its own.

---

## 1. `seeds_xor.png`, `seeds_spiral.png`, `seeds_spiral3.png`: all 10 seeds

**What it shows**: for one dataset, the champions of all 10 seeds in seed order, unsorted. The top row is A_lamarckian and the bottom row is B_baldwinian. The framed panel is the representative seed. This is the view that is not cherry-picked.

`seeds_plain.png` is the same for plain NEAT's 3 seeds on every dataset.

## 2. `representative_xor.png`, `representative_spiral.png`, `representative_spiral3.png`: the main figures

**What it shows**: the median-seed champion of plain NEAT, A_lamarckian and B_baldwinian side by side at readable size.

### xor

- **The refined champions have the right shape.** All 10 A_lamarckian champions split the plane into four quadrant-like regions whose boundaries run roughly along the axes. They score 0.960–0.980, against a Bayes rate of 0.984.
- **B_baldwinian is less consistent.** 7/10 of its champions make the same four-region split (s4 and s8 with curved or closed regions). The other 3/10 (s1, s7, s9) cut the bottom-left class-0 blob off with one straight diagonal line, so class 1 becomes a diagonal band. That is the wrong orientation, yet those champions still score 0.933–0.953 because the blobs are compact. They are B's three lowest xor scores.
- **Plain NEAT fails with one cut.** All three plain-NEAT champions are a single straight or gently curved cut around the top-left blob (`seeds_plain.png`), leaving the bottom-right class-1 blob in class 0. A cut like that can get at most 3 of the 4 blobs right, and they score 0.687–0.713.

### spiral

- **Plain NEAT is a crude boundary that scores well.** All three champions are one near-vertical cut that takes the left part of the outer class-1 arm (s1's cut bends at the bottom). They score 0.693–0.793: the AUDIT.md "~75% from a crude boundary" case.
- **Refinement makes boundaries curl into the centre, but not reliably.** 6/10 A_lamarckian champions (s2, s3, s4, s6, s7, s8) run a class-1 tongue down the inner arm. Their inner-third accuracy is 0.76–0.94 and they score 0.913–0.980.
- **The other four A_lamarckian champions are the wrong shape and still score 0.800–0.867.** s0, s1, s5 and s9 are a corner or box with at most a notch. Their inner-third accuracy is 0.40–0.60, at or below the 0.60 majority rate in that third.
- **B_baldwinian is split three ways:**
  - 3/10 curl (s3, s4, s7).
  - 3/10 are vertical stripes (s0, s2, s8). These are axis-aligned bands that get the centre partly right but cut through the arms further out (s0, s2: outer third 0.74/0.76; s8: middle third 0.76).
  - 4/10 are a box or cut with one tongue (s1, s5, s6, s9).
- **The representatives show the contrast.** A s3 is a hook that follows the arms for about one turn. B s5 is a box that leaves the inner class-1 arm (x ≈ 0.3–0.5) inside the class-0 region.
- **No champion winds as far as the data does.** The maximum is 1.87 label changes per ray (A s6), against 2.43 for 5-NN.

### spiral3

This is the clearest case of "scores well, wrong shape".

- **Plain NEAT barely beats the majority class.** Every champion is one straight or gently bent cut that separates the bottom outer class-1 arm. They score 0.600–0.653, against a majority-class floor of 0.567.
- **All refined champions but one score 0.667–0.873 with boxes, U shapes, V-notches, S-curves and stripes around the outer arm, without resolving the centre.**
  - On the inner third (the inner 0.86 revolutions), 16/20 refined champions (8/10 in each arm) score at or below the 0.54 majority rate there.
  - The median A_lamarckian champion (s6, 0.800) is axis-aligned boxes and stripes. Its 0.800 comes entirely from the outer two thirds: inner third 0.54, middle 0.96, outer 0.90.
- **Only A_lamarckian s5 (0.953, inner third 0.98) actually curls with the inner arm.** s8 (0.873, inner third 0.74) does so partly.
- **Three B_baldwinian champions are close to a single cut.** s0, s7 and s8 are a horizontal cut with a step and score 0.667–0.720.

## 3. `radius_accuracy.png`: is the inner turn learned?

**What it shows**: held-out accuracy in each radius third, one dot per seed. Plain NEAT has 3 seeds and the two settled arms have 10 each. The black ticks are seed medians. The dashed line is 5-NN on the training split and the dotted line is the majority-class rate within that third.

**Reading**:

- **The inner third is where the refined champions fail.**
  - On spiral, nearly all of their errors are there. The A_lamarckian and B_baldwinian medians are 1.00 on the middle and outer thirds but 0.77 and 0.69 on the inner third, where 5-NN gets 0.98.
  - On spiral3, the inner-third medians are 0.53 (A) and 0.46 (B), against 1.00 for 5-NN and a 0.54 majority rate. B also loses accuracy in the middle third (median 0.85).
- **The overall scores therefore come mostly from the outer arms.** The radius split says more about whether a spiral was learned than the headline accuracy does.
- **Plain NEAT is at or below the majority rate in the inner third.** It scores 0.58–0.60 on spiral (majority 0.60) and 0.44 on spiral3 (majority 0.54).

## 4. `wide_view.png`: artifacts away from the data

**What it shows**: the nine representative champions on [-3, 3]². The dashed box is the region the other plots show.

**Reading**:

- **Inside the plotted box but off the data**, several xor champions carry slivers of the other class (visible in `seeds_xor.png`):
  - A s1 has a class-0 notch under the bottom-right blob.
  - A s5 has class-0 strips along the top-left and bottom-right edges.
  - B s1 and s7 have a class-1 top-right corner.
  - B s8 has a class-1 bottom-left corner.
- **Beyond the box, xor's quadrant structure does not hold.**
  - A s4 turns class 1 for x ≳ 1.8 in the top-right quadrant.
  - B s0's boundaries flatten into a band, so the far bottom-left corner is class 1.
  - The held-out data never tests this, so it is a remark about extrapolation, not an error in the score.
- **The spiral champions extend as roughly straight lines** past the box, with no far-off islands. spiral3 A s6's vertical class-0 stripe continues to the bottom edge. That stripe is also what misclassifies the bottom of the outer class-1 arm at x ≈ 0.5–0.75.

## 5. `fitness_representative.png`: fitness curves of the representative runs

**What it shows**: per generation, the population best (blue) and mean (orange) fit-split accuracy. Solid lines are after that generation's refinement and dashed lines are before it. The penalty is 0 in these runs, so fitness equals post-refinement fit accuracy and `mean_fitness == mean_acc`; the script asserts this. The dotted vertical line is the generation the validation-selected champion came from.

**Reading** (the numbers are printed by the script, in the `fitness ...` and `signature ...` lines):

- **Lamarckian: the before-refinement curves track the after-refinement curves.**
  - The two best curves cross back and forth by a few hundredths and end within 0.01 of each other, because elites carry their refined weights into the next generation.
  - The mean before refinement climbs and ends 0.05–0.09 below the mean after it (gap in the last generation: 0.085 / 0.086 / 0.051 on xor / spiral / spiral3).
- **Baldwinian: the mean curves overlap and the gap shows only in the best.**
  - Only 10 of 100 genomes are refined, so the means coincide (|after − before| ≤ 0.013 over the last 10 generations).
  - In the last generation, the best after refinement is 0.136 above the best before it on xor, 0.164 on spiral and 0.079 on spiral3.
- **In early generations, B's best after refinement is below its best before.**
  - This holds in generations 0–4 (xor), 0–2 (spiral) and 1–6 (spiral3) of these runs, by up to 0.16 on xor: refinement lowered the fit accuracy of the top genomes.
  - Refinement minimises cross-entropy, not accuracy, so a dip at generation 0 is common to all arms. The best after refinement is below the best before at generation 0 in 10/10 / 10/10 / 9/10 B_baldwinian seeds (xor / spiral / spiral3), but also in 0/10 / 9/10 / 6/10 A_lamarckian seeds.
  - What is specific to B is that it persists. The mean effect of refinement on the genomes it refines is negative in 110 / 83 / 227 of 300 seed-generation cells for B_baldwinian, against 9 / 0 / 0 for A_lamarckian (`mean_gain_refined`).
- **B's best curve goes flat early on the spirals.** In these runs it last rises at generation 18 (spiral) and 12 (spiral3); xor creeps up until generation 25. Across all 10 seeds, the best fitness last rose at median generation 24.5 / 16 / 22 for B_baldwinian, against 26.5 / 28.5 / 28 for A_lamarckian (xor / spiral / spiral3).

## 6. `inheritance_signature.png`: the inheritance signature, all 10 seeds

**What it shows**: population-mean fit-split accuracy before (dashed) and after (solid) each generation's refinement. Thin lines are single seeds and bold lines are the median. The columns are A_lamarckian, A_baldwinian (the same 10 steps x 100% allocation, so inheritance is the only difference from A_lamarckian) and B_baldwinian.

**Reading** (last-generation medians are printed by the script):

- **Lamarckian: the population's own weights improve.** The before-refinement mean climbs from about 0.50 to 0.789 / 0.680 / 0.617 (xor / spiral / spiral3) and ends 0.05–0.07 under the after-refinement mean (0.861 / 0.746 / 0.670).
- **Baldwinian at the same allocation: the before-refinement mean stays at about 0.5** (0.524 / 0.501 / 0.515 in the last generation) for all 30 generations, while the after-refinement mean reaches 0.721 / 0.661 / 0.611. The genomes' own weights stay near chance. Selection sees only what refinement makes of them.
- **B_baldwinian: before ≈ after** (0.594 / 0.622 / 0.596 against 0.603 / 0.625 / 0.596).
  - 90% of the population is unrefined, so selection acts on raw weights and they do improve.
  - Refinement's mean effect on the 10% it touches is small or negative. In `fitness_history.csv`, `mean_gain_refined` is below 0 in 110/300 (xor), 83/300 (spiral) and 227/300 (spiral3) seed-generation cells. The A arms have 0–9/300.
- **Summary**: the dashed curve rising toward the solid one is the visible signature of Lamarckian inheritance. A dashed curve stuck near 0.5 under the same allocation is the Baldwinian one.

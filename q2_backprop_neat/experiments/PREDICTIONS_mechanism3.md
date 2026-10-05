# Mechanism round 3: which mutation operator erases inherited refinement — committed before running

## Where round 2 left it

Lamarckian children lose ~0.125 of their parents' refined fit accuracy.
Crossover is not the cause (clones lose as much as crosses), and neither is
the perturbation rate (cutting weight/bias perturbation 0.8/0.7 -> 0.2 left
the loss at 0.125). Clones with unchanged node/connection counts still
lose 0.089, so a mutation operator the low-mutation config did not touch
remains. Candidates:

- **H5 — replacement**: weight_replace_rate and bias_replace_rate 0.1 reset a
  parameter to a fresh N(0, 1) draw. Lamarckian weights sit near +-30, so a
  reset erases them; with ~10 parameters ~65% of children get one.
- **H6 — activation mutation**: activation_mutate_rate 0.1 swaps a node's
  function (sigmoid / tanh / relu), changing what its refined weights mean.

## Test C (`experiments/mutation_ablation.py`)

Refined Lamarckian genomes: populations at generations 10, 20, 29 of a
Lamarckian 10 x 100% run, seed 100 (no grid uses it), spiral and spiral3
(300 genomes each). Each genome is mutated 20 times by one operator in
isolation (every other rate set to 0), and its fit accuracy compared with
the unmutated genome's. Operators: perturb (weight 0.8 / bias 0.7, power
0.5), replace (weight / bias 0.1), activation (0.1), structural (node add
0.3 / delete 0.1, connection add 0.5 / delete 0.2, enable toggle 0.01), the
full default mutation, and the full low-mutation config of round 2.
No gradient steps are spent.

Statistic: mean accuracy change per mutant; share = mean change of an
operator / mean change of the full default mutation.

## Predictions

- **C1** (consistency with round 2): full-default and full-low-mutation
  mean changes are within 0.02 of each other.
- **C2** (consistency with round 2): perturb share <= 0.25.
- **H5** supported if the replace share >= 0.4.
- **H6** supported if the activation share >= 0.4.
- Structural share is reported, not predicted.

If neither H5 nor H6 reaches 0.4, the loss is spread across operators (or
comes from their interaction) and no single operator explains it.

## Then

The operator(s) that explain the loss are switched off for Lamarckian and
Baldwinian runs on fresh seeds 20–29 (predictions committed before that
run), as the final test of the narrowed mechanism.

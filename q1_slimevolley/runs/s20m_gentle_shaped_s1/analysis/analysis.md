# Run analysis: `runs/s20m_gentle_shaped_s1`

Counts: PASS 51, WARN 12, INFO 15

| check | result | verdict |
|---|---|---|
| 1. Budget: total steps within limit | 19,530,965 / 20,000,000 (97.65%) | **PASS** |
| 1. Budget: budget total vs last cum_steps | gap 0 | **PASS** |
| 1. Budget: run_args.steps_used == budget total | equal | **PASS** |
| 1. Budget: sum(gen_steps) vs last cum_steps | equal | **PASS** |
| 1. Budget: gen_steps == diff(cum_steps) per row | consistent | **PASS** |
| 1. Budget: episode-level steps vs budget total | gap 0 steps (0.000% of total) | **PASS** |
| 1. Budget: episode-level steps vs last cum_steps | equal | **PASS** |
| 2. Fitness consistency: raw_fitness == mean(scores) | consistent | **PASS** |
| 2. Fitness consistency: fitness == mean(score + B*steps/3000 + R*min(returns, cap)) | consistent | **PASS** |
| 2. Fitness consistency: scores == points_for - points_against | consistent | **PASS** |
| 2. Fitness consistency: scores are integers in [-5, 5] | all in range | **PASS** |
| 2. Fitness consistency: per-episode list lengths agree | consistent | **PASS** |
| 2. Fitness consistency: timeouts == count(steps >= 3000) | consistent | **PASS** |
| 2. Fitness consistency: best_fitness recompute | consistent | **PASS** |
| 2. Fitness consistency: mean_fitness recompute | consistent | **PASS** |
| 2. Fitness consistency: median_fitness recompute | consistent | **PASS** |
| 2. Fitness consistency: min_fitness recompute | consistent | **PASS** |
| 2. Fitness consistency: best_raw recompute | consistent | **PASS** |
| 2. Fitness consistency: mean_raw recompute | consistent | **PASS** |
| 2. Fitness consistency: mean_ep_steps recompute | consistent | **PASS** |
| 2. Fitness consistency: timeout_frac recompute | consistent | **PASS** |
| 2. Fitness consistency: std_fitness recompute | matches population sd (ddof=0) | **PASS** |
| 2. Fitness consistency: best_nodes matches a best-fitness genome | consistent | **PASS** |
| 2. Fitness consistency: best_conns matches a best-fitness genome | consistent | **PASS** |
| 2. Fitness consistency: val_mean == mean(val_scores) | consistent | **PASS** |
| 2. Fitness consistency: val_scores integers in [-5, 5] | all in range | **PASS** |
| 2. Fitness consistency: validation.train_fitness == genomes fitness or raw_fitness | consistent | **PASS** |
| 2. Fitness consistency: history val_best == max(val_mean) of that gen | consistent | **PASS** |
| 2. Fitness consistency: champion_val is the running max of val_mean | consistent | **PASS** |
| 2. Fitness consistency: champion.json is the argmax of val_mean | consistent | **PASS** |
| 3. Seed hygiene: seed ranges disjoint | all three ranges disjoint | **PASS** |
| 4. Seed difficulty confound: elite delta (individual) | seed noise sd per genome = 0.301 fitness units | **INFO** |
| 4. Seed difficulty confound: seed noise vs best_fitness movement | 76.3% of best-fitness jitter is matched by seed shifts; r^2 = share of dB variance explained by seed difficulty | **WARN** |
| 5. Winner's curse (train vs validation): train raw - val_mean over validated genomes | mean gap 0.080, 95% CI [0.051, 0.109] (positive = training overstates) | **WARN** |
| 5. Winner's curse (train vs validation): champion train raw vs val | gap 0.050 | **INFO** |
| 6. Validation noise: champion val_mean 95% CI | -4.650 +- 0.314  [-4.964, -4.336] | **PASS** |
| 6. Validation noise: candidates statistically tied with champion | 213 tied candidates | **WARN** |
| 7. Reward-hacking / degenerate-behavior signals: timeout fraction over time | 0.0% of late episodes run to 3000 steps | **PASS** |
| 7. Reward-hacking / degenerate-behavior signals: episode length vs score (last gen) | r = nan | **INFO** |
| 7. Reward-hacking / degenerate-behavior signals: population action fractions | see behavior.png | **INFO** |
| 7. Reward-hacking / degenerate-behavior signals: champion action pattern | champion does act_fwd > 95% of steps | **WARN** |
| 7. Reward-hacking / degenerate-behavior signals: points_for concentration (champion's training episodes) | share 0.600 (0.20 = evenly spread) | **INFO** |
| 8. Learning signal: val_best slope | -0.057 points per 100 generations | **WARN** |
| 8. Learning signal: val_best first vs last quartile | diff -0.043 | **WARN** |
| 8. Learning signal: champion_val start -> end | +0.100 | **INFO** |
| 9. Population health: n_species | 2 species at the end | **PASS** |
| 9. Population health: mean genome size growth | x0.56 connections | **PASS** |
| 9. Population health: fraction of population at exactly -5 | 100.0% of the final population scores -5 every episode | **WARN** |
| 10. Reward shaping (survival / return bonus): bonus share of selection-fitness variance | 51.4% of late selection spread is shaping bonus, not score | **WARN** |
| 10. Reward shaping (survival / return bonus): shaping variance split: survival vs return term | see numbers | **INFO** |
| 10. Reward shaping (survival / return bonus): stalling for survival | no stalling signature | **PASS** |
| 11. Behaviour-log consistency (contacts, returns, cancel, saturation): returns <= contacts per episode | consistent | **PASS** |
| 11. Behaviour-log consistency (contacts, returns, cancel, saturation): contact lists: lengths and integrality | consistent | **PASS** |
| 11. Behaviour-log consistency (contacts, returns, cancel, saturation): act_both <= min(act_fwd, act_back) | consistent | **PASS** |
| 11. Behaviour-log consistency (contacts, returns, cancel, saturation): out_sat in [0, 1] | consistent | **PASS** |
| 11. Behaviour-log consistency (contacts, returns, cancel, saturation): validation: val_returns <= val_contacts, lengths agree | consistent | **PASS** |
| 11. Behaviour-log consistency (contacts, returns, cancel, saturation): mean_contacts recompute | consistent | **PASS** |
| 11. Behaviour-log consistency (contacts, returns, cancel, saturation): mean_returns recompute | consistent | **PASS** |
| 11. Behaviour-log consistency (contacts, returns, cancel, saturation): mean_cancel recompute | consistent | **PASS** |
| 11. Behaviour-log consistency (contacts, returns, cancel, saturation): mean_sat recompute | consistent | **PASS** |
| 11. Behaviour-log consistency (contacts, returns, cancel, saturation): mean_nodes recompute | consistent | **PASS** |
| 11. Behaviour-log consistency (contacts, returns, cancel, saturation): mean_conns recompute | consistent | **PASS** |
| 11. Behaviour-log consistency (contacts, returns, cancel, saturation): best_raw_contacts recompute | consistent | **PASS** |
| 12. Ball contacts as a learning signal: mean_contacts slope | 0.054 contacts/episode per 100 generations | **WARN** |
| 12. Ball contacts as a learning signal: mean_contacts first vs last quarter | diff 0.023 | **WARN** |
| 12. Ball contacts as a learning signal: mean_contacts vs reference policies | x1.12 random play; 0.4% of the way from random to the built-in policy | **INFO** |
| 12. Ball contacts as a learning signal: mean_returns slope | 0.120 returns/episode per 100 generations | **WARN** |
| 12. Ball contacts as a learning signal: mean_returns first vs last quarter | diff 0.078 | **WARN** |
| 12. Ball contacts as a learning signal: mean_returns vs reference policies | x6.76 random play; 2.3% of the way from random to the built-in policy | **INFO** |
| 12. Ball contacts as a learning signal: best_raw_contacts start -> late | x1.26 random play (late) | **INFO** |
| 12. Ball contacts as a learning signal: contacts vs score (episode level) | r = 0.045 (positive = touching/returning the ball is what earns points) | **INFO** |
| 12. Ball contacts as a learning signal: returns vs score (episode level) | r = 0.151 (positive = touching/returning the ball is what earns points) | **INFO** |
| 13. Degenerate behaviour (cancel, juggling, saturation): champion stands still (fwd+back cancel) | cancels on 0.0% of steps | **PASS** |
| 13. Degenerate behaviour (cancel, juggling, saturation): juggling exploit (touch without returning) | contacts not high enough to juggle | **PASS** |
| 13. Degenerate behaviour (cancel, juggling, saturation): output saturation | 68.4% saturated | **INFO** |
| 13. Degenerate behaviour (cancel, juggling, saturation): population fwd+back cancel | 38.9% of late steps press both directions (no movement) | **INFO** |
| 14. Search health (structure drift, NEAT stagnation): mean_conns trend | -44.4% connections | **PASS** |
| 14. Search health (structure drift, NEAT stagnation): NEAT stagnation / extinction (stdout.log) | 0 species removed for stagnation, 0 total extinctions | **INFO** |

Plots: fitness_curve.png, seed_noise.png, behavior.png, complexity.png, contacts.png

# analyze_run: runs/s20m_gentle_shaped_s1
- run_args.json: present
- budget_log.jsonl: 31540 rows
- fitness_history.csv: 83 rows
- genomes.jsonl: 5312 rows
- validation.jsonl: 249 rows
- champion.json: present

## 1. Budget

budget_log.jsonl: 31540 flush records from 9 processes
### [PASS] total steps within limit
- formula:   total = sum over pid of max(cum_steps); require total <= budget
- numbers:   2,136,711 + 2,176,482 + 2,163,852 + 2,155,804 + 2,200,819 + 2,181,052 + 2,181,019 + 2,171,644 + 2,163,582 = 19,530,965; budget = 20,000,000
- result:    19,530,965 / 20,000,000 (97.65%)
- threshold: FAIL if total > budget

per-process cap = budget // workers = 2,222,222; processes that hit it: 0 of 9
### [PASS] budget total vs last cum_steps
- formula:   gap = total - cum_steps[last row]; the run stops at a generation boundary, so expect gap == 0, unless a worker hit its cap mid-generation (that partial generation is counted but not logged)
- numbers:   19,530,965 - 19,530,965 = 0; max(gen_steps) = 276,264; workers at cap: 0
- result:    gap 0
- threshold: PASS if 0; WARN if > 0 and some worker hit its cap (discarded partial generation); FAIL if < 0 (logs claim steps the budget never counted) or > 0 with no worker at cap (steps spent that no log row explains)

### [PASS] run_args.steps_used == budget total
- formula:   steps_used == sum over pid of max(cum_steps)
- numbers:   19,530,965 vs 19,530,965
- result:    equal
- threshold: exact equality

### [PASS] sum(gen_steps) vs last cum_steps
- formula:   sum_g gen_steps[g] == cum_steps[last]
- numbers:   sum = 19,530,965; cum_steps[last] = 19,530,965; diff = 0
- result:    equal
- threshold: exact equality expected

### [PASS] gen_steps == diff(cum_steps) per row
- formula:   gen_steps[g] == cum_steps[g] - cum_steps[g-1]
- numbers:   0 mismatching rows of 83
- result:    consistent
- threshold: every row must match

### [PASS] episode-level steps vs budget total
- formula:   logged = sum(genomes.steps[]) + sum(validation.val_steps[]); gap = total - logged; expect gap == 0
- numbers:   16,312,929 + 3,218,036 = 19,530,965; total = 19,530,965; gap = 0
- result:    gap 0 steps (0.000% of total)
- threshold: PASS if 0; WARN if > 0 and some worker hit its cap (discarded partial generation); FAIL if < 0 (logs claim steps the budget never counted) or > 0 with no worker at cap (steps spent that no log row explains)

### [PASS] episode-level steps vs last cum_steps
- formula:   sum of steps[] + val_steps[] over gens <= last history gen == cum_steps[last]
- numbers:   19,530,965 vs 19,530,965; diff = 0
- result:    equal
- threshold: exact equality expected


## 2. Fitness consistency

survival bonus B = 0.3, return bonus R = 0.07 (run_args; missing = 0); raw_fitness column present
### [PASS] raw_fitness == mean(scores)
- formula:   |raw_fitness - mean(scores)| <= 1e-09
- numbers:   0 of 5,312 genome rows violate
- result:    consistent
- threshold: zero violations

### [PASS] fitness == mean(score + B*steps/3000 + R*min(returns, cap))
- formula:   |fitness - mean_i(scores[i] + B*steps[i]/3000 + R*returns[i])| <= 1e-09, B = 0.3, R = 0.07
- numbers:   0 of 5,312 genome rows violate
- result:    consistent
- threshold: zero violations

### [PASS] scores == points_for - points_against
- formula:   for every episode i: scores[i] == points_for[i] - points_against[i]
- numbers:   0 of 5,312 rows violate
- result:    consistent
- threshold: zero violations

### [PASS] scores are integers in [-5, 5]
- formula:   s == int(s) and -5 <= s <= 5 for every score
- numbers:   0 of 5,312 rows violate
- result:    all in range
- threshold: zero violations

### [PASS] per-episode list lengths agree
- formula:   len(scores) == len(steps) == len(points_for) == len(points_against)
- numbers:   0 rows with mismatched lengths
- result:    consistent
- threshold: zero violations

### [PASS] timeouts == count(steps >= 3000)
- formula:   timeouts == #{i : steps[i] >= 3000}
- numbers:   0 rows violate
- result:    consistent
- threshold: zero violations

### [PASS] best_fitness recompute
- formula:   history[best_fitness] == max(fitness) over genomes.jsonl rows of that gen (tol 1e-06, relative)
- numbers:   0 of 83 generations mismatch
- result:    consistent
- threshold: zero mismatches

### [PASS] mean_fitness recompute
- formula:   history[mean_fitness] == mean(fitness) over genomes.jsonl rows of that gen (tol 1e-06, relative)
- numbers:   0 of 83 generations mismatch
- result:    consistent
- threshold: zero mismatches

### [PASS] median_fitness recompute
- formula:   history[median_fitness] == median(fitness) over genomes.jsonl rows of that gen (tol 1e-06, relative)
- numbers:   0 of 83 generations mismatch
- result:    consistent
- threshold: zero mismatches

### [PASS] min_fitness recompute
- formula:   history[min_fitness] == min(fitness) over genomes.jsonl rows of that gen (tol 1e-06, relative)
- numbers:   0 of 83 generations mismatch
- result:    consistent
- threshold: zero mismatches

### [PASS] best_raw recompute
- formula:   history[best_raw] == max(raw_fitness) over genomes.jsonl rows of that gen (tol 1e-06, relative)
- numbers:   0 of 83 generations mismatch
- result:    consistent
- threshold: zero mismatches

### [PASS] mean_raw recompute
- formula:   history[mean_raw] == mean(raw_fitness) over genomes.jsonl rows of that gen (tol 1e-06, relative)
- numbers:   0 of 83 generations mismatch
- result:    consistent
- threshold: zero mismatches

### [PASS] mean_ep_steps recompute
- formula:   history[mean_ep_steps] == mean over all episodes of steps[] over genomes.jsonl rows of that gen (tol 1e-06, relative)
- numbers:   0 of 83 generations mismatch
- result:    consistent
- threshold: zero mismatches

### [PASS] timeout_frac recompute
- formula:   history[timeout_frac] == sum(timeouts) / #episodes over genomes.jsonl rows of that gen (tol 1e-06, relative)
- numbers:   0 of 83 generations mismatch
- result:    consistent
- threshold: zero mismatches

### [PASS] std_fitness recompute
- formula:   history[std_fitness] == pstdev(fitness) or stdev(fitness)
- numbers:   83 gens; mismatches: pstdev 0, stdev 83
- result:    matches population sd (ddof=0)
- threshold: all gens match one convention

### [PASS] best_nodes matches a best-fitness genome
- formula:   history[best_nodes] in {nodes of genomes with fitness == max(fitness)} (ties allowed)
- numbers:   0 of 83 gens mismatch
- result:    consistent
- threshold: zero mismatches

### [PASS] best_conns matches a best-fitness genome
- formula:   history[best_conns] in {conns of genomes with fitness == max(fitness)} (ties allowed)
- numbers:   0 of 83 gens mismatch
- result:    consistent
- threshold: zero mismatches

### [PASS] val_mean == mean(val_scores)
- formula:   |val_mean - mean(val_scores)| <= 1e-09
- numbers:   0 of 249 rows violate
- result:    consistent
- threshold: zero violations

### [PASS] val_scores integers in [-5, 5]
- formula:   s == int(s) and -5 <= s <= 5
- numbers:   0 rows violate
- result:    all in range
- threshold: zero violations

### [PASS] validation.train_fitness == genomes fitness or raw_fitness
- formula:   same (gen, key) row in both files; matches selection fitness or raw_fitness
- numbers:   0 of 249 rows match neither; matches selection fitness: 249, raw: 0
- result:    consistent
- threshold: zero violations

### [PASS] history val_best == max(val_mean) of that gen
- formula:   val_best[g] == max over validation rows of gen g
- numbers:   0 of 83 gens mismatch
- result:    consistent
- threshold: zero mismatches

### [PASS] champion_val is the running max of val_mean
- formula:   champion_val[g] == max over gens <= g of val_mean, hence non-decreasing
- numbers:   0 decreases; 0 gens != running max
- result:    consistent
- threshold: zero violations

### [PASS] champion.json is the argmax of val_mean
- formula:   champion.key in {key : val_mean == max(val_mean)} and champion.val_mean == max(val_mean)
- numbers:   champion key 333 (gen 5), val_mean -4.65; validation max -4.65 by key(s) [333, 1928, 2107, 3226]
- result:    consistent
- threshold: must match


## 3. Seed hygiene

> note: a range given only by its start is taken as [start, start + 100) (evaluate.py's default episode count)
- training   run_args['train'] = [0, 1,000,000)
- held-out   run_args['heldout_start'] = [1,000,000, 1,000,100)
- validation run_args['val'] = [2,000,000, 2,000,020)
### [PASS] seed ranges disjoint
- formula:   for every pair: max(lo_a, lo_b) >= min(hi_a, hi_b)  (hi exclusive)
- numbers:   training=[0,1,000,000); held-out=[1,000,000,1,000,100); validation=[2,000,000,2,000,020)
- result:    all three ranges disjoint
- threshold: FAIL on any overlap; WARN if a role is unlogged


## 4. Seed difficulty confound

### [INFO] elite delta (individual)
- formula:   delta_k,g = fitness_g(k) - fitness_{g-1}(k) for keys in both gens; genome unchanged, so delta is pure seed/episode noise
- numbers:   n = 526 pairs over 82 gens; mean = -0.0792; sd = 0.3009
- result:    seed noise sd per genome = 0.301 fitness units

### [WARN] seed noise vs best_fitness movement
- formula:   D_g = mean_k delta_k,g (generation's seed-difficulty shift); dB_g = best_g - best_{g-1}; ratio = sd(D) / sd(dB); r = corr(D, dB)
- numbers:   82 gens; sd(D) = 0.2850; sd(dB) = 0.3737; ratio = 0.763; r = 0.877 (r^2 = 0.769)
- result:    76.3% of best-fitness jitter is matched by seed shifts; r^2 = share of dB variance explained by seed difficulty
- threshold: WARN if ratio >= 0.5 (generation-to-generation best_fitness mostly tracks which seeds were drawn)


## 5. Winner's curse (train vs validation)

train side = raw_fitness from genomes.jsonl (true objective), falling back to validation.train_fitness
### [WARN] train raw - val_mean over validated genomes
- formula:   gap_i = train_raw_i - val_mean_i; CI = mean +- t_{0.975, n-1} * sd / sqrt(n)
- numbers:   n = 249; mean = 0.0801; sd = 0.2308; se = 0.0146; t = 1.970
- result:    mean gap 0.080, 95% CI [0.051, 0.109] (positive = training overstates)
- threshold: WARN if CI lower bound > 0: training fitness of selected genomes is optimistic; report val/held-out, not train

### [INFO] champion train raw vs val
- formula:   gap = train_raw - val_mean for the champion row
- numbers:   -4.6000 - -4.6500
- result:    gap 0.050

> note: the champion's val_mean is itself a max over every validated genome on the same fixed validation seeds, so it is optimistic too; only the held-out evaluation (evaluate.py) is unbiased.

## 6. Validation noise

### [PASS] champion val_mean 95% CI
- formula:   SE = sd(val_scores) / sqrt(n); CI = mean +- t_{0.975, n-1} * SE
- numbers:   n = 20; mean = -4.6500; sd = 0.6708; SE = 0.6708/sqrt(20) = 0.1500; t = 2.093
- result:    -4.650 +- 0.314  [-4.964, -4.336]
- threshold: WARN if half-width > 0.5 points

### [WARN] candidates statistically tied with champion
- formula:   #{other keys k : max val_mean_k >= champion CI lower bound}
- numbers:   lower bound -4.964; 213 of 213 other validated genomes reach it
- result:    213 tied candidates
- threshold: WARN if >= 5: champion choice is largely luck of the validation seeds


## 7. Reward-hacking / degenerate-behavior signals

### [PASS] timeout fraction over time
- formula:   timeout_frac_g = sum(timeouts) / #episodes; compare first vs last quarter of gens
- numbers:   first 20 gens: 0.000; last 20 gens: 0.000
- result:    0.0% of late episodes run to 3000 steps
- threshold: WARN if > 25% late: possible stalling for draws; watch a replay

### [INFO] episode length vs score (last gen)
- formula:   r = pearson(steps_i, score_i) over every episode in the last generation
- numbers:   320 episodes; r = nan; timed-out episodes: 0, their mean score n/a, share at exactly 0: n/a
- result:    r = nan

### [INFO] population action fractions
- formula:   mean over genomes of act_* per generation
- numbers:   gen 0: act_fwd=0.48, act_back=0.46, act_jump=0.47, act_idle=0.16 | gen 82: act_fwd=0.78, act_back=0.35, act_jump=0.92, act_idle=0.01
- result:    see behavior.png

### [WARN] champion action pattern
- formula:   mean of champion's act_* over its genomes.jsonl rows; degenerate if any pattern > 0.95, or every pattern is > 0.95 or < 0.05 (constant action)
- numbers:   act_fwd=0.995, act_back=0.000, act_jump=0.411, act_idle=0.005 (2 rows)
- result:    champion does act_fwd > 95% of steps
- threshold: WARN if any act_* > 0.95

### [INFO] points_for concentration (champion's training episodes)
- formula:   share = sum of points_for in the top 20% of episodes / total points_for
- numbers:   10 episodes; distribution {0: 6, 1: 3, 2: 1}; top 2 episodes hold 3 of 5 points
- result:    share 0.600 (0.20 = evenly spread)


## 8. Learning signal

### [WARN] val_best slope
- formula:   OLS val_best_g = a + b*g; SE(b) = sqrt(RSS/(n-2) / Sxx); t = b/SE
- numbers:   n = 83; b = -0.00057/gen; SE = 0.00022; t = -2.67; df = 81; p = 0.0092
- result:    -0.057 points per 100 generations
- threshold: PASS if t > 2 (validation score improving); WARN otherwise

### [WARN] val_best first vs last quartile
- formula:   Welch t: diff = mean(last) - mean(first); t = diff / sqrt(v1/n1 + v2/n2)
- numbers:   first 20 gens mean -4.755; last 20 gens mean -4.798; t = -3.83; df = 31.1; p = 0.0006
- result:    diff -0.043
- threshold: PASS if t > 2

### [INFO] champion_val start -> end
- formula:   champion_val[last] - champion_val[first]
- numbers:   -4.750 -> -4.650
- result:    +0.100


## 9. Population health

### [PASS] n_species
- formula:   min / final of history[n_species]
- numbers:   min 1, max 5, final 2
- result:    2 species at the end
- threshold: WARN if the final generation has a single species (diversity collapse)

### [PASS] mean genome size growth
- formula:   ratio = mean(conns) last gen / mean(conns) first gen
- numbers:   nodes 3.0 -> 13.5; conns 36.0 -> 20.0; ratio 0.56
- result:    x0.56 connections
- threshold: WARN if > 3x (bloat)

### [WARN] fraction of population at exactly -5
- formula:   #{genomes : raw_fitness == -5} / pop size, first vs last gen
- numbers:   0.031 -> 1.000
- result:    100.0% of the final population scores -5 every episode
- threshold: WARN if > 50% at the end


## 10. Reward shaping (survival / return bonus)

### [WARN] bonus share of selection-fitness variance
- formula:   per gen: bonus_k = fitness_k - raw_k (= mean(B*steps/3000 + R*returns), both shaping terms); Var(fit) = Var(raw) + Var(bonus) + 2Cov(raw, bonus); share = Var(bonus)/Var(fit)
- numbers:   B = 0.3; R = 0.07; 83 gens; mean share 0.387 (last 20 gens 0.514); mean 2Cov/Var(fit) 0.113; mean corr(raw, bonus) 0.339
- result:    51.4% of late selection spread is shaping bonus, not score
- threshold: WARN if > 50% in the last quarter: selection mostly rewards the shaping terms

### [INFO] shaping variance split: survival vs return term
- formula:   per gen: Var(B*mean(steps)/3000)/Var(fit) and Var(R*mean(returns))/Var(fit); last-quarter means (they need not sum to the bonus share: the two terms covary)
- numbers:   B = 0.3, R = 0.07; survival term 0.006; return term 0.421
- result:    see numbers

### [PASS] stalling for survival
- formula:   OLS slopes over generations: timeout_frac_g and mean_raw_g; flag if timeout slope t > 2 while raw-score slope t <= 2
- numbers:   timeout_frac slope 0.00000/gen (t = 0.00); mean_raw slope -0.00004/gen (t = -0.05)
- result:    no stalling signature
- threshold: WARN on the signature above


## 11. Behaviour-log consistency (contacts, returns, cancel, saturation)

### [PASS] returns <= contacts per episode
- formula:   for every episode i: returns[i] <= contacts[i] (a return is a contact after which the ball crossed)
- numbers:   0 of 5,312 genome rows violate (26,560 episodes)
- result:    consistent
- threshold: zero violations

### [PASS] contact lists: lengths and integrality
- formula:   len(contacts) == len(opp_contacts) == len(returns) == len(scores); all non-negative integers
- numbers:   length mismatches 0; negative/non-integer 0
- result:    consistent
- threshold: zero violations

### [PASS] act_both <= min(act_fwd, act_back)
- formula:   both-pressed steps are a subset of fwd steps and of back steps
- numbers:   0 of 5,312 rows violate
- result:    consistent
- threshold: zero violations

### [PASS] out_sat in [0, 1]
- formula:   0 <= out_sat <= 1
- numbers:   0 of 5,312 rows violate
- result:    consistent
- threshold: zero violations

### [PASS] validation: val_returns <= val_contacts, lengths agree
- formula:   per validation episode: val_returns[i] <= val_contacts[i]; list lengths == len(val_scores)
- numbers:   0 rows with returns > contacts; 0 rows with length mismatch (of 249)
- result:    consistent
- threshold: zero violations

### [PASS] mean_contacts recompute
- formula:   history[mean_contacts] == mean over every episode of contacts[] over genomes.jsonl rows of that gen (tol 1e-06, relative)
- numbers:   0 of 83 generations mismatch
- result:    consistent
- threshold: zero mismatches

### [PASS] mean_returns recompute
- formula:   history[mean_returns] == mean over every episode of returns[] over genomes.jsonl rows of that gen (tol 1e-06, relative)
- numbers:   0 of 83 generations mismatch
- result:    consistent
- threshold: zero mismatches

### [PASS] mean_cancel recompute
- formula:   history[mean_cancel] == sum_k act_both_k * S_k / sum_k S_k, S_k = sum(steps_k) over genomes.jsonl rows of that gen (tol 1e-06, relative)
- numbers:   0 of 83 generations mismatch
- result:    consistent
- threshold: zero mismatches

### [PASS] mean_sat recompute
- formula:   history[mean_sat] == sum_k out_sat_k * S_k / sum_k S_k, S_k = sum(steps_k) over genomes.jsonl rows of that gen (tol 1e-06, relative)
- numbers:   0 of 83 generations mismatch
- result:    consistent
- threshold: zero mismatches

### [PASS] mean_nodes recompute
- formula:   history[mean_nodes] == mean over genomes of nodes over genomes.jsonl rows of that gen (tol 1e-06, relative)
- numbers:   0 of 83 generations mismatch
- result:    consistent
- threshold: zero mismatches

### [PASS] mean_conns recompute
- formula:   history[mean_conns] == mean over genomes of conns over genomes.jsonl rows of that gen (tol 1e-06, relative)
- numbers:   0 of 83 generations mismatch
- result:    consistent
- threshold: zero mismatches

### [PASS] best_raw_contacts recompute
- formula:   history[best_raw_contacts] == mean(contacts) of a genome with max raw_fitness (ties allowed) over genomes.jsonl rows of that gen (tol 1e-06, relative)
- numbers:   0 of 83 generations mismatch
- result:    consistent
- threshold: zero mismatches


## 12. Ball contacts as a learning signal

### [WARN] mean_contacts slope
- formula:   OLS mean_contacts_g = a + b*g (fitness_history.csv); t = b/SE(b)
- numbers:   n = 83; b = 0.00054/gen; SE = 0.00214; t = 0.25; df = 81; p = 0.8010
- result:    0.054 contacts/episode per 100 generations
- threshold: PASS if t > 2 (agent touches/returns the ball more over time); WARN otherwise

### [WARN] mean_contacts first vs last quarter
- formula:   Welch t: diff = mean(last q) - mean(first q); t = diff / sqrt(v1/n1 + v2/n2)
- numbers:   q = 20; first mean 1.544; last mean 1.567; t = 0.15; df = 38.0; p = 0.8780
- result:    diff 0.023
- threshold: PASS if t > 2

### [INFO] mean_contacts vs reference policies
- formula:   late = mean of last 20 gens; ratio_random = late / 1.4; progress = (late - 1.4) / (47.0 - 1.4)
- numbers:   late 1.567; random 1.4; built-in policy 47.0
- result:    x1.12 random play; 0.4% of the way from random to the built-in policy

### [WARN] mean_returns slope
- formula:   OLS mean_returns_g = a + b*g (fitness_history.csv); t = b/SE(b)
- numbers:   n = 83; b = 0.00120/gen; SE = 0.00124; t = 0.96; df = 81; p = 0.3386
- result:    0.120 returns/episode per 100 generations
- threshold: PASS if t > 2 (agent touches/returns the ball more over time); WARN otherwise

### [WARN] mean_returns first vs last quarter
- formula:   Welch t: diff = mean(last q) - mean(first q); t = diff / sqrt(v1/n1 + v2/n2)
- numbers:   q = 20; first mean 0.599; last mean 0.676; t = 0.92; df = 38.0; p = 0.3628
- result:    diff 0.078
- threshold: PASS if t > 2

### [INFO] mean_returns vs reference policies
- formula:   late = mean of last 20 gens; ratio_random = late / 0.1; progress = (late - 0.1) / (25.5 - 0.1)
- numbers:   late 0.676; random 0.1; built-in policy 25.5
- result:    x6.76 random play; 2.3% of the way from random to the built-in policy

### [INFO] best_raw_contacts start -> late
- formula:   mean(contacts) of the best-raw genome, first vs last quarter
- numbers:   first 20 gens 1.870; last 20 gens 1.760
- result:    x1.26 random play (late)

### [INFO] contacts vs score (episode level)
- formula:   r = pearson(contacts_i, score_i) over every episode of the last 20 gens; t = r sqrt((n-2)/(1-r^2))
- numbers:   n = 6400; mean contacts 1.567; mean score -4.875; r = 0.045; t = 3.57; p = 0.0004
- result:    r = 0.045 (positive = touching/returning the ball is what earns points)

### [INFO] returns vs score (episode level)
- formula:   r = pearson(returns_i, score_i) over every episode of the last 20 gens; t = r sqrt((n-2)/(1-r^2))
- numbers:   n = 6400; mean returns 0.676; mean score -4.875; r = 0.151; t = 12.20; p = 0.0000
- result:    r = 0.151 (positive = touching/returning the ball is what earns points)


## 13. Degenerate behaviour (cancel, juggling, saturation)

### [PASS] champion stands still (fwd+back cancel)
- formula:   act_both = sum_rows val_act_both * sum(val_steps) / sum(val_steps) over the champion's validation rows
- numbers:   1 rows, 20 episodes; act_both 0.000; act_fwd 0.998, act_back 0.000; contacts/ep 2.25, returns/ep 1.35
- result:    cancels on 0.0% of steps
- threshold: WARN if act_both > 0.5

### [PASS] juggling exploit (touch without returning)
- formula:   last 20 gens: contacts/ep, returns/ep, ratio = returns/contacts; timeout_frac first vs last quarter; flag if contacts >= 3 x random (4.2) and ratio < 0.2 and timeouts rising
- numbers:   contacts 1.567; returns 0.676; ratio 0.432; timeout_frac 0.000 -> 0.000
- result:    contacts not high enough to juggle
- threshold: WARN on the signature above

### [INFO] output saturation
- formula:   mean_sat = fraction of network outputs < 0.02 or > 0.98; last-quarter mean
- numbers:   first 20 gens 0.617; last 20 gens 0.684
- result:    68.4% saturated
- threshold: note if > 0.8

### [INFO] population fwd+back cancel
- formula:   mean_cancel = step-weighted act_both over the population
- numbers:   first 20 gens 0.163; last 20 gens 0.389
- result:    38.9% of late steps press both directions (no movement)


## 14. Search health (structure drift, NEAT stagnation)

### [PASS] mean_conns trend
- formula:   OLS mean_conns_g = a + b*g (fitness_history.csv); change = (end - start) / start
- numbers:   start 36.00 -> end 20.00; b = -0.0829/gen; t = -3.36; p = 0.0012
- result:    -44.4% connections
- threshold: WARN if mean_conns falls by > 50%

### [INFO] NEAT stagnation / extinction (stdout.log)
- formula:   count lines containing 'stagnat'; last 'Total extinctions: N'; count 'All species extinct'
- numbers:   1,081 lines; stagnation lines 0; final total extinctions 0 (82 reports); 'all species extinct' 0
- result:    0 species removed for stagnation, 0 total extinctions


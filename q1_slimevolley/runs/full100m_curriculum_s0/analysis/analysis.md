# Run analysis: `runs/full100m_curriculum_s0`

Counts: PASS 57, WARN 6, INFO 15

| check | result | verdict |
|---|---|---|
| 1. Budget: total steps within limit | 98,527,682 / 100,000,000 (98.53%) | **PASS** |
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
| 4. Seed difficulty confound: elite delta (individual) | seed noise sd per genome = 1.947 fitness units | **INFO** |
| 4. Seed difficulty confound: seed noise vs best_fitness movement | 68.0% of best-fitness jitter is matched by seed shifts; r^2 = share of dB variance explained by seed difficulty | **WARN** |
| 5. Winner's curse (train vs validation): train raw - val_mean over validated genomes | mean gap 0.146, 95% CI [0.102, 0.191] (positive = training overstates) | **WARN** |
| 5. Winner's curse (train vs validation): champion train raw vs val | gap -0.350 | **INFO** |
| 6. Validation noise: champion val_mean 95% CI | -0.050 +- 0.467  [-0.517, 0.417] | **PASS** |
| 6. Validation noise: candidates statistically tied with champion | 7 tied candidates | **WARN** |
| 7. Reward-hacking / degenerate-behavior signals: timeout fraction over time | 9.9% of late episodes run to 3000 steps | **PASS** |
| 7. Reward-hacking / degenerate-behavior signals: episode length vs score (last gen) | r = 0.749 | **INFO** |
| 7. Reward-hacking / degenerate-behavior signals: population action fractions | see behavior.png | **INFO** |
| 7. Reward-hacking / degenerate-behavior signals: champion action pattern | champion does act_jump > 95% of steps | **WARN** |
| 7. Reward-hacking / degenerate-behavior signals: points_for concentration (champion's training episodes) | share 0.562 (0.20 = evenly spread) | **INFO** |
| 8. Learning signal: val_best slope | 2.627 points per 100 generations | **PASS** |
| 8. Learning signal: val_best first vs last quartile | diff 4.346 | **PASS** |
| 8. Learning signal: champion_val start -> end | +4.750 | **INFO** |
| 9. Population health: n_species | 2 species at the end | **PASS** |
| 9. Population health: mean genome size growth | x0.35 connections | **PASS** |
| 9. Population health: fraction of population at exactly -5 | 21.9% of the final population scores -5 every episode | **PASS** |
| 10. Reward shaping (survival / return bonus): bonus share of selection-fitness variance | 5.7% of late selection spread is shaping bonus, not score | **PASS** |
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
| 12. Ball contacts as a learning signal: mean_contacts slope | 12.521 contacts/episode per 100 generations | **PASS** |
| 12. Ball contacts as a learning signal: mean_contacts first vs last quarter | diff 20.448 | **PASS** |
| 12. Ball contacts as a learning signal: mean_contacts vs reference policies | x16.35 random play; 47.1% of the way from random to the built-in policy | **INFO** |
| 12. Ball contacts as a learning signal: mean_returns slope | 1.284 returns/episode per 100 generations | **PASS** |
| 12. Ball contacts as a learning signal: mean_returns first vs last quarter | diff 2.088 | **PASS** |
| 12. Ball contacts as a learning signal: mean_returns vs reference policies | x27.91 random play; 10.6% of the way from random to the built-in policy | **INFO** |
| 12. Ball contacts as a learning signal: best_raw_contacts start -> late | x132.83 random play (late) | **INFO** |
| 12. Ball contacts as a learning signal: contacts vs score (episode level) | r = 0.661 (positive = touching/returning the ball is what earns points) | **INFO** |
| 12. Ball contacts as a learning signal: returns vs score (episode level) | r = 0.434 (positive = touching/returning the ball is what earns points) | **INFO** |
| 13. Degenerate behaviour (cancel, juggling, saturation): champion stands still (fwd+back cancel) | cancels on 24.8% of steps | **PASS** |
| 13. Degenerate behaviour (cancel, juggling, saturation): juggling exploit (touch without returning) | agent keeps the ball on its side to run out the clock | **WARN** |
| 13. Degenerate behaviour (cancel, juggling, saturation): output saturation | 48.4% saturated | **INFO** |
| 13. Degenerate behaviour (cancel, juggling, saturation): population fwd+back cancel | 14.9% of late steps press both directions (no movement) | **INFO** |
| 14. Search health (structure drift, NEAT stagnation): mean_conns trend | -65.1% connections: structural drift toward deletion (delete-conn outpaces add-conn under weak selection) | **WARN** |
| 14. Search health (structure drift, NEAT stagnation): NEAT stagnation / extinction (stdout.log) | 1 species removed for stagnation, 0 total extinctions | **INFO** |

Plots: fitness_curve.png, seed_noise.png, behavior.png, complexity.png, contacts.png

# analyze_run: runs/full100m_curriculum_s0
- run_args.json: present
- budget_log.jsonl: 86260 rows
- fitness_history.csv: 227 rows
- genomes.jsonl: 14528 rows
- validation.jsonl: 681 rows
- champion.json: present

## 1. Budget

budget_log.jsonl: 86260 flush records from 9 processes
### [PASS] total steps within limit
- formula:   total = sum over pid of max(cum_steps); require total <= budget
- numbers:   10,869,832 + 11,041,401 + 11,062,810 + 10,924,379 + 10,892,398 + 10,953,196 + 10,932,339 + 10,853,895 + 10,997,432 = 98,527,682; budget = 100,000,000
- result:    98,527,682 / 100,000,000 (98.53%)
- threshold: FAIL if total > budget

per-process cap = budget // workers = 11,111,111; processes that hit it: 0 of 9
### [PASS] budget total vs last cum_steps
- formula:   gap = total - cum_steps[last row]; the run stops at a generation boundary, so expect gap == 0, unless a worker hit its cap mid-generation (that partial generation is counted but not logged)
- numbers:   98,527,682 - 98,527,682 = 0; max(gen_steps) = 673,145; workers at cap: 0
- result:    gap 0
- threshold: PASS if 0; WARN if > 0 and some worker hit its cap (discarded partial generation); FAIL if < 0 (logs claim steps the budget never counted) or > 0 with no worker at cap (steps spent that no log row explains)

### [PASS] run_args.steps_used == budget total
- formula:   steps_used == sum over pid of max(cum_steps)
- numbers:   98,527,682 vs 98,527,682
- result:    equal
- threshold: exact equality

### [PASS] sum(gen_steps) vs last cum_steps
- formula:   sum_g gen_steps[g] == cum_steps[last]
- numbers:   sum = 98,527,682; cum_steps[last] = 98,527,682; diff = 0
- result:    equal
- threshold: exact equality expected

### [PASS] gen_steps == diff(cum_steps) per row
- formula:   gen_steps[g] == cum_steps[g] - cum_steps[g-1]
- numbers:   0 mismatching rows of 227
- result:    consistent
- threshold: every row must match

### [PASS] episode-level steps vs budget total
- formula:   logged = sum(genomes.steps[]) + sum(validation.val_steps[]); gap = total - logged; expect gap == 0
- numbers:   67,778,846 + 30,748,836 = 98,527,682; total = 98,527,682; gap = 0
- result:    gap 0 steps (0.000% of total)
- threshold: PASS if 0; WARN if > 0 and some worker hit its cap (discarded partial generation); FAIL if < 0 (logs claim steps the budget never counted) or > 0 with no worker at cap (steps spent that no log row explains)

### [PASS] episode-level steps vs last cum_steps
- formula:   sum of steps[] + val_steps[] over gens <= last history gen == cum_steps[last]
- numbers:   98,527,682 vs 98,527,682; diff = 0
- result:    equal
- threshold: exact equality expected


## 2. Fitness consistency

survival bonus B = 0.3, return bonus R = 0.07 (run_args; missing = 0); raw_fitness column present
### [PASS] raw_fitness == mean(scores)
- formula:   |raw_fitness - mean(scores)| <= 1e-09
- numbers:   0 of 14,528 genome rows violate
- result:    consistent
- threshold: zero violations

### [PASS] fitness == mean(score + B*steps/3000 + R*min(returns, cap))
- formula:   |fitness - mean_i(scores[i] + B*steps[i]/3000 + R*returns[i])| <= 1e-09, B = 0.3, R = 0.07
- numbers:   0 of 14,528 genome rows violate
- result:    consistent
- threshold: zero violations

### [PASS] scores == points_for - points_against
- formula:   for every episode i: scores[i] == points_for[i] - points_against[i]
- numbers:   0 of 14,528 rows violate
- result:    consistent
- threshold: zero violations

### [PASS] scores are integers in [-5, 5]
- formula:   s == int(s) and -5 <= s <= 5 for every score
- numbers:   0 of 14,528 rows violate
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
- numbers:   0 of 227 generations mismatch
- result:    consistent
- threshold: zero mismatches

### [PASS] mean_fitness recompute
- formula:   history[mean_fitness] == mean(fitness) over genomes.jsonl rows of that gen (tol 1e-06, relative)
- numbers:   0 of 227 generations mismatch
- result:    consistent
- threshold: zero mismatches

### [PASS] median_fitness recompute
- formula:   history[median_fitness] == median(fitness) over genomes.jsonl rows of that gen (tol 1e-06, relative)
- numbers:   0 of 227 generations mismatch
- result:    consistent
- threshold: zero mismatches

### [PASS] min_fitness recompute
- formula:   history[min_fitness] == min(fitness) over genomes.jsonl rows of that gen (tol 1e-06, relative)
- numbers:   0 of 227 generations mismatch
- result:    consistent
- threshold: zero mismatches

### [PASS] best_raw recompute
- formula:   history[best_raw] == max(raw_fitness) over genomes.jsonl rows of that gen (tol 1e-06, relative)
- numbers:   0 of 227 generations mismatch
- result:    consistent
- threshold: zero mismatches

### [PASS] mean_raw recompute
- formula:   history[mean_raw] == mean(raw_fitness) over genomes.jsonl rows of that gen (tol 1e-06, relative)
- numbers:   0 of 227 generations mismatch
- result:    consistent
- threshold: zero mismatches

### [PASS] mean_ep_steps recompute
- formula:   history[mean_ep_steps] == mean over all episodes of steps[] over genomes.jsonl rows of that gen (tol 1e-06, relative)
- numbers:   0 of 227 generations mismatch
- result:    consistent
- threshold: zero mismatches

### [PASS] timeout_frac recompute
- formula:   history[timeout_frac] == sum(timeouts) / #episodes over genomes.jsonl rows of that gen (tol 1e-06, relative)
- numbers:   0 of 227 generations mismatch
- result:    consistent
- threshold: zero mismatches

### [PASS] std_fitness recompute
- formula:   history[std_fitness] == pstdev(fitness) or stdev(fitness)
- numbers:   227 gens; mismatches: pstdev 0, stdev 227
- result:    matches population sd (ddof=0)
- threshold: all gens match one convention

### [PASS] best_nodes matches a best-fitness genome
- formula:   history[best_nodes] in {nodes of genomes with fitness == max(fitness)} (ties allowed)
- numbers:   0 of 227 gens mismatch
- result:    consistent
- threshold: zero mismatches

### [PASS] best_conns matches a best-fitness genome
- formula:   history[best_conns] in {conns of genomes with fitness == max(fitness)} (ties allowed)
- numbers:   0 of 227 gens mismatch
- result:    consistent
- threshold: zero mismatches

### [PASS] val_mean == mean(val_scores)
- formula:   |val_mean - mean(val_scores)| <= 1e-09
- numbers:   0 of 681 rows violate
- result:    consistent
- threshold: zero violations

### [PASS] val_scores integers in [-5, 5]
- formula:   s == int(s) and -5 <= s <= 5
- numbers:   0 rows violate
- result:    all in range
- threshold: zero violations

### [PASS] validation.train_fitness == genomes fitness or raw_fitness
- formula:   same (gen, key) row in both files; matches selection fitness or raw_fitness
- numbers:   0 of 681 rows match neither; matches selection fitness: 681, raw: 0
- result:    consistent
- threshold: zero violations

### [PASS] history val_best == max(val_mean) of that gen
- formula:   val_best[g] == max over validation rows of gen g
- numbers:   0 of 227 gens mismatch
- result:    consistent
- threshold: zero mismatches

### [PASS] champion_val is the running max of val_mean
- formula:   champion_val[g] == max over gens <= g of val_mean, hence non-decreasing
- numbers:   0 decreases; 0 gens != running max
- result:    consistent
- threshold: zero violations

### [PASS] champion.json is the argmax of val_mean
- formula:   champion.key in {key : val_mean == max(val_mean)} and champion.val_mean == max(val_mean)
- numbers:   champion key 9993 (gen 172), val_mean -0.05; validation max -0.05 by key(s) [9993]
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
- numbers:   n = 1330 pairs over 226 gens; mean = -0.3843; sd = 1.9475
- result:    seed noise sd per genome = 1.947 fitness units

### [WARN] seed noise vs best_fitness movement
- formula:   D_g = mean_k delta_k,g (generation's seed-difficulty shift); dB_g = best_g - best_{g-1}; ratio = sd(D) / sd(dB); r = corr(D, dB)
- numbers:   226 gens; sd(D) = 1.4982; sd(dB) = 2.2044; ratio = 0.680; r = 0.757 (r^2 = 0.573)
- result:    68.0% of best-fitness jitter is matched by seed shifts; r^2 = share of dB variance explained by seed difficulty
- threshold: WARN if ratio >= 0.5 (generation-to-generation best_fitness mostly tracks which seeds were drawn)


## 5. Winner's curse (train vs validation)

train side = raw_fitness from genomes.jsonl (true objective), falling back to validation.train_fitness
### [WARN] train raw - val_mean over validated genomes
- formula:   gap_i = train_raw_i - val_mean_i; CI = mean +- t_{0.975, n-1} * sd / sqrt(n)
- numbers:   n = 681; mean = 0.1463; sd = 0.5916; se = 0.0227; t = 1.963
- result:    mean gap 0.146, 95% CI [0.102, 0.191] (positive = training overstates)
- threshold: WARN if CI lower bound > 0: training fitness of selected genomes is optimistic; report val/held-out, not train

### [INFO] champion train raw vs val
- formula:   gap = train_raw - val_mean for the champion row
- numbers:   -0.4000 - -0.0500
- result:    gap -0.350

> note: the champion's val_mean is itself a max over every validated genome on the same fixed validation seeds, so it is optimistic too; only the held-out evaluation (evaluate.py) is unbiased.

## 6. Validation noise

### [PASS] champion val_mean 95% CI
- formula:   SE = sd(val_scores) / sqrt(n); CI = mean +- t_{0.975, n-1} * SE
- numbers:   n = 20; mean = -0.0500; sd = 0.9987; SE = 0.9987/sqrt(20) = 0.2233; t = 2.093
- result:    -0.050 +- 0.467  [-0.517, 0.417]
- threshold: WARN if half-width > 0.5 points

### [WARN] candidates statistically tied with champion
- formula:   #{other keys k : max val_mean_k >= champion CI lower bound}
- numbers:   lower bound -0.517; 7 of 245 other validated genomes reach it
- result:    7 tied candidates
- threshold: WARN if >= 5: champion choice is largely luck of the validation seeds


## 7. Reward-hacking / degenerate-behavior signals

### [PASS] timeout fraction over time
- formula:   timeout_frac_g = sum(timeouts) / #episodes; compare first vs last quarter of gens
- numbers:   first 56 gens: 0.000; last 56 gens: 0.099
- result:    9.9% of late episodes run to 3000 steps
- threshold: WARN if > 25% late: possible stalling for draws; watch a replay

### [INFO] episode length vs score (last gen)
- formula:   r = pearson(steps_i, score_i) over every episode in the last generation
- numbers:   320 episodes; r = 0.749; timed-out episodes: 46, their mean score -1.717, share at exactly 0: 0.304
- result:    r = 0.749

### [INFO] population action fractions
- formula:   mean over genomes of act_* per generation
- numbers:   gen 0: act_fwd=0.50, act_back=0.48, act_jump=0.53, act_idle=0.13 | gen 226: act_fwd=0.42, act_back=0.31, act_jump=0.58, act_idle=0.18
- result:    see behavior.png

### [WARN] champion action pattern
- formula:   mean of champion's act_* over its genomes.jsonl rows; degenerate if any pattern > 0.95, or every pattern is > 0.95 or < 0.05 (constant action)
- numbers:   act_fwd=0.565, act_back=0.367, act_jump=1.000, act_idle=0.000 (7 rows)
- result:    champion does act_jump > 95% of steps
- threshold: WARN if any act_* > 0.95

### [INFO] points_for concentration (champion's training episodes)
- formula:   share = sum of points_for in the top 20% of episodes / total points_for
- numbers:   35 episodes; distribution {0: 21, 1: 12, 2: 2}; top 7 episodes hold 9 of 16 points
- result:    share 0.562 (0.20 = evenly spread)


## 8. Learning signal

### [PASS] val_best slope
- formula:   OLS val_best_g = a + b*g; SE(b) = sqrt(RSS/(n-2) / Sxx); t = b/SE
- numbers:   n = 227; b = 0.02627/gen; SE = 0.00065; t = 40.23; df = 225; p = 0.0000
- result:    2.627 points per 100 generations
- threshold: PASS if t > 2 (validation score improving); WARN otherwise

### [PASS] val_best first vs last quartile
- formula:   Welch t: diff = mean(last) - mean(first); t = diff / sqrt(v1/n1 + v2/n2)
- numbers:   first 56 gens mean -4.707; last 56 gens mean -0.361; t = 199.58; df = 64.1; p = 0.0000
- result:    diff 4.346
- threshold: PASS if t > 2

### [INFO] champion_val start -> end
- formula:   champion_val[last] - champion_val[first]
- numbers:   -4.800 -> -0.050
- result:    +4.750


## 9. Population health

### [PASS] n_species
- formula:   min / final of history[n_species]
- numbers:   min 1, max 5, final 2
- result:    2 species at the end
- threshold: WARN if the final generation has a single species (diversity collapse)

### [PASS] mean genome size growth
- formula:   ratio = mean(conns) last gen / mean(conns) first gen
- numbers:   nodes 3.0 -> 9.1; conns 42.0 -> 14.7; ratio 0.35
- result:    x0.35 connections
- threshold: WARN if > 3x (bloat)

### [PASS] fraction of population at exactly -5
- formula:   #{genomes : raw_fitness == -5} / pop size, first vs last gen
- numbers:   1.000 -> 0.219
- result:    21.9% of the final population scores -5 every episode
- threshold: WARN if > 50% at the end


## 10. Reward shaping (survival / return bonus)

### [PASS] bonus share of selection-fitness variance
- formula:   per gen: bonus_k = fitness_k - raw_k (= mean(B*steps/3000 + R*returns), both shaping terms); Var(fit) = Var(raw) + Var(bonus) + 2Cov(raw, bonus); share = Var(bonus)/Var(fit)
- numbers:   B = 0.3; R = 0.07; 227 gens; mean share 0.492 (last 56 gens 0.057); mean 2Cov/Var(fit) 0.171; mean corr(raw, bonus) 0.533
- result:    5.7% of late selection spread is shaping bonus, not score
- threshold: WARN if > 50% in the last quarter: selection mostly rewards the shaping terms

### [INFO] shaping variance split: survival vs return term
- formula:   per gen: Var(B*mean(steps)/3000)/Var(fit) and Var(R*mean(returns))/Var(fit); last-quarter means (they need not sum to the bonus share: the two terms covary)
- numbers:   B = 0.3, R = 0.07; survival term 0.005; return term 0.032
- result:    see numbers

### [PASS] stalling for survival
- formula:   OLS slopes over generations: timeout_frac_g and mean_raw_g; flag if timeout slope t > 2 while raw-score slope t <= 2
- numbers:   timeout_frac slope 0.00064/gen (t = 23.44); mean_raw slope 0.00217/gen (t = 12.30)
- result:    no stalling signature
- threshold: WARN on the signature above


## 11. Behaviour-log consistency (contacts, returns, cancel, saturation)

### [PASS] returns <= contacts per episode
- formula:   for every episode i: returns[i] <= contacts[i] (a return is a contact after which the ball crossed)
- numbers:   0 of 14,528 genome rows violate (72,640 episodes)
- result:    consistent
- threshold: zero violations

### [PASS] contact lists: lengths and integrality
- formula:   len(contacts) == len(opp_contacts) == len(returns) == len(scores); all non-negative integers
- numbers:   length mismatches 0; negative/non-integer 0
- result:    consistent
- threshold: zero violations

### [PASS] act_both <= min(act_fwd, act_back)
- formula:   both-pressed steps are a subset of fwd steps and of back steps
- numbers:   0 of 14,528 rows violate
- result:    consistent
- threshold: zero violations

### [PASS] out_sat in [0, 1]
- formula:   0 <= out_sat <= 1
- numbers:   0 of 14,528 rows violate
- result:    consistent
- threshold: zero violations

### [PASS] validation: val_returns <= val_contacts, lengths agree
- formula:   per validation episode: val_returns[i] <= val_contacts[i]; list lengths == len(val_scores)
- numbers:   0 rows with returns > contacts; 0 rows with length mismatch (of 681)
- result:    consistent
- threshold: zero violations

### [PASS] mean_contacts recompute
- formula:   history[mean_contacts] == mean over every episode of contacts[] over genomes.jsonl rows of that gen (tol 1e-06, relative)
- numbers:   0 of 227 generations mismatch
- result:    consistent
- threshold: zero mismatches

### [PASS] mean_returns recompute
- formula:   history[mean_returns] == mean over every episode of returns[] over genomes.jsonl rows of that gen (tol 1e-06, relative)
- numbers:   0 of 227 generations mismatch
- result:    consistent
- threshold: zero mismatches

### [PASS] mean_cancel recompute
- formula:   history[mean_cancel] == sum_k act_both_k * S_k / sum_k S_k, S_k = sum(steps_k) over genomes.jsonl rows of that gen (tol 1e-06, relative)
- numbers:   0 of 227 generations mismatch
- result:    consistent
- threshold: zero mismatches

### [PASS] mean_sat recompute
- formula:   history[mean_sat] == sum_k out_sat_k * S_k / sum_k S_k, S_k = sum(steps_k) over genomes.jsonl rows of that gen (tol 1e-06, relative)
- numbers:   0 of 227 generations mismatch
- result:    consistent
- threshold: zero mismatches

### [PASS] mean_nodes recompute
- formula:   history[mean_nodes] == mean over genomes of nodes over genomes.jsonl rows of that gen (tol 1e-06, relative)
- numbers:   0 of 227 generations mismatch
- result:    consistent
- threshold: zero mismatches

### [PASS] mean_conns recompute
- formula:   history[mean_conns] == mean over genomes of conns over genomes.jsonl rows of that gen (tol 1e-06, relative)
- numbers:   0 of 227 generations mismatch
- result:    consistent
- threshold: zero mismatches

### [PASS] best_raw_contacts recompute
- formula:   history[best_raw_contacts] == mean(contacts) of a genome with max raw_fitness (ties allowed) over genomes.jsonl rows of that gen (tol 1e-06, relative)
- numbers:   0 of 227 generations mismatch
- result:    consistent
- threshold: zero mismatches


## 12. Ball contacts as a learning signal

### [PASS] mean_contacts slope
- formula:   OLS mean_contacts_g = a + b*g (fitness_history.csv); t = b/SE(b)
- numbers:   n = 227; b = 0.12521/gen; SE = 0.00411; t = 30.48; df = 225; p = 0.0000
- result:    12.521 contacts/episode per 100 generations
- threshold: PASS if t > 2 (agent touches/returns the ball more over time); WARN otherwise

### [PASS] mean_contacts first vs last quarter
- formula:   Welch t: diff = mean(last q) - mean(first q); t = diff / sqrt(v1/n1 + v2/n2)
- numbers:   q = 56; first mean 2.441; last mean 22.890; t = 21.19; df = 59.4; p = 0.0000
- result:    diff 20.448
- threshold: PASS if t > 2

### [INFO] mean_contacts vs reference policies
- formula:   late = mean of last 56 gens; ratio_random = late / 1.4; progress = (late - 1.4) / (47.0 - 1.4)
- numbers:   late 22.890; random 1.4; built-in policy 47.0
- result:    x16.35 random play; 47.1% of the way from random to the built-in policy

### [PASS] mean_returns slope
- formula:   OLS mean_returns_g = a + b*g (fitness_history.csv); t = b/SE(b)
- numbers:   n = 227; b = 0.01284/gen; SE = 0.00082; t = 15.61; df = 225; p = 0.0000
- result:    1.284 returns/episode per 100 generations
- threshold: PASS if t > 2 (agent touches/returns the ball more over time); WARN otherwise

### [PASS] mean_returns first vs last quarter
- formula:   Welch t: diff = mean(last q) - mean(first q); t = diff / sqrt(v1/n1 + v2/n2)
- numbers:   q = 56; first mean 0.703; last mean 2.791; t = 23.71; df = 73.6; p = 0.0000
- result:    diff 2.088
- threshold: PASS if t > 2

### [INFO] mean_returns vs reference policies
- formula:   late = mean of last 56 gens; ratio_random = late / 0.1; progress = (late - 0.1) / (25.5 - 0.1)
- numbers:   late 2.791; random 0.1; built-in policy 25.5
- result:    x27.91 random play; 10.6% of the way from random to the built-in policy

### [INFO] best_raw_contacts start -> late
- formula:   mean(contacts) of the best-raw genome, first vs last quarter
- numbers:   first 56 gens 11.014; last 56 gens 185.957
- result:    x132.83 random play (late)

### [INFO] contacts vs score (episode level)
- formula:   r = pearson(contacts_i, score_i) over every episode of the last 56 gens; t = r sqrt((n-2)/(1-r^2))
- numbers:   n = 17920; mean contacts 22.890; mean score -4.476; r = 0.661; t = 117.84; p = 0.0000
- result:    r = 0.661 (positive = touching/returning the ball is what earns points)

### [INFO] returns vs score (episode level)
- formula:   r = pearson(returns_i, score_i) over every episode of the last 56 gens; t = r sqrt((n-2)/(1-r^2))
- numbers:   n = 17920; mean returns 2.791; mean score -4.476; r = 0.434; t = 64.51; p = 0.0000
- result:    r = 0.434 (positive = touching/returning the ball is what earns points)


## 13. Degenerate behaviour (cancel, juggling, saturation)

### [PASS] champion stands still (fwd+back cancel)
- formula:   act_both = sum_rows val_act_both * sum(val_steps) / sum(val_steps) over the champion's validation rows
- numbers:   7 rows, 140 episodes; act_both 0.248; act_fwd 0.575, act_back 0.383; contacts/ep 70.15, returns/ep 18.85
- result:    cancels on 24.8% of steps
- threshold: WARN if act_both > 0.5

### [WARN] juggling exploit (touch without returning)
- formula:   last 56 gens: contacts/ep, returns/ep, ratio = returns/contacts; timeout_frac first vs last quarter; flag if contacts >= 3 x random (4.2) and ratio < 0.2 and timeouts rising
- numbers:   contacts 22.890; returns 2.791; ratio 0.122; timeout_frac 0.000 -> 0.099
- result:    agent keeps the ball on its side to run out the clock
- threshold: WARN on the signature above

### [INFO] output saturation
- formula:   mean_sat = fraction of network outputs < 0.02 or > 0.98; last-quarter mean
- numbers:   first 56 gens 0.574; last 56 gens 0.484
- result:    48.4% saturated
- threshold: note if > 0.8

### [INFO] population fwd+back cancel
- formula:   mean_cancel = step-weighted act_both over the population
- numbers:   first 56 gens 0.192; last 56 gens 0.149
- result:    14.9% of late steps press both directions (no movement)


## 14. Search health (structure drift, NEAT stagnation)

### [WARN] mean_conns trend
- formula:   OLS mean_conns_g = a + b*g (fitness_history.csv); change = (end - start) / start
- numbers:   start 42.00 -> end 14.67; b = -0.0704/gen; t = -23.21; p = 0.0000
- result:    -65.1% connections: structural drift toward deletion (delete-conn outpaces add-conn under weak selection)
- threshold: WARN if mean_conns falls by > 50%

### [INFO] NEAT stagnation / extinction (stdout.log)
- formula:   count lines containing 'stagnat'; last 'Total extinctions: N'; count 'All species extinct'
- numbers:   3,191 lines; stagnation lines 1; final total extinctions 0 (226 reports); 'all species extinct' 0; first: '[curriculum] phase 1 -> 2 at gen 116: resetting species stagnation history'
- result:    1 species removed for stagnation, 0 total extinctions


# Run log format (`runs/<run-id>/`)

Every run writes enough to re-derive every number in the report.

| file | one row per | contents |
|---|---|---|
| `run_args.json` | run | CLI args, git commit, start/end time, seed ranges, worker count |
| `config.ini` | run | exact copy of the NEAT config used — evaluate/replay load this |
| `budget_log.jsonl` | flush | scaffold step accounting (do not edit) |
| `fitness_history.csv` | generation | see columns below |
| `genomes.jsonl` | genome per generation | `gen, key, species, nodes, conns, fitness, scores[], steps[], points_for[], points_against[], timeouts, act_fwd, act_back, act_jump, act_idle` |
| `validation.jsonl` | validated genome per generation | `gen, key, train_fitness, val_mean, val_scores[], val_steps[]` |
| `champion.pkl` | run | genome with the best **validation** mean (not training fitness) |
| `champion.json` | run | champion key, generation found, train fitness, val mean, nodes, conns |

`fitness_history.csv` columns: `generation, best_fitness, mean_fitness` (scaffold),
then `median_fitness, std_fitness, min_fitness, n_species, best_nodes, best_conns,
mean_ep_steps, timeout_frac, gen_steps, cum_steps, val_best, champion_val`.

Seed ranges (disjoint by construction):
- training: drawn from `[0, 1_000_000)`, fresh per generation, shared by all genomes in it
- held-out (evaluate.py): `1_000_000 + i`
- validation (champion selection): `2_000_000 + i`, fixed for the whole run

Fitness / scores are game reward: +1 per point we win, -1 per point lost, in [-5, 5].
`act_*` are the fraction of steps each action pattern was taken (fwd/back/jump may
co-occur; idle = no buttons). `timeouts` = episodes that hit the 3000-step limit.

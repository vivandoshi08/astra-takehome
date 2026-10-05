# Run log format (`runs/<run-id>/`)

Every run writes enough to re-derive every number in the report.

| file | one row per | contents |
|---|---|---|
| `run_args.json` | run | CLI args, git commit, start/end time, seed ranges, worker count |
| `config.ini` | run | exact copy of the NEAT config used — evaluate/replay load this |
| `stdout.log` | run | everything printed, incl. NEAT's species / stagnation / extinction report |
| `budget_log.jsonl` | flush | scaffold step accounting (do not edit) |
| `fitness_history.csv` | generation | see columns below |
| `genomes.jsonl` | genome per generation | `gen, key, species, nodes, conns, fitness, raw_fitness, scores[], steps[], points_for[], points_against[], timeouts, act_fwd, act_back, act_jump, act_idle, act_both, out_sat, contacts[], opp_contacts[], returns[], moved[], jumps[], jump_hits[], zone_net, zone_mid, zone_back` |
| `validation.jsonl` | validated genome per generation | `gen, key, train_fitness, val_mean, val_scores[], val_steps[]`, plus `val_`-prefixed versions of the genomes.jsonl episode aggregates (`val_contacts[]`, `val_act_jump`, ...) |
| `champion.pkl` | run | genome with the best **validation** mean (not training fitness) |
| `snapshots/<N>M/` | every `--snapshot-every` steps (10M) | `champion.pkl/json` (best val so far) and `gen_best.pkl/json` (that generation's best validated genome); `replay_snapshots.py` turns them into `snapshots/progress.md` + GIFs on fixed held-out seeds |
| `champion.json` | run | champion key, generation found, train fitness, val mean, nodes, conns |

`fitness_history.csv` columns: `generation, best_fitness, mean_fitness` (scaffold),
then `median_fitness, std_fitness, min_fitness, n_species, best_nodes, best_conns,
mean_ep_steps, timeout_frac, gen_steps, cum_steps, val_best, champion_val, best_raw, mean_raw, mean_nodes, mean_conns, mean_contacts,
mean_returns, best_raw_contacts, mean_cancel, mean_sat, gen_seconds, mean_moved,
mean_jumps, jump_hit_rate, zone_net_frac, zone_back_frac`.

`fitness` / `best_fitness` / `mean_fitness` are the *selection* fitness: with
`--survival-bonus B` (in run_args) each episode counts `score + B*steps/3000`, so
`fitness == mean(score + B*steps/3000 + R*min(returns, C))` with `--return-bonus R`,
`--return-cap C` (B + R*C < 1 is enforced, so a point outweighs all shaping). `raw_fitness` / `best_raw` / `mean_raw` are
plain mean game score. Validation and champion selection always use raw score.

Seed ranges (disjoint by construction):
- training: drawn from `[0, 1_000_000)`, fresh per generation, shared by all genomes in it
- held-out (evaluate.py): `1_000_000 + i`, range `[1_000_000, 2_000_000)`
- validation (champion selection): `2_000_000 + i`, fixed for the whole run

Fitness / scores are game reward: +1 per point we win, -1 per point lost, in [-5, 5].
`act_*` are the fraction of steps each action pattern was taken (fwd/back/jump may
co-occur; idle = no buttons). `timeouts` = episodes that hit the 3000-step limit.

Episode-level behaviour fields (from `episode.run_episode`):
- `contacts` / `opp_contacts`: ball bounces off our agent / the opponent, counted by
  wrapping `Particle.bounce` (physics unchanged). Random play ~1.4 per episode, the
  built-in policy playing our side ~47.
- `returns`: our contacts after which the ball reached the opponent's half.
- `act_both`: fraction of steps with forward and backward both pressed (they cancel).
- `out_sat`: fraction of output values < 0.02 or > 0.98 (all-or-nothing outputs, where
  small weight mutations change no action).

Clarifications:
- `validation.jsonl` `train_fitness` is the genome's *selection* fitness that
  generation (shaped if `--survival-bonus` > 0); validation is always re-played,
  never cached, so `val_steps` are real budgeted steps.
- Every genome, including elites carried over unchanged, is re-scored each generation
  on that generation's seeds. A key in consecutive generations is the same genome.
- `gen` (jsonl) and `generation` (csv) are the same 0-based counter.
- `moved`: horizontal distance travelled per episode; `jumps`: takeoffs; `jump_hits`:
  jumps with a ball contact before landing (jump timing); `zone_*`: fraction of steps in
  the net / middle / back third of our half. References: net camper moves ~10, 21 jumps,
  3% hit, 99% at net; hand-written tracker ~180, 80% hit; built-in policy ~555, 52% hit.

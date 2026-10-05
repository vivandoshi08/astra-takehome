# Snapshot progress: `s20m_relative_s1`

Each row: genome scored on held-out seeds 1000000..1000019 (same seeds for every row). GIFs/sheets use the first 2.

| snapshot | which | gen | val mean | held-out mean [95% CI] | steps | timeouts | contacts | returns | fwd | back | jump | both | replays |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 10M | champion | 0 | -4.70 | -4.70 [-4.92, -4.48] | 628 | 0 | 1.7 | 0.7 | 0.22 | 0.00 | 1.00 | 0.00 | [1000000](replays/10M_champion_1000000.gif) (-5) [1000001](replays/10M_champion_1000001.gif) (-5) |
| 10M | gen best | 42 | -4.80 | -4.80 [-4.99, -4.61] | 630 | 0 | 3.0 | 0.7 | 0.49 | 0.44 | 0.52 | 0.32 | [1000000](replays/10M_gen_best_1000000.gif) (-5) [1000001](replays/10M_gen_best_1000001.gif) (-5) |
| final | champion | 0 | -4.70 | -4.70 [-4.92, -4.48] | 628 | 0 | 1.7 | 0.7 | 0.22 | 0.00 | 1.00 | 0.00 | [1000000](replays/final_champion_1000000.gif) (-5) [1000001](replays/final_champion_1000001.gif) (-5) |

Reference (reward_check.py probe seeds): random 1.4 contacts / 0.1 returns; hand-written tracker 13.6 / 3.0; built-in policy 45 / 24.

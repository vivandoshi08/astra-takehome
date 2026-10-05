# Snapshot progress: `s20m_relative_s0`

Each row: genome scored on held-out seeds 1000000..1000019 (same seeds for every row). GIFs/sheets use the first 2.

| snapshot | which | gen | val mean | held-out mean [95% CI] | steps | timeouts | contacts | returns | fwd | back | jump | both | replays |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 10M | champion | 20 | -4.60 | -4.75 [-4.96, -4.54] | 616 | 0 | 1.7 | 0.8 | 1.00 | 0.04 | 0.29 | 0.04 | [1000000](replays/10M_champion_1000000.gif) (-5) [1000001](replays/10M_champion_1000001.gif) (-5) |
| 10M | gen best | 42 | -4.75 | -4.70 [-4.97, -4.43] | 586 | 0 | 1.4 | 0.4 | 0.16 | 0.00 | 0.78 | 0.00 | [1000000](replays/10M_gen_best_1000000.gif) (-5) [1000001](replays/10M_gen_best_1000001.gif) (-5) |
| final | champion | 20 | -4.60 | -4.75 [-4.96, -4.54] | 616 | 0 | 1.7 | 0.8 | 1.00 | 0.04 | 0.29 | 0.04 | [1000000](replays/final_champion_1000000.gif) (-5) [1000001](replays/final_champion_1000001.gif) (-5) |

Reference (reward_check.py probe seeds): random 1.4 contacts / 0.1 returns; hand-written tracker 13.6 / 3.0; built-in policy 45 / 24.

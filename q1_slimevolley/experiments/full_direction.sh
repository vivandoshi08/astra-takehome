#!/usr/bin/env bash
# Full-budget replication of the curriculum with the 'direction' mapping + response 1.0.
# Identical to the full100m_curriculum_s{0,1,2} command except --config, on the same
# seeds 0-2 (old method: 1 of 3 escaped; held-out -0.37 / -4.77 / -4.79).
#
# Pre-registered before any of these runs: all three seeds are reported with their
# 100-episode held-out CIs. If one is named the headline, it is the run with the best
# final champion VALIDATION mean (champion.json val_mean) — never chosen on held-out.
#
# 3 runs concurrently x 3 workers; then evaluate.py (100 held-out) and snapshot replays.
set -uo pipefail
cd "$(dirname "$0")/.."
PY="${PYTHON:-python3}"
for seed in 0 1 2; do
  (
    id="full100m_direction_s$seed"
    mkdir -p "runs/$id"
    $PY evolve.py --seed "$seed" --config configs/direction_r1.ini --workers 3 \
      --phase1-steps 40000000 --phase1-survival 0.3 \
      --survival-bonus 0.3 --return-bonus 0.07 --return-cap 9 \
      --run-id "$id" > /dev/null 2> "runs/$id/stderr.log"
    $PY evaluate.py --winner "runs/$id/champion.pkl" --episodes 100 \
      --out "runs/$id/eval_heldout100.json" > "runs/$id/eval_heldout100.txt" 2>> "runs/$id/stderr.log"
    $PY replay_snapshots.py "runs/$id" --episodes 20 --gifs 1 > /dev/null 2>> "runs/$id/stderr.log"
    echo "done $id"
  ) &
done
wait
for seed in 0 1 2; do
  echo "== seed $seed"; sed -n 2,6p "runs/full100m_direction_s$seed/eval_heldout100.txt"
done

#!/usr/bin/env bash
# Why does the curriculum escape the net-camper trap on seed 0 but not seeds 1-2?
# One arm per hypothesis, each a single change from configs/relative.ini, phase 1
# only (15M steps, returns-based selection), on fresh seeds 10-12 shared by all arms.
#   control      configs/relative.ini
#   H1 wide      larger initial weights + mutation steps
#   H2 pop       population 64 -> 192
#   H3 episodes  10 games per genome instead of 5
#   H4 nodelete  connection / node delete probability -> 0.05
#   H5 species   compatibility threshold 2.0 -> 1.0
# Runs 3 at a time with 3 workers each (same total throughput as 1 x 9).
# Summarise with: python experiments/summarize_discovery.py
set -uo pipefail
cd "$(dirname "$0")/.."
PY="${PYTHON:-python3}"
COMMON="--budget 15000000 --phase1-steps 15000000 --phase1-survival 0.3 --workers 3 --snapshot-every 5000000"

jobs=()
for seed in 10 11 12; do
  jobs+=("dh_control_s$seed|--config configs/relative.ini --seed $seed")
  jobs+=("dh_h1wide_s$seed|--config configs/h1_wide.ini --seed $seed")
  jobs+=("dh_h2pop_s$seed|--config configs/h2_pop.ini --seed $seed")
  jobs+=("dh_h3episodes_s$seed|--config configs/relative.ini --episodes 10 --seed $seed")
  jobs+=("dh_h4nodelete_s$seed|--config configs/h4_nodelete.ini --seed $seed")
  jobs+=("dh_h5species_s$seed|--config configs/h5_species.ini --seed $seed")
done

for job in "${jobs[@]}"; do
  id="${job%%|*}"; args="${job#*|}"
  [ -f "runs/$id/run_args.json" ] && grep -q end_time "runs/$id/run_args.json" && continue
  while [ "$(jobs -rp | wc -l)" -ge 3 ]; do sleep 5; done
  echo "start $id"
  # shellcheck disable=SC2086
  $PY evolve.py $COMMON $args --run-id "$id" > /dev/null 2>&1 &
done
wait
echo "all done"

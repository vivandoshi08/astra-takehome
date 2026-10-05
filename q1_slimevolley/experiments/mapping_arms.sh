#!/usr/bin/env bash
# Does the 'direction' action mapping remove the net-camper valley in evolution?
# Same protocol as discovery_hypotheses.sh (phase 1 only, 15M steps, seeds 10-12),
# so the dh_control_s{10,11,12} runs are the control (0/3 escaped).
#   direction       configs/direction.ini       (mapping only)
#   direction_r1    configs/direction_r1.ini    (+ response 1.0)
#   direction_wide  configs/direction_wide.ini  (+ weight init 3, mutation 1.0)
# Summarise with: python experiments/summarize_discovery.py --arms direction direction_r1 direction_wide
set -uo pipefail
cd "$(dirname "$0")/.."
PY="${PYTHON:-python3}"
COMMON="--budget 15000000 --phase1-steps 15000000 --phase1-survival 0.3 --workers 3 --snapshot-every 5000000"

jobs=()
for seed in 10 11 12; do
  for arm in direction direction_r1 direction_wide; do
    jobs+=("dh_${arm}_s$seed|--config configs/$arm.ini --seed $seed")
  done
done

for job in "${jobs[@]}"; do
  id="${job%%|*}"; args="${job#*|}"
  [ -f "runs/$id/run_args.json" ] && grep -q end_time "runs/$id/run_args.json" && continue
  while [ "$(jobs -rp | wc -l)" -ge 3 ]; do sleep 5; done
  echo "start $id"
  mkdir -p "runs/$id"
  # shellcheck disable=SC2086
  $PY evolve.py $COMMON $args --run-id "$id" > /dev/null 2> "runs/$id/stderr.log" &
done
wait
echo "all done"

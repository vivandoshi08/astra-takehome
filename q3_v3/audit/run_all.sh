#!/usr/bin/env bash
# Re-run the whole Q3 scaffold audit (AUDIT.md). Text results land in
# audit/results/{math,loop}/*.txt; scratch files (npz, png, run dirs) go to
# ${Q3_AUDIT_OUT:-$TMPDIR/q3_audit}, never the repo. No network: fake clients only.
#   PY=/path/to/python bash audit/run_all.sh        (~4 min)
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
Q3="$(cd "$HERE/.." && pwd)"
export PY="${PY:-python}" PYTHONDONTWRITEBYTECODE=1
unset OPENROUTER_API_KEY
R="$HERE/results"; mkdir -p "$R/math" "$R/loop"

cd "$Q3"    # the math checks import data/baselines/vae from the CWD
for s in check_data check_evaluate check_floor check_vae_math \
         check_vae_clip_practice check_misc; do
  echo "##### math/$s"; "$PY" "$HERE/math/$s.py" > "$R/math/$s.txt" 2>&1; echo "exit $?"
done
echo "##### math/check_baselines"; "$PY" "$HERE/math/check_baselines.py" --vae-control > "$R/math/check_baselines.txt" 2>&1; echo "exit $?"
echo "##### math/check_vae_retrain"; "$PY" "$HERE/math/check_vae_retrain.py" 2 > "$R/math/check_vae_retrain.txt" 2>&1; echo "exit $?"
echo "##### race_sandboxed"; bash "$HERE/run_race_sandboxed.sh" > "$R/math/race_sandboxed.txt" 2>&1; echo "exit $?"

cd "$HERE/loop"
for s in check_accounting check_main check_extract check_isolation \
         check_population check_spend check_mock check_cost check_timeout \
         check_misc; do
  echo "##### loop/$s"; "$PY" "$s.py" > "$R/loop/$s.txt" 2>&1; echo "exit $?"
done
echo "##### loop/check_packaging"; bash check_packaging.sh > "$R/loop/check_packaging.txt" 2>&1; echo "exit $?"

echo; echo "repo status (only audit/results should be new):"
git -C "$Q3" status --short --ignored -- . | grep -v DS_Store

#!/usr/bin/env bash
# race.py and plots.py write PNGs into the current directory, so run them from
# a temp dir. Also checks race.py's --images handling (HELD_OUT, PUBLIC_CLEAN,
# and SIGMA, which is not an image array and crashes).
set -uo pipefail
Q3="$(cd "$(dirname "$0")/.." && pwd)"
PY="${PY:-python}"
W="${Q3_AUDIT_OUT:-${TMPDIR:-/tmp}/q3_audit}/race_out"
mkdir -p "$W/fake_runs/seedrun" && cd "$W" || exit 1
PYTHONPATH="$Q3" "$PY" -c "from tasks import ellipses; print(ellipses.SEED)" > fake_runs/seedrun/best.py
PYTHONPATH="$Q3" "$PY" "$Q3/race.py" fake_runs/seedrun/best.py
PYTHONPATH="$Q3" "$PY" "$Q3/race.py" --images HELD_OUT --no-plots fake_runs/seedrun/best.py
PYTHONPATH="$Q3" "$PY" "$Q3/race.py" --images PUBLIC_CLEAN --no-plots
PYTHONPATH="$Q3" "$PY" "$Q3/race.py" --images SIGMA --no-plots 2>&1 | tail -2
PYTHONPATH="$Q3" "$PY" "$Q3/plots.py"
echo "PNGs in $W"

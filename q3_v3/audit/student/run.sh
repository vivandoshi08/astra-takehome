#!/bin/bash
# Independent verification of student.py (see results/ and DEVLOG).
# usage: PY=/path/to/python bash audit/student/run.sh t1_scores.py [args]
# Runs with the CWD in a temp dir (scratch npz files go there, never the
# repo), imports from q3_v3, writes no bytecode, and has no API key.
SCR="$(cd "$(dirname "$0")" && pwd)"
Q3="$(cd "$SCR/../.." && pwd)"
W="${Q3_AUDIT_OUT:-${TMPDIR:-/tmp}/q3_audit}/student"
mkdir -p "$W/out" && cd "$W" || exit 1
unset OPENROUTER_API_KEY
script="$1"; shift
case "$script" in /*) ;; *) script="$SCR/$script" ;; esac
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$Q3:$SCR" "${PY:-python}" "$script" "$@"

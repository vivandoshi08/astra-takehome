#!/usr/bin/env bash
# Runs package_submission.sh on a scratch copy of the base commit's Q3 files
# (never inside the repo) and lists what the tarball actually contains.
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/../../.." && pwd)"
W="${Q3_AUDIT_OUT:-${TMPDIR:-/tmp}/q3_audit}/loop/packaging"
rm -rf "$W"; mkdir -p "$W/copy"
cd "$W/copy"
git -C "$REPO" archive d23a4bd package_submission.sh .gitignore README.md QUESTIONS.md q3_v3 | tar -xf -
git init -q && git add -A && git -c user.email=a@b -c user.name=audit commit -qm base
# A finished-looking run that the user has not committed yet:
mkdir -p q3_v3/runs/r1
echo '{"generation": 0}' > q3_v3/runs/r1/run_log.jsonl
echo '{"call": 1}' > q3_v3/runs/r1/token_log.jsonl
printf 'def encode(img):\n    return [0]*6\n' > q3_v3/runs/r1/best.py

echo "=== A. base-commit Q3 files + one UNTRACKED run dir"
TMPDIR="$W" bash package_submission.sh 2>&1 | sed 's/^/  | /'
T=$(ls astra_submission_*.tar.gz)
echo "  tarball entries under q3_v3/:"
tar -tzf "$T" | grep 'q3_v3/' | grep -v '/$' | sed 's/^/    /'
echo "  best.py in tarball file copies: $(tar -tzf "$T" | grep -c 'runs/r1/best.py')"
echo "  student.py in tarball file copies: $(tar -tzf "$T" | grep -c 'q3_v3/student.py')"
rm -f "$T"

echo
echo "=== B. same, after committing the run dir"
git add -A && git -c user.email=a@b -c user.name=audit commit -qm run
TMPDIR="$W" bash package_submission.sh 2>&1 | grep -E 'WARN|Everything' | sed 's/^/  | /'
T=$(ls astra_submission_*.tar.gz)
echo "  best.py in tarball file copies: $(tar -tzf "$T" | grep -c 'runs/r1/best.py')"
mkdir -p unpack && tar -xzf "$T" -C unpack
B=$(ls unpack/*/repo.bundle)
echo "  best.py in repo.bundle (committed tree): $(git bundle list-heads "$B" >/dev/null && git clone -q "$B" unpack/clone 2>/dev/null; ls unpack/clone/q3_v3/runs/r1/best.py 2>/dev/null | wc -l | tr -d ' ')"
rm -f "$T"

echo
echo "=== C. does a Q3 report check ever fire? (finding.md template ships)"
ls q3_v3/*.md | sed 's/^/    /'

echo
echo "=== D. student.py with everything BETWEEN the markers replaced (markers kept, as instructed)"
/opt/anaconda3/envs/astra/bin/python - <<'PY'
import re, pathlib
p = pathlib.Path("q3_v3/student.py")
s = p.read_text()
start = s.index("# TODO: replace everything between these markers")
start = s.index("\n# ----", start) + 1          # end of the opening marker block
start = s.index("\n", start) + 1
end = s.index("# ---------------------------------------------------------------------------\n# end of your model")
s = s[:start] + "\ndef encode(img):\n    return [8.0]*6   # my own model\n\ndef decode(z):\n    return [0.0]*256\n\n" + s[end:]
p.write_text(s)
PY
sed -n '24,40p' q3_v3/student.py | sed 's/^/    /'
if grep -q 'TODO: replace everything between these markers' q3_v3/student.py; then
  echo "  -> package_submission.sh would WARN 'student.py still contains the placeholder' (false alarm)"
fi

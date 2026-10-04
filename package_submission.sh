#!/usr/bin/env bash
# Packages your submission into one timestamped tarball.
#
#   bash package_submission.sh
#
# Collects your git history, reports (whatever you named them), run logs,
# budget logs, and plots. Prints what it found and warns about anything we
# expect but couldn't find. Warnings don't stop the tarball from being
# written — read them and decide.

set -uo pipefail
cd "$(dirname "$0")"

STAMP="$(date +%Y%m%d_%H%M%S)"
NAME="astra_submission_${STAMP}"
STAGE="$(mktemp -d "${TMPDIR:-/tmp}/astra_pkg.XXXXXX")"
trap 'rm -rf "$STAGE"' EXIT
mkdir -p "$STAGE/$NAME"

warned=0
warn() { printf 'WARN  %s\n' "$1"; warned=1; }
info() { printf '      %s\n' "$1"; }

echo "ASTRA take-home — packaging submission"
echo

# ------------------------------------------------------------------ git history
if [ -d .git ]; then
  ncommits="$(git rev-list --all --count 2>/dev/null || echo 0)"
  if [ "$ncommits" -eq 0 ]; then
    warn "no commits yet — there is no history to bundle, and we read the history"
  elif git bundle create "$STAGE/$NAME/repo.bundle" --all >/dev/null 2>&1; then
    info "git history: $ncommits commits bundled"
  else
    warn "git bundle failed — history will be missing from the tarball"
  fi
  git log --all --pretty=format:'%h|%at|%an|%s' > "$STAGE/$NAME/git_log.txt" 2>/dev/null
  if ! git diff --quiet 2>/dev/null || ! git diff --cached --quiet 2>/dev/null; then
    git diff HEAD > "$STAGE/$NAME/uncommitted.diff" 2>/dev/null
    warn "you have uncommitted changes — included as uncommitted.diff, but commit them if you can"
  fi
  if [ -n "$(git ls-files --others --exclude-standard 2>/dev/null)" ]; then
    warn "you have untracked files — the file copies below include them, but the git history does not"
  fi
else
  warn "no .git directory here — no history to include, and history is part of what we read"
fi

# ------------------------------------------------- reports, logs, plots, models
# Everything matching below is copied with its path preserved, so files with
# the same name in different question directories cannot collide.
LIST0="$STAGE/files0"        # null-delimited, for tar
LIST="$STAGE/files.txt"      # newline-delimited, for counting

find . \
    \( -name .git -o -name .venv -o -name venv -o -name env \
       -o -name __pycache__ -o -name checkpoints \) -prune \
    -o -type f \
    \( -name '*.md' -o -name '*.pdf' -o -name '*.html' -o -name '*.ipynb' \
       -o -name '*.docx' -o -name '*.txt' \
       -o -name '*.jsonl' -o -name '*.csv' -o -name '*.pkl' \
       -o -name '*.png' -o -name '*.jpg' -o -name '*.jpeg' -o -name '*.svg' \
       -o -name '*.gif' -o -name '*.mp4' \
       -o -path './agent_trace/*' \) \
    ! -name 'astra_submission_*.tar.gz' -print0 > "$LIST0"
tr '\0' '\n' < "$LIST0" > "$LIST"

if [ -s "$LIST0" ]; then
  if ! tar -cf - --null -T "$LIST0" | tar -xf - -C "$STAGE/$NAME"; then
    warn "some files could not be copied — check the tar errors above before sending"
  fi
else
  warn "found no reports, logs, or plots at all — is this the right directory?"
fi

count() { grep -c -E "$1" "$LIST" 2>/dev/null || true; }
info "documents:   $(count '\.(md|pdf|html|ipynb|docx|txt)$') (.md/.pdf/.html/.ipynb/.docx/.txt)"
info "logs:        $(count '\.(jsonl|csv)$') (.jsonl/.csv)"
info "plots/media: $(count '\.(png|jpe?g|svg|gif|mp4)$')"
info "pickles:     $(count '\.pkl$')"
n_trace=$(count '^\./agent_trace/')
if [ "$n_trace" -gt 0 ]; then
  info "agent trace: $n_trace (optional, not graded — check you scrubbed any keys)"
fi

# -------------------------------------------------------- per-question checks
for qdir in q*/; do
  [ -d "$qdir" ] || continue
  q="${qdir%/}"

  # A report: any document in the question directory that isn't the README.
  if ! find "$qdir" -maxdepth 2 -type f \
        \( -name '*.md' -o -name '*.pdf' -o -name '*.html' -o -name '*.ipynb' \
           -o -name '*.docx' \) ! -name 'README.md' 2>/dev/null | grep -q .; then
    warn "$q: no report found (any .md/.pdf/.html/.ipynb/.docx besides README.md counts)"
  fi

  case "$q" in
    q1_slimevolley|q2_backprop_neat)
      if ! find "$qdir" -path '*/runs/*' -name 'fitness_history.csv' 2>/dev/null | grep -q .; then
        warn "$q: no runs/*/fitness_history.csv — no training run recorded"
      fi
      ;;
    q3_*)
      if ! find "$qdir" -path '*/runs/*' -name 'run_log.jsonl' 2>/dev/null | grep -q .; then
        warn "$q: no runs/*/run_log.jsonl — no loop run recorded"
      fi
      if ! find "$qdir" -path '*/runs/*' -name 'token_log.jsonl' 2>/dev/null | grep -q .; then
        warn "$q: no runs/*/token_log.jsonl — token accounting is part of the submission"
      fi
      if [ -f "$qdir/student.py" ] \
         && grep -q 'TODO: replace everything between these markers' "$qdir/student.py"; then
        warn "$q: student.py still contains the placeholder — your own model is a deliverable"
      fi
      ;;
  esac
  if [ "$q" = "q1_slimevolley" ]; then
    if ! find "$qdir" -path '*/runs/*' -name 'budget_log.jsonl' 2>/dev/null | grep -q .; then
      warn "$q: no runs/*/budget_log.jsonl — step accounting is part of the submission"
    fi
  fi
done

# -------------------------------------------------------------------- tarball
OUT="${NAME}.tar.gz"
if ! tar -czf "$OUT" -C "$STAGE" "$NAME"; then
  echo
  echo "FAILED to write $OUT" >&2
  exit 1
fi

echo
echo "Wrote $OUT ($(du -h "$OUT" | cut -f1 | tr -d ' '))"
if [ "$warned" -ne 0 ]; then
  echo "There were warnings above. The tarball was still written — review them"
  echo "before you send it."
else
  echo "Everything expected was found. Email us the tarball."
fi

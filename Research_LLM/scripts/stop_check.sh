#!/usr/bin/env bash
# Stop hook for Research_LLM (separate git repo, pure Python — no Next.js
# stack here, unlike the parent Consulting Dashboard).
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

changed=$(git status --porcelain --untracked-files=no 2>/dev/null | awk '{print $2}')
status=0

py_changed=$(echo "$changed" | grep -E '\.py$' || true)
if [ -n "$py_changed" ]; then
  echo "-- Python files changed: syntax check --"
  for f in $py_changed; do
    [ -f "$f" ] && { python3 -m py_compile "$f" || status=1; }
  done
fi

if python3 -m pytest --version >/dev/null 2>&1 && [ -d tests ]; then
  echo "-- Running pytest --"
  python3 -m pytest tests/ -q || status=1
else
  echo "-- pytest not installed / no tests/ dir, skipping test run --"
fi

exit $status

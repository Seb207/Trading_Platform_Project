#!/usr/bin/env bash
# Runs on Claude Code's Stop hook. Branches by which part of this monorepo
# actually changed, since it mixes a Next.js/FastAPI stack (Consulting
# Dashboard) with plain Python (Market Regime). Exits non-zero (blocking
# the turn from ending cleanly) if anything fails.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

changed=$(git status --porcelain --untracked-files=no 2>/dev/null | awk '{print $2}')
status=0

if echo "$changed" | grep -q '^Consulting Dashboard/'; then
  echo "-- Consulting Dashboard changed: npm lint + build --"
  (cd "Consulting Dashboard" && npm run lint && npm run build) || status=1
fi

py_changed=$(echo "$changed" | grep -E '^(Market Regime|Research_LLM)/.*\.py$' || true)
if [ -n "$py_changed" ]; then
  echo "-- Python files changed: syntax check --"
  for f in $py_changed; do
    [ -f "$f" ] && { python3 -m py_compile "$f" || status=1; }
  done
fi

exit $status

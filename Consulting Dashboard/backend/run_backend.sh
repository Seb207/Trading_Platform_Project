#!/usr/bin/env bash
# Always launches uvicorn with the project's own .venv, by absolute path —
# immune to shell PATH state. Why this exists: this machine's ~/.zshrc runs
# `conda init`, which auto-activates the anaconda "base" env in every new
# terminal and puts /opt/anaconda3/bin ahead of everything else in PATH. A
# bare `uvicorn` or `python3` (even after `cd`-ing into this project) then
# silently resolves to conda's env instead of this project's .venv. Conda's
# pyarrow is an older version than the one that wrote
# `Market Regime/data/factors_weekly.parquet`, so reading it fails with:
#   OSError: Repetition level histogram size mismatch
# which surfaces in the browser as a Market Regime page load/TypeError.
# Use this script (not a bare `uvicorn` command) so the correct Python is
# guaranteed regardless of shell/conda state.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DASHBOARD_DIR="$(dirname "$SCRIPT_DIR")"
VENV_PYTHON="$DASHBOARD_DIR/../.venv/bin/python3"

if [ ! -x "$VENV_PYTHON" ]; then
  echo "Project venv not found at: $VENV_PYTHON" >&2
  echo "Create it first:" >&2
  echo "  cd \"$(dirname "$DASHBOARD_DIR")\"" >&2
  echo "  python3 -m venv .venv" >&2
  echo "  source .venv/bin/activate" >&2
  echo "  pip install -r \"$DASHBOARD_DIR/backend/requirements.txt\"" >&2
  exit 1
fi

cd "$DASHBOARD_DIR"
exec "$VENV_PYTHON" -m uvicorn backend.main:app --reload --port 8000

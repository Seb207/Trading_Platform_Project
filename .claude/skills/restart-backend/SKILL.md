---
name: restart-backend
description: Safely restart the Consulting Dashboard FastAPI backend and verify it came back up correctly, including both the Market Regime and Paper2Alpha bridge endpoints. Use whenever backend code, dependencies, or config changed and a restart is needed — never restart with a bare `uvicorn` command.
---

# Restart Backend

## Why this exists

This machine's `~/.zshrc` runs `conda init`, which auto-activates anaconda's
`base` env in every new terminal and puts it ahead of the project `.venv` on
PATH. A bare `uvicorn backend.main:app` or `python3 -m uvicorn ...` — even
from inside the project directory — can silently launch under conda's Python
instead of the project's `.venv`, which has caused two separate production
bugs (pyarrow version mismatch breaking Market Regime, missing
chromadb/sentence-transformers breaking Paper2Alpha). See
`Trading_Platform_Project/CLAUDE.md` §2 for the full history.

## Steps

1. **Kill any running instance**:
   ```bash
   pkill -f "uvicorn backend.main"
   ```
2. **Launch via the wrapper script — never a bare `uvicorn`/`python3` command**:
   ```bash
   cd "Consulting Dashboard"
   nohup ./backend/run_backend.sh > /tmp/backend_restart.log 2>&1 &
   ```
   `run_backend.sh` resolves the venv Python by absolute path
   (`$DASHBOARD_DIR/../.venv/bin/python3`), so it works regardless of what
   shell/conda state is active.
3. **Wait ~4s for cold start** (pandas/numpy/scipy imports plus uvicorn's
   `--reload` watcher subprocess take a few seconds).
4. **Verify health**:
   ```bash
   curl -s http://localhost:8000/health
   ```
5. **Verify both bridge families didn't regress** — a change to one bridge's
   dependencies can silently break the other if `run_backend.sh` picks up a
   different environment than expected:
   ```bash
   curl -s http://localhost:8000/api/regime/factors | head -c 200
   curl -s -X POST http://localhost:8000/api/papers/search/abstract \
     -H "Content-Type: application/json" \
     -d '{"query":"momentum factor equity","n_results":3,"category":""}' | head -c 300
   ```
   Both must return 200 with real data, not a stack trace or `Phase 4
   requires additional packages` message.
6. If either check fails, don't assume — read `/tmp/backend_restart.log` for
   the actual import error, and check `which python3` / `sys.executable`
   inside the failing process against the expected `.venv` path before
   looking for a code-level bug (see the `diagnose-env-mismatch` skill).

## If `run_backend.sh` itself is missing or fails

It expects the project venv at `<repo root>/.venv`. If that doesn't exist:
```bash
cd "Trading_Platform_Project"
python3 -m venv .venv
source .venv/bin/activate
pip install -r "Consulting Dashboard/backend/requirements.txt"
```

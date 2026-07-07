---
name: add-bridge-dependency
description: Procedure for adding a new Python package dependency needed by Market Regime or Research_LLM when the Consulting Dashboard backend bridges to it. Use whenever a bridge module (backend/modules/regime/ or backend/modules/research/) starts failing with an ImportError, or when adding a new feature to Market Regime/ or Research_LLM/ that pulls in a new package.
---

# Add Bridge Dependency

## Why this exists

`Consulting Dashboard/backend/` imports code directly from `Market Regime/`
and `Research_LLM/` (the bridge pattern — see `CLAUDE.md` §1) rather than
depending on them as installed packages. Nothing enforces that
`backend/requirements.txt` stays in sync with what those modules actually
need — no package manager flags the drift. This exact gap caused the
Paper2Alpha regression (`"Phase 4 requires additional packages"`) after an
unrelated backend fix. See `CLAUDE.md` §3.

## Steps

1. **Identify the source of truth**:
   - Research_LLM: `Research_LLM/requirements.txt` is authoritative.
   - Market Regime: no separate requirements file — versions are pinned
     directly in `backend/requirements.txt`. If Market Regime code starts
     needing a new package, add it there with a comment explaining why.
2. **Add the package to the source of truth first** (if it's a Research_LLM
   dependency and not already listed there).
3. **Update `Consulting Dashboard/backend/requirements.txt`** in the matching
   section (Market Regime bridge / Research_LLM bridge), keeping the existing
   comment blocks that explain *why* each group of packages is there — extend
   them rather than replacing them.
4. **Install into the project `.venv`** (not conda — see
   `diagnose-env-mismatch` if unsure which is active):
   ```bash
   "Trading_Platform_Project/.venv/bin/pip" install -q <package>
   ```
5. **Verify the import succeeds in that exact venv**:
   ```bash
   "Trading_Platform_Project/.venv/bin/python3" -c "import <package>; print(<package>.__version__)"
   ```
6. **Restart the backend and verify both bridge families** — use the
   `restart-backend` skill in full, not just the one endpoint you changed.
   A dependency install can have side effects on the other bridge (e.g. a
   version bump satisfying one requirement can break another's pin).

## Common pitfall

Don't just `pip install` and move on — if you skip updating
`backend/requirements.txt`, the fix works locally but silently regresses the
next time someone provisions a fresh venv from that file. The file is the
actual contract; the installed venv is just its current materialization.

---
name: diagnose-env-mismatch
description: First-response checklist for bugs that are inconsistent across runs/restarts, work "sometimes," or fail with missing-package/version errors that don't match what's in requirements.txt. Use before assuming a code bug when behavior doesn't match what the code says it should do.
---

# Diagnose Environment Mismatch

## Why this exists

This project has a recurring failure class that looks like a code bug but
isn't: the running process is using a different Python interpreter/env than
the one that was edited or reasoned about. Root cause is `conda init` in
`~/.zshrc` auto-activating anaconda's `base` env ahead of the project's
`.venv` on PATH in every new terminal. This produced two distinct incidents
(pyarrow version mismatch, missing chromadb/sentence-transformers) that both
initially looked like application bugs. See `CLAUDE.md` §2 for full history.

## When to reach for this

- An error mentions a package version that doesn't match what's pinned in
  `requirements.txt`.
- A fix that should have worked didn't, or worked once and then didn't.
- Behavior differs between "just restarted" and "been running a while," or
  between different terminal windows/sessions.
- An import error for a package you're confident is installed.

## Steps

1. **Check which interpreter is actually running the failing process** —
   don't assume it's the project venv:
   ```bash
   ps aux | grep uvicorn   # or the relevant process
   ```
   Cross-reference the binary path against
   `Trading_Platform_Project/.venv/bin/python3`. If it points anywhere else
   (e.g. `/opt/anaconda3/...`), that's very likely the whole bug.
2. **From inside Python, confirm directly** rather than trusting the shell:
   ```python
   import sys
   print(sys.executable)
   ```
3. **If it's the wrong environment**: don't patch around it (e.g. installing
   the missing package into conda too) — that just papers over the PATH
   hijack and the bug will resurface differently later, as it did twice
   already. Use `restart-backend`, which launches via
   `run_backend.sh`'s absolute venv path specifically to route around this.
   If the backend wasn't started via `run_backend.sh` at all — e.g. it was
   started by `/Applications/Quant Dashboard.app` (a hand-built launcher;
   see `CLAUDE.md` §2 for its own venv fix) — check `ps aux` for how it was
   actually invoked before assuming `run_backend.sh` is even the process
   you're dealing with.
4. **If it's confirmed to be the right environment** and the bug persists,
   only then move on to treating it as an actual code/logic bug.

## Rule of thumb

Any bug report containing "requires additional packages" or a version
number that contradicts `requirements.txt` should start here, not with
reading application code.

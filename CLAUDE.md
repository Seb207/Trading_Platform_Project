# Trading Platform Project — Claude Working Notes

This file is the accumulated operating memory for working on this repo with
Claude Code. It is **not a static spec** — it is a living document. It holds
only **cross-module** conventions and gotchas; module-specific detail lives
in each module's own `CLAUDE.md` (linked in §1).

## Maintenance Protocol (read this first)

- After any session where you hit a bug, made a wrong assumption, discovered
  an environment quirk, or the user corrected your approach: check whether it
  generalizes beyond the one-off fix. If it does, add or update a section
  below (or in the relevant module's `CLAUDE.md`) before ending the session.
  Don't wait to be asked.
- This protocol applies to every `CLAUDE.md` in this repo, not just this
  root file — `Market Regime/CLAUDE.md`, `Consulting Dashboard/CLAUDE.md`,
  and `Research_LLM/CLAUDE.md` all need the same upkeep.
- **Where a fact belongs**: if it only matters while working inside one
  module, put it in that module's file. If it spans modules (the bridge
  pattern, environment setup, dependency sync, cross-cutting discipline),
  it belongs here. When in doubt, prefer the narrower file — it's the one
  that's actually in view when someone is heads-down in that module.
- Prefer editing an existing section over appending a new one — merge
  related facts instead of letting any of these files sprawl into
  duplicates.
- If a documented gotcha turns out to be wrong, outdated, or fixed for good
  (e.g. a root cause is structurally eliminated, not just patched), remove
  or correct it rather than leaving stale guidance for the next session.
- Every entry should carry a **why**, not just a **what** — the reasoning is
  what lets a future session judge edge cases the entry didn't anticipate.
- Repeated multi-step procedures (not one-off facts) belong in
  `.claude/skills/`, not in any `CLAUDE.md` — these files are for standing
  context and gotchas; skills are for runbooks. Cross-reference by name
  when relevant.

## 1. Module Map

Monorepo with three active modules, plus a bridge layer connecting them.
Each has its own `CLAUDE.md` for module-specific notes — read this file
first for cross-cutting context, then the module file for anything
specific to where you're working:

- **[`Consulting Dashboard/`](Consulting%20Dashboard/CLAUDE.md)** —
  Next.js frontend + FastAPI backend. The primary UI surface.
  `backend/modules/` contains thin bridge routers that import logic living
  in the other two modules rather than reimplementing it.
- **[`Market Regime/`](Market%20Regime/CLAUDE.md)** — standalone Python
  analysis engine (no server of its own). Computes factor datasets and
  similarity search. Bridged into the dashboard via
  `Consulting Dashboard/backend/modules/regime/regime_bridge.py`, which
  lazy-imports `similarity_engine.py`, `validation.py`, and
  `factor_schema.py` directly from this directory.
- **[`Research_LLM/`](Research_LLM/CLAUDE.md)** — Paper2Alpha: arXiv MCP
  server + semantic search over downloaded papers (`arxiv_client.py`,
  `mcp_server.py`). **This directory is its own nested git repo** (has its
  own `.git`), separate from this top-level repo. Because of that git
  boundary, a session opened directly inside `Research_LLM/` may not
  auto-load this root file — its `CLAUDE.md` duplicates the
  cross-cutting facts it needs rather than assuming they're inherited.
  Bridged into the dashboard via
  `Consulting Dashboard/backend/modules/research/arxiv_bridge.py`.

**Why the bridge pattern matters**: neither `Market Regime/` nor
`Research_LLM/` know they're being used by a web backend. All the framework
plumbing (FastAPI routes, request/response schemas) lives in
`Consulting Dashboard/backend/`. When a bug is a data/logic issue, look in the
source module; when it's an HTTP/serialization issue, look in the bridge.

## 2. Environment Gotchas

### conda hijacks PATH — always launch the backend via `run_backend.sh`
`~/.zshrc` runs `conda init`, which auto-activates anaconda's `base` env in
every new terminal and puts `/opt/anaconda3/bin` ahead of the project's
`.venv` on PATH. A bare `uvicorn` or `python3` — even after `cd`-ing into the
project, and regardless of which module's code you're running — silently
resolves to conda's Python, not the project venv.

This caused two separate incidents:
- Conda's `pyarrow` was older than the one that wrote
  `Market Regime/data/factors_weekly.parquet`, so reading it raised
  `OSError: Repetition level histogram size mismatch`, surfacing in the
  browser as a Market Regime page load failure / TypeError. Detail:
  `Market Regime/CLAUDE.md`.
- Conda's `base` env happened to have `chromadb`/`sentence-transformers`
  installed from unrelated prior work, masking the fact that the project
  `.venv` didn't — until the backend was switched to launch from `.venv`
  correctly, at which point Paper2Alpha search broke with
  `"Phase 4 requires additional packages"`. Detail: `Research_LLM/CLAUDE.md`.

**Fix**: never start the backend with a bare `uvicorn`/`python3` command. Use
`Consulting Dashboard/backend/run_backend.sh`, which resolves the venv python
by absolute path (`$DASHBOARD_DIR/../.venv/bin/python3`), immune to shell/conda
state. See the `restart-backend` skill. If a bug looks inconsistent across
runs or mentions a package/version that contradicts a `requirements.txt`,
check the interpreter first — see the `diagnose-env-mismatch` skill.

## 3. Dependency Sync Rule (bridge modules)

`Consulting Dashboard/backend/requirements.txt` must be manually kept in
sync with the source-of-truth requirements of the modules it bridges to —
it duplicates rather than depends on them, since the backend imports their
code directly (bridge pattern above) rather than installing them as
packages. Nothing enforces this automatically; no package manager flags the
drift. This exact gap caused the Paper2Alpha regression referenced above.

- Market Regime deps (`pandas`, `numpy`, `scipy`, `pyarrow`) — pinned
  directly in `backend/requirements.txt`; see `Market Regime/CLAUDE.md` for
  the pyarrow version-floor coupling.
- Research_LLM deps (`mcp`, `requests`, `beautifulsoup4`, `markdownify`,
  `chromadb`, `sentence-transformers`, `pypdf`) — `Research_LLM/requirements.txt`
  is the source of truth; see `Research_LLM/CLAUDE.md` for detail.

See the `add-bridge-dependency` skill for the procedure when adding a new
one.

## 4. Verification Discipline

Do not declare a change complete based on code review alone. Repeatedly in
this project, bugs that looked fine on inspection were only caught by
actually running the change and checking real output: DOM fill-color order,
`getBoundingClientRect()` dimensions, live network responses, actual
pip/import checks in the specific environment in question (not the one
assumed to be active). Use the preview tool's console/network/DOM
inspection for frontend work (see the `verify-ui-change` skill), or direct
curl/python checks against the running process for backend work — never
assume a fix worked without observing it.

## 5. Design Precedent

When scoping a new analysis feature anywhere in this platform, the Market
Regime Detector's locked scope decision is the reference — see
`Market Regime/CLAUDE.md` for the full rationale: retrieve and present
historical context first; treat statistical validation as opt-in, not a
blocking gate.

## 6. Credentials & Local Services — Never Substitute, Always Ask

When a task needs the user's confidential information (API keys,
credentials, tokens) — or would require starting/driving a local service on
their machine (e.g. a local LLM runtime like Ollama) — **stop and ask the
user first.** Do not fabricate a placeholder/fake key to route around a
missing credential, and do not autonomously switch to or spin up a local
service as a workaround instead of asking.

**Why**: during Paper2Alpha critic-loop testing, a fake OpenRouter API key
and an autonomous switch to a locally-running Ollama model were both used
to avoid asking the user for real credentials. The user flagged this as an
overstep — decisions involving their credentials or side effects on their
local environment (starting local model servers, consuming rate-limited
keys) are theirs to make, not defaults an assistant should choose
unilaterally.

**How to apply**: If a task requires an API key/secret not already present
in the environment or request, pause mid-task and ask the user for it (or
ask whether a mock/test value is acceptable for a specific verification
step) rather than substituting one. Never autonomously launch or drive a
local LLM/service to work around a missing credential — ask first, every
time, even mid-task.

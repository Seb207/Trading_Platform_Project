# Trading Platform Project — Claude Working Notes

This file is the accumulated operating memory for working on this repo with
Claude Code. It is **not a static spec** — it is a living document.

## Maintenance Protocol (read this first)

- After any session where you hit a bug, made a wrong assumption, discovered
  an environment quirk, or the user corrected your approach: check whether it
  generalizes beyond the one-off fix. If it does, add or update a section
  below before ending the session. Don't wait to be asked.
- Prefer editing an existing section over appending a new one — merge related
  facts instead of letting the file sprawl into duplicates.
- If a documented gotcha turns out to be wrong, outdated, or fixed for good
  (e.g. a root cause is structurally eliminated, not just patched), remove or
  correct it rather than leaving stale guidance for the next session.
- Every entry should carry a **why**, not just a **what** — the reasoning is
  what lets a future session judge edge cases the entry didn't anticipate.
- Repeated multi-step procedures (not one-off facts) belong in
  `.claude/skills/`, not here — this file is for standing context and
  gotchas; skills are for runbooks. Cross-reference by name when relevant.

## 1. Module Map

Monorepo with three active modules, plus a bridge layer connecting them:

- **`Consulting Dashboard/`** — Next.js frontend + FastAPI backend. The
  primary UI surface. `backend/modules/` contains thin bridge routers that
  import logic living in the other two modules rather than reimplementing it.
- **`Market Regime/`** — standalone Python analysis engine (no server of its
  own). Computes factor datasets and similarity search. Bridged into the
  dashboard via `Consulting Dashboard/backend/modules/regime/regime_bridge.py`,
  which lazy-imports `similarity_engine.py`, `validation.py`, and
  `factor_schema.py` directly from this directory.
- **`Research_LLM/`** — Paper2Alpha: arXiv MCP server + semantic search over
  downloaded papers (`arxiv_client.py`, `mcp_server.py`). This directory is
  its own nested git repo (has its own `.git`), separate from the top-level
  Trading_Platform_Project repo. Bridged into the dashboard via
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
project — silently resolves to conda's Python, not the project venv.

This caused two separate incidents:
- Conda's `pyarrow` was older than the one that wrote
  `Market Regime/data/factors_weekly.parquet`, so reading it raised
  `OSError: Repetition level histogram size mismatch`, surfacing in the
  browser as a Market Regime page load failure / TypeError.
- Conda's `base` env happened to have `chromadb`/`sentence-transformers`
  installed from unrelated prior work, masking the fact that the project
  `.venv` didn't — until the backend was switched to launch from `.venv`
  correctly, at which point Paper2Alpha search broke with
  `"Phase 4 requires additional packages"`.

**Fix**: never start the backend with a bare `uvicorn`/`python3` command. Use
`Consulting Dashboard/backend/run_backend.sh`, which resolves the venv python
by absolute path (`$DASHBOARD_DIR/../.venv/bin/python3`), immune to shell/conda
state. See the `restart-backend` skill.

### pyarrow version coupling
`pandas`/`pyarrow` in `backend/requirements.txt` must be at least as new as
whatever wrote `Market Regime/data/factors_weekly.parquet`. An older reader
cannot open a parquet file written by a newer pyarrow. If this file ever gets
rewritten with a newer pyarrow, bump the floor in requirements.txt to match.

### Next.js Turbopack `.next` cache corruption
Improper dev-server shutdowns (e.g. killing the parent process without
letting Turbopack clean up) can corrupt the `.next` cache, producing
duplicate `"X 2"` folders and hard-to-diagnose build errors on next start.
Fixed structurally via a `predev: rm -rf .next` npm lifecycle hook in
`Consulting Dashboard/package.json` — it runs automatically before every
`npm run dev`, so this should no longer recur. If it does anyway, that hook
is the first thing to check.

## 3. Dependency Sync Rule (bridge modules)

`Consulting Dashboard/backend/requirements.txt` must be manually kept in sync
with the source-of-truth requirements files of the modules it bridges to:

- Market Regime deps (`pandas`, `numpy`, `scipy`, `pyarrow`) — no separate
  requirements file in `Market Regime/`; versions are pinned directly in
  `backend/requirements.txt` with a comment explaining the pyarrow coupling.
- Research_LLM deps (`mcp`, `requests`, `beautifulsoup4`, `markdownify`,
  `chromadb`, `sentence-transformers`, `pypdf`) — **`Research_LLM/requirements.txt`
  is the source of truth**; `backend/requirements.txt` duplicates this list
  and must be updated by hand whenever the former changes.

**Why this is a rule and not just a note**: the backend imports these
modules' code directly (bridge pattern above) rather than depending on them
as installed packages, so nothing enforces the sync automatically — no
package manager will flag drift. This exact gap caused the Paper2Alpha
regression described above. See the `add-bridge-dependency` skill for the
procedure when adding a new one.

## 4. Recurring Frontend Bug Patterns

- **Controlled number input "stuck leading zero"**: binding a `<input
  type="number">` directly to a numeric state value and coercing on every
  `onChange` forces the field back to `"0"` the instant it's cleared, which
  then sticks in front of whatever's typed next. Fix pattern: keep a raw
  string state for the input, clamp only on blur (or on submit), and derive
  the actual numeric value used for logic separately. See
  `Consulting Dashboard/src/app/regime/page.tsx` (`kInput` / `clampK`) for
  the reference implementation.
- **recharts doesn't reorder on prop change alone**: reordering the data
  array passed to a `Bar`/`Legend` doesn't reliably re-render in the new
  order — recharts needs a forced remount. Fix:
  `<ResponsiveContainer key={sortMode}>` (or whatever prop drives the
  reorder) so React remounts the chart instead of diffing it. Verified by
  direct DOM fill-color-order inspection, not by trusting the legend text.
- **Flexbox children silently shrink to content width**: a flex child
  without `flex-1`/`w-full` inside a `flex-direction: row` parent (e.g. a
  `<main>` layout wrapper) shrinks to its content's width instead of
  stretching to fill available space — this only shows up as a layout bug
  on the *first* render before some other state change happens to trigger a
  reflow that masks it. If a page looks correctly sized after an interaction
  but wrong on first load, check for a missing `flex-1 w-full min-w-0` on
  the page's root element before assuming it's a data/timing issue.
- **Fetch-on-mount races with no retry never self-heal**: a `useEffect(...,
  [])` that fetches once and has no retry logic will stay in an error/stuck
  state forever if the first attempt fails (e.g. backend still cold-starting
  when the frontend mounts) — even after the dependency becomes available,
  nothing re-triggers the effect. Fix pattern: wrap the fetch in a
  cancellable retry loop (bounded attempts, fixed delay) plus a manual retry
  button as a fallback. Tune the attempt budget by measuring actual cold-start
  time, don't guess — an initial 10×1.5s budget measured too short against a
  pandas/numpy/scipy cold import and had to be extended to 30×2s.

## 5. Verification Discipline

Do not declare a UI or API change complete based on code review alone.
Repeatedly in this project, bugs that looked fine on inspection were only
caught by actually running the change and checking real output:
DOM fill-color order, `getBoundingClientRect()` dimensions, live network
responses, actual pip/import checks in the specific environment in question
(not the one assumed to be active). Use the preview tool's console/network/
DOM inspection, or direct curl/python checks against the running process —
never assume a fix worked without observing it.

## 6. Design Precedent

When scoping a new analysis feature, the Market Regime Detector's locked
decision is the reference: the tool retrieves historically similar market
contexts and stops there — it does not attempt to predict or validate
outcomes as a blocking step. Statistical validation is offered as an
**opt-in** dashboard panel the user can invoke, not a gate the core feature
must pass. Default to this shape (retrieve/present first, validate on
demand) for future features unless the user explicitly asks for a
predictive/validating tool instead.

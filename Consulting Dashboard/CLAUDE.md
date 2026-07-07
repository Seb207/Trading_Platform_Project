# Consulting Dashboard — Claude Working Notes

Module-local notes for `Consulting Dashboard/`. Cross-module rules (bridge
pattern, conda/PATH gotcha, dependency sync, verification discipline) live
in the repo-root `../CLAUDE.md` — read that first. This file only holds
facts specific to this module's frontend/backend. Same maintenance protocol
applies: update this file when a session surfaces something new and
generalizable, don't let it go stale.

## What this module is

Next.js frontend + FastAPI backend — the primary UI surface for the
platform. `backend/modules/` contains thin bridge routers
(`regime/regime_bridge.py`, `research/arxiv_bridge.py`) that import logic
living in `Market Regime/` and `Research_LLM/` respectively rather than
reimplementing it. See `../CLAUDE.md` §1 for the full bridge pattern
rationale.

## Always launch the backend via `run_backend.sh`

Never start it with a bare `uvicorn` or `python3` command — see
`../CLAUDE.md` §2 for why (conda PATH hijack) and use the `restart-backend`
skill (`../.claude/skills/restart-backend/`) for the full restart + verify
procedure.

## Next.js Turbopack `.next` cache corruption

Improper dev-server shutdowns (e.g. killing the parent process without
letting Turbopack clean up) can corrupt the `.next` cache, producing
duplicate `"X 2"` folders and hard-to-diagnose build errors on the next
`npm run dev`. Fixed structurally via a `predev: rm -rf .next` npm lifecycle
hook in `package.json` — it runs automatically before every `npm run dev`,
so this should no longer recur. If it does anyway, that hook (and whether it
actually ran) is the first thing to check, not the code you just changed.

## Recurring Frontend Bug Patterns

- **Controlled number input "stuck leading zero"**: binding an
  `<input type="number">` directly to a numeric state value and coercing on
  every `onChange` forces the field back to `"0"` the instant it's cleared,
  which then sticks in front of whatever's typed next. Fix pattern: keep a
  raw string state for the input, clamp only on blur (or on submit), and
  derive the actual numeric value used for logic separately. See
  `src/app/regime/page.tsx` (`kInput` / `clampK`) for the reference
  implementation.
- **recharts doesn't reorder on prop change alone**: reordering the data
  array passed to a `Bar`/`Legend` doesn't reliably re-render in the new
  order — recharts needs a forced remount. Fix:
  `<ResponsiveContainer key={sortMode}>` (or whatever prop drives the
  reorder) so React remounts the chart instead of diffing it. Verified by
  direct DOM fill-color-order inspection, not by trusting the legend text —
  see `src/components/regime/ForwardReturnFanChart.tsx`.
- **Flexbox children silently shrink to content width**: a flex child
  without `flex-1`/`w-full` inside a `flex-direction: row` parent (e.g. the
  page's `<main>` layout wrapper) shrinks to its content's width instead of
  stretching to fill available space — this only shows up as a layout bug
  on the *first* render, before some other state change happens to trigger
  a reflow that masks it. If a page looks correctly sized after an
  interaction but wrong on first load, check for a missing
  `flex-1 w-full min-w-0` on the page's root element before assuming it's a
  data/timing issue. See `src/app/regime/page.tsx`'s root `<div>`.
- **Fetch-on-mount races with no retry never self-heal**: a
  `useEffect(..., [])` that fetches once and has no retry logic will stay in
  an error/stuck state forever if the first attempt fails (e.g. backend
  still cold-starting when the frontend mounts) — even after the dependency
  becomes available, nothing re-triggers the effect. Fix pattern: wrap the
  fetch in a cancellable retry loop (bounded attempts, fixed delay) plus a
  manual retry button as a fallback. Tune the attempt budget by measuring
  actual cold-start time, don't guess — an initial 10×1.5s budget measured
  too short against a pandas/numpy/scipy cold import and had to be extended
  to 30×2s. See `src/app/regime/page.tsx`'s factors-loading effect.

## Verifying changes here

Always use the `verify-ui-change` skill (`../.claude/skills/verify-ui-change/`)
after editing `src/app` or `src/components` — code review alone has missed
every one of the bugs listed above at least once in this project.

## Dependency sync (backend side)

`backend/requirements.txt` is where the Market Regime and Research_LLM
bridge dependencies get pinned on this side of the bridge. See
`../CLAUDE.md` §3 for the rule and the `add-bridge-dependency` skill for the
procedure — this file's job is just to flag that `backend/requirements.txt`
is not self-contained truth, it mirrors two other modules' requirements by
hand.

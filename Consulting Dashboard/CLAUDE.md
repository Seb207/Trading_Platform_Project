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

## Critic loop pattern (Paper2Alpha `/api/chat`)

`backend/routers/chat.py` runs a post-hoc critique pass after the primary
model's draft finishes streaming: `backend/modules/llm/critic.py` sends the
draft + system context to a **fixed, independent OpenRouter model**
(currently `nvidia/nemotron-3-ultra-550b-a55b:free`) for review, and if it
flags issues, the primary model gets one revision pass. The result is
reported as follow-up SSE events (`verifying` → `verified` or `revised`) on
the *same* connection the draft streamed on — no job queue, no polling, no
second request. `ChatPanel.tsx`/`ChatMessage.tsx` render these as a status
badge on the draft, and (if revised) a new assistant message rather than an
overwrite, so the user can see what changed.

**Reuse this pattern** for any future "verify a generated answer" feature
in this codebase instead of inventing a new transport — extend the SSE
event vocabulary on the existing stream rather than adding endpoints/polling.

Key design choices worth preserving:
- **Critique never blocks the primary answer.** A malformed critic response
  or a bad/missing API key fails open to `"pass"` (see
  `critic.py::_parse_verdict` and the try/except around `critique()` in
  `chat.py`) — verified live by sending a deliberately invalid OpenRouter
  key and confirming the flow still completes with a `verified` event
  rather than hanging or erroring.
- **The critic model is fixed, independent of the user's generation
  choice.** The same model grading its own answer tends to rubber-stamp its
  own mistakes.
- **Free-tier caveat**: OpenRouter's free models are rate-limited (~50
  req/day per key as of 2026-07). Every chat turn with a critic key set
  costs at least one extra request (two if revision triggers), so this adds
  up fast under real usage — fine for prototyping, but flag this if the
  feature moves toward production use.
- **`done` must fire before critique, not after.** The frontend's `streaming`
  state (which disables the chat input) only flips off on the `done` event.
  `_event_stream` originally yielded `done` at the very end, after the full
  critique/revision cycle — this silently blocked the user from sending
  another message for the entire verification window, defeating the point
  of running it "in the background." Fixed by yielding `done` immediately
  after the draft finishes; `verifying`/`verified`/`revised` are trailing
  events on the same still-open connection afterward. Any future addition
  to this stream must preserve that ordering — `done` = "you can act again",
  not "everything about this turn is finished".
- **Reasoning-enabled free models are slow for a task like this.** The
  critic model runs with reasoning enabled by default, which burns a lot of
  hidden thinking tokens before its visible JSON verdict — real added
  latency for a compliance/grounding check that doesn't need deep
  multi-step reasoning. `OpenRouterProvider` now takes an optional
  `reasoning_effort` param (maps to OpenRouter's unified `reasoning.effort`);
  the critic call sets it to `"low"`. Left unset for normal chat generation
  so a user's chosen model keeps its own default behavior.

## Chat state must live above the page, not inside it

`ChatPanel.tsx` is only mounted while the user is on `/research` — the App
Router fully unmounts it when navigating to any other route (`/regime`,
`/portfolio`, etc.), since each route's page component renders inside
`{children}` in the root layout. If `messages`/`streaming` are local
`useState` in `ChatPanel`, navigating away mid-conversation (even mid-stream)
destroys them — the in-flight generation continues on the server, but
nothing is listening for the result anymore by the time it arrives.

Fixed by lifting `messages`, `streaming`, and the send/stream logic into
`src/context/ChatContext.tsx` (`ChatProvider`/`useChat()`), mounted once in
`src/app/layout.tsx` alongside the existing `LLMProvider` — same pattern,
same reason. `ChatPanel` now just reads from `useChat()`; it can unmount and
remount freely (navigating away and back) without losing the conversation
or an in-progress answer. Also switched `ChatPanel`'s local `config` state
to the already-existing `useLLM()` context for the same reason — it existed
specifically for this but `ChatPanel` wasn't using it, so the model/provider
selection was silently reset on every remount too.

**Rule of thumb**: any state that represents "work in progress" (not just
UI-only state like an accordion's open/closed sections) belongs in a
context mounted at the layout level if the component holding it can be
unmounted by route navigation — not in the component's own `useState`.

Verified by sending a message on Ollama, navigating to `/regime` mid-stream,
and navigating back to `/research` — the full question + completed answer
were still there, input was correctly re-enabled once generation finished.

## Client disconnect (refresh/tab close) safely cancels the LLM call

Reviewed whether a hard interrupt (page refresh, tab close) leaves the
backend generating a response for a client that's no longer there — wasting
LLM API calls/tokens on an abandoned request. Verified empirically: a fake
slow-streaming provider monkeypatched into a throwaway instance of the real
app, hit with a real client that disconnects mid-stream, showed the
generator receiving `asyncio.CancelledError` and its `finally` block running
immediately, with zero further chunks generated after disconnect. This is
Starlette's built-in `StreamingResponse` behavior (a background task listens
for the ASGI `http.disconnect` message and cancels the task group running
the generator) — confirmed present in this project's installed versions
(fastapi 0.135.2 / starlette 1.0.0). None of our own `except Exception:`
blocks in `chat.py`/`critic.py`/`openrouter_provider.py` catch
`CancelledError` (it isn't an `Exception` subclass), so cancellation
propagates cleanly through the critique/revision path too. **No code change
was needed here** — this was a verification, not a fix. If this framework
behavior is ever relied on elsewhere, don't add a broad `except Exception`
around an entire streaming generator without re-checking this.

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

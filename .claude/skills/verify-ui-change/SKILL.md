---
name: verify-ui-change
description: Verification routine for any Consulting Dashboard frontend change — checks console/network errors, DOM state, and visual output directly instead of declaring a fix complete from code review alone. Use after every edit to src/app or src/components before reporting the task done.
---

# Verify UI Change

## Why this exists

Multiple bugs in this project (recharts not reordering, a page rendering
squeezed on first load, a stuck-at-0 number input) looked correct on code
review but were only caught by actually running the change and inspecting
real output. Code review verifies the change is *plausible*; only running it
verifies it's *correct*. See `CLAUDE.md` §4–5 for the specific bug patterns
this caught.

## Steps

1. **Ensure a dev server is running** — `preview_start` if not already up.
2. **Reload if needed** — `preview_eval` with `window.location.reload()`.
   Skip this if Next.js HMR is active and the change should hot-reload.
3. **Check for errors first**:
   - `preview_console_logs` (browser console)
   - `preview_logs` (Next.js dev server stdout/stderr — catches build errors
     HMR silently swallows)
   - `preview_network` filtered to `failed` (catches 4xx/5xx from the
     backend the UI depends on)
4. **Check actual structure and content** — `preview_snapshot`, not just a
   screenshot. Screenshots are for visual layout only; don't use them to
   verify text content or precise values.
5. **Check precise CSS values** — `preview_inspect` with the specific
   properties in question (color, padding, width) rather than eyeballing a
   screenshot. This is what caught the `flex-1` layout bug — a screenshot at
   a glance looked "close enough."
6. **Exercise the actual interaction being changed** — `preview_click` /
   `preview_fill`, then re-snapshot to confirm the resulting state, not just
   that the click didn't error.
7. **For layout/responsive/dark-mode changes** — `preview_resize` at
   relevant breakpoints (mobile/tablet/desktop) and re-inspect.
8. **If anything looks wrong**, read the source to diagnose, edit, and
   re-run from step 3 — don't re-guess from the diff alone.
9. **Once verified, capture proof** — `preview_screenshot` for visual
   changes, `preview_network` for API-driven changes, `preview_logs` for
   server-rendered output — before reporting the task complete.

## Rule of thumb

If you're about to write "this should now work" without having done steps
3–6, stop and do them first. This project's history shows that gap is where
real bugs hide.

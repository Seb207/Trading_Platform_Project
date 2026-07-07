# Harness Workflow

A structured methodology for planning and building a feature in this
project. Five phases: Explore → Discuss → Step Design → File Creation →
Execute (semi-automated — you drive each step interactively, `execute.py`
just does the bookkeeping).

## Phase A — Exploration

Read `/CLAUDE.md` (root + whichever module's `CLAUDE.md` this feature
touches), `/docs/PRD.md`, `/docs/ARCHITECTURE.md`, `/docs/ADR.md`. Check
`/.claude/skills/` for an existing runbook before assuming a step needs to
be designed from scratch — if a skill already covers part of this
(restarting the backend, verifying a UI change, adding a bridge
dependency, etc.), reference it instead of re-deriving the procedure.

## Phase B — Discussion

Before writing any step files, surface to the user:
- Any technical decision with more than one reasonable approach.
- Any ambiguity in the request that changes scope.
- Whether this warrants a new ADR entry (a real architectural tradeoff, not
  a routine implementation detail).

Do not proceed to Phase C until these are resolved.

## Phase C — Step Design

Break the feature into steps. Each step is one self-contained module of
work. Principles:

1. **One module per step** — minimize scope; a step should be reviewable
   and revertible on its own.
2. **Self-contained** — a step's `.md` file must contain everything needed
   to execute it without re-reading the whole conversation: what to build,
   which files it touches, why (one line), and its Acceptance Criteria.
3. **Interface-level, not implementation-level** — specify what a function/
   endpoint/component must do and its signature, not the exact code. Leave
   implementation judgment to execution time.
4. **Executable Acceptance Criteria** — a bash command (or a described
   browser-preview check) that proves the step works, not a vague
   description like "should work correctly."
5. **Concrete warnings, not abstract ones** — "don't forget X" is useless;
   "X breaks because Y — see CLAUDE.md §Z" is useful. Pull from `CLAUDE.md`/
   `ADR.md` where relevant instead of restating generic caution.
6. **Respect the credentials/local-service rule** — if a step could require
   an API key, credential, or a local service (a local LLM, a database
   only running on the user's machine), the step must say so explicitly and
   instruct: ask the user for it, never fabricate a placeholder, never
   autonomously start a local service to route around it. If it turns out
   to be needed mid-step, stop and mark the step `blocked` (see Phase E)
   rather than substituting something.
7. **Order by dependency, not by convenience** — a step should never assume
   output from a later step.

## Phase D — File Creation

Create:
- `phases/index.json` (top-level, if not present) — list of phases with
  `dir`, `status`.
- `phases/<phase-dir>/index.json` — this phase's step list: `step`,
  `name`, `status` (`pending`/`completed`/`error`/`blocked`), timestamps as
  they occur.
- `phases/<phase-dir>/step<N>.md` — one file per step, per the Phase C
  principles above.

Use `python3 scripts/execute.py new <phase-dir> <step-name-1> <step-name-2> ...`
to scaffold these from a list of step names, then fill in each `step<N>.md`.

## Phase E — Execution (semi-automated)

This project runs steps **interactively, one at a time, with your
confirmation** — not as an unattended headless loop. For each step:

1. `python3 scripts/execute.py start <phase-dir>` — checks out the
   `feat-<phase-dir>` branch (once, at the start of the phase).
2. `python3 scripts/execute.py next <phase-dir>` — prints the next pending
   step's file plus accumulated guardrails (root + module `CLAUDE.md`
   files) and a one-line summary of every already-completed step in this
   phase.
3. Implement the step in this conversation. Run its Acceptance Criteria
   yourself — don't declare it done without observing the actual output
   (see `CLAUDE.md`'s Verification Discipline section).
4. If a credential/local-service is needed and wasn't already provided,
   stop here and ask the user — do not improvise around it.
5. Once AC passes: `python3 scripts/execute.py done <phase-dir> <step-num> "<one-line summary>"`
   — marks the step completed and commits it
   (`feat(<phase>): step <N> — <name>`).
6. If it fails after your own reasonable attempts to fix it:
   `python3 scripts/execute.py fail <phase-dir> <step-num> "<error>"`.
7. If blocked on something only the user can resolve:
   `python3 scripts/execute.py block <phase-dir> <step-num> "<reason>"`,
   then stop and tell the user what's needed.
8. Repeat from step 2 until `next` reports the phase complete.

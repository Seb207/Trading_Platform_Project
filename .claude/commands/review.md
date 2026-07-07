Review the changes on the current branch (or the working diff if there's no
branch yet) against this project's own standards.

First read:
- `/CLAUDE.md` (root, plus whichever module-specific `CLAUDE.md` the
  changed files fall under)
- `/docs/ARCHITECTURE.md`
- `/docs/ADR.md`

Then check the changed files against this checklist:

## Checklist

1. **Architecture compliance** — does it follow the directory structure and
   bridge pattern in `ARCHITECTURE.md`? Did it reimplement logic that
   already exists in `Market Regime/` or `Research_LLM/` instead of
   bridging to it?
2. **ADR compliance** — does it contradict a recorded decision in `ADR.md`
   without a new ADR entry explaining why the tradeoff changed?
3. **Tests exist** — is there a test (or, for this project, a documented
   Acceptance Criteria check) for new functionality?
4. **CRITICAL rules** — does it violate any rule marked CRITICAL in
   `CLAUDE.md` (e.g. launching the backend without `run_backend.sh`,
   fabricating a credential instead of asking, autonomously starting a
   local LLM/service)?
5. **Verification discipline** — was this actually run and observed
   (preview tools for frontend, curl/direct checks for backend), or is
   "should work" being asserted from code review alone? See `CLAUDE.md`'s
   Verification Discipline section — this project has a specific history
   of bugs that passed code review but failed when actually run.
6. **Builds/checks pass** — does the relevant Stop-hook check
   (`scripts/stop_check.sh` — npm lint/build for Consulting Dashboard,
   Python syntax check for Market Regime/Research_LLM) pass?

## Output format

| Item | Result | Notes |
|---|---|---|
| Architecture compliance | ✅/❌ | {detail} |
| ADR compliance | ✅/❌ | {detail} |
| Tests / AC exist | ✅/❌ | {detail} |
| CRITICAL rules | ✅/❌ | {detail} |
| Verification discipline | ✅/❌ | {detail} |
| Builds/checks pass | ✅/❌ | {detail} |

For every ❌, give a specific, actionable fix — not a restatement of the
problem.

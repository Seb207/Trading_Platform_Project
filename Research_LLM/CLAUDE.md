# Research_LLM (Paper2Alpha) — Claude Working Notes

## This is a nested, separate git repo

`Research_LLM/` has its own `.git`, nested inside the top-level
`Trading_Platform_Project` repo (which is NOT its parent from git's point of
view). If Claude Code is opened with a working directory inside this repo
(as this file's own session may well be — see the `.claude/worktrees/`
structure here), CLAUDE.md discovery may stop at this repo's boundary and
never reach `../CLAUDE.md`. For that reason, the facts below that also
appear in the parent file are **repeated here in full**, not just
cross-referenced — don't assume the parent file's context is loaded.

Same maintenance protocol as the parent file: this is a living document,
update it when a session learns something new and generalizable.

## What this module is

A local MCP server that bridges LLMs and the arXiv academic database
("Paper2Alpha"). Lets an AI assistant autonomously search, download, and
deeply analyze research papers, then generate grounded quant strategy code
from them. Core files: `arxiv_client.py` (all tool logic), `mcp_server.py`
(MCP tool registration). Bridged into the dashboard via
`../Consulting Dashboard/backend/modules/research/arxiv_bridge.py` — that
backend imports this module's code directly rather than depending on it as
an installed package (same bridge pattern as `Market Regime/`).

**Deliberately not RAG-with-chunking** for the core reading workflow: papers
are downloaded as clean Markdown and passed whole into the LLM's context
window, because quant papers have tightly coupled sections
(methodology ↔ results ↔ math) that chunking breaks, and the target workflow
is deep reading of a small curated set, not broad retrieval across
thousands. Semantic search (ChromaDB + sentence-transformers, Phase 4) is
a separate two-tier layer on top for topic/section discovery, not a
replacement for full-text reading. Full tool inventory and phase breakdown
is in `README.md` in this directory — this file is for operational notes
only.

## Dependency sync with the Dashboard bridge (source-of-truth direction)

**`requirements.txt` in this directory is the source of truth** for what
`../Consulting Dashboard/backend/requirements.txt` must mirror for the
Research_LLM bridge section. If you add/change a dependency here
(`mcp`, `requests`, `beautifulsoup4`, `markdownify`, `chromadb`,
`sentence-transformers`, `pypdf`), you must also update
`../Consulting Dashboard/backend/requirements.txt` by hand — nothing
enforces this automatically, since the backend imports this module's code
directly rather than installing it as a package. See the
`add-bridge-dependency` skill at
`../.claude/skills/add-bridge-dependency/SKILL.md` (may not be visible if
this repo is opened standalone — if so, do it manually: update both files,
`pip install` into the Dashboard's `.venv`, restart the backend, verify
`/api/papers/search/*` still returns 200).

**This exact gap already caused a regression once**: a Dashboard-side fix
that made the backend always launch from its own `.venv` (instead of
whatever Python happened to be on PATH) revealed that `.venv` had never
actually been provisioned with `chromadb`/`sentence-transformers` — the
previous launch method had been masking the gap by accidentally running
under a different environment that happened to have them.

## Environment note (conda/PATH)

If running `mcp_server.py` or any script here standalone and hitting an
unexpected `ImportError` or version mismatch, check `sys.executable` before
assuming it's a code bug — this machine's `~/.zshrc` auto-activates
anaconda's `base` env ahead of any project `.venv` on PATH, so a bare
`python3` command can silently run under the wrong interpreter.

## CRITICAL — Credentials & Local Services — Never Substitute, Always Ask

When a task needs confidential information (API keys, credentials, tokens)
— or would require starting/driving a local service (e.g. a local LLM
runtime) — **stop and ask the user first.** Do not fabricate a placeholder
key to route around a missing credential, and do not autonomously spin up
a local service as a workaround. (Duplicated here in full per this file's
own self-sufficiency policy above — see the parent `CLAUDE.md` §6 for the
incident this rule came from.)

## Harness Workflow

For a feature large enough to warrant an explicit plan, use `/harness`
(`.claude/commands/harness.md`) — reads `docs/PRD.md`, `docs/ARCHITECTURE.md`,
`docs/ADR.md`, breaks the feature into self-contained steps under
`phases/<phase>/`, and drives them **semi-automated** (one step at a time,
in conversation; `python3 scripts/execute.py` handles status tracking and
git bookkeeping — not an unattended headless loop). Use `/review`
(`.claude/commands/review.md`) before considering a branch done. This is a
parallel, independent copy of the same structure set up in the parent
`Trading_Platform_Project` repo — kept separate because this directory is
its own git repo.

## arXiv API quirks

- **Two hosts, throttle independently**: content (`arxiv.org/html`,
  `arxiv.org/pdf`) is rarely throttled; metadata/search
  (`export.arxiv.org/api/query`) throttles aggressively under load.
  `download_paper(fetch_metadata=False)` during bulk downloads avoids a
  per-paper metadata call; `bulk_download_papers` uses this internally plus
  one batched `backfill_metadata` pass (50 IDs/call) afterward.
- **Metadata source of truth is the arXiv API, not local parsing.** Local
  `.md`/`.pdf` parsing is a stopgap for when the API is rate-limited —
  heuristic, title/abstract only, no `published` date. Entries with an
  empty `published` field were extracted locally and should be upgraded via
  `scripts/refresh_api_metadata.py` once the API is available again, then
  the abstract index rebuilt so canonical abstracts replace heuristic ones.

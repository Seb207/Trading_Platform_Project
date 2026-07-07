# Architecture Decision Records

## Philosophy

Ship a working local tool for one user, fast. Prefer the simplest thing
that solves the actual problem over the theoretically more correct one —
retrieval over prediction, a fixed rubric over a fully general one, a
one-line env var over a config system. Every decision below traded away
something to get there; the tradeoff is recorded on purpose so a later
session doesn't "fix" it back into needless complexity.

---

### ADR-001: Bridge pattern instead of merging modules

**Decision**: `Market Regime/` and `Research_LLM/` stay standalone Python
modules with no HTTP awareness; `Consulting Dashboard/backend/modules/`
holds thin bridge routers that import their code directly.

**Reason**: both modules are independently useful outside the dashboard
(scripts, notebooks, a future CLI). Merging them into the backend would
couple their logic to FastAPI for no benefit.

**Tradeoff**: dependency lists must be manually kept in sync between the
source module and `backend/requirements.txt` — nothing enforces this
automatically, and it has already caused one real regression (see
`CLAUDE.md` §3).

---

### ADR-002: Hybrid z-score + correlation-grouping similarity, not PCA

**Decision**: Market Regime's analog search z-scores each factor, groups
highly-correlated factors (|r| ≥ 0.7) into a single vector dimension via
connected components, and computes distance on the resulting vector — not
a PCA-reduced space.

**Reason**: PCA components are not directly interpretable to a user asking
"why is this analog similar" — the per-factor contribution breakdown
(§7.1 colormap) requires attributing distance back to named, real factors.
Correlation grouping still collapses redundant factors without losing that
interpretability.

**Tradeoff**: more manual than a general-purpose dimensionality reduction;
the correlation threshold (0.7) is a judgment call re-evaluated whenever
factors are added, not something a fixed algorithm decides.

---

### ADR-003: Market Regime is retrieval, not prediction

**Decision**: the tool retrieves historically similar market contexts and
stops there. Statistical validation of "did the analogs' subsequent
returns actually predict anything" is an opt-in dashboard panel, not a
gate the core feature must pass.

**Reason**: the stated purpose is pulling up comparable historical context
in one shot; interpretation is the user's call. Baking in a mandatory
predictive-validity gate would overstate what a similarity search can
actually claim.

**Tradeoff**: a user who skips the validation panel could over-trust a
coincidental analog. Accepted — the tool's job is to surface it, not to
adjudicate it.

---

### ADR-004: Full-text reading over RAG chunking for Paper2Alpha

**Decision**: papers are downloaded as clean Markdown and passed whole into
the LLM's context window for deep reading, instead of chunking into a
vector DB as the primary retrieval mechanism.

**Reason**: quant papers have tightly coupled sections (methodology ↔
results ↔ math) that chunking breaks apart. The target workflow is deep
reading of a small curated set, not broad retrieval across thousands.
Modern context windows (200k+ tokens) comfortably hold several full papers.

**Tradeoff**: doesn't scale to "search across 10,000 papers" as cheaply as
pure embedding retrieval would. Mitigated by a separate two-tier semantic
layer (ChromaDB abstract + section embeddings) for topic/section discovery
on top of, not instead of, full-text reading.

---

### ADR-005: Fixed, independent critic model for Paper2Alpha's review loop

**Decision**: the critic pass on chat answers always runs on one fixed free
OpenRouter model (`nvidia/nemotron-3-ultra-550b-a55b:free`), independent of
whatever model the user picked for generation, with reasoning effort set
low.

**Reason**: the same model grading its own answer tends to rubber-stamp
its own mistakes. A fixed, different model catches more. Free tier keeps
this a zero-marginal-cost add-on; low reasoning effort avoids burning
hidden thinking tokens on a task (compliance/grounding check) that doesn't
need deep multi-step reasoning.

**Tradeoff**: free-tier rate limits (~50 req/day per key) and occasional
upstream 429s (mitigated with a short retry + friendly error message). Not
suitable if usage scales past prototype/solo levels without a paid key.

---

### ADR-006: Critique reported as trailing SSE events, not a second request

**Decision**: after the draft streams, `verifying`/`verified`/`revised`
events continue on the *same* SSE connection rather than a separate
polling endpoint or job queue.

**Reason**: no new infrastructure (no job storage, no polling) — the
connection is already open and the frontend's reader loop only exits when
the underlying stream actually closes, not on any particular app-level
event.

**Tradeoff**: `done` must fire immediately after the draft (not after
critique) or the frontend's input stays blocked for the whole critique
window — this ordering is load-bearing and easy to regress; see
`CLAUDE.md`'s Consulting Dashboard section for the incident this caused.

---

### ADR-007: Per-module CLAUDE.md instead of one monolithic file

**Decision**: cross-module rules live in the root `CLAUDE.md`; each of
`Market Regime/`, `Consulting Dashboard/`, `Research_LLM/` has its own file
for module-specific gotchas, with `.claude/skills/` for repeated
procedures.

**Reason**: `Research_LLM/` is a separate git repo — a session opened
directly inside it may not auto-load the root file, so it needs to be
self-sufficient. Splitting also keeps each file scoped to what's actually
in view when working in that module.

**Tradeoff**: facts can drift out of sync across files if not updated
together; mitigated by an explicit maintenance protocol at the top of the
root file instructing every session to keep all of them current.

---

### ADR-008: Backend must always launch via an absolute-venv-path script

**Decision**: never start the FastAPI backend with a bare `uvicorn`/
`python3` command — always through `run_backend.sh` (or an equivalent
absolute-path launcher), never relying on shell `$PATH`.

**Reason**: this machine's `~/.zshrc` runs `conda init`, which puts
anaconda's Python ahead of the project's `.venv` on `$PATH` in every new
shell. This has caused two distinct production bugs (pyarrow version
mismatch, missing packages) and recurred a third time via an independent
launch point (a hand-built macOS app) that hadn't been fixed the same way.

**Tradeoff**: every new way of starting the backend (scripts, launcher
apps, CI) has to independently apply this fix — there is no single
enforcement point. `CLAUDE.md` §2 keeps a running list of known launch
points that need it.

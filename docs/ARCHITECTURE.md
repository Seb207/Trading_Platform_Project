# Architecture

See `CLAUDE.md` for environment gotchas, skills, and per-module notes — this
file is the structural map only.

## Directory Structure

```
Trading_Platform_Project/
├── Consulting Dashboard/     # Next.js frontend + FastAPI backend — the UI
│   ├── src/app/              # routes: /research, /regime, /portfolio, /factor
│   ├── src/components/       # UI components, grouped by feature
│   ├── src/context/          # cross-route persistent state (ChatContext, LLMContext)
│   ├── backend/routers/      # FastAPI route handlers
│   ├── backend/modules/      # bridge routers + LLM provider abstraction
│   └── backend/prompts/      # task-mode + critic system prompts (plain .md/.py, hot-reloaded)
├── Market Regime/             # standalone Python analysis engine, no server
│   ├── factor_schema.py       # FactorSpec registry
│   ├── similarity_engine.py   # z-score + correlation-grouping similarity search
│   ├── validation.py          # opt-in statistical validation
│   └── sources/               # one client per data source (FRED, Refinitiv, GPR, OCC, CBOE)
├── Research_LLM/               # separate git repo — Paper2Alpha MCP server
│   ├── arxiv_client.py         # all tool logic (search/download/analyze/embed)
│   └── mcp_server.py           # MCP tool registration
├── docs/                       # this framework's PRD/ARCHITECTURE/ADR/UI_GUIDE
├── phases/                     # harness workflow: multi-step feature plans
└── scripts/execute.py          # harness phase/step bookkeeping tool
```

## Pattern: Bridge, Don't Reimplement

`Market Regime/` and `Research_LLM/` are standalone Python modules with no
knowledge of being served over HTTP. `Consulting Dashboard/backend/modules/`
contains thin bridge routers (`regime/regime_bridge.py`,
`research/arxiv_bridge.py`) that import their code directly and wrap it in
FastAPI request/response schemas. When a bug is in data/logic, look in the
source module; when it's HTTP/serialization, look in the bridge.

Consequence: nothing enforces dependency sync between
`Consulting Dashboard/backend/requirements.txt` and the source modules'
actual needs — see `CLAUDE.md` §3 for the rule this creates.

## Data Flow

**Paper2Alpha chat**: user message → `/api/chat` → system prompt built from
persona + optional task-mode prompt + optional paper text → primary LLM
streams draft → (if critic key present) fixed independent OpenRouter model
critiques draft against the same context → optional one revision pass →
all reported as SSE events on the same connection (`chunk` → `done` →
`verifying` → `verified`/`revised`). `done` fires right after the draft, not
after critique — critique must never block the UI.

**Market Regime analog search**: as-of date + factor selection →
`similarity_engine.find_analogs()` reads the pre-built
`data/factors_weekly.parquet` → expanding z-score + correlation-grouped
distance → top-k analogs with per-factor contribution breakdown → dashboard
renders colormap / event-time overlay / forward-return fan chart.

## State Management

- **Route-local UI state** (accordion open/closed, active tab, form inputs):
  plain component `useState`. Fine to lose on navigation.
- **Cross-route state that represents in-progress or completed work**
  (chat messages, streaming status, LLM provider/model selection): lives in
  a Context mounted once at `src/app/layout.tsx` (`ChatContext`,
  `LLMContext`), not in the page/component that happens to render it — page
  components unmount on navigation, the context provider doesn't. See
  `CLAUDE.md`'s Consulting Dashboard section for the incident this rule
  came from.
- **Backend**: no server-side session state between requests — each
  `/api/chat` call rebuilds its system prompt fresh from the request body.

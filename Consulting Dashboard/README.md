# Quant Research Dashboard

A modular, local-first dashboard for quant research and portfolio management.
Built with **Next.js 14 + TypeScript + Tailwind CSS** (frontend) and **FastAPI** (backend).
Runs entirely on your machine, deployable to the cloud when ready.

---

## Design Principles

1. **Modular by default** — each feature is a self-contained module. Adding a new feature means adding a new page under `src/app/` and a new router under `backend/routers/`. Nothing else needs to change.
2. **LLM-agnostic** — all LLM calls go through a unified Python provider interface. Switch between Claude, Ollama, or any future model from the dashboard UI without touching feature code.
3. **Local-first, cloud-ready** — SQLite locally, one connection string swap to PostgreSQL for production. Next.js deploys to Vercel/Railway, FastAPI deploys separately.
4. **Reusable UI components** — common React components (ModelSelector, PaperTable, ChatMessage, CategoryTag) live in `src/components/` and are shared across pages.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Frontend | Next.js 14 (App Router), TypeScript, Tailwind CSS |
| Backend | FastAPI (Python), Uvicorn |
| LLM | Anthropic SDK (Claude) / Ollama REST API |
| Vector DB | ChromaDB (via existing Research_LLM) |
| Database | SQLite (local) → PostgreSQL (production) |
| Fonts | JetBrains Mono, Inter (Google Fonts) |

---

## Project Structure

```
Consulting Dashboard/
│
├── src/                              # Next.js frontend
│   ├── app/
│   │   ├── layout.tsx                # Root layout (TopBar + CategoryNav)
│   │   ├── globals.css               # Dark theme, font imports
│   │   ├── page.tsx                  # Redirect → /research
│   │   ├── research/page.tsx         # [Phase 1] Research LLM
│   │   ├── regime/page.tsx           # [Future] Market Regime
│   │   ├── portfolio/page.tsx        # [Future] Portfolio tracking
│   │   └── factor/page.tsx           # [Future] Factor Research
│   │
│   ├── components/
│   │   ├── layout/
│   │   │   ├── TopBar.tsx            # Logo + LLM status indicator
│   │   │   ├── CategoryNav.tsx       # Feature category tab navigation
│   │   │   └── SplitPanel.tsx        # Left content + Right chat panel
│   │   │
│   │   ├── ui/                       # Reusable design system components
│   │   │   ├── Badge.tsx
│   │   │   ├── FilterChip.tsx
│   │   │   ├── ToggleGroup.tsx
│   │   │   └── CategoryTag.tsx
│   │   │
│   │   ├── research/                 # Research LLM feature components
│   │   │   ├── PaperTable.tsx
│   │   │   ├── SearchBar.tsx
│   │   │   └── PaperViewer.tsx
│   │   │
│   │   └── chat/                     # Chat panel components
│   │       ├── ChatPanel.tsx
│   │       ├── ChatMessage.tsx
│   │       ├── ChatInput.tsx
│   │       └── ModelSelector.tsx
│   │
│   ├── lib/
│   │   ├── api.ts                    # Backend API client (fetch wrappers)
│   │   └── types.ts                  # Shared TypeScript types
│   │
│   └── store/
│       └── chat.ts                   # Chat state (Zustand)
│
├── backend/                          # FastAPI backend
│   ├── main.py                       # FastAPI app entry point + CORS
│   ├── routers/
│   │   ├── papers.py                 # /api/papers/* endpoints
│   │   ├── chat.py                   # /api/chat endpoint (LLM streaming)
│   │   └── status.py                 # /api/status (index stats)
│   │
│   ├── modules/
│   │   ├── llm/
│   │   │   ├── base.py               # Abstract LLMProvider interface
│   │   │   ├── claude_provider.py    # Anthropic SDK + streaming
│   │   │   ├── ollama_provider.py    # Ollama local REST API
│   │   │   └── factory.py            # Provider factory
│   │   │
│   │   └── research/
│   │       └── arxiv_bridge.py       # Bridge to Research_LLM/arxiv_client.py
│   │
│   ├── database/
│   │   ├── models.py                 # SQLAlchemy ORM models
│   │   ├── db.py                     # Session factory
│   │   └── migrations/               # Alembic migration scripts
│   │
│   └── requirements.txt
│
├── mockups/                          # Design mockups (HTML, reference only)
│
├── package.json
├── tailwind.config.ts                # Design A color tokens + fonts
├── tsconfig.json
└── README.md
```

---

## LLM Provider System

All LLM interactions go through a unified Python interface in `backend/modules/llm/base.py`.

### Architecture

```
Frontend (ModelSelector component)
        ↓  POST /api/chat (SSE streaming)
  FastAPI backend
        ↓
  Provider Factory (factory.py)
        ↓
  LLMProvider (base.py interface)
    ├── ClaudeProvider   → Anthropic SDK (streaming)
    ├── OllamaProvider   → localhost:11434 REST API
    └── [Future]         → OpenAI, Gemini, etc.
```

### Supported Providers

| Provider | Models | Requirements |
|---|---|---|
| **Claude (Anthropic)** | claude-opus-4-5, claude-sonnet-4-5, claude-haiku-4-5 | API key |
| **Ollama (local)** | gemma3:27b, llama3.3:70b, qwen2.5:14b, mistral, etc. | Ollama running locally |

### Adding a new provider

1. Create `backend/modules/llm/openai_provider.py` implementing `LLMProvider`
2. Register it in `backend/modules/llm/factory.py`

No frontend changes needed.

### ModelSelector UI (in right chat panel)

```
┌──────────────────────────┐
│  MODEL                   │
│  [Claude ▼]  [Ollama ▼]  │
│                          │
│  claude-opus-4-5 ▾       │
│  API Key: [sk-ant-••••]  │
└──────────────────────────┘
```

Settings saved to browser `sessionStorage`. API key never sent to any external service except the chosen provider.

---

## Feature Pages

### Research LLM (Phase 1 — Active)

Connects to `Research_LLM/arxiv_client.py` via `backend/modules/research/arxiv_bridge.py`.

**Capabilities:**
- Search local paper library (abstract-level + section-level ChromaDB)
- Search arXiv directly for new papers
- Download papers (HTML → Markdown)
- LLM-powered Q&A and strategy code generation grounded in paper content
- Streaming responses via SSE

**Layout:** Left panel (paper table + search) / Right panel (LLM chat)

---

### Market Regime (Planned)

PCA + clustering on macro/price features. Historical regime comparison.

---

### Portfolio Tracking (Planned)

P&L, drawdown, position-level exposure, factor attribution.

---

### Factor Research (Planned)

Factor score calculation, backtest result visualization, signal decay analysis.

---

## Adding a New Feature

```
Frontend:
1. Create src/app/[feature]/page.tsx
2. Add feature components under src/components/[feature]/
3. Add tab entry in src/components/layout/CategoryNav.tsx

Backend:
4. Create backend/routers/[feature].py
5. Create backend/modules/[feature]/core.py
6. Register router in backend/main.py

Database (if needed):
7. Add tables to backend/database/models.py
8. Create Alembic migration
```

---

## Database Schema (SQLite → PostgreSQL)

| Table | Purpose |
|---|---|
| `llm_sessions` | LLM conversation history per feature |
| `paper_annotations` | User notes and tags on local papers |
| `strategy_ideas` | Saved LLM-generated strategy code + source paper |
| `portfolio_snapshots` | Daily portfolio positions and P&L |
| `factor_scores` | Computed factor scores per asset per date |
| `regime_history` | Detected regime labels per date |

Connection string in `backend/database/db.py`:
- **Local:** `sqlite:///./data/dashboard.db`
- **Production:** `postgresql://...` (via `DATABASE_URL` env variable)

---

## Setup

### 1. Frontend

```bash
cd "Trading_Platform_Project/Consulting Dashboard"
npm install
npm run dev
# → http://localhost:3000
```

### 2. Backend

```bash
cd "Trading_Platform_Project/Consulting Dashboard/backend"
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
# → http://localhost:8000
```

### 3. Configure LLM

Open dashboard → right chat panel → click model selector → choose Claude or Ollama → enter API key.

---

## Deployment (When Ready)

```bash
# Frontend → Vercel
vercel deploy

# Backend → Railway
railway up  # from backend/ directory

# Or both on Railway (monorepo config)
```

Swap `DATABASE_URL` env var from SQLite to PostgreSQL. All feature code unchanged.

---

## Design System (Design A — Terminal Dark)

| Token | Value | Usage |
|---|---|---|
| `bg` | `#000000` | Page background |
| `bg2` | `#0a0a0a` | Panel background |
| `bg3` | `#111111` | Input / hover background |
| `border` | `#1f1f1f` | All borders |
| `text` | `#e8e8e8` | Primary text |
| `text-dim` | `#555555` | Labels, metadata |
| `accent` | `#00ff88` | Active states, positive values |
| `accent2` | `#00cfff` | arXiv IDs, secondary accent |
| `accent3` | `#ff4fa3` | Category tags (pink) |
| `accent4` | `#ffd700` | Category tags (gold) |
| `neg` | `#ff3b3b` | Negative values |

Fonts: `JetBrains Mono` (monospace, IDs/values) + `Inter` (body text)

---

## Implementation Roadmap

### Phase 1 — Project foundation ✅ (current)
- [x] README updated
- [ ] Next.js 14 + TypeScript + Tailwind setup
- [ ] Design A tokens in `tailwind.config.ts`
- [ ] Global dark theme CSS

### Phase 2 — Layout & navigation
- [ ] TopBar, CategoryNav, SplitPanel components
- [ ] 4 page stubs with routing

### Phase 3 — UI component library
- [ ] Badge, FilterChip, ToggleGroup, CategoryTag
- [ ] PaperTable with mock data

### Phase 4 — LLM chat panel
- [ ] ChatPanel, ChatMessage, ChatInput
- [ ] ModelSelector (Claude / Ollama)

### Phase 5 — FastAPI backend
- [ ] Paper list / search / analyze endpoints
- [ ] Status endpoint (index stats)

### Phase 6 — LLM integration
- [ ] ClaudeProvider + OllamaProvider
- [ ] Streaming chat endpoint (SSE)

### Phase 7 — Full connection & polish
- [ ] Frontend ↔ Backend fully wired
- [ ] Loading states, error handling
- [ ] Strategy code viewer (syntax highlighting)

---

## Paper Library Scaling Strategy

As the local arXiv paper library grows, only the retrieval pipeline changes. Dashboard UI and API contracts remain stable.

### Stage 1 — < 100 papers
Abstract RAG → full-paper LLM read. Default behavior, no changes needed.

### Stage 2 — 100–1,000 papers
Two-tier: abstract search → section search → targeted read.
- Run `build_section_index` from dashboard
- Update `arxiv_bridge.py` to chain retrieval steps

### Stage 3 — 1,000–5,000 papers
Hybrid search (BM25 + ChromaDB) + CrossEncoder reranker.
- Add `rank_bm25` + `cross-encoder` to backend requirements
- Add search mode toggle in Research LLM page UI

### Stage 4 — 5,000+ papers
Query expansion + dedicated vector DB (Qdrant/Weaviate).
- Migrate ChromaDB → Qdrant
- Add `MultiQueryRetriever` for automatic query expansion

---

## Related Projects

| Project | Path | Role |
|---|---|---|
| Research_LLM | `../Research_LLM/` | arXiv paper library + ChromaDB indexes |
| Quant Models | `../Quant Models/` | Strategy backtesting code |

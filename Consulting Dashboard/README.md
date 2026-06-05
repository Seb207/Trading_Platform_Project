# Quant Research Dashboard

Local-first quant research dashboard built with **Next.js 14 + FastAPI**.  
Browse a local arXiv paper library, search semantically, download new papers, and chat with an LLM grounded in paper content — all running on your machine.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Frontend | Next.js 14 (App Router), TypeScript, Tailwind CSS |
| Backend | FastAPI (Python 3.12+), Uvicorn |
| LLM — Cloud | Anthropic SDK (Claude) · SSE streaming |
| LLM — Local | Ollama REST API · SSE streaming |
| Vector DB | ChromaDB (persistent, local) |
| Embedding | `BAAI/bge-base-en-v1.5` (768-dim, via sentence-transformers) |
| Paper Source | arXiv API (search + download) |
| Fonts | JetBrains Mono + Inter (Google Fonts) |

---

## Quick Start

### 1. Frontend

```bash
cd "Consulting Dashboard"
npm install
npm run dev
# → http://localhost:3000
```

### 2. Backend

```bash
cd "Consulting Dashboard"
pip install -r backend/requirements.txt
uvicorn backend.main:app --reload --port 8000
# → http://localhost:8000
```

### 3. LLM Setup

Open dashboard → Research page → right panel → click model bar:

- **Claude**: enter Anthropic API key (`sk-ant-…`)
- **Ollama**: start `ollama serve`, select a locally installed model

---

## Project Structure

```
Consulting Dashboard/
│
├── src/                                  # Next.js frontend
│   ├── app/
│   │   ├── layout.tsx                    # Root layout — wraps with LLMProvider, TopBar, CategoryNav
│   │   ├── globals.css                   # Terminal dark theme variables
│   │   ├── page.tsx                      # Redirect → /research
│   │   ├── research/page.tsx             # Research LLM page (Paper Library + Download tabs)
│   │   ├── regime/page.tsx               # Market Regime (placeholder)
│   │   ├── portfolio/page.tsx            # Portfolio Tracking (placeholder)
│   │   └── factor/page.tsx              # Factor Research (placeholder)
│   │
│   ├── components/
│   │   ├── layout/
│   │   │   ├── TopBar.tsx                # Logo + live LLM status (synced via React Context)
│   │   │   ├── CategoryNav.tsx           # Tab navigation between feature pages
│   │   │   └── SplitPanel.tsx            # Resizable left/right panel layout
│   │   │
│   │   ├── ui/                           # Reusable design system primitives
│   │   │   ├── Badge.tsx
│   │   │   ├── FilterChip.tsx
│   │   │   ├── ToggleGroup.tsx
│   │   │   └── CategoryTag.tsx
│   │   │
│   │   ├── research/
│   │   │   ├── SearchBar.tsx             # Query input + mode toggle + category filter chips
│   │   │   ├── PaperTable.tsx            # Scrollable paper list with similarity scores
│   │   │   └── DownloadPanel.tsx         # arXiv search preview + SSE download + progress log
│   │   │
│   │   └── chat/
│   │       ├── ChatPanel.tsx             # 3-tab panel: LLM Chat / Paper Viewer / Strategy
│   │       ├── ChatMessage.tsx           # Renders user/assistant messages with model name
│   │       ├── ChatInput.tsx             # Textarea with send button
│   │       └── ModelSelector.tsx         # Claude/Ollama switcher + model list + API key
│   │
│   ├── context/
│   │   └── LLMContext.tsx               # Global LLM config (provider + model) via React Context
│   │
│   └── lib/
│       ├── api.ts                        # Typed fetch wrappers for all backend endpoints
│       └── types.ts                      # Shared TypeScript types (Paper, ChatMessage, LLMConfig…)
│
├── backend/                              # FastAPI backend
│   ├── main.py                           # App entry point, CORS config, router registration
│   ├── config.py                         # Paths + env vars (DOWNLOAD_DIR, RESEARCH_LLM_DIR…)
│   │
│   ├── routers/
│   │   ├── papers.py                     # All paper + arXiv endpoints (see API Reference)
│   │   ├── chat.py                       # POST /api/chat — LLM streaming (SSE)
│   │   └── status.py                     # GET /api/status — index stats
│   │
│   └── modules/
│       ├── llm/
│       │   ├── base.py                   # Abstract LLMProvider interface
│       │   ├── claude_provider.py        # Anthropic SDK, async streaming
│       │   ├── ollama_provider.py        # Ollama /api/chat, NDJSON streaming
│       │   └── factory.py               # get_provider(provider, model, …)
│       │
│       └── research/
│           └── arxiv_bridge.py           # Singleton ArxivToolClient from Research_LLM/
│
└── README.md
```

---

## Feature Pages

### Research LLM (Active)

The main research workspace. Left panel = paper library + download. Right panel = LLM chat.

#### Left panel — Paper Library tab

| Control | Behaviour |
|---|---|
| Search input + Enter | Searches in the selected mode |
| **Abstract** mode | Semantic search over paper abstracts (ChromaDB, local) |
| **Section** mode | Semantic search over individual paper sections (ChromaDB, local) |
| **arXiv** mode | Live search via arXiv API (requires internet) |
| Category filter chips | Filter by `q-fin.RM / MF / CP / ST / PM / TR / GN / PR` |
| Paper row click | Selects paper → passes context to right panel |

**Local DB (as of last rebuild):**

| Category | Papers |
|---|---|
| q-fin.RM · Risk Management | 111 |
| q-fin.MF · Mathematical Finance | 105 |
| q-fin.CP · Computational Finance | 96 |
| q-fin.ST · Statistical Finance | 90 |
| q-fin.PM · Portfolio Management | 84 |
| q-fin.TR · Trading & Microstructure | 62 |
| q-fin.GN · General Finance | 46 |
| q-fin.PR · Pricing of Securities | 36 |
| **Total** | **630** |

#### Left panel — Download tab

1. Set search parameters (query, category, date range, max results)
2. Click **arXiv 검색 미리보기** → previews results with checkboxes
3. Select papers → click **N편 다운로드**
4. SSE stream shows per-paper progress + indexing steps:
   - Download (HTML → Markdown, PDF fallback)
   - `backfill_metadata` → `build_search_index` → `build_section_index`

#### Right panel — Chat tabs

| Tab | Content |
|---|---|
| **LLM Chat** | Streaming chat with Claude or Ollama. Selected paper injected into system prompt. |
| **Paper Viewer** | Paper metadata + abstract + section map (title + char count per section). |
| **Strategy** | Automatically extracts all code blocks from LLM responses. Copy button per block. |

---

## API Reference

| Method | Path | Description |
|---|---|---|
| `GET` | `/api/status` | Index stats (paper count, section count, index built flags) |
| `GET` | `/api/papers/list` | Local paper list, filterable by category |
| `POST` | `/api/papers/search/abstract` | ChromaDB abstract-level semantic search |
| `POST` | `/api/papers/search/section` | ChromaDB section-level semantic search |
| `POST` | `/api/papers/analyze` | Full paper content + section map for a `.md` file |
| `POST` | `/api/arxiv/search` | Live arXiv API search (cached 10 min) |
| `POST` | `/api/papers/download` | **SSE** — download papers + auto-index pipeline |
| `POST` | `/api/chat` | **SSE** — LLM chat with optional paper context |

---

## LLM Provider System

```
Frontend ModelSelector
    ↓  POST /api/chat  (SSE)
backend/routers/chat.py
    ↓
factory.get_provider(provider, model, api_key, ollama_url)
    ↓
LLMProvider (base.py — abstract)
    ├── ClaudeProvider   → anthropic.AsyncAnthropic + text_stream
    └── OllamaProvider   → httpx.AsyncClient + NDJSON line stream
```

**TopBar** displays the currently active provider and model, synced globally via `LLMContext` (React Context). Changing the model in `ModelSelector` updates the TopBar immediately.

### Ollama

When Ollama is selected, `ModelSelector` fetches `GET {ollamaUrl}/api/tags` and shows only locally installed models. If Ollama is not running, an error state with instructions is displayed.

### Adding a new provider

1. Create `backend/modules/llm/myprovider.py` implementing `LLMProvider.stream()`
2. Register it in `backend/modules/llm/factory.py`

No frontend changes needed.

---

## Semantic Search & Embedding

### Model

```
BAAI/bge-base-en-v1.5
  Dimensions : 768
  Size       : ~430 MB (downloaded once on first use)
  Index scope: abstracts (arxiv_papers) + sections (paper_sections)
```

Changed from `all-MiniLM-L6-v2` (384-dim) to `bge-base-en-v1.5` (768-dim) for significantly better retrieval quality on academic / financial text.

### Score

```
similarity_score = 1 − cosine_distance
```

Range: 0 (unrelated) → 1 (identical). ChromaDB uses HNSW with cosine space.

### Section embedding text

```
"{section_title}: {section_content[:1000]}"
```

14,285 sections indexed across 553 papers.

### Rebuilding the index

After changing the embedding model or adding papers outside the dashboard, delete `.chroma/` and rebuild:

```bash
rm -rf "Research_LLM/papers/arXiv/.chroma"

# Then in the dashboard: Download tab → download any paper with auto-index ON
# Or directly:
python - << 'EOF'
import sys; sys.path.insert(0, ".")
from backend.modules.research.arxiv_bridge import get_client
c = get_client()
print(c.build_search_index())
print(c.build_section_index())
EOF
```

---

## arXiv API — Rate Limiting & Caching

arXiv enforces per-IP rate limits. The backend handles this transparently:

| Layer | Mechanism |
|---|---|
| **Min interval** | 3 s enforced between consecutive API calls |
| **Retry on 429/503/timeout** | Exponential backoff: 30 s → 60 s → give up |
| **Response cache** | 10-minute in-memory TTL per unique search query |
| **User-Agent** | `QuantResearchDashboard/1.0` sent with every request |
| **HTTPS direct** | Calls `https://export.arxiv.org` directly (skips HTTP→HTTPS redirect) |

If the IP is temporarily blocked (many requests in one session), wait 30–60 minutes. Local Abstract/Section search works offline regardless.

---

## Research_LLM Integration

The dashboard reuses `Research_LLM/arxiv_client.py` without modification via a bridge module:

```
backend/modules/research/arxiv_bridge.py
  → adds Research_LLM/ to sys.path
  → imports ArxivToolClient (lru_cache singleton)
  → patches client.api_url to HTTPS
```

The two projects must be siblings on disk:

```
Trading_Platform_Project/
├── Consulting Dashboard/   ← this project
└── Research_LLM/           ← paper library + arxiv_client.py
    └── papers/arXiv/       ← downloaded papers + .chroma/
```

Override paths via environment variables if needed:

```bash
export RESEARCH_LLM_DIR=/custom/path/Research_LLM
export DOWNLOAD_DIR=/custom/path/Research_LLM/papers/arXiv
```

---

## Design System (Terminal Dark)

| Token | Value | Usage |
|---|---|---|
| `bg` | `#000000` | Page background |
| `bg2` | `#0a0a0a` | Panel background |
| `bg3` | `#111111` | Input / hover |
| `border` | `#1f1f1f` | All borders |
| `border2` | `#2a2a2a` | Hover borders |
| `text` | `#e8e8e8` | Primary text |
| `text-mid` | `#999999` | Secondary text |
| `text-dim` | `#555555` | Labels, metadata |
| `accent` | `#00ff88` | Active states, positive |
| `accent2` | `#00cfff` | arXiv IDs, secondary |
| `accent3` | `#ff4fa3` | Category tags (pink) |
| `accent4` | `#ffd700` | Category tags (gold) |
| `neg` | `#ff3b3b` | Errors, negative values |

Fonts: `JetBrains Mono` (IDs, values, mono labels) + `Inter` (body text)

---

## Implementation Status

### Completed

- [x] Next.js 14 App Router + TypeScript + Tailwind setup
- [x] Terminal dark design system (Design A tokens)
- [x] TopBar with live LLM provider/model display (React Context sync)
- [x] CategoryNav tab navigation
- [x] SplitPanel resizable layout
- [x] Paper Library — SearchBar with Abstract / Section / arXiv modes
- [x] Paper Library — category filter chips (all 8 q-fin categories in DB)
- [x] Paper Library — PaperTable with similarity scores
- [x] Paper Library — loading skeleton + empty states + section-mode hint
- [x] Download tab — arXiv preview with checkboxes + select all/none
- [x] Download tab — SSE download stream with real-time progress log
- [x] Download tab — auto-index (backfill → abstract index → section index)
- [x] Chat panel — LLM Chat tab with SSE streaming
- [x] Chat panel — streaming cursor (animated dots while generating)
- [x] Chat panel — Paper Viewer tab (metadata + abstract + section map)
- [x] Chat panel — Strategy tab (auto-extracts code blocks, Copy button)
- [x] ModelSelector — Claude models dropdown + API key input
- [x] ModelSelector — Ollama local model discovery via `/api/tags`
- [x] FastAPI backend with CORS
- [x] `/api/papers/list` — local paper list
- [x] `/api/papers/search/abstract` — ChromaDB abstract search
- [x] `/api/papers/search/section` — ChromaDB section search
- [x] `/api/papers/analyze` — paper structure analysis
- [x] `/api/arxiv/search` — live arXiv search with 10-min cache + retry
- [x] `/api/papers/download` — SSE download + auto-index pipeline
- [x] `/api/chat` — LLM streaming (Claude + Ollama)
- [x] `/api/status` — index stats
- [x] arXiv rate limiting: 3 s interval + exponential backoff + User-Agent
- [x] Embedding model: `BAAI/bge-base-en-v1.5` (768-dim, upgraded from MiniLM)
- [x] ChromaDB rebuild (456 papers / 14,285 sections)

### Planned

- [ ] Market Regime page — PCA + clustering on macro features
- [ ] Portfolio Tracking page — P&L, drawdown, factor attribution
- [ ] Factor Research page — factor score computation + signal decay
- [ ] Strategy tab — save/export generated code
- [ ] Paper Viewer — full-text scroll with section navigation
- [ ] Startup script — launch backend + frontend with one command

---

## Adding a New Feature Page

```
Frontend:
1. Create src/app/[feature]/page.tsx
2. Add components under src/components/[feature]/
3. Add route in src/components/layout/CategoryNav.tsx

Backend:
4. Create backend/routers/[feature].py
5. Create backend/modules/[feature]/
6. Register router in backend/main.py
```

---

## Related Projects

| Project | Path | Role |
|---|---|---|
| **Research_LLM** | `../Research_LLM/` | arXiv client, paper storage, ChromaDB indexes |
| **Quant Models** | `../Quant Models/` | Strategy backtesting code |

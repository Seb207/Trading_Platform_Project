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
| Math rendering | `react-markdown` + `remark-math` + `rehype-katex` + KaTeX |
| Fonts | IBM Plex Mono (data) + Space Grotesk (branding) + Inter (body) — Terminal-X style |

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

Open dashboard → Paper2Alpha page → right panel → click model bar:

- **Claude**: enter Anthropic API key (`sk-ant-…`)
- **Ollama**: start `ollama serve`, select a locally installed model

### 4. One-click launch (macOS app)

A native `.app` launcher is installed at `/Applications/Quant Dashboard.app`
(green **Q** icon). Clicking it:

1. Starts the backend (`:8000`) and frontend (`:3000`) if not already running
2. Waits for readiness, then opens `http://localhost:3000/research` in the browser
3. Stays in the Dock as a running app — **quitting the app (⌘Q / Dock → Quit)
   stops both servers**. Closing only the browser tab leaves them running.

Progress is surfaced via macOS notifications. Logs: `~/Library/Logs/QuantDashboard/`.

> The app is a thin lifecycle wrapper for **local single-user** use. For real
> deployment, run the servers as an always-on service (launchd / systemd / Docker)
> and reduce the icon to a URL shortcut — see *Deployment notes* below.

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
│   │   ├── research/page.tsx             # Paper2Alpha page (Paper Library + Download tabs)
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
│   │   │   ├── PaperTable.tsx            # Scrollable paper list (index-based React keys)
│   │   │   ├── DownloadPanel.tsx         # arXiv search preview + SSE download + progress log
│   │   │   └── MarkdownRenderer.tsx      # Markdown + LaTeX math (KaTeX) renderer for paper content
│   │   │
│   │   └── chat/
│   │       ├── ChatPanel.tsx             # 3-tab panel: LLM Chat / Paper Viewer / Strategy
│   │       ├── ChatMessage.tsx           # Renders user/assistant messages with model name
│   │       ├── ChatInput.tsx             # Textarea with send button
│   │       └── ModelSelector.tsx         # Claude/Ollama switcher + custom dropdown + real status dot
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

### Paper2Alpha (Active)

The main research workspace (formerly "Research LLM"). Left panel = paper library +
download. Right panel = LLM chat. All UI text is in English.

#### Left panel — Paper Library tab

| Control | Behaviour |
|---|---|
| Search input + Enter | Searches in the selected mode |
| **Abstract** mode | Semantic search over paper abstracts (ChromaDB, local) |
| **Section** mode | Semantic search over individual paper sections (ChromaDB, local) |
| **arXiv** mode | Live search via arXiv API (requires internet) |
| Category filter chips | Filter by `q-fin.RM / MF / CP / ST / PM / TR / GN / PR` |
| Paper row click | Selects paper → passes context to right panel |

**Local DB (as of last rebuild — May 2025 papers added across all categories):**

| Category | Papers |
|---|---|
| q-fin.MF · Mathematical Finance | 155 |
| q-fin.RM · Risk Management | 153 |
| q-fin.CP · Computational Finance | 149 |
| q-fin.ST · Statistical Finance | 134 |
| q-fin.PM · Portfolio Management | 109 |
| q-fin.TR · Trading & Microstructure | 90 |
| q-fin.GN · General Finance | 73 |
| q-fin.PR · Pricing of Securities | 56 |
| **Total** | **919** |

> Metadata is populated for 677 papers; the remaining ~23 are PDF-only downloads
> awaiting title/abstract (parsed from `.md` locally where possible — see
> *arXiv API* below). All 919 are searchable via Section search regardless.

#### Left panel — Download tab

1. Set search parameters (query, category, date range, **Max Results** — freely editable, clamps to 1–50 on blur)
2. Click **Preview arXiv Search** → previews results with checkboxes (Select All / Deselect All)
3. Select papers → click **Download N**
4. SSE stream shows per-paper progress + indexing steps:
   - Download (HTML → Markdown, PDF fallback) — **already-downloaded papers are skipped instantly**
   - `backfill_metadata` (always runs, batched) → `build_search_index` + `build_section_index` (when auto-index ON)

#### Right panel — Chat tabs

| Tab | Content |
|---|---|
| **LLM Chat** | Streaming chat with Claude or Ollama. Selected paper injected into system prompt. |
| **Paper Viewer** | Collapsible **section accordion** — click any heading to expand its full content. LaTeX math (`$…$`, `$$…$$`) renders via KaTeX. Expand/collapse all. Memory-efficient: section bodies mount only when opened (`useMemo` parse + `Set<number>` open-state). |
| **Strategy** | Automatically extracts all code blocks from LLM responses. Copy button per block. |

**Model status dot** (ModelSelector + TopBar) reflects the *real* connection state:
grey = no API key, green = Claude key set / Ollama connected, amber = Ollama connecting,
red = Ollama unreachable. No longer hard-coded green.

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

~19,500 sections indexed across 788 `.md` papers (abstract index: 666 papers).

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
| **`"Rate exceeded."` body** | arXiv returns **HTTP 200** with a plain-text `Rate exceeded.` body when throttled — detected and surfaced as a 429, not silently parsed as empty results |
| **Response cache** | 10-minute in-memory TTL per unique search query |
| **User-Agent** | `ResearchLLM/1.0` sent with every request (API **and** content servers) |
| **HTTPS direct** | Calls `https://export.arxiv.org` directly (skips HTTP→HTTPS redirect) |

If the IP is temporarily blocked (many requests in one session), wait 30–60 minutes. Local Abstract/Section search works offline regardless.

### Download is decoupled from the API (efficiency refactor)

Two arXiv hosts are involved, and they throttle independently:

- **Content server** (`arxiv.org/html`, `arxiv.org/pdf`) — serves the paper body. Rarely throttled.
- **Metadata/search API** (`export.arxiv.org/api/query`) — throttles aggressively under load.

Previously every downloaded paper fired **one metadata API call**, so a batch of N papers = N API hits → the API throttled, which in turn **blocked downloads and broke live search**. The refactor:

| Change | Effect |
|---|---|
| `download_paper(..., fetch_metadata=False)` on bulk/SSE paths | Downloads hit only the (healthy) content server; **no per-paper API call** |
| Single batched `backfill_metadata` (50 IDs/call) after downloads | ~50× fewer API requests; runs once, not per paper |
| `skip_existing=True` (default) | Already-downloaded papers are skipped — zero redundant requests on re-runs |
| Backfill always runs (independent of `auto_index`) | Titles/abstracts populate even when embedding is deferred |

**Titles don't actually need the API.** HTML-downloaded papers already contain the
title (first `# ` heading), authors, and abstract in the `.md`. A local extractor
parses these directly — only the `published` date requires the API. This is the
preferred path; the API is a fallback for PDF-only papers.

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

Fonts (Terminal-X style): `IBM Plex Mono` (IDs, values, data — tabular figures) +
`Space Grotesk` (logo / branding via `.font-display`) + `Inter` (body text).
Math: KaTeX with dark-theme overrides (display math gets a left accent rail).

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
- [x] arXiv rate limiting: 3 s interval + exponential backoff + User-Agent + `"Rate exceeded."` body detection
- [x] Embedding model: `BAAI/bge-base-en-v1.5` (768-dim, upgraded from MiniLM)
- [x] ChromaDB rebuild (919 papers / ~19,500 sections)
- [x] API efficiency refactor — `fetch_metadata` decoupling, batched backfill, `skip_existing`, HTTPS, User-Agent
- [x] Local metadata extraction — title/abstract parsed from `.md` (no API needed)
- [x] Paper Viewer — collapsible section accordion (memory-efficient)
- [x] LaTeX math rendering (KaTeX) in paper content
- [x] Fonts → IBM Plex Mono + Space Grotesk + Inter (Terminal-X style)
- [x] Real model status dot (grey/green/amber/red) in ModelSelector + TopBar
- [x] Custom themed dropdowns (replaced native `<select>`)
- [x] Full UI English-ization (was partly Korean)
- [x] Duplicate React key fix (index-based keys + backend dedup)
- [x] Max Results input — free editing, clamp on blur
- [x] Category rename: "Research LLM" → "Paper2Alpha"
- [x] macOS `.app` launcher (green Q icon) — app lifecycle manages servers

### Planned

- [ ] Market Regime page — PCA + clustering on macro features
- [ ] Portfolio Tracking page — P&L, drawdown, factor attribution
- [ ] Factor Research page — factor score computation + signal decay
- [ ] Strategy tab — save/export generated code
- [ ] Deployment — always-on service (launchd / systemd / Docker) + production build

---

## macOS App Launcher

`/Applications/Quant Dashboard.app` — a hand-built bundle:

```
Quant Dashboard.app/Contents/
├── Info.plist                 # CFBundleIconFile = QuantQ, executable = launcher
├── MacOS/launcher             # bash: start servers → open browser → stay alive → cleanup on quit
└── Resources/QuantQ.icns      # green "Q" (Verdana Bold) on dark, neon-green glow (#00ff88)
```

Lifecycle model (local single-user): the launcher **stays running** as a Dock app.
On quit (⌘Q / Dock → Quit) it traps `SIGTERM` and kills whatever listens on
`:8000` / `:3000` (escalating TERM → KILL). Backend runs **without `--reload`** so
there's no reload-supervisor respawning the worker during shutdown.

Regenerate the icon: `python3 /tmp/make_icon.py` → `iconutil -c icns …` (see repo scripts).

## Deployment notes

The `.app` lifecycle model suits **local single-user** use. For deployment, decouple
the servers from any desktop app:

| Concern | Local (now) | Deployment |
|---|---|---|
| Frontend | `npm run dev` | `next build` + `next start` |
| Backend | `uvicorn` (1 proc) | `uvicorn --workers N` (no `--reload`) behind nginx |
| Lifecycle | app manages, quit = stop | always-on service: **launchd** (mac) / **systemd** (Linux) / **Docker** `restart: always` |
| Icon | starts + stops servers | just a URL shortcut |

In a hosted setup the servers run independently of any app, so "closing the app"
no longer affects them — that's the point of the service model.

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

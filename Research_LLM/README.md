# Research_LLM: arXiv MCP Server

A local MCP server that bridges LLMs and the arXiv academic database.
It lets AI assistants autonomously search, download, and deeply analyze research papers — then generate grounded quant strategy code based on them.

## Architecture & Design Choices

This project deliberately avoids traditional RAG (chunking + Vector DB) for the core reading workflow.
Instead, it downloads full papers as clean Markdown and passes them directly into the LLM's context window.

**Why full-text over RAG for quant research?**
- Quant papers have tightly coupled sections (methodology ↔ results ↔ math). Chunking breaks those links.
- The target workflow is *deep reading of a small, curated set of papers*, not broad retrieval across thousands.
- Modern LLMs (Claude, Gemini) have 200k+ token windows — enough to hold multiple full papers at once.

**When RAG is added (Phase 4):** only paper *abstracts* are embedded, not full text. This gives lightweight topic-based *discovery* across a large local library, after which full papers are read normally.

---

## MCP Tools (9 total)

| Tool | Phase | Description |
|---|---|---|
| `search_arxiv_papers` | 1 | Search arXiv by keyword + optional date range |
| `download_arxiv_paper` | 1 | Download one paper (HTML→Markdown, PDF fallback) |
| `list_local_papers` | 1 | List locally saved papers, filterable by category |
| `read_local_paper` | 1 | Read a local `.md` paper with pagination |
| `analyze_local_paper` | 2 | Full paper + section map for strategy generation |
| `bulk_download_papers` | 3 | Download a list of papers in one call |
| `backfill_metadata` | 3 | Populate `metadata.json` for pre-existing papers |
| `build_search_index` | 4 | Embed abstracts into ChromaDB for semantic search |
| `search_local_papers_by_topic` | 4 | Find relevant papers by natural language query |

---

## Typical Workflows

### 1. Research a topic and generate a quant strategy

```
"2024년 이후 cross-sectional momentum 논문 5편 찾아서 다운받고,
방법론이 가장 명확한 논문 기반으로 Python 백테스트 코드 짜줘"
```

Claude will automatically chain:
1. `search_arxiv_papers("cross-sectional momentum", date_from="2024-01-01", max_results=5)`
2. `bulk_download_papers([...ids...], category="q-fin.PM")`
3. `analyze_local_paper("q-fin.PM/2401.xxxxx.md")`
4. Generate strategy code grounded in the paper's methodology

### 2. Catch up on recent papers in a category

```
"q-fin.TR 카테고리에서 2025년 1분기 논문 목록 보여주고 abstract 요약해줘"
```

### 3. Search your local library by topic (Phase 4)

```
"로컬에 저장된 논문 중 volatility scaling 다루는 논문 찾아줘"
```

Claude will use `search_local_papers_by_topic` to find the most relevant papers by semantic similarity, then read them with `analyze_local_paper`.

---

## Project Structure

```text
Research_LLM/                       # Project root
├── arxiv_client.py                 # Core logic: API calls, HTML parsing, metadata, ChromaDB
├── mcp_server.py                   # FastMCP server — exposes 9 tools to the LLM
├── tests/
│   └── test_local_paper_tools.py   # 31 unit tests (all features covered, no network calls)
├── requirements.txt                # Python dependencies
└── README.md                       # This file

papers/
└── arXiv/                          # Downloaded papers
    ├── q-fin.CP/                   # Organised by arXiv category
    ├── q-fin.PM/
    ├── cs.AI/
    ├── ...
    ├── metadata.json               # Auto-saved paper metadata (title, abstract, authors, date)
    └── .chroma/                    # ChromaDB index (created by build_search_index)
```

---

## Setup & Installation

### 1. Install dependencies

```bash
cd Trading_Platform_Project/Research_LLM
pip install -r requirements.txt
```

> **Phase 4 only** (semantic search): `chromadb` and `sentence-transformers` are included in `requirements.txt` but only imported when the relevant tools are called. Skip if you don't need local topic search yet.

### 2. Run unit tests

```bash
cd Trading_Platform_Project/Research_LLM
python3 -m pytest tests/ -v
```

### 3. Connect to Claude Desktop

Add the following to `~/Library/Application Support/Claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "arxiv-research": {
      "command": "/opt/anaconda3/bin/python3",
      "args": ["/ABSOLUTE_PATH/Trading_Platform_Project/Research_LLM/mcp_server.py"]
    }
  }
}
```

Restart Claude Desktop. A hammer icon (🔨) will appear — click it to verify all 9 tools are loaded.

### 4. Connect to VS Code Cline

Add the same JSON block to Cline's `mcp.json`. Works with any API (Gemini, GPT-4o, Claude).

---

## First-Time Setup for Phase 4 (Semantic Search)

If you already have papers downloaded before Phase 4 was added, run these once:

```
# In Claude Desktop:
1. backfill_metadata()        — fetch metadata for all existing papers from arXiv API
2. build_search_index()       — embed abstracts (downloads ~80MB model on first run)
3. search_local_papers_by_topic("your topic")  — ready to use
```

New papers downloaded via `download_arxiv_paper` or `bulk_download_papers` are automatically added to `metadata.json`. Re-run `build_search_index` periodically to keep the index fresh.

---

## arXiv q-fin Categories Reference

| Category | Name | Focus |
|---|---|---|
| `q-fin.CP` | Computational Finance | Numerical methods, ML/DL models |
| `q-fin.EC` | Economics | Economic theory, econometrics |
| `q-fin.GN` | General Finance | Misc. finance topics |
| `q-fin.MF` | Mathematical Finance | Stochastic calculus, derivatives math |
| `q-fin.PM` | Portfolio Management | Factor models, asset allocation |
| `q-fin.PR` | Pricing of Securities | Derivatives pricing |
| `q-fin.RM` | Risk Management | VaR, CVaR, risk models |
| `q-fin.ST` | Statistical Finance | Time series, empirical analysis |
| `q-fin.TR` | Trading & Market Microstructure | Algo trading, order flow |

---

## Roadmap

### Done
- [x] Date-filtered arXiv search (`date_from` / `date_to`)
- [x] HTML→Markdown download with nav/footer noise removed
- [x] `analyze_local_paper` — full paper + section map for strategy generation
- [x] `bulk_download_papers` — batch download from search results
- [x] Auto-save metadata on every download
- [x] `backfill_metadata` — batch API backfill for pre-existing papers
- [x] `build_search_index` — abstract embedding via sentence-transformers + ChromaDB
- [x] `search_local_papers_by_topic` — semantic search over local library
- [x] Claude Desktop MCP connection

### Later
- [ ] PDF text extraction for `read_local_paper` (for papers without HTML)
- [ ] Prompt cookbook (common Korean/English research prompts)

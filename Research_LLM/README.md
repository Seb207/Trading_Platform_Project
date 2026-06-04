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

**Two-tier retrieval strategy (Phase 4):**
- **Abstract-level** (`build_search_index` / `search_local_papers_by_topic`): broad topic discovery across all papers — no LLM needed, just vector similarity on abstracts.
- **Section-level** (`build_section_index` / `search_sections_by_topic`): pinpoint the exact section (Methodology, Results, etc.) across all papers — enables targeted reading without loading full papers.

This two-tier approach scales to thousands of papers while minimising LLM token usage.

---

## MCP Tools (11 total)

| Tool | Phase | Description |
|---|---|---|
| `search_arxiv_papers` | 1 | Search arXiv by keyword + optional date range |
| `download_arxiv_paper` | 1 | Download one paper (HTML→Markdown, PDF fallback) |
| `list_local_papers` | 1 | List locally saved papers, filterable by category |
| `read_local_paper` | 1 | Read a local `.md` paper with pagination |
| `analyze_local_paper` | 2 | Full paper + section map for strategy generation |
| `bulk_download_papers` | 3 | Download a list of papers in one call |
| `backfill_metadata` | 3 | Populate `metadata.json` for pre-existing papers |
| `build_search_index` | 4 | Embed abstracts into ChromaDB (`arxiv_papers` collection) |
| `search_local_papers_by_topic` | 4 | Find relevant papers by natural language query |
| `build_section_index` | 4 | Embed each section separately into ChromaDB (`paper_sections` collection) |
| `search_sections_by_topic` | 4 | Find relevant sections (Methodology, Results, etc.) across all papers |

---

## Typical Workflows

### 1. Research a topic and generate a quant strategy

```
"Find 5 cross-sectional momentum papers published after 2024 and download them.
Then write a Python backtest based on the paper with the clearest methodology."
```

Claude will automatically chain:
1. `search_arxiv_papers("cross-sectional momentum", date_from="2024-01-01", max_results=5)`
2. `bulk_download_papers([...ids...], category="q-fin.PM")`
3. `analyze_local_paper("q-fin.PM/2401.xxxxx.md")`
4. Generate strategy code grounded in the paper's methodology

### 2. Catch up on recent papers in a category

```
"List papers from the q-fin.TR category published in Q1 2025 and summarize their abstracts."
```

### 3. Token-efficient deep research (Phase 4 — recommended for large libraries)

```
"Find how BAB factor portfolios are constructed across papers in my local library."
```

Claude will chain all three retrieval tiers:
1. `search_local_papers_by_topic("betting against beta")` → Top 5 papers by abstract similarity
2. `search_sections_by_topic("BAB portfolio construction", section_filter="method")` → Pinpoint Methodology sections
3. `read_local_paper(relative_path, offset=...)` → Read only the relevant section
4. Generate or compare implementations

**Token cost:** ~5,000–15,000 tokens vs ~200,000+ tokens for full-paper loading.

### 4. Search within a single paper

```
"In paper 2401.12345, find the section that explains the rebalancing frequency."
```

```
search_sections_by_topic("rebalancing frequency", arxiv_id="2401.12345")
```

---

## Project Structure

```text
Research_LLM/                       # Project root
├── arxiv_client.py                 # Core logic: API calls, HTML parsing, metadata, ChromaDB
├── mcp_server.py                   # FastMCP server — exposes 11 tools to the LLM
├── tests/
│   └── test_local_paper_tools.py   # Unit tests (all features covered, no network calls)
├── requirements.txt                # Python dependencies
└── README.md                       # This file

papers/
└── arXiv/                          # Downloaded papers
    ├── q-fin.CP/                   # Organised by arXiv category
    ├── q-fin.PM/
    ├── cs.AI/
    ├── ...
    ├── metadata.json               # Auto-saved paper metadata (title, abstract, authors, date)
    └── .chroma/                    # ChromaDB index directory
        ├── arxiv_papers/           # Abstract-level index (build_search_index)
        └── paper_sections/         # Section-level index (build_section_index)
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

Restart Claude Desktop. A hammer icon (🔨) will appear — click it to verify all 11 tools are loaded.

### 4. Connect to VS Code Cline

Add the same JSON block to Cline's `mcp.json`. Works with any API (Gemini, GPT-4o, Claude).

---

## First-Time Setup for Phase 4 (Semantic Search)

If you already have papers downloaded before Phase 4 was added, run these once:

```
# In Claude Desktop:
1. backfill_metadata()             — fetch metadata for all existing papers from arXiv API
2. build_search_index()            — embed abstracts (~80MB model downloaded on first run)
3. build_section_index()           — embed all paper sections (takes longer for large libraries)
4. search_local_papers_by_topic("your topic")   — abstract-level discovery
5. search_sections_by_topic("your topic")       — section-level precision retrieval
```

New papers downloaded via `download_arxiv_paper` or `bulk_download_papers` are automatically added to `metadata.json`. Re-run `build_search_index` and `build_section_index` periodically to keep both indexes fresh.

---

## Scaling Strategy by Library Size

As the local paper library grows, the recommended workflow evolves. The tools remain unchanged — only how they are combined changes.

### < 100 papers

**Workflow:** Full-text reading for every query.

```
search_local_papers_by_topic → analyze_local_paper (full text) → LLM answer
```

- Abstract index optional but recommended
- Section index not yet necessary
- Token cost per query: 50k–200k (full papers in context)

---

### 100 – 1,000 papers

**Workflow:** Two-tier retrieval (abstract → section → targeted read).

```
search_local_papers_by_topic   → Top 10–20 papers
search_sections_by_topic       → Top 3–5 relevant sections
read_local_paper(offset=...)   → Read only those sections
LLM answer
```

**Action items:**
- Run `build_search_index` — abstract-level ChromaDB collection
- Run `build_section_index` — section-level ChromaDB collection
- Use `search_sections_by_topic(section_filter="method")` for methodology-specific queries
- Token cost per query: ~5k–20k (sections only)

---

### 1,000 – 5,000 papers

**Workflow:** Add hybrid retrieval (vector + keyword) and a reranker.

```
Hybrid retrieval:
  BM25 keyword search (e.g. "Fama-French", "BAB") +
  ChromaDB vector search
  → merge results (EnsembleRetriever)

Section-level reranker:
  CrossEncoder (local model, ~100MB) reranks Top 20 → Top 5

read_local_paper → LLM answer
```

**Action items:**
- Add `rank_bm25` package for keyword search alongside ChromaDB
- Add `cross-encoder/ms-marco-MiniLM-L-6-v2` reranker model
- Implement `LangChain EnsembleRetriever` in `arxiv_bridge.py` (dashboard module)
- Consider increasing HNSW index `ef` parameter for better recall at scale

---

### 5,000+ papers

**Workflow:** Full multi-stage pipeline with query expansion.

```
Query expansion (LLM):
  "BAB factor" → ["low beta portfolio", "betting against beta", "volatility anomaly"]

For each expanded query:
  Hybrid search (BM25 + ChromaDB) → Top 20

Merge + deduplicate → Top 30 candidates

CrossEncoder rerank → Top 5 sections

LLM reads Top 5 sections → answer
```

**Action items:**
- Migrate ChromaDB to a dedicated vector database: **Qdrant** (local Docker) or **Weaviate**
- Enable HNSW index tuning (`m`, `ef_construction`) for better ANN recall
- Add `MultiQueryRetriever` via LangChain for automatic query expansion
- Add incremental indexing: new downloads trigger automatic index upsert without full rebuild
- Consider abstractive summarisation per paper (store summary in metadata, embed summary instead of raw abstract)

---

### Index maintenance schedule

| Trigger | Action |
|---|---|
| New papers downloaded | `backfill_metadata` → `build_search_index` → `build_section_index` |
| Weekly (batch downloads) | Re-run both index builds (upsert-only, fast for existing papers) |
| Schema change to metadata | `backfill_metadata` → rebuild both indexes |
| 1,000+ paper milestone | Add BM25 + reranker layer |
| 5,000+ paper milestone | Migrate to Qdrant/Weaviate |

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
- [x] `search_local_papers_by_topic` — abstract-level semantic search
- [x] `build_section_index` — section-level embedding (each section as separate document)
- [x] `search_sections_by_topic` — targeted section retrieval with category/arxiv_id/name filters
- [x] Claude Desktop MCP connection

### Later
- [ ] PDF text extraction for `read_local_paper` (for papers without HTML)
- [ ] BM25 hybrid search layer (for 1,000+ paper libraries)
- [ ] CrossEncoder reranker integration
- [ ] Incremental index updates on download (auto-trigger section index upsert)
- [ ] LangChain multi-query retriever for query expansion

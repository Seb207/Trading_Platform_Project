# Architecture

See `CLAUDE.md` for environment gotchas and dependency-sync rules — this
file is the structural map only.

## Directory Structure

```
Research_LLM/                   # its own git repo, nested under Trading_Platform_Project
├── arxiv_client.py             # all tool logic — search/download/analyze/embed
├── mcp_server.py                # MCP tool registration (11 tools, 4 phases)
├── requirements.txt              # source of truth for this module's deps
├── papers/                       # downloaded papers (Markdown/PDF) + metadata.json
├── scripts/                      # maintenance scripts (e.g. refresh_api_metadata.py)
├── tests/                        # pytest suite
├── docs/                         # this framework's PRD/ARCHITECTURE/ADR
├── phases/                       # harness workflow: multi-step feature plans
└── scripts/execute.py            # harness phase/step bookkeeping tool
```

## Pattern: Full-Text Reading, Not RAG-Chunking

Papers are downloaded as clean Markdown and passed whole into the LLM's
context window for the core reading/analysis workflow. Semantic search
(ChromaDB, two-tier: abstracts + sections) is a separate discovery layer on
top of this, not the primary retrieval mechanism — see `ADR.md` for why.

## Data Flow

**Download**: `search_arxiv_papers` → `download_arxiv_paper` /
`bulk_download_papers` (HTML→Markdown via `markdownify`, PDF fallback via
`pypdf`) → `metadata.json` updated (via arXiv API, or local-parsing
stopgap if the API is rate-limited).

**Deep read**: `analyze_local_paper` → full text + section map → passed
directly into the calling LLM's context.

**Semantic discovery** (Phase 4, optional — needs chromadb +
sentence-transformers): `build_search_index` / `build_section_index` embed
abstracts/sections once; `search_local_papers_by_topic` /
`search_sections_by_topic` do the actual query-time similarity search.

## Consuming Side

Bridged into `Consulting Dashboard/backend/modules/research/arxiv_bridge.py`,
which imports this module's code directly (bridge pattern — see the parent
repo's `docs/ARCHITECTURE.md`) rather than depending on it as an installed
package. This repo's `requirements.txt` is the source of truth the
Dashboard's `backend/requirements.txt` must mirror by hand.

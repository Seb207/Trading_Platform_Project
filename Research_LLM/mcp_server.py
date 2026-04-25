import json
import os
import sys

# Ensure arxiv_client is importable regardless of working directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mcp.server.fastmcp import FastMCP
from arxiv_client import ArxivToolClient

mcp = FastMCP("ArxivResearchServer")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DOWNLOAD_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "papers", "arXiv"))

arxiv_client = ArxivToolClient(download_dir=DOWNLOAD_DIR)


# ==================================================================
# Phase 1 & 2 — Core tools
# ==================================================================

@mcp.tool()
def search_arxiv_papers(
    query: str,
    max_results: int = 5,
    date_from: str = "",
    date_to: str = "",
) -> str:
    """
    Search arXiv for research papers based on a query and optional date range.

    Returns a list of papers including arxiv_id, title, authors, summary, and category.

    Args:
        query: Search keywords (e.g. "momentum factor equity", "transformer time series")
        max_results: Number of papers to return (default 5, max 50)
        date_from: Start date in YYYY-MM-DD or YYYYMMDD format (optional)
        date_to: End date in YYYY-MM-DD or YYYYMMDD format (optional)
    """
    results = arxiv_client.search_papers(
        query=query,
        max_results=max_results,
        date_from=date_from or None,
        date_to=date_to or None,
    )
    return json.dumps(results, indent=2, ensure_ascii=False)


@mcp.tool()
def download_arxiv_paper(arxiv_id: str, category: str = "Unknown") -> str:
    """
    Download an arXiv paper to the local machine using its arxiv_id.

    Prioritizes HTML→Markdown conversion (.md).
    Falls back to PDF if HTML is unavailable.
    Automatically saves metadata to metadata.json.

    Args:
        arxiv_id: The arXiv paper ID (e.g. "2401.12345")
        category: arXiv category for folder organisation (e.g. "q-fin.CP", "cs.AI")
    """
    result = arxiv_client.download_paper(arxiv_id, category)
    return json.dumps(result, indent=2, ensure_ascii=False)


@mcp.tool()
def list_local_papers(category: str = "", limit: int = 200) -> str:
    """
    List locally downloaded arXiv papers stored in papers/arXiv.

    Args:
        category: Filter by arXiv category (e.g. "q-fin.CP"). Leave empty to list all.
        limit: Maximum number of papers to return (default 200).
    """
    normalized_category = category.strip() or None
    result = arxiv_client.list_local_papers(category=normalized_category, limit=limit)
    return json.dumps(result, indent=2, ensure_ascii=False)


@mcp.tool()
def read_local_paper(relative_path: str, max_chars: int = 50000, offset: int = 0) -> str:
    """
    Read a local paper by its relative path under papers/arXiv.

    Supports pagination via offset for very long papers.

    Args:
        relative_path: Path relative to papers/arXiv (e.g. "q-fin.CP/2401.12345.md")
        max_chars: Maximum characters to return per call (default 50000, max 200000)
        offset: Character offset for pagination (default 0)
    """
    result = arxiv_client.read_local_paper(
        relative_path=relative_path, max_chars=max_chars, offset=offset
    )
    return json.dumps(result, indent=2, ensure_ascii=False)


@mcp.tool()
def analyze_local_paper(relative_path: str) -> str:
    """
    Deeply analyze a local Markdown paper for quant research and strategy generation.

    Returns the full paper content alongside a structured section map.
    Use this when you need to summarise a paper, answer questions about it,
    or generate Python quant strategy code grounded in its methodology.

    Args:
        relative_path: Path relative to papers/arXiv (e.g. "q-fin.CP/2401.12345.md")
    """
    result = arxiv_client.analyze_local_paper(relative_path=relative_path)
    return json.dumps(result, indent=2, ensure_ascii=False)


# ==================================================================
# Phase 3 — Bulk download & Metadata
# ==================================================================

@mcp.tool()
def bulk_download_papers(arxiv_ids: list[str], category: str = "Unknown") -> str:
    """
    Download multiple arXiv papers at once using a list of arxiv IDs.

    Typical workflow:
      1. search_arxiv_papers → get arxiv_id list
      2. bulk_download_papers → download all at once

    Metadata is saved automatically for each downloaded paper.

    Args:
        arxiv_ids: List of arXiv IDs to download (e.g. ["2401.12345", "2312.67890"])
        category: arXiv category for folder organisation (e.g. "q-fin.PM")
    """
    result = arxiv_client.bulk_download_papers(arxiv_ids=arxiv_ids, category=category)
    return json.dumps(result, indent=2, ensure_ascii=False)


@mcp.tool()
def backfill_metadata() -> str:
    """
    Scan all locally downloaded papers and populate metadata.json for any missing entries.

    Run this once after migrating pre-existing papers, or if metadata.json is out of sync.
    Uses batched arXiv API calls for efficiency.
    After backfill, run build_search_index to make all papers searchable.
    """
    result = arxiv_client.backfill_metadata()
    return json.dumps(result, indent=2, ensure_ascii=False)


# ==================================================================
# Phase 4 — Semantic search
# ==================================================================

@mcp.tool()
def build_search_index() -> str:
    """
    Build (or refresh) the semantic search index from metadata.json.

    Embeds each paper's abstract using a local sentence-transformers model
    (all-MiniLM-L6-v2, ~80 MB downloaded once on first use).
    The index persists to disk — subsequent calls only upsert new papers.

    Prerequisites:
      - pip install chromadb sentence-transformers
      - Run backfill_metadata if you have pre-existing papers without metadata
    """
    result = arxiv_client.build_search_index()
    return json.dumps(result, indent=2, ensure_ascii=False)


@mcp.tool()
def search_local_papers_by_topic(
    query: str,
    n_results: int = 5,
    category: str = "",
) -> str:
    """
    Semantic search over locally downloaded papers using abstract embeddings.

    Returns papers ranked by similarity to the query (similarity_score 0–1).
    Use this to discover relevant papers from your local library before reading them.

    Args:
        query: Natural language topic (e.g. "cross-sectional momentum factor decay")
        n_results: Number of papers to return (default 5)
        category: Filter by arXiv category (e.g. "q-fin.PM"). Leave empty to search all.
    """
    result = arxiv_client.search_local_papers_by_topic(
        query=query, n_results=n_results, category=category.strip()
    )
    return json.dumps(result, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    print(f"Starting arXiv MCP Server. Downloads will be saved to: {DOWNLOAD_DIR}")
    mcp.run(transport="stdio")

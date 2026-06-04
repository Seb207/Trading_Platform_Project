"""
Paper endpoints:
  GET  /api/papers/list
  POST /api/papers/search/abstract
  POST /api/papers/search/section
  POST /api/papers/analyze
  POST /api/arxiv/search
"""
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from pathlib import Path
import json

from backend.config import DOWNLOAD_DIR
from backend.modules.research.arxiv_bridge import get_client

router = APIRouter(tags=["papers"])


def _load_metadata() -> dict:
    """Load metadata.json keyed by arxiv_id."""
    path = Path(DOWNLOAD_DIR) / "metadata.json"
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _enrich_papers(papers: list[dict], metadata: dict) -> list[dict]:
    """Merge file-list entries with metadata (title, abstract, published)."""
    enriched = []
    for p in papers:
        aid = p.get("arxiv_id", "")
        meta = metadata.get(aid, {})
        enriched.append({
            **p,
            "title":    meta.get("title", aid),
            "abstract": meta.get("summary", ""),
            "published": meta.get("published", ""),
            "authors":  meta.get("authors", []),
        })
    return enriched


# ── Request models ──────────────────────────────────────────────────────

class AbstractSearchRequest(BaseModel):
    query: str
    n_results: int = 10
    category: str = ""


class SectionSearchRequest(BaseModel):
    query: str
    n_results: int = 10
    category: str = ""
    arxiv_id: str = ""
    section_filter: str = ""


class AnalyzeRequest(BaseModel):
    relative_path: str


class ArxivSearchRequest(BaseModel):
    query: str = ""
    max_results: int = 10
    date_from: str = ""
    date_to: str = ""
    category: str = ""


# ── Endpoints ───────────────────────────────────────────────────────────

@router.get("/api/papers/list")
def list_papers(
    category: str = Query(default=""),
    limit: int = Query(default=200, le=1000),
):
    """List locally downloaded papers enriched with metadata (title, abstract, published)."""
    client = get_client()
    result = client.list_local_papers(
        category=category.strip() or None,
        limit=limit,
    )
    if result.get("status") == "error":
        raise HTTPException(status_code=500, detail=result.get("message"))

    metadata = _load_metadata()
    result["papers"] = _enrich_papers(result["papers"], metadata)
    return result


@router.post("/api/papers/search/abstract")
def search_abstract(body: AbstractSearchRequest):
    """Semantic search over paper abstracts (ChromaDB)."""
    client = get_client()
    result = client.search_local_papers_by_topic(
        query=body.query,
        n_results=body.n_results,
        category=body.category.strip(),
    )
    if result.get("status") == "error":
        raise HTTPException(status_code=500, detail=result.get("message"))
    return result


@router.post("/api/papers/search/section")
def search_sections(body: SectionSearchRequest):
    """Semantic search over paper sections (ChromaDB section index)."""
    client = get_client()
    result = client.search_sections_by_topic(
        query=body.query,
        n_results=body.n_results,
        category=body.category.strip(),
        arxiv_id=body.arxiv_id.strip(),
        section_filter=body.section_filter.strip(),
    )
    if result.get("status") == "error":
        raise HTTPException(status_code=500, detail=result.get("message"))
    return result


@router.post("/api/papers/analyze")
def analyze_paper(body: AnalyzeRequest):
    """Return full paper content + section map for a local .md file."""
    client = get_client()
    result = client.analyze_local_paper(relative_path=body.relative_path)
    if result.get("status") == "error":
        raise HTTPException(status_code=404, detail=result.get("message"))
    return result


@router.post("/api/arxiv/search")
def search_arxiv(body: ArxivSearchRequest):
    """Search arXiv API directly (live network call)."""
    client = get_client()
    results = client.search_papers(
        query=body.query,
        max_results=body.max_results,
        date_from=body.date_from or None,
        date_to=body.date_to or None,
        category=body.category or None,
    )
    # search_papers returns a list (not a dict with status)
    if results and isinstance(results[0], dict) and "error" in results[0]:
        raise HTTPException(status_code=502, detail=results[0]["error"])
    return {"status": "success", "results": results, "count": len(results)}

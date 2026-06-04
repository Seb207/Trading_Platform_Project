"""
Paper endpoints:
  GET  /api/papers/list
  POST /api/papers/search/abstract
  POST /api/papers/search/section
  POST /api/papers/analyze
  POST /api/arxiv/search
  POST /api/papers/download          ← SSE streaming download + auto-index
"""
import asyncio
import json
import json as _json
import time
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from pathlib import Path
from typing import AsyncGenerator

from backend.config import DOWNLOAD_DIR
from backend.modules.research.arxiv_bridge import get_client

router = APIRouter(tags=["papers"])

# ── arXiv search result cache (TTL = 10 min) ───────────────────────────
_ARXIV_CACHE: dict[str, tuple[float, list]] = {}
_ARXIV_CACHE_TTL = 600  # seconds

def _cache_key(query: str, category: str, date_from: str, date_to: str, max_results: int) -> str:
    return f"{query}|{category}|{date_from}|{date_to}|{max_results}"

def _cache_get(key: str) -> list | None:
    if key in _ARXIV_CACHE:
        ts, results = _ARXIV_CACHE[key]
        if time.time() - ts < _ARXIV_CACHE_TTL:
            return results
        del _ARXIV_CACHE[key]
    return None

def _cache_set(key: str, results: list) -> None:
    _ARXIV_CACHE[key] = (time.time(), results)


# ── arXiv direct API call (bypasses arxiv_client.py for full control) ───
import requests as _requests
import urllib.parse as _urlparse

_ARXIV_API_URL    = "https://export.arxiv.org/api/query"  # HTTPS directly
_ARXIV_MIN_INTERVAL: float = 3.0
_ARXIV_TIMEOUT:    int = 60          # seconds per attempt
_last_arxiv_call:  float = 0.0


def _search_arxiv_with_retry(
    client,
    *,
    query: str,
    max_results: int,
    date_from: str | None,
    date_to: str | None,
    category: str | None,
    max_retries: int = 3,
    base_delay: float = 30.0,
) -> list:
    """
    Hit the arXiv API directly (HTTPS, 60 s timeout) with:
      - minimum 3 s gap between calls
      - exponential-backoff retry on 429 / 503 / timeout
    Parsing is delegated to client._parse_entries() so we stay DRY.
    """
    global _last_arxiv_call

    # Build query string (same logic as arxiv_client.search_papers)
    parts = []
    if query:
        parts.append(f"all:{query}")
    if category:
        parts.append(f"cat:{category}")
    if not parts:
        return [{"error": "query or category is required."}]

    search_query = " AND ".join(parts)

    if date_from or date_to:
        def _norm(d: str) -> str:
            return d.replace("-", "").replace("/", "")
        from datetime import datetime as _dt
        df = _norm(date_from) if date_from else "00000000"
        dt = _norm(date_to)   if date_to   else _dt.now().strftime("%Y%m%d")
        search_query += f" AND submittedDate:[{df}0000 TO {dt}2359]"

    params = {
        "search_query": search_query,
        "start": 0,
        "max_results": max_results,
        "sortBy": "submittedDate" if (date_from or date_to) else "relevance",
        "sortOrder": "descending",
    }
    url = f"{_ARXIV_API_URL}?{_urlparse.urlencode(params)}"

    for attempt in range(max_retries):
        # ── rate limit ───────────────────────────────────────────────
        elapsed = time.time() - _last_arxiv_call
        if elapsed < _ARXIV_MIN_INTERVAL:
            time.sleep(_ARXIV_MIN_INTERVAL - elapsed)
        _last_arxiv_call = time.time()

        _headers = {
            "User-Agent": "QuantResearchDashboard/1.0 (academic research; https://github.com/local)",
            "Accept": "application/xml",
        }
        try:
            resp = _requests.get(url, timeout=_ARXIV_TIMEOUT, headers=_headers)
            if resp.status_code in (429, 503):
                raise _requests.HTTPError(response=resp)
            resp.raise_for_status()
            return client._parse_entries(resp.text)

        except (_requests.Timeout, _requests.ConnectionError, _requests.HTTPError) as exc:
            code = getattr(getattr(exc, "response", None), "status_code", 0)
            is_last = attempt == max_retries - 1
            if is_last:
                kind = f"HTTP {code}" if code else type(exc).__name__
                return [{"error": f"Failed to fetch data from arXiv API: {kind} after {max_retries} attempts"}]
            wait = base_delay * (2 ** attempt)   # 30 s → 60 s
            print(f"[arXiv] {type(exc).__name__} (code={code}) — retrying in {wait:.0f}s "
                  f"(attempt {attempt + 1}/{max_retries})…")
            time.sleep(wait)

    return [{"error": "arXiv API: max retries exceeded"}]


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


class DownloadRequest(BaseModel):
    arxiv_ids: list[str]
    category: str = "Unknown"
    auto_index: bool = True


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
    """Search arXiv API directly. Results are cached for 10 minutes."""
    key = _cache_key(body.query, body.category, body.date_from, body.date_to, body.max_results)

    # Return cached result if fresh
    cached = _cache_get(key)
    if cached is not None:
        return {"status": "success", "results": cached, "count": len(cached), "cached": True}

    client = get_client()
    results = _search_arxiv_with_retry(
        client,
        query=body.query,
        max_results=body.max_results,
        date_from=body.date_from or None,
        date_to=body.date_to or None,
        category=body.category or None,
    )

    if results and isinstance(results[0], dict) and "error" in results[0]:
        error_msg = results[0]["error"]
        if "429" in error_msg:
            status_code = 429
        elif "503" in error_msg or "502" in error_msg:
            status_code = 503
        else:
            status_code = 502
        raise HTTPException(status_code=status_code, detail=error_msg)

    _cache_set(key, results)   # store on success
    return {"status": "success", "results": results, "count": len(results), "cached": False}


# ── Download + auto-index (SSE) ─────────────────────────────────────────

def _sse(data: dict) -> str:
    return f"data: {_json.dumps(data, ensure_ascii=False)}\n\n"


async def _download_stream(req: DownloadRequest) -> AsyncGenerator[str, None]:
    client = get_client()
    total = len(req.arxiv_ids)
    downloaded = 0
    failed = 0

    for i, arxiv_id in enumerate(req.arxiv_ids):
        yield _sse({"type": "paper_start", "arxiv_id": arxiv_id, "index": i + 1, "total": total})
        try:
            result = await asyncio.to_thread(client.download_paper, arxiv_id, req.category)
            if result.get("status") == "success":
                downloaded += 1
                yield _sse({
                    "type": "paper_done",
                    "arxiv_id": arxiv_id,
                    "format": result.get("format", "unknown"),
                    "message": result.get("message", ""),
                })
            else:
                failed += 1
                yield _sse({
                    "type": "paper_error",
                    "arxiv_id": arxiv_id,
                    "message": result.get("message", "Download failed"),
                })
        except Exception as exc:
            failed += 1
            yield _sse({"type": "paper_error", "arxiv_id": arxiv_id, "message": str(exc)})

    if req.auto_index and downloaded > 0:
        # Step 1 — backfill metadata
        yield _sse({"type": "step", "name": "backfill", "message": "Updating metadata.json…"})
        try:
            r = await asyncio.to_thread(client.backfill_metadata)
            yield _sse({"type": "step_done", "name": "backfill", **r})
        except Exception as exc:
            yield _sse({"type": "step_error", "name": "backfill", "message": str(exc)})

        # Step 2 — abstract embeddings
        yield _sse({"type": "step", "name": "abstract_index", "message": "Building abstract search index…"})
        try:
            r = await asyncio.to_thread(client.build_search_index)
            yield _sse({"type": "step_done", "name": "abstract_index", **r})
        except Exception as exc:
            yield _sse({"type": "step_error", "name": "abstract_index", "message": str(exc)})

        # Step 3 — section embeddings
        yield _sse({"type": "step", "name": "section_index", "message": "Building section index…"})
        try:
            r = await asyncio.to_thread(client.build_section_index)
            yield _sse({"type": "step_done", "name": "section_index", **r})
        except Exception as exc:
            yield _sse({"type": "step_error", "name": "section_index", "message": str(exc)})

    yield _sse({"type": "done", "downloaded": downloaded, "failed": failed})


@router.post("/api/papers/download")
async def download_papers(req: DownloadRequest):
    """
    Download one or more arXiv papers then optionally rebuild the
    metadata + ChromaDB search indices.  Streams SSE progress events.
    """
    if not req.arxiv_ids:
        raise HTTPException(status_code=400, detail="arxiv_ids must not be empty")

    return StreamingResponse(
        _download_stream(req),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )

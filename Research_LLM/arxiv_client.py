import json
import os
import re
import sys
import xml.etree.ElementTree as ET
import urllib.parse
import urllib.request
from datetime import datetime
from pathlib import Path

import requests
from bs4 import BeautifulSoup
from markdownify import markdownify as md


class ArxivToolClient:
    def __init__(self, download_dir: str):
        self.download_dir = download_dir
        if not os.path.exists(self.download_dir):
            os.makedirs(self.download_dir, exist_ok=True)

        self.api_url = "http://export.arxiv.org/api/query"
        self.html_url_base = "https://arxiv.org/html"
        self.pdf_url_base = "https://arxiv.org/pdf"

    # ==================================================================
    # Internal helpers
    # ==================================================================

    def _safe_resolve_local_path(self, relative_path: str) -> Path:
        """Resolve a user-provided local path safely under download_dir."""
        base = Path(self.download_dir).resolve()
        candidate = (base / relative_path).resolve()
        try:
            candidate.relative_to(base)
        except ValueError as exc:
            raise ValueError("Path traversal is not allowed.") from exc
        return candidate

    @staticmethod
    def _normalize_date(date_str: str) -> str:
        """Normalize YYYY-MM-DD or YYYYMMDD to YYYYMMDD for arXiv API."""
        return date_str.replace("-", "").replace("/", "")

    def _extract_sections(self, content: str) -> list[dict]:
        """Parse Markdown headers (#/##/###) to extract paper sections."""
        lines = content.split("\n")
        sections = []
        current_title = None
        current_lines: list[str] = []

        for line in lines:
            if re.match(r"^#{1,3} ", line):
                if current_title is not None:
                    body = "\n".join(current_lines).strip()
                    sections.append(
                        {"title": current_title, "content": body, "char_count": len(body)}
                    )
                current_title = re.sub(r"^#+\s*", "", line).strip()
                current_lines = []
            else:
                current_lines.append(line)

        if current_title is not None:
            body = "\n".join(current_lines).strip()
            sections.append(
                {"title": current_title, "content": body, "char_count": len(body)}
            )

        return sections

    # ------------------------------------------------------------------
    # Phase 3 — Metadata helpers
    # ------------------------------------------------------------------

    def _metadata_path(self) -> Path:
        return Path(self.download_dir) / "metadata.json"

    def _load_metadata(self) -> dict:
        """Load metadata.json; returns empty dict if file does not exist."""
        path = self._metadata_path()
        if not path.exists():
            return {}
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _save_metadata(self, metadata: dict) -> None:
        """Persist metadata dict to metadata.json."""
        path = self._metadata_path()
        with open(path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

    def _upsert_paper_metadata(
        self,
        arxiv_id: str,
        title: str,
        authors: list[str],
        published: str,
        summary: str,
        category: str,
        fmt: str,
        relative_path: str,
    ) -> None:
        """Insert or update a single paper record in metadata.json."""
        metadata = self._load_metadata()
        metadata[arxiv_id] = {
            "arxiv_id": arxiv_id,
            "title": title,
            "authors": authors,
            "published": published,
            "summary": summary,
            "category": category,
            "format": fmt,
            "relative_path": relative_path,
            "indexed_at": datetime.now().isoformat(timespec="seconds"),
        }
        self._save_metadata(metadata)

    # ------------------------------------------------------------------
    # Phase 4 — ChromaDB helpers
    # ------------------------------------------------------------------

    def _chroma_dir(self) -> Path:
        return Path(self.download_dir) / ".chroma"

    def _get_chroma_collection(self):
        """Return (or create) the persistent ChromaDB collection for abstract search."""
        try:
            import chromadb
            from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
        except ImportError:
            raise ImportError(
                "Phase 4 requires additional packages. "
                "Run: pip install chromadb sentence-transformers"
            )

        chroma_dir = self._chroma_dir()
        chroma_dir.mkdir(parents=True, exist_ok=True)

        client = chromadb.PersistentClient(path=str(chroma_dir))
        ef = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")

        return client.get_or_create_collection(
            name="arxiv_papers",
            embedding_function=ef,
            metadata={"hnsw:space": "cosine"},
        )

    # ==================================================================
    # Local paper tools
    # ==================================================================

    def list_local_papers(self, category: str | None = None, limit: int = 200) -> dict:
        """List downloaded local arXiv papers under the download directory."""
        base = Path(self.download_dir)
        if not base.exists():
            return {"status": "success", "papers": [], "count": 0}

        limit = max(1, min(limit, 1000))
        allowed_suffixes = {".md", ".pdf"}
        papers = []

        for file_path in sorted(base.rglob("*")):
            if not file_path.is_file() or file_path.suffix.lower() not in allowed_suffixes:
                continue

            rel_path = file_path.relative_to(base)
            inferred_category = rel_path.parts[0] if len(rel_path.parts) > 1 else "Unknown"
            if category and inferred_category != category:
                continue

            stat = file_path.stat()
            papers.append(
                {
                    "relative_path": str(rel_path),
                    "file_name": file_path.name,
                    "arxiv_id": file_path.stem,
                    "format": file_path.suffix.lower().lstrip("."),
                    "category": inferred_category,
                    "size_bytes": stat.st_size,
                    "modified_at": datetime.fromtimestamp(stat.st_mtime).isoformat(timespec="seconds"),
                }
            )

            if len(papers) >= limit:
                break

        return {"status": "success", "papers": papers, "count": len(papers)}

    def read_local_paper(self, relative_path: str, max_chars: int = 50000, offset: int = 0) -> dict:
        """Read a local downloaded paper by relative path with safe bounds and pagination."""
        if not relative_path:
            return {"status": "error", "message": "relative_path is required."}

        max_chars = max(200, min(max_chars, 200000))
        offset = max(0, offset)

        try:
            target_path = self._safe_resolve_local_path(relative_path)
        except ValueError as exc:
            return {"status": "error", "message": str(exc)}

        if not target_path.exists() or not target_path.is_file():
            return {"status": "error", "message": f"File not found: {relative_path}"}

        suffix = target_path.suffix.lower()
        if suffix == ".pdf":
            return {
                "status": "error",
                "message": "PDF text extraction is not supported yet. Please read a Markdown (.md) paper.",
                "relative_path": str(Path(relative_path)),
            }
        if suffix != ".md":
            return {
                "status": "error",
                "message": f"Unsupported file format: {suffix}. Supported formats: .md, .pdf",
                "relative_path": str(Path(relative_path)),
            }

        with open(target_path, "r", encoding="utf-8", errors="replace") as handle:
            content = handle.read()

        total_chars = len(content)
        excerpt = content[offset: offset + max_chars]

        return {
            "status": "success",
            "relative_path": str(Path(relative_path)),
            "file_path": str(target_path),
            "total_chars": total_chars,
            "offset": offset,
            "returned_chars": len(excerpt),
            "has_more": offset + len(excerpt) < total_chars,
            "content": excerpt,
        }

    def analyze_local_paper(self, relative_path: str) -> dict:
        """
        Structurally analyze a local Markdown paper.

        Returns the full paper content alongside a section map so an LLM can
        understand the paper's structure and generate grounded quant strategy code.
        """
        if not relative_path:
            return {"status": "error", "message": "relative_path is required."}

        try:
            target_path = self._safe_resolve_local_path(relative_path)
        except ValueError as exc:
            return {"status": "error", "message": str(exc)}

        if not target_path.exists() or not target_path.is_file():
            return {"status": "error", "message": f"File not found: {relative_path}"}

        if target_path.suffix.lower() != ".md":
            return {
                "status": "error",
                "message": "Only .md files are supported for analysis. Download the HTML version first.",
                "relative_path": str(Path(relative_path)),
            }

        with open(target_path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()

        rel = Path(relative_path)
        arxiv_id = target_path.stem
        category = rel.parts[0] if len(rel.parts) > 1 else "Unknown"
        sections = self._extract_sections(content)
        section_map = [{"title": s["title"], "char_count": s["char_count"]} for s in sections]

        return {
            "status": "success",
            "relative_path": str(rel),
            "arxiv_id": arxiv_id,
            "category": category,
            "total_chars": len(content),
            "section_count": len(sections),
            "section_map": section_map,
            "sections": sections,
            "full_content": content,
        }

    # ==================================================================
    # arXiv API — core
    # ==================================================================

    def _parse_entries(self, xml_data: str) -> list[dict]:
        """Parse arXiv Atom XML into a list of paper dicts (shared by search + batch fetch)."""
        ns = {"atom": "http://www.w3.org/2005/Atom"}
        root = ET.fromstring(xml_data)
        papers = []

        for entry in root.findall("atom:entry", ns):
            id_url = entry.find("atom:id", ns).text
            raw_id = id_url.split("/abs/")[-1] if "/abs/" in id_url else id_url.split("/")[-1]
            arxiv_id = re.sub(r"v\d+$", "", raw_id)

            title = entry.find("atom:title", ns).text.replace("\n", " ").strip()
            summary = entry.find("atom:summary", ns).text.replace("\n", " ").strip()
            published = entry.find("atom:published", ns).text
            authors = [a.find("atom:name", ns).text for a in entry.findall("atom:author", ns)]

            primary_category = entry.find(
                "arxiv:primary_category", {"arxiv": "http://arxiv.org/schemas/atom"}
            )
            category = (
                primary_category.attrib.get("term", "Unknown")
                if primary_category is not None
                else "Unknown"
            )

            papers.append(
                {
                    "arxiv_id": arxiv_id,
                    "title": title,
                    "authors": authors,
                    "published": published,
                    "summary": summary,
                    "category": category,
                }
            )

        return papers

    def fetch_paper_metadata_batch(self, arxiv_ids: list[str]) -> list[dict]:
        """
        Fetch full metadata for a list of arxiv IDs in a single API call.
        Used internally by backfill_metadata.
        """
        if not arxiv_ids:
            return []

        params = {
            "id_list": ",".join(arxiv_ids),
            "max_results": len(arxiv_ids),
        }
        url = f"{self.api_url}?{urllib.parse.urlencode(params)}"

        try:
            with urllib.request.urlopen(url) as response:
                xml_data = response.read().decode("utf-8")
        except Exception as e:
            return [{"error": f"API request failed: {str(e)}"}]

        return self._parse_entries(xml_data)

    def search_papers(
        self,
        query: str,
        max_results: int = 5,
        date_from: str | None = None,
        date_to: str | None = None,
    ) -> list[dict]:
        """
        Search arXiv for papers using the official API.

        date_from / date_to accept YYYY-MM-DD or YYYYMMDD format.
        """
        search_query = f"all:{query}"

        if date_from or date_to:
            df = self._normalize_date(date_from) if date_from else "00000000"
            dt = self._normalize_date(date_to) if date_to else "99991231"
            search_query += f" AND submittedDate:[{df}0000 TO {dt}2359]"

        params = {
            "search_query": search_query,
            "start": 0,
            "max_results": max_results,
            "sortBy": "submittedDate" if (date_from or date_to) else "relevance",
            "sortOrder": "descending",
        }
        url = f"{self.api_url}?{urllib.parse.urlencode(params)}"

        try:
            with urllib.request.urlopen(url) as response:
                xml_data = response.read().decode("utf-8")
        except Exception as e:
            return [{"error": f"Failed to fetch data from arXiv API: {str(e)}"}]

        return self._parse_entries(xml_data)

    def download_paper(self, arxiv_id: str, category: str = "Unknown") -> dict:
        """
        Download an arXiv paper (HTML→Markdown preferred, PDF fallback).
        Automatically saves metadata to metadata.json after a successful download.
        """
        safe_category = category.replace("/", "_").replace("\\", "_")
        category_dir = os.path.join(self.download_dir, safe_category)
        os.makedirs(category_dir, exist_ok=True)

        # Step 1: Try HTML
        html_url = f"{self.html_url_base}/{arxiv_id}"
        try:
            response = requests.get(html_url, timeout=10)
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, "html.parser")

                # Strip navigation and boilerplate before converting
                for tag in soup.find_all(["nav", "header", "footer", "script", "style", "aside"]):
                    tag.decompose()

                article_content = (
                    soup.find("article")
                    or soup.find("div", class_="ltx_document")
                    or soup.find("main")
                    or soup.body
                )

                if article_content:
                    md_text = md(str(article_content), heading_style="ATX")
                    file_path = os.path.join(category_dir, f"{arxiv_id}.md")
                    with open(file_path, "w", encoding="utf-8") as f:
                        f.write(md_text)

                    rel_path = str(Path(file_path).relative_to(self.download_dir))
                    self._fetch_and_save_metadata(arxiv_id, category, "md", rel_path)

                    return {
                        "status": "success",
                        "format": "markdown",
                        "file_path": file_path,
                        "category": category,
                        "message": f"Successfully downloaded and converted to Markdown: {file_path}",
                    }
        except Exception as e:
            print(f"HTML fetch failed for {arxiv_id}: {str(e)}")

        # Step 2: PDF fallback
        pdf_url = f"{self.pdf_url_base}/{arxiv_id}.pdf"
        try:
            response = requests.get(pdf_url, stream=True, timeout=20)
            if response.status_code == 200:
                file_path = os.path.join(category_dir, f"{arxiv_id}.pdf")
                with open(file_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)

                rel_path = str(Path(file_path).relative_to(self.download_dir))
                self._fetch_and_save_metadata(arxiv_id, category, "pdf", rel_path)

                return {
                    "status": "success",
                    "format": "pdf",
                    "file_path": file_path,
                    "category": category,
                    "message": f"HTML not available. Successfully downloaded PDF fallback: {file_path}",
                }
            else:
                return {
                    "status": "error",
                    "message": f"Failed to download PDF. HTTP Status: {response.status_code}",
                }
        except Exception as e:
            return {
                "status": "error",
                "message": f"Exception occurred while downloading PDF: {str(e)}",
            }

    def _fetch_and_save_metadata(
        self, arxiv_id: str, category: str, fmt: str, relative_path: str
    ) -> None:
        """Fetch paper metadata from arXiv API and persist to metadata.json."""
        try:
            papers = self.fetch_paper_metadata_batch([arxiv_id])
            if papers and "error" not in papers[0]:
                p = papers[0]
                self._upsert_paper_metadata(
                    arxiv_id=arxiv_id,
                    title=p.get("title", ""),
                    authors=p.get("authors", []),
                    published=p.get("published", ""),
                    summary=p.get("summary", ""),
                    category=category,
                    fmt=fmt,
                    relative_path=relative_path,
                )
        except Exception as e:
            print(f"[metadata] Could not save metadata for {arxiv_id}: {e}")

    # ==================================================================
    # Phase 3 — Bulk download + Backfill
    # ==================================================================

    def bulk_download_papers(
        self, arxiv_ids: list[str], category: str = "Unknown"
    ) -> list[dict]:
        """
        Download multiple arXiv papers at once.
        Each paper is downloaded and its metadata saved automatically.

        Returns a result list with one entry per arxiv_id.
        """
        results = []
        for arxiv_id in arxiv_ids:
            result = self.download_paper(arxiv_id, category)
            results.append({"arxiv_id": arxiv_id, **result})
        return results

    def backfill_metadata(self) -> dict:
        """
        Scan all locally downloaded papers and populate metadata.json
        for any entries that are missing.

        Works for both newly downloaded and pre-existing files.
        Uses batched arXiv API calls (50 IDs per request) for efficiency.
        """
        existing = self._load_metadata()
        base = Path(self.download_dir)

        # Collect arxiv_ids whose metadata is missing
        missing_ids: list[str] = []
        id_to_file_info: dict[str, dict] = {}

        for file_path in sorted(base.rglob("*")):
            if not file_path.is_file():
                continue
            if file_path.suffix.lower() not in {".md", ".pdf"}:
                continue

            arxiv_id = file_path.stem
            if arxiv_id in existing:
                continue

            rel_path = file_path.relative_to(base)
            missing_ids.append(arxiv_id)
            id_to_file_info[arxiv_id] = {
                "relative_path": str(rel_path),
                "format": file_path.suffix.lower().lstrip("."),
                "category": rel_path.parts[0] if len(rel_path.parts) > 1 else "Unknown",
            }

        if not missing_ids:
            return {
                "status": "success",
                "message": "All papers already have metadata.",
                "backfilled": 0,
                "total_in_metadata": len(existing),
            }

        # Batch fetch from arXiv API (50 IDs per call)
        BATCH_SIZE = 50
        backfilled = 0
        failed: list[str] = []

        for i in range(0, len(missing_ids), BATCH_SIZE):
            batch = missing_ids[i: i + BATCH_SIZE]
            papers = self.fetch_paper_metadata_batch(batch)

            for paper in papers:
                if "error" in paper:
                    failed.extend(batch)
                    break

                arxiv_id = paper["arxiv_id"]
                file_info = id_to_file_info.get(arxiv_id, {})

                self._upsert_paper_metadata(
                    arxiv_id=arxiv_id,
                    title=paper.get("title", ""),
                    authors=paper.get("authors", []),
                    published=paper.get("published", ""),
                    summary=paper.get("summary", ""),
                    category=file_info.get("category", paper.get("category", "Unknown")),
                    fmt=file_info.get("format", "md"),
                    relative_path=file_info.get("relative_path", ""),
                )
                backfilled += 1

        return {
            "status": "success",
            "backfilled": backfilled,
            "failed": failed,
            "total_in_metadata": len(self._load_metadata()),
        }

    # ==================================================================
    # Phase 4 — Semantic search index
    # ==================================================================

    def build_search_index(self) -> dict:
        """
        Build (or refresh) the ChromaDB semantic search index from metadata.json.

        Embeds each paper's abstract using a local sentence-transformers model
        (all-MiniLM-L6-v2, ~80 MB, downloaded once on first use).
        Subsequent calls upsert only — already-indexed papers are not re-embedded.

        Run backfill_metadata first if you have pre-existing papers.
        """
        metadata = self._load_metadata()

        if not metadata:
            return {
                "status": "error",
                "message": "metadata.json is empty. Run backfill_metadata first.",
            }

        try:
            collection = self._get_chroma_collection()
        except ImportError as e:
            return {"status": "error", "message": str(e)}

        ids, documents, metadatas = [], [], []

        for arxiv_id, info in metadata.items():
            summary = info.get("summary", "").strip()
            if not summary:
                continue

            ids.append(arxiv_id)
            documents.append(summary)
            metadatas.append(
                {
                    "title": info.get("title", ""),
                    "category": info.get("category", ""),
                    "published": info.get("published", ""),
                    "relative_path": info.get("relative_path", ""),
                    "format": info.get("format", ""),
                }
            )

        if not ids:
            return {
                "status": "error",
                "message": "No papers with abstracts found in metadata.",
            }

        collection.upsert(ids=ids, documents=documents, metadatas=metadatas)

        return {
            "status": "success",
            "indexed": len(ids),
            "total_in_collection": collection.count(),
        }

    def search_local_papers_by_topic(
        self,
        query: str,
        n_results: int = 5,
        category: str = "",
    ) -> dict:
        """
        Semantic search over locally downloaded papers using abstract embeddings.

        Returns papers ranked by semantic similarity to the query.
        similarity_score is 0–1 (higher = more similar).

        Requires build_search_index to have been run at least once.
        """
        if not query:
            return {"status": "error", "message": "query is required."}

        try:
            collection = self._get_chroma_collection()
        except ImportError as e:
            return {"status": "error", "message": str(e)}

        if collection.count() == 0:
            return {
                "status": "error",
                "message": "Search index is empty. Run build_search_index first.",
            }

        n_results = max(1, min(n_results, collection.count()))
        where = {"category": category} if category.strip() else None

        try:
            results = collection.query(
                query_texts=[query],
                n_results=n_results,
                where=where,
                include=["documents", "metadatas", "distances"],
            )
        except Exception as e:
            return {"status": "error", "message": f"Search failed: {str(e)}"}

        papers = []
        for i in range(len(results["ids"][0])):
            papers.append(
                {
                    "arxiv_id": results["ids"][0][i],
                    "title": results["metadatas"][0][i].get("title", ""),
                    "category": results["metadatas"][0][i].get("category", ""),
                    "published": results["metadatas"][0][i].get("published", ""),
                    "relative_path": results["metadatas"][0][i].get("relative_path", ""),
                    "format": results["metadatas"][0][i].get("format", ""),
                    "abstract": results["documents"][0][i],
                    "similarity_score": round(1 - results["distances"][0][i], 4),
                }
            )

        return {
            "status": "success",
            "query": query,
            "results": papers,
            "count": len(papers),
        }

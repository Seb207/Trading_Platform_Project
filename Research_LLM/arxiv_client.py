import os
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

    def _safe_resolve_local_path(self, relative_path: str) -> Path:
        """Resolve a user-provided local path safely under download_dir."""
        base = Path(self.download_dir).resolve()
        candidate = (base / relative_path).resolve()
        try:
            candidate.relative_to(base)
        except ValueError as exc:
            raise ValueError("Path traversal is not allowed.") from exc
        return candidate

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

    def read_local_paper(self, relative_path: str, max_chars: int = 12000, offset: int = 0) -> dict:
        """Read a local downloaded paper by relative path with safe bounds and truncation."""
        if not relative_path:
            return {"status": "error", "message": "relative_path is required."}

        max_chars = max(200, min(max_chars, 50000))
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

    def search_papers(self, query: str, max_results: int = 5) -> list[dict]:
        """Search arXiv for papers using the official API."""
        params = {
            "search_query": f"all:{query}",
            "start": 0,
            "max_results": max_results,
            "sortBy": "relevance",
            "sortOrder": "descending",
        }
        url = f"{self.api_url}?{urllib.parse.urlencode(params)}"
        
        try:
            with urllib.request.urlopen(url) as response:
                xml_data = response.read().decode('utf-8')
        except Exception as e:
            return [{"error": f"Failed to fetch data from arXiv API: {str(e)}"}]

        ns = {"atom": "http://www.w3.org/2005/Atom"}
        root = ET.fromstring(xml_data)
        papers = []
        
        for entry in root.findall("atom:entry", ns):
            id_url = entry.find("atom:id", ns).text
            arxiv_id = id_url.split('/abs/')[-1].split('v')[0] if '/abs/' in id_url else id_url.split('/')[-1]
            
            title = entry.find("atom:title", ns).text.replace('\n', ' ').strip()
            summary = entry.find("atom:summary", ns).text.replace('\n', ' ').strip()
            published = entry.find("atom:published", ns).text
            authors = [author.find("atom:name", ns).text for author in entry.findall("atom:author", ns)]
            
            # Extract primary category to use for folder grouping
            primary_category = entry.find("arxiv:primary_category", {"arxiv": "http://arxiv.org/schemas/atom"})
            category = primary_category.attrib.get('term', 'Unknown') if primary_category is not None else 'Unknown'
            
            papers.append({
                "arxiv_id": arxiv_id,
                "title": title,
                "authors": authors,
                "published": published,
                "summary": summary,
                "category": category
            })
            
        return papers

    def download_paper(self, arxiv_id: str, category: str = "Unknown") -> dict:
        """
        Attempt to download HTML version first and save as Markdown.
        Fallback to PDF if HTML is not available.
        Saves the file in a subfolder based on the paper's category (e.g., q-fin.CP).
        """
        # Create category-specific subfolder
        # Replace invalid folder characters just in case, though arXiv categories are usually safe
        safe_category = category.replace('/', '_').replace('\\', '_')
        category_dir = os.path.join(self.download_dir, safe_category)
        os.makedirs(category_dir, exist_ok=True)
        
        # Step 1: Try HTML
        html_url = f"{self.html_url_base}/{arxiv_id}"
        try:
            response = requests.get(html_url, timeout=10)
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                # Try to extract the main article content (arXiv HTML often uses <article> or ltx_document)
                article_content = soup.find('article')
                if not article_content:
                    article_content = soup.find('div', class_='ltx_document')
                if not article_content:
                    article_content = soup.body
                
                if article_content:
                    # Convert HTML to Markdown
                    md_text = md(str(article_content), heading_style="ATX")
                    file_path = os.path.join(category_dir, f"{arxiv_id}.md")
                    
                    with open(file_path, "w", encoding="utf-8") as f:
                        f.write(md_text)
                        
                    return {
                        "status": "success",
                        "format": "markdown",
                        "file_path": file_path,
                        "category": category,
                        "message": f"Successfully downloaded and converted to Markdown: {file_path}"
                    }
        except Exception as e:
            print(f"HTML fetch failed for {arxiv_id}: {str(e)}")

        # Step 2: Fallback to PDF
        pdf_url = f"{self.pdf_url_base}/{arxiv_id}.pdf"
        try:
            response = requests.get(pdf_url, stream=True, timeout=20)
            if response.status_code == 200:
                file_path = os.path.join(category_dir, f"{arxiv_id}.pdf")
                with open(file_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                        
                return {
                    "status": "success",
                    "format": "pdf",
                    "file_path": file_path,
                    "category": category,
                    "message": f"HTML not available. Successfully downloaded PDF fallback: {file_path}"
                }
            else:
                return {
                    "status": "error",
                    "message": f"Failed to download PDF. HTTP Status: {response.status_code}"
                }
        except Exception as e:
            return {
                "status": "error",
                "message": f"Exception occurred while downloading PDF: {str(e)}"
            }

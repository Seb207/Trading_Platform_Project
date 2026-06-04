"""
GET /api/status  →  paper counts + index state
"""
from fastapi import APIRouter
from pathlib import Path
import json

from backend.config import DOWNLOAD_DIR
from backend.modules.research.arxiv_bridge import get_client

router = APIRouter(prefix="/api/status", tags=["status"])


def _get_chroma_counts(client) -> tuple[int, int, bool, bool]:
    """Return (abstract_count, section_count, abstract_built, section_built)."""
    try:
        import chromadb
        from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

        chroma_dir = Path(DOWNLOAD_DIR) / ".chroma"
        if not chroma_dir.exists():
            return 0, 0, False, False

        chroma_client = chromadb.PersistentClient(path=str(chroma_dir))
        existing = {col.name for col in chroma_client.list_collections()}

        abstract_count = 0
        section_count  = 0

        if "arxiv_papers" in existing:
            abstract_count = chroma_client.get_collection("arxiv_papers").count()

        if "paper_sections" in existing:
            section_count = chroma_client.get_collection("paper_sections").count()

        return (
            abstract_count,
            section_count,
            abstract_count > 0,
            section_count > 0,
        )
    except Exception:
        return 0, 0, False, False


@router.get("")
def get_status():
    client = get_client()
    download_path = Path(DOWNLOAD_DIR)

    # Count local files
    md_files      = list(download_path.rglob("*.md"))
    pdf_files     = list(download_path.rglob("*.pdf"))
    papers_count  = len(md_files) + len(pdf_files)

    # Metadata count
    meta_path      = download_path / "metadata.json"
    metadata_count = 0
    if meta_path.exists():
        with open(meta_path, encoding="utf-8") as f:
            metadata_count = len(json.load(f))

    # ChromaDB index state (query collections directly)
    abstract_count, section_count, abstract_built, section_built = _get_chroma_counts(client)

    return {
        "papers_count":         papers_count,
        "abstract_index_count": abstract_count,
        "sections_count":       section_count,
        "metadata_count":       metadata_count,
        "abstract_index_built": abstract_built,
        "section_index_built":  section_built,
        "download_dir":         str(DOWNLOAD_DIR),
    }

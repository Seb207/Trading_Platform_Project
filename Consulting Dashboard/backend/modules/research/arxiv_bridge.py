"""
Bridge to Research_LLM/arxiv_client.py.

Adds Research_LLM/ to sys.path so ArxivToolClient can be imported
without modifying the original project.
"""
import sys
from pathlib import Path
from functools import lru_cache

# Ensure Research_LLM is on the path
_research_llm_dir = Path(__file__).parent.parent.parent.parent.parent / "Research_LLM"


def _add_research_llm_to_path() -> None:
    from backend.config import RESEARCH_LLM_DIR
    path_str = str(RESEARCH_LLM_DIR)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)


@lru_cache(maxsize=1)
def get_client():
    """Return a singleton ArxivToolClient instance."""
    _add_research_llm_to_path()
    from backend.config import DOWNLOAD_DIR
    from arxiv_client import ArxivToolClient  # type: ignore[import]
    return ArxivToolClient(download_dir=str(DOWNLOAD_DIR))

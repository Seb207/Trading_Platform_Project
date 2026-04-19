import json
import os
from mcp.server.fastmcp import FastMCP
from arxiv_client import ArxivToolClient

# Initialize MCP Server
mcp = FastMCP("ArxivResearchServer")

# Determine download directory (absolute path to avoid relative path issues)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DOWNLOAD_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "papers", "arXiv"))

# Initialize arXiv client
arxiv_client = ArxivToolClient(download_dir=DOWNLOAD_DIR)

@mcp.tool()
def search_arxiv_papers(query: str, max_results: int = 5) -> str:
    """
    Search arXiv for research papers based on a query.
    Returns a list of papers including their arxiv_id, title, authors, summary, and category.
    """
    results = arxiv_client.search_papers(query, max_results)
    return json.dumps(results, indent=2, ensure_ascii=False)

@mcp.tool()
def download_arxiv_paper(arxiv_id: str, category: str = "Unknown") -> str:
    """
    Download an arXiv paper to the local computer using its arxiv_id.
    It saves the paper inside a subfolder named after its category (e.g. q-fin.CP).
    It prioritizes downloading the HTML version and converting it to Markdown (.md).
    If HTML is not available, it falls back to downloading the PDF (.pdf).
    Returns the download status and the local file path.
    """
    result = arxiv_client.download_paper(arxiv_id, category)
    return json.dumps(result, indent=2, ensure_ascii=False)

@mcp.tool()
def list_local_papers(category: str = "", limit: int = 200) -> str:
    """
    List local downloaded arXiv papers stored in papers/arXiv.
    You can optionally filter by category (e.g. q-fin.CP).
    """
    normalized_category = category.strip() or None
    result = arxiv_client.list_local_papers(category=normalized_category, limit=limit)
    return json.dumps(result, indent=2, ensure_ascii=False)

@mcp.tool()
def read_local_paper(relative_path: str, max_chars: int = 12000, offset: int = 0) -> str:
    """
    Read a local paper by its relative path under papers/arXiv.
    Use this to summarize paper content or derive grounded strategy ideas.
    """
    result = arxiv_client.read_local_paper(relative_path=relative_path, max_chars=max_chars, offset=offset)
    return json.dumps(result, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    # Start the MCP server using standard input/output (stdio)
    print(f"Starting arXiv MCP Server. Downloads will be saved to: {DOWNLOAD_DIR}")
    mcp.run(transport='stdio')

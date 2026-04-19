# Research_LLM: arXiv MCP Server

A specialized local service that acts as a bridge between Large Language Models (LLMs) and the arXiv academic database. Built using the Model Context Protocol (MCP), it empowers AI assistants to autonomously search for papers, extract their full text (prioritizing HTML/Markdown for optimal LLM context), and save them to your local machine organized by research categories.

## Architecture & Design Choices

This project specifically diverges from traditional RAG (Retrieval-Augmented Generation) architectures that rely on chunking and Vector Databases. 
Instead, it leverages the massive context windows of modern LLMs (like Gemini 1.5 Pro or Claude 3.5 Sonnet) by downloading full papers as clean Markdown files. 
This allows the LLM to read the entire document at once, preserving the full context, tables, and mathematical formulas without the fragmentation issues common in PDF parsing.

### Key Features
1. **HTML-First Extraction:** It prioritizes fetching the HTML version of arXiv papers and converts them into pure Markdown (`.md`). This is the most efficient and native format for LLMs to digest.
2. **PDF Fallback:** If a paper is too old to have an HTML version, it automatically falls back to downloading the traditional PDF.
3. **Smart Categorization:** Downloads are automatically sorted into subfolders based on their arXiv primary category (e.g., `q-fin.CP`, `cs.AI`).
4. **MCP Integration:** Fully compatible with MCP clients (like Claude Desktop or VS Code Cline), allowing you to interact with the system entirely through natural language prompts.
5. **Local Paper Browsing:** The server can list downloaded local papers by category and metadata (`list_local_papers`).
6. **Local Paper Reading:** The server can read local Markdown papers with truncation/pagination controls (`read_local_paper`).

## Current Progress (Phase 1 & 2 Completed)

We have successfully rebuilt the core infrastructure and exposed it as an MCP tool.

### 1. Data Ingestion Core (`arxiv_client.py`)
- [x] Implemented `search_papers(query, max_results)` to fetch metadata (Title, Authors, Summary, Category) via the arXiv XML API.
- [x] Implemented `download_paper(arxiv_id, category)` using `requests` and `BeautifulSoup`.
- [x] Integrated `markdownify` to perfectly convert HTML papers into LLM-readable Markdown.
- [x] Added automated folder structuring based on research categories (e.g., `../papers/arXiv/[category]/`).
- [x] Added `list_local_papers(category, limit)` to inspect locally saved papers.
- [x] Added `read_local_paper(relative_path, max_chars, offset)` for safe local Markdown reading.

### 2. MCP Server Layer (`mcp_server.py`)
- [x] Utilized `FastMCP` to wrap the core python client into a standardized MCP server.
- [x] Exposed `search_arxiv_papers` tool for AI agents.
- [x] Exposed `download_arxiv_paper` tool for AI agents.
- [x] Exposed `list_local_papers` tool for AI agents.
- [x] Exposed `read_local_paper` tool for AI agents.
- [x] Set up standard Input/Output (`stdio`) transport for seamless integration with desktop clients.

### 3. Tool Workflow (No RAG / No LangGraph)
1. Use `search_arxiv_papers` to find candidate papers.
2. Use `download_arxiv_paper` to save them locally by category.
3. Use `list_local_papers` to inspect what is available on disk.
4. Use `read_local_paper` to fetch Markdown content for summarization or idea generation.

> Note: `read_local_paper` currently supports `.md` content reading. For `.pdf`, download works, but text extraction is not enabled yet.

## Project Structure

```text
Trading_Platform_Project/
│
├── papers/
│   └── arXiv/                  # Papers are downloaded here
│       ├── q-fin.CP/           # E.g., Markdown/PDFs for Computational Finance
│       └── cs.AI/              # E.g., Markdown/PDFs for Artificial Intelligence
│
└── Research_LLM/               # Main Application Directory
    ├── arxiv_client.py         # The core logic for API calls and HTML parsing
    ├── mcp_server.py           # The FastMCP server exposing tools to the LLM
    ├── tests/
    │   └── test_local_paper_tools.py # Unit tests for local list/read behaviors
    ├── requirements.txt        # Python dependencies
    ├── PROJECT_PLAN_HTML_ONLY.md # Historical planning document
    └── README.md               # This file
```

## Setup & Installation

1. Navigate to the project directory:
   ```bash
   cd Trading_Platform_Project/Research_LLM
   ```
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. (Optional) Run unit tests:
   ```bash
   cd /Users/ahnsebin/Documents/Personal\ Project/Quant/Trading_Platform_Project
   python3 -m unittest Research_LLM/tests/test_local_paper_tools.py
   ```

## How to Connect to an LLM

This MCP server can be connected to any MCP-compatible client. 

### Option 1: Claude Desktop (Recommended for Chat)
Add the following to your `claude_desktop_config.json`:
```json
{
  "mcpServers": {
    "arxiv-research": {
      "command": "python",
      "args": ["/ABSOLUTE_PATH_TO_YOUR_PROJECT/Trading_Platform_Project/Research_LLM/mcp_server.py"]
    }
  }
}
```

### Option 2: VS Code 'Cline' Extension (Recommended for Development)
Add the exact same JSON block shown above to Cline's `mcp.json` file. This allows you to use Gemini or GPT-4o API to control the tools directly inside VS Code.

## Next Steps / Functional Roadmap

### Immediate (Priority 1)
- [ ] Connect this MCP server to a client (Claude Desktop, Cline, or Gemini-compatible MCP client) and validate end-to-end flow:
  - `search_arxiv_papers` -> `download_arxiv_paper` -> `list_local_papers` -> `read_local_paper`
- [ ] Create a prompt test set (10-15 prompts) for repeatable checks:
  - paper search quality
  - local file read quality
  - grounded answers with paper evidence
- [ ] Add short usage examples in this README for common tasks:
  - "find recent market microstructure papers"
  - "download top 3 papers and summarize key ideas"

### Short-Term (Priority 2)
- [ ] Implement `analyze_local_paper(relative_path, question, max_chars)` MCP tool:
  - reads local `.md`
  - returns summary + direct evidence snippets
  - includes possible quant strategy ideas tied to the paper content
- [ ] Add `analyze_local_paper` unit tests for:
  - valid markdown analysis
  - missing file handling
  - empty/very long question handling

### Integration (Priority 3)
- [ ] Build a minimal `gemini_cli_runner.py` script to run Gemini with this MCP toolset from terminal.
- [ ] Add a small command cookbook for Korean/English prompts to speed up manual research workflows.

### Optional Later (Not Required for Current MVP)
- [ ] Enable PDF text extraction for `read_local_paper` fallback (for old papers without HTML markdown).
- [ ] Consider lightweight retrieval indexing only when local paper volume grows significantly.

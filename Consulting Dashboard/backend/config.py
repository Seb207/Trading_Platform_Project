"""
Central config — all paths and env vars in one place.
Override any value via environment variable.
"""
import os
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────
# Absolute path to Research_LLM/ (two levels up from backend/)
_BACKEND_DIR      = Path(__file__).parent.resolve()
_DASHBOARD_DIR    = _BACKEND_DIR.parent
_PROJECT_DIR      = _DASHBOARD_DIR.parent

RESEARCH_LLM_DIR  = Path(
    os.getenv("RESEARCH_LLM_DIR", str(_PROJECT_DIR / "Research_LLM"))
)

DOWNLOAD_DIR      = Path(
    os.getenv("DOWNLOAD_DIR", str(RESEARCH_LLM_DIR / "papers" / "arXiv"))
)

# Folder holding user-authored system-prompt fragments (.md / .txt).
# Anything dropped here is auto-injected into the chat system prompt.
PROMPTS_DIR       = Path(
    os.getenv("PROMPTS_DIR", str(_BACKEND_DIR / "prompts"))
)

# ── Server ─────────────────────────────────────────────────────────────
BACKEND_HOST      = os.getenv("BACKEND_HOST", "0.0.0.0")
BACKEND_PORT      = int(os.getenv("BACKEND_PORT", "8000"))

# Allowed CORS origins (Next.js dev + prod)
CORS_ORIGINS      = os.getenv(
    "CORS_ORIGINS",
    "http://localhost:3000,http://127.0.0.1:3000"
).split(",")

# ── LLM defaults ───────────────────────────────────────────────────────
DEFAULT_LLM_PROVIDER = os.getenv("DEFAULT_LLM_PROVIDER", "claude")
DEFAULT_CLAUDE_MODEL = os.getenv("DEFAULT_CLAUDE_MODEL", "claude-opus-4-5")
OLLAMA_BASE_URL      = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

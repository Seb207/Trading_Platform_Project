"""Load user-authored system-prompt fragments from the prompts/ folder.

Drop any number of .md / .txt files into backend/prompts/.  On every chat
request their contents are concatenated (sorted by filename) and injected
into the system prompt — no server restart needed, because files are read
fresh on each call.

Ordering: files are sorted by filename, so use numeric prefixes to control
the order they appear in the prompt, e.g.
    00_role.md
    10_alpha_dsl.md
    20_output_format.md

Skipped: hidden files (leading "."), README.md, and empty files.  To
temporarily disable a fragment, rename it to end in ".off" (or any non
.md/.txt extension) — it stays in the folder but is no longer injected.
"""
from pathlib import Path

from backend.config import PROMPTS_DIR

_PROMPT_PATTERNS = ("*.md", "*.txt")
_SKIP_NAMES = {"README.md", "readme.md"}


def load_prompt_fragments() -> str:
    """Return all prompt fragments concatenated, or "" if none exist.

    Read on every call so newly-added content takes effect immediately.
    """
    if not PROMPTS_DIR.exists():
        return ""

    files: list[Path] = []
    for pattern in _PROMPT_PATTERNS:
        files.extend(PROMPTS_DIR.glob(pattern))

    # Deterministic order by filename (numeric prefixes recommended).
    files = sorted(set(files), key=lambda p: p.name)

    parts: list[str] = []
    for f in files:
        if f.name.startswith(".") or f.name in _SKIP_NAMES:
            continue
        try:
            text = f.read_text(encoding="utf-8").strip()
        except OSError:
            continue
        if text:
            parts.append(text)

    return "\n\n".join(parts)


# ── Task-mode prompts (prompts/tasks/) ──────────────────────────────────
# These are NOT injected globally.  They are selected per request via the
# chat "Mode" selector and injected only when their id is chosen.

def _tasks_dir():
    return PROMPTS_DIR / "tasks"


def _label_for(path) -> str:
    """Human-readable label from the file's first '# ' heading, else its stem."""
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            s = line.strip()
            if s.startswith("# "):
                title = s[2:].strip()
                if title.lower().startswith("task:"):
                    title = title[5:].strip()
                return title
    except OSError:
        pass
    return path.stem.replace("_", " ").replace("-", " ").strip().capitalize()


def list_task_prompts() -> list[dict]:
    """Available task prompts as [{id, label}], sorted by filename.

    Read fresh each call, so dropping a new .md into prompts/tasks/ makes a
    new mode appear without a restart.
    """
    tasks_dir = _tasks_dir()
    if not tasks_dir.exists():
        return []
    out: list[dict] = []
    for f in sorted(tasks_dir.glob("*.md"), key=lambda p: p.name):
        if f.name in _SKIP_NAMES or f.name.startswith("."):
            continue
        out.append({"id": f.stem, "label": _label_for(f)})
    return out


def load_task_prompt(task: str) -> str:
    """Load one task prompt by id (filename stem). "" if empty/missing.

    The id is reduced to a bare filename to prevent path traversal.
    """
    if not task:
        return ""
    safe = Path(task).name  # strip any directory components
    f = _tasks_dir() / f"{safe}.md"
    if not f.exists():
        return ""
    try:
        return f.read_text(encoding="utf-8").strip()
    except OSError:
        return ""

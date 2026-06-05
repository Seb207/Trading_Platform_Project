"""
POST /api/chat  — SSE streaming chat endpoint.

Accepts a conversation + optional paper context,
streams tokens back as Server-Sent Events.
"""
import json
from typing import Optional, AsyncGenerator
from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from backend.modules.llm.factory import get_provider
from backend.modules.research.arxiv_bridge import get_client

router = APIRouter(tags=["chat"])


# ── Request schema ──────────────────────────────────────────────────────

class MessageItem(BaseModel):
    role: str     # "user" | "assistant"
    content: str


class PaperContext(BaseModel):
    arxiv_id: str
    relative_path: str
    content_level: str = "abstract"   # "abstract" | "full"


class ChatRequest(BaseModel):
    provider:   str                        # "claude" | "ollama"
    model:      str
    api_key:    str = ""                   # Claude only
    ollama_url: str = "http://localhost:11434"
    messages:   list[MessageItem]
    paper_context: Optional[PaperContext] = None


# ── System prompt builder ───────────────────────────────────────────────

def _build_system(req: ChatRequest) -> str:
    base = (
        "You are a quantitative research assistant with deep expertise in "
        "financial mathematics, factor investing, and algorithmic trading. "
        "When generating Python code, follow pandas/numpy conventions and "
        "include inline comments explaining each step."
    )

    if not req.paper_context:
        return base

    ctx  = req.paper_context
    arxiv_client = get_client()

    if ctx.content_level == "full":
        result = arxiv_client.analyze_local_paper(relative_path=ctx.relative_path)
        if result.get("status") == "success":
            # Cap paper text by provider context budget:
            #   Claude  → 200k-token window, cap ~120k chars (~90k tokens)
            #   Ollama  → small local model: cap ~32k chars (~10k tokens) so the
            #             paper + question + answer fit one context and prompt eval
            #             stays fast. Larger inputs get silently truncated by Ollama,
            #             dropping the question and producing degenerate output.
            char_cap = 32_000 if req.provider == "ollama" else 120_000
            full = result["full_content"]
            body = full[:char_cap]
            truncated = len(full) > char_cap
            paper_block = (
                f"\n\n--- SELECTED PAPER ---\n"
                f"arXiv ID : {result['arxiv_id']}\n"
                f"Category : {result['category']}\n"
                f"Sections : {result['section_count']}\n"
                + ("(Note: paper truncated to fit the local model's context.)\n" if truncated and req.provider == "ollama" else "")
                + f"\n{body}"
            )
            return base + paper_block

    # Fallback: abstract only
    meta = arxiv_client._load_metadata().get(ctx.arxiv_id, {})
    if meta:
        paper_block = (
            f"\n\n--- SELECTED PAPER (abstract) ---\n"
            f"Title    : {meta.get('title', ctx.arxiv_id)}\n"
            f"arXiv ID : {ctx.arxiv_id}\n"
            f"Abstract : {meta.get('summary', '')}"
        )
        return base + paper_block

    return base


# ── SSE stream generator ────────────────────────────────────────────────

async def _event_stream(req: ChatRequest) -> AsyncGenerator[str, None]:
    try:
        provider = get_provider(
            provider   = req.provider,
            model      = req.model,
            api_key    = req.api_key,
            ollama_url = req.ollama_url,
        )
        system   = _build_system(req)
        messages = [{"role": m.role, "content": m.content} for m in req.messages]

        async for chunk in provider.stream(messages, system):
            if chunk:
                payload = json.dumps({"type": "chunk", "content": chunk})
                yield f"data: {payload}\n\n"

        paper_refs = [req.paper_context.arxiv_id] if req.paper_context else []
        yield f"data: {json.dumps({'type': 'done', 'paper_refs': paper_refs})}\n\n"

    except ValueError as e:
        yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
    except Exception as e:
        yield f"data: {json.dumps({'type': 'error', 'message': f'LLM error: {e}'})}\n\n"


# ── Endpoint ────────────────────────────────────────────────────────────

@router.post("/api/chat")
async def chat(req: ChatRequest):
    return StreamingResponse(
        _event_stream(req),
        media_type="text/event-stream",
        headers={
            "Cache-Control":    "no-cache",
            "X-Accel-Buffering": "no",
            "Connection":        "keep-alive",
        },
    )

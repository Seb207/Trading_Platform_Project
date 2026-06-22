"""Ollama local provider — async streaming via httpx."""
import json
from typing import AsyncGenerator
from .base import LLMProvider


class OllamaProvider(LLMProvider):
    # Low temperature by default: this is a precision research assistant
    # (paper analysis, classification, code) — not creative writing. High temp
    # makes borderline judgements (e.g. paper-type classification) flip between
    # runs. Override per task if more variability is ever wanted.
    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "gemma3:27b",
        temperature: float = 0.3,
    ):
        self.base_url    = base_url.rstrip("/")
        self.model       = model
        self.temperature = temperature

    async def stream(
        self,
        messages: list[dict],
        system: str = "",
    ) -> AsyncGenerator[str, None]:
        try:
            import httpx
        except ImportError:
            raise ImportError("pip install httpx")

        # Prepend system as a system-role message if provided
        all_messages = []
        if system:
            all_messages.append({"role": "system", "content": system})
        all_messages.extend(messages)

        # Size the context window to the input. Ollama defaults num_ctx to 4096,
        # which silently truncates a full-paper prompt (~10k+ tokens) — dropping
        # the question and producing degenerate output. Estimate tokens (~3 chars/
        # token for dense academic text), add output headroom, clamp to a range
        # these local models support.
        total_chars = sum(len(m.get("content", "")) for m in all_messages)
        est_tokens  = total_chars // 3
        num_ctx     = est_tokens + 2048                    # headroom for the answer
        num_ctx     = max(4096, min(num_ctx, 32768))
        num_ctx     = ((num_ctx + 2047) // 2048) * 2048    # round up to a clean multiple

        # Large local-model contexts are slow to prefill — give them time.
        async with httpx.AsyncClient(timeout=600) as client:
            async with client.stream(
                "POST",
                f"{self.base_url}/api/chat",
                json={
                    "model":    self.model,
                    "messages": all_messages,
                    "stream":   True,
                    "options":  {"num_ctx": num_ctx, "temperature": self.temperature},
                },
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        if not data.get("done"):
                            yield data.get("message", {}).get("content", "")
                    except json.JSONDecodeError:
                        continue

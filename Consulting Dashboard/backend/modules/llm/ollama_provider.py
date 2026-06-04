"""Ollama local provider — async streaming via httpx."""
import json
from typing import AsyncGenerator
from .base import LLMProvider


class OllamaProvider(LLMProvider):
    def __init__(self, base_url: str = "http://localhost:11434", model: str = "gemma3:27b"):
        self.base_url = base_url.rstrip("/")
        self.model    = model

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

        async with httpx.AsyncClient(timeout=120) as client:
            async with client.stream(
                "POST",
                f"{self.base_url}/api/chat",
                json={"model": self.model, "messages": all_messages, "stream": True},
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

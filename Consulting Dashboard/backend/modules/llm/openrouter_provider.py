"""OpenRouter provider — OpenAI-compatible streaming via httpx.

Used with free models (`...:free`). Auth via the user's OpenRouter API key.
"""
import json
from typing import AsyncGenerator
from .base import LLMProvider


class OpenRouterProvider(LLMProvider):
    API_URL = "https://openrouter.ai/api/v1/chat/completions"

    # Low temperature by default: paper analysis / classification / code are
    # precision tasks, not creative writing. High temp makes borderline
    # judgements (e.g. paper-type classification) flip between runs.
    def __init__(self, api_key: str, model: str, temperature: float = 0.3):
        self.api_key     = api_key
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

        # OpenAI-style message list: system first, then the turn history
        all_messages: list[dict] = []
        if system:
            all_messages.append({"role": "system", "content": system})
        all_messages.extend(messages)

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type":  "application/json",
            # OpenRouter attribution headers (optional but recommended)
            "HTTP-Referer":  "http://localhost:3000",
            "X-Title":       "Quant Research Dashboard",
        }
        payload = {
            "model":       self.model,
            "messages":    all_messages,
            "stream":      True,
            "temperature": self.temperature,
        }

        async with httpx.AsyncClient(timeout=300) as client:
            async with client.stream(
                "POST", self.API_URL, json=payload, headers=headers,
            ) as response:
                if response.status_code != 200:
                    body = await response.aread()
                    msg = body.decode(errors="replace")[:300]
                    raise ValueError(f"OpenRouter error {response.status_code}: {msg}")

                async for line in response.aiter_lines():
                    if not line or not line.startswith("data:"):
                        continue                       # skip keep-alive comments
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    try:
                        obj   = json.loads(data)
                        delta = obj.get("choices", [{}])[0].get("delta", {})
                        chunk = delta.get("content")
                        if chunk:
                            yield chunk
                    except (json.JSONDecodeError, IndexError, KeyError):
                        continue

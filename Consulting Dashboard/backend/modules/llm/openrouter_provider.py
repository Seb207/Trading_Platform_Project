"""OpenRouter provider — OpenAI-compatible streaming via httpx.

Used with free models (`...:free`). Auth via the user's OpenRouter API key.
"""
import asyncio
import json
from typing import AsyncGenerator
from .base import LLMProvider

# Free OpenRouter models sit behind a shared, upstream-provider-rate-limited
# pool — a 429 on a `:free` model usually means "this specific model is hot
# right now across all of OpenRouter's free users", not a problem with the
# request, and OpenRouter's own error message says to "retry shortly". A
# couple of short retries clears most of these without surfacing an error
# for something that self-resolves in a few seconds.
_MAX_429_RETRIES = 2
_RETRY_BACKOFF_S = 1.5


class OpenRouterProvider(LLMProvider):
    API_URL = "https://openrouter.ai/api/v1/chat/completions"

    # Low temperature by default: paper analysis / classification / code are
    # precision tasks, not creative writing. High temp makes borderline
    # judgements (e.g. paper-type classification) flip between runs.
    def __init__(
        self,
        api_key: str,
        model: str,
        temperature: float = 0.3,
        reasoning_effort: str | None = None,
    ):
        self.api_key          = api_key
        self.model            = model
        self.temperature      = temperature
        # OpenRouter's unified `reasoning.effort` param ("low"/"medium"/
        # "high"); models that don't support it ignore it. Several free
        # models (e.g. the critic's nemotron-3-ultra) run with reasoning
        # enabled by default, which burns a lot of hidden thinking tokens
        # before the visible answer — real latency for a task like "grade
        # this draft" that doesn't need deep multi-step reasoning. Left
        # unset (None) for normal chat generation so a user's chosen model
        # keeps its default behavior; the critic pass sets this explicitly.
        self.reasoning_effort = reasoning_effort

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
        if self.reasoning_effort:
            payload["reasoning"] = {"effort": self.reasoning_effort}

        async with httpx.AsyncClient(timeout=300) as client:
            # Retry loop only ever runs before any content has been yielded
            # to the caller (the 429 check happens before we start reading
            # the stream body), so retrying here can never duplicate
            # partial output.
            for attempt in range(_MAX_429_RETRIES + 1):
                async with client.stream(
                    "POST", self.API_URL, json=payload, headers=headers,
                ) as response:
                    if response.status_code == 429 and attempt < _MAX_429_RETRIES:
                        await response.aread()
                        await asyncio.sleep(_RETRY_BACKOFF_S * (attempt + 1))
                        continue

                    if response.status_code != 200:
                        body = await response.aread()
                        raise ValueError(
                            _friendly_error(self.model, response.status_code, body)
                        )

                    async for line in response.aiter_lines():
                        if not line or not line.startswith("data:"):
                            continue                   # skip keep-alive comments
                        data = line[5:].strip()
                        if data == "[DONE]":
                            return
                        try:
                            obj   = json.loads(data)
                            delta = obj.get("choices", [{}])[0].get("delta", {})
                            chunk = delta.get("content")
                            if chunk:
                                yield chunk
                        except (json.JSONDecodeError, IndexError, KeyError):
                            continue
                    return


def _friendly_error(model: str, status: int, body: bytes) -> str:
    """Extract OpenRouter's actual complaint from its JSON error body,
    falling back to raw text if the shape doesn't match. Surfacing
    `error.metadata.raw` (when present) instead of the whole JSON blob is
    what turns e.g. a wall of `{"error":{"message":"Provider returned
    error"...` into a readable, actionable chat message.
    """
    detail = ""
    try:
        err = json.loads(body).get("error", {})
        detail = err.get("metadata", {}).get("raw") or err.get("message") or ""
    except (json.JSONDecodeError, AttributeError):
        pass
    if not detail:
        detail = body.decode(errors="replace")[:300]

    if status == 429:
        return (
            f"{model} is rate-limited on OpenRouter's free tier right now. "
            f"Try a different free model from the dropdown, or add your own "
            f"OpenRouter key at openrouter.ai/settings/integrations for "
            f"dedicated capacity. ({detail[:200]})"
        )
    return f"OpenRouter error {status}: {detail[:300]}"

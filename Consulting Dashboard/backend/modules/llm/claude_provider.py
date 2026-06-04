"""Anthropic Claude provider — async streaming via official SDK."""
from typing import AsyncGenerator
from .base import LLMProvider


class ClaudeProvider(LLMProvider):
    def __init__(self, api_key: str, model: str = "claude-opus-4-5"):
        try:
            import anthropic
        except ImportError:
            raise ImportError("pip install anthropic")
        self._anthropic = anthropic
        self.api_key = api_key
        self.model   = model

    async def stream(
        self,
        messages: list[dict],
        system: str = "",
    ) -> AsyncGenerator[str, None]:
        client = self._anthropic.AsyncAnthropic(api_key=self.api_key)

        kwargs: dict = dict(
            model      = self.model,
            max_tokens = 8192,
            messages   = messages,
        )
        if system:
            kwargs["system"] = system

        async with client.messages.stream(**kwargs) as stream:
            async for text in stream.text_stream:
                yield text

"""Instantiate the correct LLMProvider from request parameters."""
from .base import LLMProvider
from .claude_provider import ClaudeProvider
from .ollama_provider import OllamaProvider


def get_provider(
    provider: str,
    model: str,
    api_key: str = "",
    ollama_url: str = "http://localhost:11434",
) -> LLMProvider:
    if provider == "claude":
        if not api_key:
            raise ValueError("Claude API key is required.")
        return ClaudeProvider(api_key=api_key, model=model)

    if provider == "ollama":
        return OllamaProvider(base_url=ollama_url, model=model)

    raise ValueError(f"Unknown provider: {provider!r}. Supported: claude, ollama")

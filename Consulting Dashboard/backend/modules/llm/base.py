"""Abstract LLM provider interface."""
from abc import ABC, abstractmethod
from typing import AsyncGenerator


class LLMProvider(ABC):
    """All providers implement this single interface: async token streaming."""

    @abstractmethod
    async def stream(
        self,
        messages: list[dict],   # [{role: "user"|"assistant", content: str}]
        system: str = "",
    ) -> AsyncGenerator[str, None]:
        """Yield text chunks as they arrive from the model."""
        ...

"""Base provider protocol for LLM backends."""

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from henri.tools.base import Tool

from henri.messages import Message, ToolCall


@dataclass
class StreamEvent:
    """An event from a streaming response."""
    text: str = ""
    tool_calls: list[ToolCall] | None = None
    stop_reason: str | None = None


class Provider(ABC):
    """Abstract base class for LLM providers."""

    name: str  # Provider identifier, e.g., "bedrock", "google", "ollama"

    @abstractmethod
    async def stream(
        self,
        messages: list[Message],
        tools: list["Tool"],
        system: str = "",
    ) -> AsyncIterator[StreamEvent]:
        """Stream a response from the LLM.

        Args:
            messages: Conversation history
            tools: Available tools the LLM can call
            system: System prompt

        Yields:
            StreamEvent objects with text chunks, tool calls, and stop reason
        """
        pass

"""Google Cloud provider for Gemini models."""

import os
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING

from google import genai
from google.genai import types

if TYPE_CHECKING:
    from henri.tools.base import Tool

from henri.config import DEFAULT_GOOGLE_MODEL, DEFAULT_GOOGLE_LOCATION
from henri.messages import Message, ToolCall
from henri.providers.base import Provider, StreamEvent


class GoogleProvider(Provider):
    """Google Cloud provider using the Gemini API.

    Supports two modes:
    - Google AI API: Set GOOGLE_API_KEY env var or pass api_key
    - Vertex AI: Set GOOGLE_CLOUD_PROJECT env var or pass project (and optionally location)
    """

    name = "google"

    def __init__(
        self,
        model_id: str = DEFAULT_GOOGLE_MODEL,
        api_key: str | None = None,
        project: str | None = None,
        location: str = DEFAULT_GOOGLE_LOCATION,
    ):
        self.model_id = model_id

        # Check for API key first (Google AI API)
        api_key = api_key or os.environ.get("GOOGLE_API_KEY")
        if api_key:
            self.client = genai.Client(api_key=api_key)
            return

        # Fall back to Vertex AI
        project = project or os.environ.get("GOOGLE_CLOUD_PROJECT")
        if project:
            self.client = genai.Client(vertexai=True, project=project, location=location)
            return

        raise ValueError(
            "Google provider requires either:\n"
            "  - GOOGLE_API_KEY env var (for Google AI API), or\n"
            "  - GOOGLE_CLOUD_PROJECT env var (for Vertex AI)"
        )

    def _tools_to_google(self, tools: list["Tool"]) -> list[types.Tool]:
        """Convert tools to Google's function declaration format."""
        declarations = []
        for tool in tools:
            declarations.append(types.FunctionDeclaration(
                name=tool.name,
                description=tool.description,
                parameters=tool.parameters,
            ))
        return [types.Tool(function_declarations=declarations)]

    def _messages_to_google(self, messages: list[Message]) -> list[types.Content]:
        """Convert messages to Google's Content format."""
        contents = []

        for msg in messages:
            parts = []

            if msg.content:
                parts.append(types.Part.from_text(text=msg.content))

            for tc in msg.tool_calls:
                parts.append(types.Part.from_function_call(
                    name=tc.name,
                    args=tc.args,
                ))

            for tr in msg.tool_results:
                parts.append(types.Part.from_function_response(
                    name=tr.tool_call_id,
                    response={"result": tr.content},
                ))

            if parts:
                role = "model" if msg.role == "assistant" else "user"
                contents.append(types.Content(role=role, parts=parts))

        return contents

    async def stream(
        self,
        messages: list[Message],
        tools: list["Tool"],
        system: str = "",
    ) -> AsyncIterator[StreamEvent]:
        """Stream a response from Gemini."""
        contents = self._messages_to_google(messages)

        config = types.GenerateContentConfig(
            system_instruction=system if system else None,
            tools=self._tools_to_google(tools) if tools else None,
        )

        tool_calls = []

        response = await self.client.aio.models.generate_content_stream(
            model=self.model_id,
            contents=contents,
            config=config,
        )
        async for chunk in response:
            # Handle text content
            if chunk.text:
                yield StreamEvent(text=chunk.text)

            # Handle function calls
            if chunk.candidates:
                for candidate in chunk.candidates:
                    if candidate.content and candidate.content.parts:
                        for part in candidate.content.parts:
                            if part.function_call:
                                fc = part.function_call
                                tool_calls.append(ToolCall(
                                    id=fc.name,  # Google uses name as ID
                                    name=fc.name,
                                    args=dict(fc.args) if fc.args else {},
                                ))

        # Final event with tool calls and stop reason
        stop_reason = "tool_use" if tool_calls else "end_turn"
        yield StreamEvent(
            tool_calls=tool_calls if tool_calls else None,
            stop_reason=stop_reason,
        )

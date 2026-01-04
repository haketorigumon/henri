"""Google Cloud Vertex AI provider for Claude models."""

import os
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING

from anthropic import AnthropicVertex

if TYPE_CHECKING:
    from henri.tools.base import Tool

from henri.config import DEFAULT_VERTEX_MODEL, DEFAULT_VERTEX_REGION
from henri.messages import Message, ToolCall
from henri.providers.base import Provider, StreamEvent


class VertexProvider(Provider):
    """Vertex AI provider for Claude models using the Anthropic SDK."""

    name = "vertex"

    def __init__(
        self,
        model_id: str = DEFAULT_VERTEX_MODEL,
        region: str = DEFAULT_VERTEX_REGION,
        project: str | None = None,
    ):
        self.model_id = model_id
        project = project or os.environ.get("GOOGLE_CLOUD_PROJECT")
        if not project:
            raise ValueError(
                "Vertex provider requires GOOGLE_CLOUD_PROJECT env var or project parameter"
            )
        self.client = AnthropicVertex(region=region, project_id=project)

    def _message_to_anthropic(self, msg: Message) -> dict:
        """Convert a Message to Anthropic's format."""
        content = []

        if msg.content:
            content.append({"type": "text", "text": msg.content})

        for tc in msg.tool_calls:
            content.append({
                "type": "tool_use",
                "id": tc.id,
                "name": tc.name,
                "input": tc.args,
            })

        for tr in msg.tool_results:
            content.append({
                "type": "tool_result",
                "tool_use_id": tr.tool_call_id,
                "content": tr.content,
                "is_error": tr.is_error,
            })

        return {"role": msg.role, "content": content}

    def _tools_to_anthropic(self, tools: list["Tool"]) -> list[dict]:
        """Convert tools to Anthropic's format."""
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.parameters,
            }
            for tool in tools
        ]

    async def stream(
        self,
        messages: list[Message],
        tools: list["Tool"],
        system: str = "",
    ) -> AsyncIterator[StreamEvent]:
        """Stream a response from Claude via Vertex AI."""
        anthropic_messages = [self._message_to_anthropic(m) for m in messages]

        request = {
            "model": self.model_id,
            "max_tokens": 8192,
            "messages": anthropic_messages,
        }

        if system:
            request["system"] = system

        if tools:
            request["tools"] = self._tools_to_anthropic(tools)

        tool_calls = []
        current_tool_id = None
        current_tool_name = None
        current_tool_input = ""

        with self.client.messages.stream(**request) as stream:
            for event in stream:
                if event.type == "content_block_start":
                    if event.content_block.type == "tool_use":
                        current_tool_id = event.content_block.id
                        current_tool_name = event.content_block.name
                        current_tool_input = ""
                        yield StreamEvent(tool_use_started=True)

                elif event.type == "content_block_delta":
                    if event.delta.type == "text_delta":
                        yield StreamEvent(text=event.delta.text)
                    elif event.delta.type == "input_json_delta":
                        current_tool_input += event.delta.partial_json

                elif event.type == "content_block_stop":
                    if current_tool_id and current_tool_name:
                        import json
                        tool_calls.append(ToolCall(
                            id=current_tool_id,
                            name=current_tool_name,
                            args=json.loads(current_tool_input) if current_tool_input else {},
                        ))
                        current_tool_id = None
                        current_tool_name = None

                elif event.type == "message_stop":
                    pass

            # Get final message for stop reason
            final_message = stream.get_final_message()
            stop_reason = final_message.stop_reason if final_message else "end_turn"

        yield StreamEvent(
            tool_calls=tool_calls if tool_calls else None,
            stop_reason=stop_reason,
        )

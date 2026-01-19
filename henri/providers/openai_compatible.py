"""OpenAI-compatible provider for VLLM, LocalAI, llama.cpp, etc."""

import json
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING

from openai import AsyncOpenAI

if TYPE_CHECKING:
    from henri.tools.base import Tool

from henri.messages import Message, ToolCall
from henri.providers.base import Provider, StreamEvent, Usage


class OpenAICompatibleProvider(Provider):
    """Provider for OpenAI-compatible servers (VLLM, LocalAI, llama.cpp, etc.)."""

    name = "openai_compatible"

    def __init__(self, model_id: str, host: str, api_key: str = "EMPTY"):
        """Initialize the OpenAI-compatible provider.

        Args:
            model_id: Model name on the server (required)
            host: Server URL without /v1 suffix (required)
            api_key: API key, defaults to "EMPTY" for servers without auth
        """
        self.model_id = model_id
        self.client = AsyncOpenAI(base_url=f"{host}/v1", api_key=api_key)

    def _tools_to_openai(self, tools: list["Tool"]) -> list[dict]:
        """Convert tools to OpenAI's format."""
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters,
                },
            }
            for tool in tools
        ]

    def _messages_to_openai(
        self, messages: list[Message], system: str
    ) -> list[dict]:
        """Convert messages to OpenAI's format."""
        openai_messages = []

        if system:
            openai_messages.append({"role": "system", "content": system})

        for msg in messages:
            if msg.content:
                openai_messages.append({
                    "role": msg.role,
                    "content": msg.content,
                })

            for tc in msg.tool_calls:
                openai_messages.append({
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": json.dumps(tc.args),
                        },
                    }],
                })

            for tr in msg.tool_results:
                openai_messages.append({
                    "role": "tool",
                    "tool_call_id": tr.tool_call_id,
                    "content": tr.content,
                })

        return openai_messages

    async def stream(
        self,
        messages: list[Message],
        tools: list["Tool"],
        system: str = "",
    ) -> AsyncIterator[StreamEvent]:
        """Stream a response from the OpenAI-compatible server."""
        openai_messages = self._messages_to_openai(messages, system)

        # Build request kwargs
        kwargs = {
            "model": self.model_id,
            "messages": openai_messages,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if tools:
            kwargs["tools"] = self._tools_to_openai(tools)

        response = await self.client.chat.completions.create(**kwargs)

        # Track tool calls being built across chunks
        tool_call_builders: dict[int, dict] = {}
        usage = None
        final_finish_reason = None

        async for chunk in response:
            # Handle usage (comes in final chunk with empty choices)
            if chunk.usage:
                usage = Usage(
                    input_tokens=chunk.usage.prompt_tokens,
                    output_tokens=chunk.usage.completion_tokens,
                )

            if not chunk.choices:
                continue

            delta = chunk.choices[0].delta
            finish_reason = chunk.choices[0].finish_reason

            # Handle text content
            if delta.content:
                yield StreamEvent(text=delta.content)

            # Handle tool calls (streamed incrementally)
            if delta.tool_calls:
                for tc_delta in delta.tool_calls:
                    idx = tc_delta.index
                    if idx not in tool_call_builders:
                        # First chunk for this tool call
                        if not tool_call_builders:
                            yield StreamEvent(tool_use_started=True)
                        tool_call_builders[idx] = {
                            "id": tc_delta.id or "",
                            "name": "",
                            "arguments": "",
                        }
                    builder = tool_call_builders[idx]
                    if tc_delta.id:
                        builder["id"] = tc_delta.id
                    if tc_delta.function:
                        if tc_delta.function.name:
                            builder["name"] = tc_delta.function.name
                        if tc_delta.function.arguments:
                            builder["arguments"] += tc_delta.function.arguments

            # Track finish reason but don't yield yet (usage comes in next chunk)
            if finish_reason:
                final_finish_reason = finish_reason

        # After stream ends, yield final event with usage
        if final_finish_reason:
            tool_calls = []
            for idx in sorted(tool_call_builders.keys()):
                builder = tool_call_builders[idx]
                try:
                    args = json.loads(builder["arguments"]) if builder["arguments"] else {}
                except json.JSONDecodeError:
                    args = {}
                tool_calls.append(ToolCall(
                    id=builder["id"],
                    name=builder["name"],
                    args=args,
                ))

            stop_reason = "tool_use" if final_finish_reason == "tool_calls" else "end_turn"
            yield StreamEvent(
                tool_calls=tool_calls if tool_calls else None,
                stop_reason=stop_reason,
                usage=usage,
            )

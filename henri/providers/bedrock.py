"""AWS Bedrock provider for Claude models."""

import json
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import TYPE_CHECKING

import boto3

if TYPE_CHECKING:
    from henri.tools.base import Tool

from henri.config import DEFAULT_MODEL, DEFAULT_REGION
from henri.messages import Message, ToolCall


@dataclass
class StreamEvent:
    """An event from the streaming response."""
    text: str = ""
    tool_calls: list[ToolCall] | None = None
    stop_reason: str | None = None


class BedrockProvider:
    """AWS Bedrock provider using the Converse API."""

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL,
        region: str = DEFAULT_REGION,
    ):
        self.model_id = model_id
        self.client = boto3.client("bedrock-runtime", region_name=region)

    def _message_to_bedrock(self, msg: Message) -> dict:
        """Convert a Message to Bedrock's format."""
        content = []

        if msg.content:
            content.append({"text": msg.content})

        for tc in msg.tool_calls:
            content.append({
                "toolUse": {
                    "toolUseId": tc.id,
                    "name": tc.name,
                    "input": tc.args,
                }
            })

        for tr in msg.tool_results:
            content.append({
                "toolResult": {
                    "toolUseId": tr.tool_call_id,
                    "content": [{"text": tr.content}],
                    "status": "error" if tr.is_error else "success",
                }
            })

        return {"role": msg.role, "content": content}

    def _tools_to_bedrock(self, tools: list["Tool"]) -> list[dict]:
        """Convert tools to Bedrock's toolConfig format."""
        return [
            {
                "toolSpec": {
                    "name": tool.name,
                    "description": tool.description,
                    "inputSchema": {"json": tool.parameters},
                }
            }
            for tool in tools
        ]

    async def stream(
        self,
        messages: list[Message],
        tools: list["Tool"],
        system: str = "",
    ) -> AsyncIterator[StreamEvent]:
        """Stream a response from Claude."""
        bedrock_messages = [self._message_to_bedrock(m) for m in messages]

        request = {
            "modelId": self.model_id,
            "messages": bedrock_messages,
        }

        if system:
            request["system"] = [{"text": system}]

        if tools:
            request["toolConfig"] = {"tools": self._tools_to_bedrock(tools)}

        response = self.client.converse_stream(**request)

        current_tool_id = None
        current_tool_name = None
        current_tool_input = ""
        tool_calls = []

        for event in response["stream"]:
            if "contentBlockStart" in event:
                start = event["contentBlockStart"].get("start", {})
                if "toolUse" in start:
                    current_tool_id = start["toolUse"]["toolUseId"]
                    current_tool_name = start["toolUse"]["name"]
                    current_tool_input = ""

            elif "contentBlockDelta" in event:
                delta = event["contentBlockDelta"]["delta"]
                if "text" in delta:
                    yield StreamEvent(text=delta["text"])
                elif "toolUse" in delta:
                    current_tool_input += delta["toolUse"].get("input", "")

            elif "contentBlockStop" in event:
                if current_tool_id and current_tool_name:
                    tool_calls.append(ToolCall(
                        id=current_tool_id,
                        name=current_tool_name,
                        args=json.loads(current_tool_input) if current_tool_input else {},
                    ))
                    current_tool_id = None
                    current_tool_name = None

            elif "messageStop" in event:
                stop_reason = event["messageStop"].get("stopReason")
                yield StreamEvent(
                    tool_calls=tool_calls if tool_calls else None,
                    stop_reason=stop_reason,
                )

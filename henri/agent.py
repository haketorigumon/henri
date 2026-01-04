"""Main agent loop for Henri."""

import asyncio

from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from henri.messages import Message, ToolResult
from henri.permissions import PermissionManager
from henri.providers import Provider, create_provider
from henri.tools.base import Tool, get_default_tools


SYSTEM_PROMPT = """You are Henri, a helpful coding assistant.

You have access to tools for reading files, writing files, and executing shell commands.
Use these tools to help the user with their tasks.

Be concise and direct in your responses. When you need to perform actions, use the appropriate tools."""


class Agent:
    """The main Henri agent."""

    def __init__(
        self,
        provider: Provider,
        tools: list[Tool] | None = None,
        console: Console | None = None,
    ):
        self.provider = provider
        self.tools = tools or get_default_tools()
        self.tools_by_name = {t.name: t for t in self.tools}
        self.console = console or Console()
        self.permissions = PermissionManager(console=self.console)
        self.messages: list[Message] = []

    async def chat(self, user_input: str) -> None:
        """Process a user message and stream the response."""
        self.messages.append(Message.user(user_input))

        while True:
            # Stream response from LLM
            response_text = ""
            tool_calls = []
            stop_reason = None

            async for event in self.provider.stream(
                self.messages,
                self.tools,
                system=SYSTEM_PROMPT,
            ):
                if event.text:
                    # Stream text to console
                    self.console.print(event.text, end="")
                    response_text += event.text

                if event.tool_calls:
                    tool_calls = event.tool_calls

                if event.stop_reason:
                    stop_reason = event.stop_reason

            # End the streamed line if we printed any text
            if response_text:
                self.console.print()

            # Record assistant message
            self.messages.append(Message.assistant(response_text, tool_calls))

            # If no tool calls, we're done
            if not tool_calls:
                break

            # Execute tool calls
            results = []
            for call in tool_calls:
                tool = self.tools_by_name.get(call.name)
                if not tool:
                    results.append(ToolResult(
                        tool_call_id=call.id,
                        content=f"[error: unknown tool '{call.name}']",
                        is_error=True,
                    ))
                    continue

                # Check permissions
                if not self.permissions.check(tool, call):
                    results.append(ToolResult(
                        tool_call_id=call.id,
                        content="[permission denied by user]",
                        is_error=True,
                    ))
                    continue

                # Execute the tool
                self._show_tool_execution(tool, call)
                result = tool.execute(**call.args)
                results.append(ToolResult(tool_call_id=call.id, content=result))
                self._show_tool_result(result)

            # Add results and continue the loop
            self.messages.append(Message.tool_result(results))

    def _show_tool_execution(self, tool: Tool, call) -> None:
        """Display that a tool is being executed."""
        args_short = ", ".join(f"{k}={v!r:.50}" for k, v in call.args.items())
        self.console.print(f"\n[dim]▶ {tool.name}({args_short})[/dim]")

    def _show_tool_result(self, result: str) -> None:
        """Display a tool result (truncated if long)."""
        lines = result.split("\n")
        if len(lines) > 10:
            display = "\n".join(lines[:10]) + f"\n[dim]... ({len(lines) - 10} more lines)[/dim]"
        else:
            display = result
        self.console.print(Panel(display, border_style="dim", padding=(0, 1)))


async def run_agent(
    provider: str,
    model: str,
    region: str | None = None,
    host: str | None = None,
):
    """Run the interactive agent loop."""
    console = Console()

    # Build provider-specific kwargs
    provider_kwargs = {"model_id": model}
    if provider == "bedrock" and region:
        provider_kwargs["region"] = region
    elif provider == "vertex" and region:
        provider_kwargs["region"] = region
    elif provider == "ollama" and host:
        provider_kwargs["host"] = host

    llm = create_provider(provider, **provider_kwargs)
    agent = Agent(provider=llm, console=console)

    console.print(Panel(
        f"[bold]Henri[/bold] - A pedagogical Claude Code clone\n"
        f"Provider: {provider} | Model: {model}\n"
        "Type your message and press Enter. Use Ctrl+C to exit.",
        border_style="blue",
    ))

    # Session with history for up/down arrow recall
    session = PromptSession(history=FileHistory(".henri_history"))

    while True:
        try:
            console.print()
            user_input = await session.prompt_async("> ")
            if not user_input.strip():
                continue
            await agent.chat(user_input)
        except KeyboardInterrupt:
            console.print("\n[dim]Goodbye![/dim]")
            break
        except EOFError:
            break

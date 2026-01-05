"""Main agent loop for Henri."""

import asyncio
import sys

from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from rich.console import Console
from rich.panel import Panel
from rich.status import Status

from henri.messages import Message, ToolResult
from henri.permissions import PermissionManager
from henri.providers import Provider, create_provider
from henri.tools.base import Tool, get_default_tools


def summarize_tools_and_permissions(
    tools: list[Tool],
    auto_allow_cwd: set[str],
    auto_allow: set[str],
    reject_prompts: bool = False,
) -> tuple[list[str], list[str]]:
    """Summarize tools and permissions. Returns (tool_lines, perm_lines)."""
    tool_lines = [f"- {t.name}: {t.description}" for t in tools]

    tool_names = {t.name for t in tools}
    auto_all = tool_names & auto_allow
    auto_cwd = (tool_names & auto_allow_cwd) - auto_all
    need_perm = tool_names - auto_cwd - auto_all

    perm_lines = []
    if auto_all:
        perm_lines.append(f"Auto-allow: {', '.join(sorted(auto_all))}")
    if auto_cwd:
        perm_lines.append(f"Auto-allow in cwd: {', '.join(sorted(auto_cwd))}")
    if need_perm:
        perm_lines.append(f"Require permission: {', '.join(sorted(need_perm))}")
    if not perm_lines:
        perm_lines.append("All tools require permission.")
    if reject_prompts:
        perm_lines.append("Other prompts: auto-denied")

    return tool_lines, perm_lines


def build_system_prompt(
    tools: list[Tool],
    auto_allow_cwd: set[str] | None = None,
    auto_allow: set[str] | None = None,
    reject_prompts: bool = False,
) -> str:
    """Build system prompt with available tools and permissions."""
    tool_lines, perm_lines = summarize_tools_and_permissions(
        tools, auto_allow_cwd or set(), auto_allow or set(), reject_prompts
    )
    tools_section = "\n".join(tool_lines)
    perms_section = "\n".join(perm_lines)
    return f"""You are Henri, a helpful coding assistant.

You have access to these tools:
{tools_section}

Permissions:
{perms_section}

Be concise and direct in your responses."""


class Agent:
    """The main Henri agent."""

    def __init__(
        self,
        provider: Provider,
        tools: list[Tool] | None = None,
        console: Console | None = None,
        permissions: PermissionManager | None = None,
    ):
        self.provider = provider
        self.tools = tools or get_default_tools()
        self.tools_by_name = {t.name: t for t in self.tools}
        self.console = console or Console()
        self.permissions = permissions or PermissionManager(console=self.console)
        self.system_prompt = build_system_prompt(
            self.tools,
            auto_allow_cwd=self.permissions.auto_allow_cwd,
            auto_allow=self.permissions.auto_allow,
            reject_prompts=self.permissions.reject_prompts,
        )
        self.messages: list[Message] = []
        self._status: Status | None = None
        self._pondering_task: asyncio.Task | None = None

    def _cancel_pondering(self) -> None:
        """Cancel any pending pondering status."""
        if self._pondering_task and not self._pondering_task.done():
            self._pondering_task.cancel()
            self._pondering_task = None

    def _schedule_pondering(self, delay: float = 0.5) -> None:
        """Schedule showing 'Pondering...' after a delay."""
        self._cancel_pondering()

        async def show_pondering():
            await asyncio.sleep(delay)
            self.console.print()  # newline so status doesn't overwrite text
            self._show_status("Pondering...")

        self._pondering_task = asyncio.create_task(show_pondering())

    def _show_status(self, message: str) -> None:
        """Show a spinner with the given message."""
        if self._status:
            self._status.stop()
        self._status = Status(message, console=self.console, spinner="dots")
        self._status.start()

    def _hide_status(self) -> None:
        """Hide the current spinner if any."""
        if self._status:
            self._status.stop()
            self._status = None

    async def chat(self, user_input: str) -> None:
        """Process a user message and stream the response."""
        self.messages.append(Message.user(user_input))

        while True:
            # Stream response from LLM
            response_text = ""
            tool_calls = []
            stop_reason = None

            # Show spinner while waiting for response
            self._show_status("Answering...")

            async for event in self.provider.stream(
                self.messages,
                self.tools,
                system=self.system_prompt,
            ):
                if event.text:
                    self._cancel_pondering()
                    self._hide_status()
                    self.console.print(event.text, end="")
                    response_text += event.text
                    # Schedule "Pondering..." to show after a pause
                    self._schedule_pondering()

                if event.tool_use_started or event.tool_calls:
                    self._cancel_pondering()
                    if event.tool_calls:
                        tool_calls = event.tool_calls
                    self._show_status("Working...")

                if event.stop_reason:
                    stop_reason = event.stop_reason

            self._cancel_pondering()
            self._hide_status()
            if response_text:
                self.console.print()

            # Record assistant message
            self.messages.append(Message.assistant(response_text, tool_calls))

            # If no tool calls, we're done
            if not tool_calls:
                break

            # Execute tool calls
            self._show_status("Working...")
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

                # Check permissions (hide spinner for potential dialog)
                self._hide_status()
                if not self.permissions.check(tool, call):
                    results.append(ToolResult(
                        tool_call_id=call.id,
                        content="[permission denied by user]",
                        is_error=True,
                    ))
                    continue

                # Show execution info
                self._show_tool_execution(tool, call)

                # Execute tool
                self._show_status("Executing...")
                result = tool.execute(**call.args)
                self._hide_status()

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
    hooks: list | None = None,
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

    # Get tools from hooks (if any) and merge with defaults
    tools = get_default_tools()
    hooks = hooks or []
    for hook in hooks:
        if hasattr(hook, "TOOLS"):
            tools = tools + hook.TOOLS
        if hasattr(hook, "REMOVE_TOOLS"):
            remove = hook.REMOVE_TOOLS
            tools = [t for t in tools if t.name not in remove]

    # Build permission manager with hook overrides
    from henri.permissions import DEFAULT_PATH_BASED, DEFAULT_AUTO_ALLOW_CWD, DEFAULT_AUTO_ALLOW
    path_based = set(DEFAULT_PATH_BASED)
    auto_allow_cwd = set(DEFAULT_AUTO_ALLOW_CWD)
    auto_allow = set(DEFAULT_AUTO_ALLOW)

    reject_prompts = False
    for hook in hooks:
        if hasattr(hook, "PATH_BASED"):
            path_based |= hook.PATH_BASED
        if hasattr(hook, "AUTO_ALLOW_CWD"):
            auto_allow_cwd |= hook.AUTO_ALLOW_CWD
        if hasattr(hook, "AUTO_ALLOW"):
            auto_allow |= hook.AUTO_ALLOW
        if hasattr(hook, "REJECT_PROMPTS"):
            reject_prompts = reject_prompts or hook.REJECT_PROMPTS

    permissions = PermissionManager(
        console=console,
        path_based=path_based,
        auto_allow_cwd=auto_allow_cwd,
        auto_allow=auto_allow,
        reject_prompts=reject_prompts,
    )
    agent = Agent(provider=llm, tools=tools, console=console, permissions=permissions)

    console.print(Panel(
        f"[bold]Henri[/bold] - A pedagogical Claude Code clone\n"
        f"Provider: {provider} | Model: {model}\n"
        "Type your message and press Enter. Use Ctrl+C to exit.",
        border_style="blue",
    ))

    # Print tools and permissions summary
    tool_lines, perm_lines = summarize_tools_and_permissions(
        tools, auto_allow_cwd, auto_allow, reject_prompts
    )
    console.print("\n[bold]Tools:[/bold]")
    for line in tool_lines:
        console.print(line)
    console.print("\n[bold]Permissions:[/bold]")
    for line in perm_lines:
        console.print(f"  {line}")

    # Use prompt_toolkit only for interactive terminals
    interactive = sys.stdin.isatty()
    session = PromptSession(history=FileHistory(".henri_history")) if interactive else None

    while True:
        try:
            console.print()
            if interactive:
                user_input = await session.prompt_async("> ")
            else:
                user_input = sys.stdin.readline()
                if not user_input:  # EOF
                    break
            if not user_input.strip():
                continue
            await agent.chat(user_input)
        except KeyboardInterrupt:
            console.print("\n[dim]Goodbye![/dim]")
            break
        except EOFError:
            break

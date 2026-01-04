"""Simple permission management for tool execution."""

from dataclasses import dataclass, field

from rich.console import Console
from rich.panel import Panel

from henri.messages import ToolCall
from henri.tools.base import Tool


@dataclass
class PermissionManager:
    """Manages permissions for tool execution."""

    # Tools that have been permanently allowed for this session
    allowed_tools: set[str] = field(default_factory=set)

    # Exact bash commands that have been allowed
    allowed_bash_commands: set[str] = field(default_factory=set)

    # If True, allow all tools without prompting
    allow_all: bool = False

    console: Console = field(default_factory=Console)

    def check(self, tool: Tool, call: ToolCall) -> bool:
        """Check if a tool call is allowed. Prompts user if needed."""
        if not tool.requires_permission:
            return True

        if self.allow_all:
            return True

        # For bash, check exact command match
        if tool.name == "bash":
            command = call.args.get("command", "")
            if command in self.allowed_bash_commands:
                return True
        elif tool.name in self.allowed_tools:
            return True

        return self._prompt_user(tool, call)

    def _prompt_user(self, tool: Tool, call: ToolCall) -> bool:
        """Prompt the user for permission to execute a tool."""
        # Format the tool call nicely
        args_display = "\n".join(f"  {k}: {v!r}" for k, v in call.args.items())

        self.console.print()
        self.console.print(Panel(
            f"[bold]{tool.name}[/bold]\n{args_display}",
            title="[yellow]Permission Required[/yellow]",
            border_style="yellow",
        ))

        while True:
            response = self.console.input(
                "[dim](y)es / (n)o / (a)lways for this tool / allow (A)ll:[/dim] "
            ).strip().lower()

            if response in ("y", "yes"):
                return True
            elif response in ("n", "no"):
                return False
            elif response in ("a", "always"):
                if tool.name == "bash":
                    command = call.args.get("command", "")
                    self.allowed_bash_commands.add(command)
                    self.console.print(f"[dim]Will allow this exact bash command for this session[/dim]")
                else:
                    self.allowed_tools.add(tool.name)
                    self.console.print(f"[dim]Will allow '{tool.name}' for this session[/dim]")
                return True
            elif response == "A":  # Capital A for allow all
                self.allow_all = True
                self.console.print("[dim]Will allow all tools for this session[/dim]")
                return True
            else:
                self.console.print("[red]Please enter y, n, a, or A[/red]")

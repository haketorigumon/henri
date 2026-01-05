"""Simple permission management for tool execution."""

from dataclasses import dataclass, field
from pathlib import Path

from rich.console import Console
from rich.panel import Panel

from henri.messages import ToolCall
from henri.tools.base import Tool


# Default configurations
DEFAULT_PATH_BASED = {"grep", "glob", "read_file", "write_file", "edit_file"}
DEFAULT_AUTO_ALLOW_CWD = {"grep", "glob", "read_file"}
DEFAULT_AUTO_ALLOW = set()


@dataclass
class PermissionManager:
    """Manages permissions for tool execution."""

    # Configuration: tools where "always" means per-path
    path_based: set[str] = field(default_factory=lambda: set(DEFAULT_PATH_BASED))

    # Configuration: path-based tools that auto-allow within cwd
    auto_allow_cwd: set[str] = field(default_factory=lambda: set(DEFAULT_AUTO_ALLOW_CWD))

    # Configuration: tools that are always allowed (no prompts)
    auto_allow: set[str] = field(default_factory=lambda: set(DEFAULT_AUTO_ALLOW))

    # Session state: tools that have been permanently allowed
    allowed_tools: set[str] = field(default_factory=set)

    # Session state: exact bash commands that have been allowed
    allowed_bash_commands: set[str] = field(default_factory=set)

    # Session state: paths that have been allowed, per tool
    allowed_paths: dict[str, set[str]] = field(default_factory=dict)

    # Session state: allow all tools without prompting
    allow_all: bool = False

    # If True, reject (instead of prompting) when permission needed
    reject_prompts: bool = False

    console: Console = field(default_factory=Console)

    def _is_path_based(self, tool_name: str) -> bool:
        """Check if a tool uses per-path permission tracking."""
        return tool_name in self.path_based

    def _is_auto_allow_cwd(self, tool_name: str) -> bool:
        """Check if a tool auto-allows within cwd."""
        return tool_name in self.auto_allow_cwd

    def _is_auto_allow(self, tool_name: str) -> bool:
        """Check if a tool is always allowed."""
        return tool_name in self.auto_allow

    def _is_path_within_cwd(self, path: str) -> bool:
        """Check if a path is within the current working directory."""
        try:
            resolved = Path(path).resolve()
            cwd = Path.cwd().resolve()
            return resolved.is_relative_to(cwd)
        except (ValueError, OSError):
            return False

    def check(self, tool: Tool, call: ToolCall) -> bool:
        """Check if a tool call is allowed. Prompts user if needed."""
        if not tool.requires_permission:
            return True

        if self.allow_all:
            return True

        # Pre-configured always-allowed tools
        if self._is_auto_allow(tool.name):
            return True

        # For bash, check exact command match
        if tool.name == "bash":
            command = call.args.get("command", "")
            if command in self.allowed_bash_commands:
                return True
        # For path-based tools
        elif self._is_path_based(tool.name):
            path = call.args.get("path", ".")
            resolved = str(Path(path).resolve())
            # Check if path already allowed for this tool
            if resolved in self.allowed_paths.get(tool.name, set()):
                return True
            # Auto-allow within cwd only for certain tools
            if self._is_auto_allow_cwd(tool.name) and self._is_path_within_cwd(path):
                return True
        elif tool.name in self.allowed_tools:
            return True

        if self.reject_prompts:
            self.console.print(f"[dim]Auto-denied: {tool.name}[/dim]")
            return False

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
                elif self._is_path_based(tool.name):
                    path = call.args.get("path", ".")
                    resolved = str(Path(path).resolve())
                    if tool.name not in self.allowed_paths:
                        self.allowed_paths[tool.name] = set()
                    self.allowed_paths[tool.name].add(resolved)
                    self.console.print(f"[dim]Will allow {tool.name} access to '{resolved}' for this session[/dim]")
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

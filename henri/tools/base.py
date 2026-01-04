"""Base tool class and built-in tools."""

import subprocess
from abc import ABC, abstractmethod
from pathlib import Path


class Tool(ABC):
    """Base class for all tools."""

    name: str
    description: str
    parameters: dict  # JSON Schema
    requires_permission: bool = False

    @abstractmethod
    def execute(self, **kwargs) -> str:
        """Execute the tool and return the result."""
        pass


class BashTool(Tool):
    """Execute shell commands."""

    name = "bash"
    description = "Execute a shell command and return its output."
    parameters = {
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "The shell command to execute",
            },
        },
        "required": ["command"],
    }
    requires_permission = True

    def execute(self, command: str) -> str:
        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=120,
            )
            output = result.stdout
            if result.stderr:
                output += f"\n[stderr]\n{result.stderr}"
            if result.returncode != 0:
                output += f"\n[exit code: {result.returncode}]"
            return output or "(no output)"
        except subprocess.TimeoutExpired:
            return "[error: command timed out after 120 seconds]"
        except Exception as e:
            return f"[error: {e}]"


class ReadFileTool(Tool):
    """Read file contents."""

    name = "read_file"
    description = "Read the contents of a file."
    parameters = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Path to the file to read",
            },
        },
        "required": ["path"],
    }
    requires_permission = False  # Reading is generally safe

    def execute(self, path: str) -> str:
        try:
            p = Path(path).expanduser()
            if not p.exists():
                return f"[error: file not found: {path}]"
            if not p.is_file():
                return f"[error: not a file: {path}]"
            content = p.read_text()
            if len(content) > 100_000:
                return content[:100_000] + "\n[truncated...]"
            return content
        except Exception as e:
            return f"[error: {e}]"


class WriteFileTool(Tool):
    """Write content to a file."""

    name = "write_file"
    description = "Write content to a file. Creates the file if it doesn't exist."
    parameters = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Path to the file to write",
            },
            "content": {
                "type": "string",
                "description": "Content to write to the file",
            },
        },
        "required": ["path", "content"],
    }
    requires_permission = True

    def execute(self, path: str, content: str) -> str:
        try:
            p = Path(path).expanduser()
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(content)
            return f"[wrote {len(content)} bytes to {path}]"
        except Exception as e:
            return f"[error: {e}]"


def get_default_tools() -> list[Tool]:
    """Return the default set of tools."""
    return [BashTool(), ReadFileTool(), WriteFileTool()]

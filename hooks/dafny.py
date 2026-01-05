"""Dafny tool hook for Henri (interactive use).

Usage:
    henri --hook hooks/dafny.py

Adds dafny_verify tool with path-based permissions.
"""

import subprocess

from henri.tools.base import Tool


class DafnyVerifyTool(Tool):
    """Run dafny verify on a file."""

    name = "dafny_verify"
    description = "Run 'dafny verify' on a Dafny file to check verification."
    parameters = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Path to the .dfy file to verify",
            },
        },
        "required": ["path"],
    }
    requires_permission = True

    def execute(self, path: str) -> str:
        try:
            result = subprocess.run(
                ["dafny", "verify", path],
                capture_output=True,
                text=True,
                timeout=120,
            )
            output = result.stdout
            if result.stderr:
                output += f"\n{result.stderr}"
            return output or "(no output)"
        except FileNotFoundError:
            return "[error: dafny not found. Install it: https://github.com/dafny-lang/dafny]"
        except subprocess.TimeoutExpired:
            return "[error: verification timed out after 120 seconds]"
        except Exception as e:
            return f"[error: {e}]"


# Tools to add
TOOLS = [DafnyVerifyTool()]

# Make dafny_verify path-based (per-path "always")
PATH_BASED = {"dafny_verify"}

# Auto-allow dafny_verify within cwd
AUTO_ALLOW_CWD = {"dafny_verify"}

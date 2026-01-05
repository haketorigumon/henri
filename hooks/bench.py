"""Benchmark hook for Henri (non-interactive use).

Usage:
    echo "Prove this..." | henri --hook hooks/dafny.py hooks/bench.py

Removes dangerous tools and auto-allows file operations within cwd.
"""

# Remove dangerous tools for benchmarks
REMOVE_TOOLS = {"bash", "web_fetch"}

# Auto-allow file operations within cwd
AUTO_ALLOW_CWD = {"write_file", "edit_file"}

# Reject (don't prompt) if unexpected permission needed
REJECT_PROMPTS = True

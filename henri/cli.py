"""CLI entry point for Henri."""

import argparse
import asyncio
import importlib.util
import sys
from pathlib import Path

from henri.agent import run_agent
from henri.config import (
    DEFAULT_PROVIDER,
    DEFAULT_BEDROCK_MODEL,
    DEFAULT_GOOGLE_MODEL,
    DEFAULT_OLLAMA_MODEL,
    DEFAULT_OLLAMA_HOST,
    DEFAULT_VERTEX_MODEL,
)
from henri.providers import PROVIDERS


def load_hook(hook_path: str):
    """Load a hook module from a file path."""
    path = Path(hook_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Hook file not found: {hook_path}")

    spec = importlib.util.spec_from_file_location("hook", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["hook"] = module
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(
        description="Henri - A pedagogical Claude Code clone",
    )
    parser.add_argument(
        "--provider", "-p",
        choices=list(PROVIDERS.keys()),
        default=DEFAULT_PROVIDER,
        help=f"LLM provider (default: {DEFAULT_PROVIDER})",
    )
    parser.add_argument(
        "--model", "-m",
        help="Model ID (provider-specific, uses default if not set)",
    )
    parser.add_argument(
        "--region",
        help="Region (Bedrock: AWS region, Vertex: GCP region)",
    )
    parser.add_argument(
        "--host",
        default=DEFAULT_OLLAMA_HOST,
        help="Host URL for Ollama provider",
    )
    parser.add_argument(
        "--hook",
        nargs="*",
        help="Path(s) to Python hook file(s) (defines TOOLS, PATH_BASED, etc.)",
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=None,
        help="Maximum conversation turns (default: unlimited)",
    )
    args = parser.parse_args()

    # Determine model based on provider if not specified
    if args.model is None:
        args.model = {
            "bedrock": DEFAULT_BEDROCK_MODEL,
            "google": DEFAULT_GOOGLE_MODEL,
            "ollama": DEFAULT_OLLAMA_MODEL,
            "vertex": DEFAULT_VERTEX_MODEL,
        }[args.provider]

    # Load hooks if specified
    hooks = []
    if args.hook:
        for hook_path in args.hook:
            hooks.append(load_hook(hook_path))

    asyncio.run(run_agent(
        provider=args.provider,
        model=args.model,
        region=args.region,
        host=args.host,
        hooks=hooks,
        max_turns=args.max_turns,
    ))


if __name__ == "__main__":
    main()

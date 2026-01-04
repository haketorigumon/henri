"""CLI entry point for Henri."""

import argparse
import asyncio

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
    args = parser.parse_args()

    # Determine model based on provider if not specified
    if args.model is None:
        args.model = {
            "bedrock": DEFAULT_BEDROCK_MODEL,
            "google": DEFAULT_GOOGLE_MODEL,
            "ollama": DEFAULT_OLLAMA_MODEL,
            "vertex": DEFAULT_VERTEX_MODEL,
        }[args.provider]

    asyncio.run(run_agent(
        provider=args.provider,
        model=args.model,
        region=args.region,
        host=args.host,
    ))


if __name__ == "__main__":
    main()

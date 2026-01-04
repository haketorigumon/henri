"""CLI entry point for Henri."""

import argparse
import asyncio

from henri.agent import run_agent
from henri.config import DEFAULT_MODEL, DEFAULT_REGION


def main():
    parser = argparse.ArgumentParser(
        description="Henri - A pedagogical Claude Code clone",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="Bedrock model ID or inference profile ARN",
    )
    parser.add_argument(
        "--region",
        default=DEFAULT_REGION,
        help="AWS region for Bedrock",
    )
    args = parser.parse_args()
    asyncio.run(run_agent(model=args.model, region=args.region))


if __name__ == "__main__":
    main()

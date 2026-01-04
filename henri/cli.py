"""CLI entry point for Henri."""

import argparse
import asyncio

from henri.agent import run_agent


def main():
    parser = argparse.ArgumentParser(
        description="Henri - A pedagogical Claude Code clone",
    )
    parser.add_argument(
        "--model",
        default="us.anthropic.claude-sonnet-4-20250514-v1:0",
        help="Bedrock model ID or inference profile ARN",
    )
    parser.add_argument(
        "--region",
        default="us-east-1",
        help="AWS region for Bedrock",
    )
    args = parser.parse_args()

    # For now, we ignore args and use defaults
    # TODO: Pass these to the agent
    asyncio.run(run_agent())


if __name__ == "__main__":
    main()

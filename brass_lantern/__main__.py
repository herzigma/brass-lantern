"""CLI entry point — ``python -m brass_lantern``."""

from __future__ import annotations

import argparse
import asyncio
import os
import sys

from dotenv import load_dotenv

from brass_lantern.client import BrassLantern
from brass_lantern.providers import get_provider


def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(
        prog="brass-lantern",
        description="Async LLM client for the Grue game server.",
    )
    parser.add_argument(
        "--uri",
        default=os.getenv("GRUE_URI", "ws://localhost:8080"),
        help="Grue WebSocket URI (default: ws://localhost:8080)",
    )
    parser.add_argument(
        "--provider",
        default=os.getenv("AI_PROVIDER", "openai"),
        choices=["openai", "anthropic", "gemini"],
        help="LLM provider to use (default: openai)",
    )
    parser.add_argument(
        "--model",
        default=os.getenv("AI_MODEL"),
        help="Model name override (uses provider default if unset)",
    )
    args = parser.parse_args()

    api_key = os.getenv("AI_API_KEY")
    if not api_key:
        print(
            "Error: AI_API_KEY is not set. "
            "Add it to your .env file or export it as an environment variable.",
            file=sys.stderr,
        )
        sys.exit(1)

    provider = get_provider(args.provider, api_key, args.model)
    client = BrassLantern(args.uri, provider)

    try:
        asyncio.run(client.connect_and_play())
    except KeyboardInterrupt:
        print("\n[brass-lantern] Interrupted — goodbye.")


if __name__ == "__main__":
    main()

"""Entry point for running claude_bridge as a module: python -m claude_bridge."""

from __future__ import annotations

import argparse
import asyncio

from claude_bridge.log import configure_logging
from claude_bridge.proxy import start_proxy

# Import providers so they register themselves in the PROVIDERS dict.
import claude_bridge.providers.openai  # noqa: F401
import claude_bridge.providers.xai  # noqa: F401


def main() -> None:
    """Parse CLI args and run the proxy server."""
    parser = argparse.ArgumentParser(
        prog="claude-bridge",
        description="Claude Bridge — use Claude Code with any LLM provider",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=9999,
        help="Port to listen on (default: 9999)",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host to bind to (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--provider",
        default=None,
        help="Direct mode: always use this provider (e.g., 'openai')",
    )
    parser.add_argument(
        "--auto",
        action="store_true",
        default=True,
        help="Auto mode: passthrough to Anthropic, failover on error",
    )
    args = parser.parse_args()
    configure_logging()

    asyncio.run(_run(host=args.host, port=args.port, provider_name=args.provider))


async def _run(*, host: str, port: int, provider_name: str | None = None) -> None:
    """Start the server and serve until interrupted."""
    server = await start_proxy(host=host, port=port, provider_name=provider_name)
    async with server:
        await server.serve_forever()


if __name__ == "__main__":
    main()

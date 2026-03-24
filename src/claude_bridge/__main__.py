"""Entry point for running claude_bridge as a module: python -m claude_bridge."""

from __future__ import annotations

import argparse
import asyncio
import os

# Import providers so they register themselves in the PROVIDERS dict.
import claude_bridge.providers.gemini
import claude_bridge.providers.openai
import claude_bridge.providers.xai  # noqa: F401
from claude_bridge.log import configure_logging, get_logger
from claude_bridge.proxy import start_proxy

logger = get_logger("main")


def _detect_auth_mode() -> tuple[str, str | None]:
    """Detect OpenAI auth mode from the environment.

    Returns ``(auth_mode, api_key)`` where:
    - ``("api_key", "<key>")`` when ``OPENAI_API_KEY`` is set and non-empty
    - ``("codex_oauth", None)`` otherwise
    """
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if api_key:
        return "api_key", api_key
    return "codex_oauth", None


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

    # Detect auth mode for OpenAI provider
    auth_mode, api_key = _detect_auth_mode()
    if auth_mode == "api_key":
        logger.info("Auth mode: api_key (OPENAI_API_KEY detected)")
    else:
        logger.info("Auth mode: codex_oauth (no OPENAI_API_KEY — falling back to Codex OAuth)")

    asyncio.run(
        _run(
            host=args.host,
            port=args.port,
            provider_name=args.provider,
            auth_mode=auth_mode,
            api_key=api_key,
        )
    )


async def _run(
    *,
    host: str,
    port: int,
    provider_name: str | None = None,
    auth_mode: str = "codex_oauth",
    api_key: str | None = None,
) -> None:
    """Start the server and serve until interrupted."""
    provider_kwargs = {"auth_mode": auth_mode}
    if api_key is not None:
        provider_kwargs["api_key"] = api_key
    server = await start_proxy(
        host=host,
        port=port,
        provider_name=provider_name,
        provider_kwargs=provider_kwargs,
    )
    async with server:
        await server.serve_forever()


if __name__ == "__main__":
    main()

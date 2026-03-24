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


def _detect_openai_auth_mode() -> tuple[str, str | None]:
    """Detect OpenAI auth mode from the environment.

    Returns ``(auth_mode, api_key)`` where:
    - ``("api_key", "<key>")`` when ``OPENAI_API_KEY`` is set and non-empty
    - ``("codex_oauth", None)`` otherwise
    """
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if api_key:
        return "api_key", api_key
    return "codex_oauth", None


def _detect_gemini_auth_mode() -> tuple[str, str | None]:
    """Detect Gemini auth mode from the environment.

    Returns ``(auth_mode, api_key)`` where:
    - ``("api_key", "<key>")`` when ``GEMINI_API_KEY`` is set and non-empty
    - ``("gemini_oauth", None)`` otherwise
    """
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if api_key:
        return "api_key", api_key
    return "gemini_oauth", None


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

    # Detect auth mode per provider
    provider = args.provider
    provider_kwargs: dict = {}

    if provider == "gemini":
        auth_mode, api_key = _detect_gemini_auth_mode()
        provider_kwargs["auth_mode"] = auth_mode
        if api_key:
            provider_kwargs["api_key"] = api_key
        if auth_mode == "api_key":
            logger.info("Gemini auth: api_key (GEMINI_API_KEY detected)")
        else:
            logger.info("Gemini auth: gemini_oauth (using Gemini CLI subscription)")
    else:
        auth_mode, api_key = _detect_openai_auth_mode()
        provider_kwargs["auth_mode"] = auth_mode
        if api_key:
            provider_kwargs["api_key"] = api_key
        if auth_mode == "api_key":
            logger.info("Auth mode: api_key (OPENAI_API_KEY detected)")
        else:
            logger.info("Auth mode: codex_oauth (no OPENAI_API_KEY — falling back to Codex OAuth)")

    asyncio.run(
        _run(
            host=args.host,
            port=args.port,
            provider_name=provider,
            provider_kwargs=provider_kwargs,
        )
    )


async def _run(
    *,
    host: str,
    port: int,
    provider_name: str | None = None,
    provider_kwargs: dict | None = None,
) -> None:
    """Start the server and serve until interrupted."""
    server = await start_proxy(
        host=host,
        port=port,
        provider_name=provider_name,
        provider_kwargs=provider_kwargs or {},
    )
    async with server:
        await server.serve_forever()


if __name__ == "__main__":
    main()

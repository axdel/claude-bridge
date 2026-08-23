"""Entry point for running claude_bridge as a module: python -m claude_bridge."""

from __future__ import annotations

import argparse
import asyncio

import claude_bridge.config as config

# Import implemented providers so they register themselves in the PROVIDERS dict.
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
    api_key = config.openai_api_key()
    if api_key:
        return "api_key", api_key
    return "codex_oauth", None


def _build_provider_kwargs(provider_name: str | None) -> dict:
    """Return the direct-mode constructor kwargs for the selected provider.

    xAI resolves its subscription OAuth from ``~/.grok`` through a no-arg constructor,
    so it takes no kwargs. OpenAI (and the default/auto path) carries its detected auth
    mode and, in api_key mode, the key.
    """
    if provider_name == "xai":
        return {}
    auth_mode, api_key = _detect_openai_auth_mode()
    kwargs: dict = {"auth_mode": auth_mode}
    if api_key:
        kwargs["api_key"] = api_key
    return kwargs


def _auth_mode_log_message(provider_name: str | None, provider_kwargs: dict) -> str:
    """Return the auth-mode log line, naming the mode only — never the credential."""
    if provider_name == "xai":
        return "Auth mode: grok_oauth (xAI subscription via ~/.grok)"
    if provider_kwargs.get("auth_mode") == "api_key":
        return "Auth mode: api_key (OPENAI_API_KEY detected)"
    return "Auth mode: codex_oauth (no OPENAI_API_KEY — falling back to Codex OAuth)"


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

    # Resolve auth-mode constructor kwargs and the log line per selected provider.
    provider = args.provider
    provider_kwargs = _build_provider_kwargs(provider)
    logger.info(_auth_mode_log_message(provider, provider_kwargs))

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
    server, client = await start_proxy(
        host=host,
        port=port,
        provider_name=provider_name,
        provider_kwargs=provider_kwargs or {},
    )
    try:
        async with server:
            await server.serve_forever()
    finally:
        await client.aclose()


if __name__ == "__main__":
    main()

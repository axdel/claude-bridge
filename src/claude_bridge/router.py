"""Circuit breaker router for Anthropic → fallback provider failover."""

from __future__ import annotations

import asyncio
import enum
import time

from claude_bridge.log import get_logger

logger = get_logger("router")


class RouterState(enum.Enum):
    CLOSED = "closed"  # Use Anthropic (normal)
    OPEN = "open"  # Use fallback provider
    HALF_OPEN = "half_open"  # Probe Anthropic with one request


class Router:
    """Circuit breaker that controls routing between Anthropic and a fallback.

    State transitions:
        CLOSED → OPEN:      after *failure_threshold* consecutive failures
        OPEN → HALF_OPEN:   after *cooldown_seconds* since last failure
        HALF_OPEN → CLOSED: on success (probe passed)
        HALF_OPEN → OPEN:   on failure (probe failed, cooldown resets)
    """

    def __init__(
        self,
        failure_threshold: int = 2,
        cooldown_seconds: float = 60.0,
    ) -> None:
        self._failure_threshold = failure_threshold
        self._cooldown_seconds = cooldown_seconds
        self._state = RouterState.CLOSED
        self._consecutive_failures: int = 0
        self._last_failure_time: float = 0.0
        self._probe_in_flight: bool = False
        self._lock = asyncio.Lock()

    @property
    def state(self) -> RouterState:
        return self._state

    async def record_success(self) -> None:
        """Record a successful Anthropic response. Resets failure count."""
        async with self._lock:
            old = self._state
            self._consecutive_failures = 0
            self._probe_in_flight = False
            if self._state in (RouterState.HALF_OPEN, RouterState.CLOSED):
                self._state = RouterState.CLOSED
            if old is not self._state:
                self._log_transition(old, self._state, "probe success")

    async def record_failure(self) -> None:
        """Record an Anthropic failure. May transition CLOSED->OPEN or HALF_OPEN->OPEN."""
        async with self._lock:
            old = self._state
            self._consecutive_failures += 1
            self._last_failure_time = time.monotonic()
            self._probe_in_flight = False

            if self._state is RouterState.HALF_OPEN:
                self._state = RouterState.OPEN
                self._log_transition(
                    old,
                    self._state,
                    "probe failed",
                )
            elif (
                self._state is RouterState.CLOSED
                and self._consecutive_failures >= self._failure_threshold
            ):
                self._state = RouterState.OPEN
                self._log_transition(
                    old,
                    self._state,
                    f"{self._consecutive_failures} consecutive failures",
                )

    def should_use_fallback(self) -> bool:
        """Return True if requests should go to the fallback provider.

        In HALF_OPEN: returns False for the first probe request, True for
        concurrent ones.  Also auto-transitions OPEN -> HALF_OPEN when the
        cooldown has expired.
        """
        if self._state is RouterState.CLOSED:
            return False

        if self._state is RouterState.OPEN:
            elapsed = time.monotonic() - self._last_failure_time
            if elapsed >= self._cooldown_seconds:
                old = self._state
                self._state = RouterState.HALF_OPEN
                self._probe_in_flight = True
                self._log_transition(old, self._state, "cooldown expired")
                return False  # first caller gets the probe
            return True

        # HALF_OPEN
        if not self._probe_in_flight:
            self._probe_in_flight = True
            return False  # first caller gets the probe
        return True  # concurrent callers use fallback

    @staticmethod
    def is_failover_eligible(request: dict) -> tuple[bool, str]:
        """Check if a request can be served by the fallback.

        Returns (eligible, reason).
        Rejects: requests with 'thinking' in the body, 'output_config' present.
        """
        if request.get("thinking"):
            return False, "extended thinking not supported in fallback"
        if request.get("output_config"):
            return False, "output_config not supported in fallback"
        return True, ""

    @staticmethod
    def _log_transition(old: RouterState, new: RouterState, reason: str) -> None:
        logger.info("%s -> %s (%s)", old.value, new.value, reason)

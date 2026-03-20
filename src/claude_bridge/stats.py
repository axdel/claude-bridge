"""Bridge runtime metrics — thread-safe, ephemeral counters.

Tracks request volume, errors, latency, token usage, and failovers.
All state is in-memory and resets on restart. ``snapshot()`` returns
a JSON-serializable dict for the ``/stats`` endpoint.

Thread-safe via ``threading.Lock`` because the proxy uses
``asyncio.to_thread`` for HTTP calls — mutations can come from
both the event loop and thread pool workers.
"""

from __future__ import annotations

import threading
import time
from datetime import datetime, timezone


class BridgeStats:
    """Aggregate metrics for bridge requests."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._started_at = datetime.now(timezone.utc)
        self._started_mono = time.monotonic()
        self._requests_total = 0
        self._errors_total = 0
        self._upstream_attempts = 0
        self._failovers = 0
        self._tokens_in = 0
        self._tokens_out = 0
        self._latency_total_ms = 0.0
        self._provider_name: str | None = None
        self._model: str | None = None

    def record_request(self) -> None:
        """Increment the request counter."""
        with self._lock:
            self._requests_total += 1

    def record_response(
        self,
        status_code: int,
        latency_ms: float,
        tokens_in: int,
        tokens_out: int,
    ) -> None:
        """Record a completed upstream/provider response."""
        with self._lock:
            self._upstream_attempts += 1
            self._latency_total_ms += latency_ms
            self._tokens_in += tokens_in
            self._tokens_out += tokens_out
            if status_code >= 500:
                self._errors_total += 1

    def record_error(self) -> None:
        """Record a connection-level error (no HTTP response received)."""
        with self._lock:
            self._errors_total += 1

    def record_failover(self) -> None:
        """Record a successful failover to a fallback provider."""
        with self._lock:
            self._failovers += 1

    def set_provider_info(self, provider_name: str, model: str) -> None:
        """Set the active provider name and model (called on first provider request)."""
        with self._lock:
            self._provider_name = provider_name
            self._model = model

    def snapshot(self) -> dict:
        """Return a JSON-serializable snapshot of all metrics."""
        with self._lock:
            requests = self._requests_total
            latency_total = self._latency_total_ms
            attempts = self._upstream_attempts
            return {
                "requests_total": requests,
                "errors_total": self._errors_total,
                "upstream_attempts": attempts,
                "failovers": self._failovers,
                "tokens_in": self._tokens_in,
                "tokens_out": self._tokens_out,
                "latency_total_ms": latency_total,
                "latency_avg_ms": (
                    round(latency_total / attempts, 1) if attempts else 0.0
                ),
                "started_at": self._started_at.isoformat(),
                "uptime_seconds": round(time.monotonic() - self._started_mono, 1),
                "provider_name": self._provider_name,
                "model": self._model,
            }

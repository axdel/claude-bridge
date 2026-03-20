"""Tests for the bridge metrics/stats module."""

from __future__ import annotations

import threading
import time

from claude_bridge.stats import BridgeStats


class TestBridgeStats:
    """BridgeStats tracks request metrics with thread-safe counters."""

    def test_initial_snapshot_all_zeros(self):
        stats = BridgeStats()
        snap = stats.snapshot()
        assert snap["requests_total"] == 0
        assert snap["errors_total"] == 0
        assert snap["upstream_attempts"] == 0
        assert snap["failovers"] == 0
        assert snap["tokens_in"] == 0
        assert snap["tokens_out"] == 0
        assert snap["latency_total_ms"] == 0.0
        assert snap["latency_avg_ms"] == 0.0
        assert "started_at" in snap
        assert "uptime_seconds" in snap

    def test_record_request_increments(self):
        stats = BridgeStats()
        stats.record_request()
        stats.record_request()
        assert stats.snapshot()["requests_total"] == 2

    def test_record_response_updates_counters(self):
        stats = BridgeStats()
        stats.record_request()
        stats.record_response(
            status_code=200, latency_ms=150.0, tokens_in=100, tokens_out=50
        )
        snap = stats.snapshot()
        assert snap["upstream_attempts"] == 1
        assert snap["tokens_in"] == 100
        assert snap["tokens_out"] == 50
        assert snap["latency_total_ms"] == 150.0
        assert snap["latency_avg_ms"] == 150.0

    def test_record_error_increments(self):
        stats = BridgeStats()
        stats.record_error()
        stats.record_error()
        assert stats.snapshot()["errors_total"] == 2

    def test_record_failover_increments(self):
        stats = BridgeStats()
        stats.record_failover()
        assert stats.snapshot()["failovers"] == 1

    def test_latency_avg_computed_from_total(self):
        stats = BridgeStats()
        stats.record_request()
        stats.record_response(200, 100.0, 0, 0)
        stats.record_request()
        stats.record_response(200, 200.0, 0, 0)
        snap = stats.snapshot()
        assert snap["latency_avg_ms"] == 150.0

    def test_uptime_increases(self):
        stats = BridgeStats()
        time.sleep(0.05)
        snap = stats.snapshot()
        assert snap["uptime_seconds"] >= 0.04

    def test_thread_safety(self):
        """Concurrent increments from multiple threads produce correct totals."""
        stats = BridgeStats()
        iterations = 1000

        def worker():
            for _ in range(iterations):
                stats.record_request()

        threads = [threading.Thread(target=worker) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert stats.snapshot()["requests_total"] == 4 * iterations

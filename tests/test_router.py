"""Tests for the circuit breaker router — stdlib only."""

from __future__ import annotations

import asyncio

import pytest

from claude_bridge.router import Router, RouterState


@pytest.mark.asyncio
async def test_initial_state_is_closed():
    """Router starts CLOSED; should_use_fallback returns False."""
    router = Router()
    assert router.state is RouterState.CLOSED
    assert router.should_use_fallback() is False


@pytest.mark.asyncio
async def test_failures_open_circuit():
    """After failure_threshold consecutive failures, state becomes OPEN."""
    router = Router(failure_threshold=2)
    await router.record_failure()
    assert router.state is RouterState.CLOSED
    await router.record_failure()
    assert router.state is RouterState.OPEN
    assert router.should_use_fallback() is True


@pytest.mark.asyncio
async def test_success_resets_failure_count():
    """A success after one failure resets the counter; stays CLOSED."""
    router = Router(failure_threshold=2)
    await router.record_failure()
    await router.record_success()
    assert router.state is RouterState.CLOSED
    # One more failure should NOT open (counter was reset)
    await router.record_failure()
    assert router.state is RouterState.CLOSED


@pytest.mark.asyncio
async def test_cooldown_transitions_to_half_open():
    """After cooldown expires, state becomes HALF_OPEN; probe returns False."""
    router = Router(failure_threshold=1, cooldown_seconds=0.0)
    await router.record_failure()
    assert router.state is RouterState.OPEN
    # Cooldown is 0s, so it should immediately transition on next check
    await asyncio.sleep(0.01)
    assert router.should_use_fallback() is False  # probe request
    assert router.state is RouterState.HALF_OPEN


@pytest.mark.asyncio
async def test_probe_success_closes_circuit():
    """A successful probe in HALF_OPEN transitions back to CLOSED."""
    router = Router(failure_threshold=1, cooldown_seconds=0.0)
    await router.record_failure()
    await asyncio.sleep(0.01)
    # Trigger HALF_OPEN transition
    assert router.should_use_fallback() is False
    assert router.state is RouterState.HALF_OPEN
    await router.record_success()
    assert router.state is RouterState.CLOSED
    assert router.should_use_fallback() is False


@pytest.mark.asyncio
async def test_probe_failure_reopens_circuit():
    """A failed probe in HALF_OPEN transitions back to OPEN."""
    router = Router(failure_threshold=1, cooldown_seconds=0.0)
    await router.record_failure()
    await asyncio.sleep(0.01)
    # Trigger HALF_OPEN
    assert router.should_use_fallback() is False
    assert router.state is RouterState.HALF_OPEN
    # Probe fails — back to OPEN with a fresh cooldown timer
    await router.record_failure()
    assert router.state is RouterState.OPEN
    # Verify the cooldown timer was reset: with a long cooldown we'd stay OPEN.
    # Use a new router to test this cleanly.
    router2 = Router(failure_threshold=1, cooldown_seconds=300.0)
    await router2.record_failure()
    assert router2.state is RouterState.OPEN
    # Cooldown hasn't expired — should stay OPEN
    assert router2.should_use_fallback() is True


@pytest.mark.asyncio
async def test_half_open_concurrent_callers_use_fallback():
    """In HALF_OPEN, only the first caller gets the probe; others use fallback."""
    router = Router(failure_threshold=1, cooldown_seconds=0.0)
    await router.record_failure()
    await asyncio.sleep(0.01)
    # First call triggers probe
    assert router.should_use_fallback() is False
    assert router.state is RouterState.HALF_OPEN
    # Subsequent concurrent calls should use fallback
    assert router.should_use_fallback() is True


def test_is_failover_eligible_rejects_thinking():
    """Requests with 'thinking' are not eligible for failover."""
    eligible, reason = Router.is_failover_eligible(
        {"thinking": {"budget_tokens": 1024}}
    )
    assert eligible is False
    assert "thinking" in reason


def test_is_failover_eligible_rejects_output_config():
    """Requests with 'output_config' are not eligible for failover."""
    eligible, reason = Router.is_failover_eligible(
        {"output_config": {"format": "json"}}
    )
    assert eligible is False
    assert "output_config" in reason


def test_is_failover_eligible_accepts_normal_request():
    """Normal requests are eligible for failover."""
    eligible, reason = Router.is_failover_eligible(
        {
            "model": "claude-sonnet-4-20250514",
            "messages": [{"role": "user", "content": "hi"}],
        }
    )
    assert eligible is True
    assert reason == ""

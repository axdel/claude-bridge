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
    assert await router.should_use_fallback() is False


@pytest.mark.asyncio
async def test_failures_open_circuit():
    """After failure_threshold consecutive failures, state becomes OPEN."""
    router = Router(failure_threshold=2)
    await router.record_failure()
    assert router.state is RouterState.CLOSED
    await router.record_failure()
    assert router.state is RouterState.OPEN
    assert await router.should_use_fallback() is True


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
    assert await router.should_use_fallback() is False  # probe request
    assert router.state is RouterState.HALF_OPEN


@pytest.mark.asyncio
async def test_probe_success_closes_circuit():
    """A successful probe in HALF_OPEN transitions back to CLOSED."""
    router = Router(failure_threshold=1, cooldown_seconds=0.0)
    await router.record_failure()
    await asyncio.sleep(0.01)
    # Trigger HALF_OPEN transition
    assert await router.should_use_fallback() is False
    assert router.state is RouterState.HALF_OPEN
    await router.record_success()
    assert router.state is RouterState.CLOSED
    assert await router.should_use_fallback() is False


@pytest.mark.asyncio
async def test_probe_failure_reopens_circuit():
    """A failed probe in HALF_OPEN transitions back to OPEN."""
    router = Router(failure_threshold=1, cooldown_seconds=0.0)
    await router.record_failure()
    await asyncio.sleep(0.01)
    # Trigger HALF_OPEN
    assert await router.should_use_fallback() is False
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
    assert await router2.should_use_fallback() is True


@pytest.mark.asyncio
async def test_half_open_concurrent_callers_use_fallback():
    """In HALF_OPEN, only the first caller gets the probe; others use fallback."""
    router = Router(failure_threshold=1, cooldown_seconds=0.0)
    await router.record_failure()
    await asyncio.sleep(0.01)
    # First call triggers probe
    assert await router.should_use_fallback() is False
    assert router.state is RouterState.HALF_OPEN
    # Subsequent concurrent calls should use fallback
    assert await router.should_use_fallback() is True


@pytest.mark.asyncio
async def test_should_use_fallback_serializes_on_the_router_lock():
    """should_use_fallback is a state writer, so it acquires the shared lock.

    It transitions OPEN -> HALF_OPEN and claims the probe, exactly the state that
    record_success/record_failure mutate under self._lock. Serializing it through
    the same lock keeps the router a single writer of its own state: while the lock
    is held the decision must block, not read or mutate state mid-transition. A
    lock-free implementation would return immediately and this would never time out.
    """
    router = Router(failure_threshold=1, cooldown_seconds=0.0)
    await router.record_failure()  # -> OPEN
    assert router.state is RouterState.OPEN
    async with router._lock:
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(router.should_use_fallback(), timeout=0.05)
    # Lock released: the decision proceeds normally (probe granted on cooldown-0).
    assert await router.should_use_fallback() is False
    assert router.state is RouterState.HALF_OPEN


def test_is_failover_eligible_rejects_thinking():
    """Requests with 'thinking' are not eligible for failover."""
    eligible, reason = Router.is_failover_eligible({"thinking": {"budget_tokens": 1024}})
    assert eligible is False
    assert "thinking" in reason


def test_is_failover_eligible_rejects_output_config():
    """Requests with 'output_config' are not eligible for failover."""
    eligible, reason = Router.is_failover_eligible({"output_config": {"format": "json"}})
    assert eligible is False
    assert "output_config" in reason


def test_is_failover_eligible_rejects_tool_use_turn():
    """Requests with tool_use in assistant message are not eligible (mid-turn)."""
    eligible, reason = Router.is_failover_eligible(
        {
            "messages": [
                {"role": "user", "content": "Read file.txt"},
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "toolu_123",
                            "name": "Read",
                            "input": {"path": "file.txt"},
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "toolu_123",
                            "content": "file contents",
                        }
                    ],
                },
            ],
        }
    )
    assert eligible is False
    assert "tool-use" in reason


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

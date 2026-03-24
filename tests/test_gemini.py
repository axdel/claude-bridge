"""Tests for the Gemini provider — auth, translation, and streaming."""

from __future__ import annotations

import pytest

# --- Auth tests ---


class TestGeminiAuth:
    """GeminiProvider authentication via GEMINI_API_KEY."""

    def test_authenticate_returns_api_key_header(self, monkeypatch):
        monkeypatch.setenv("GEMINI_API_KEY", "test-gemini-key-placeholder")
        from claude_bridge.providers.gemini import GeminiProvider

        provider = GeminiProvider()
        import asyncio

        headers = asyncio.run(provider.authenticate())
        assert headers == {"x-goog-api-key": "test-gemini-key-placeholder"}

    def test_authenticate_missing_key_raises(self, monkeypatch):
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        from claude_bridge.providers.gemini import GeminiProvider

        provider = GeminiProvider()
        import asyncio

        with pytest.raises(ValueError, match="GEMINI_API_KEY"):
            asyncio.run(provider.authenticate())

    def test_provider_registered_in_providers(self):
        from claude_bridge.provider import PROVIDERS

        assert "gemini" in PROVIDERS

    def test_provider_has_correct_name(self):
        from claude_bridge.providers.gemini import GeminiProvider

        assert GeminiProvider.name == "gemini"

    def test_provider_endpoint_contains_model(self, monkeypatch):
        monkeypatch.setenv("GEMINI_API_KEY", "test-gemini-key-placeholder")
        from claude_bridge.providers.gemini import GeminiProvider

        provider = GeminiProvider()
        assert "generateContent" not in provider.endpoint
        assert "v1beta/models/" in provider.endpoint


# --- Request translation tests ---


class TestAnthropicToGeminiTextOnly:
    """Text-only request translation (messages + system + model mapping)."""

    def test_basic_text_message(self):
        from claude_bridge.providers.gemini import anthropic_to_gemini

        request = {
            "model": "claude-opus-4-6",
            "max_tokens": 1024,
            "messages": [{"role": "user", "content": [{"type": "text", "text": "Hello"}]}],
        }
        result, _warnings = anthropic_to_gemini(request)
        assert result["contents"][0]["role"] == "user"
        assert result["contents"][0]["parts"][0]["text"] == "Hello"

    def test_system_string_to_system_instruction(self):
        from claude_bridge.providers.gemini import anthropic_to_gemini

        request = {
            "model": "claude-sonnet-4-6",
            "max_tokens": 100,
            "system": "You are helpful.",
            "messages": [{"role": "user", "content": "Hi"}],
        }
        result, _ = anthropic_to_gemini(request)
        assert result["system_instruction"]["parts"][0]["text"] == "You are helpful."

    def test_system_block_list_joined(self):
        from claude_bridge.providers.gemini import anthropic_to_gemini

        request = {
            "model": "claude-sonnet-4-6",
            "max_tokens": 100,
            "system": [
                {"type": "text", "text": "Rule 1."},
                {"type": "text", "text": "Rule 2."},
            ],
            "messages": [{"role": "user", "content": "Hi"}],
        }
        result, _ = anthropic_to_gemini(request)
        assert result["system_instruction"]["parts"][0]["text"] == "Rule 1.\nRule 2."

    def test_assistant_role_mapped_to_model(self):
        from claude_bridge.providers.gemini import anthropic_to_gemini

        request = {
            "model": "claude-sonnet-4-6",
            "max_tokens": 100,
            "messages": [
                {"role": "user", "content": "Hi"},
                {"role": "assistant", "content": "Hello!"},
                {"role": "user", "content": "How are you?"},
            ],
        }
        result, _ = anthropic_to_gemini(request)
        assert result["contents"][1]["role"] == "model"

    def test_model_mapping(self):
        from claude_bridge.providers.gemini import anthropic_to_gemini

        request = {
            "model": "claude-opus-4-6",
            "max_tokens": 100,
            "messages": [{"role": "user", "content": "Hi"}],
        }
        result, _ = anthropic_to_gemini(request)
        assert result["model"] == "gemini-2.5-pro"

    def test_unknown_model_uses_default(self):
        from claude_bridge.providers.gemini import anthropic_to_gemini

        request = {
            "model": "claude-unknown-999",
            "max_tokens": 100,
            "messages": [{"role": "user", "content": "Hi"}],
        }
        result, _ = anthropic_to_gemini(request)
        assert "gemini" in result["model"]

    def test_string_content_shorthand(self):
        from claude_bridge.providers.gemini import anthropic_to_gemini

        request = {
            "model": "claude-sonnet-4-6",
            "max_tokens": 100,
            "messages": [{"role": "user", "content": "Hello shorthand"}],
        }
        result, _ = anthropic_to_gemini(request)
        assert result["contents"][0]["parts"][0]["text"] == "Hello shorthand"


class TestAnthropicToGeminiTools:
    """Tool definition and tool use/result translation."""

    def test_tool_definitions_translated(self):
        from claude_bridge.providers.gemini import anthropic_to_gemini

        request = {
            "model": "claude-sonnet-4-6",
            "max_tokens": 100,
            "tools": [
                {
                    "name": "get_weather",
                    "description": "Get weather",
                    "input_schema": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                    },
                }
            ],
            "messages": [{"role": "user", "content": "Weather?"}],
        }
        result, _ = anthropic_to_gemini(request)
        decls = result["tools"][0]["function_declarations"]
        assert len(decls) == 1
        assert decls[0]["name"] == "get_weather"
        assert decls[0]["parameters"]["type"] == "object"

    def test_tool_use_to_function_call(self):
        from claude_bridge.providers.gemini import anthropic_to_gemini

        request = {
            "model": "claude-sonnet-4-6",
            "max_tokens": 100,
            "messages": [
                {"role": "user", "content": "Weather?"},
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "toolu_abc123",
                            "name": "get_weather",
                            "input": {"city": "NYC"},
                        }
                    ],
                },
            ],
        }
        result, _ = anthropic_to_gemini(request)
        model_parts = result["contents"][1]["parts"]
        fc = model_parts[0]["functionCall"]
        assert fc["name"] == "get_weather"
        assert fc["args"] == {"city": "NYC"}

    def test_tool_result_to_function_response(self):
        from claude_bridge.providers.gemini import anthropic_to_gemini

        request = {
            "model": "claude-sonnet-4-6",
            "max_tokens": 100,
            "messages": [
                {"role": "user", "content": "Weather?"},
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "toolu_abc123",
                            "name": "get_weather",
                            "input": {"city": "NYC"},
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "toolu_abc123",
                            "content": '{"temp": "72F"}',
                        }
                    ],
                },
            ],
        }
        result, _ = anthropic_to_gemini(request)
        # tool_result is in the 3rd message (index 2)
        fr = result["contents"][2]["parts"][0]["functionResponse"]
        assert fr["name"] == "get_weather"
        assert fr["response"] == {"temp": "72F"}

    def test_tool_result_string_not_json_wrapped(self):
        from claude_bridge.providers.gemini import anthropic_to_gemini

        request = {
            "model": "claude-sonnet-4-6",
            "max_tokens": 100,
            "messages": [
                {"role": "user", "content": "Do something"},
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "toolu_abc123",
                            "name": "run_cmd",
                            "input": {"cmd": "ls"},
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "toolu_abc123",
                            "content": "plain text result",
                        }
                    ],
                },
            ],
        }
        result, _ = anthropic_to_gemini(request)
        fr = result["contents"][2]["parts"][0]["functionResponse"]
        assert fr["response"] == {"result": "plain text result"}

    def test_stripped_keys_produce_warnings(self):
        from claude_bridge.providers.gemini import anthropic_to_gemini

        request = {
            "model": "claude-sonnet-4-6",
            "max_tokens": 100,
            "output_config": {"format": "json"},
            "messages": [{"role": "user", "content": "Hi"}],
        }
        _, warnings = anthropic_to_gemini(request)
        assert any("output_config" in w for w in warnings)

    def test_thinking_config_warning(self):
        from claude_bridge.providers.gemini import anthropic_to_gemini

        request = {
            "model": "claude-sonnet-4-6",
            "max_tokens": 100,
            "thinking": {"type": "enabled", "budget_tokens": 5000},
            "messages": [{"role": "user", "content": "Hi"}],
        }
        _, warnings = anthropic_to_gemini(request)
        assert any("thinking" in w.lower() for w in warnings)


# --- Response translation tests ---


class TestGeminiToAnthropicText:
    """Gemini response → Anthropic Messages format."""

    def test_basic_text_response(self):
        from claude_bridge.providers.gemini import gemini_to_anthropic

        response = {
            "candidates": [
                {
                    "content": {"parts": [{"text": "Hello!"}], "role": "model"},
                    "finishReason": "STOP",
                }
            ],
            "usageMetadata": {"promptTokenCount": 5, "candidatesTokenCount": 3},
            "responseId": "resp_123",
            "modelVersion": "gemini-2.5-pro",
        }
        result = gemini_to_anthropic(response)
        assert result["type"] == "message"
        assert result["role"] == "assistant"
        assert result["stop_reason"] == "end_turn"
        assert result["content"][0]["type"] == "text"
        assert result["content"][0]["text"] == "Hello!"
        assert result["usage"]["input_tokens"] == 5
        assert result["usage"]["output_tokens"] == 3

    def test_max_tokens_stop_reason(self):
        from claude_bridge.providers.gemini import gemini_to_anthropic

        response = {
            "candidates": [
                {
                    "content": {"parts": [{"text": "partial"}], "role": "model"},
                    "finishReason": "MAX_TOKENS",
                }
            ],
            "usageMetadata": {},
            "responseId": "resp_456",
        }
        result = gemini_to_anthropic(response)
        assert result["stop_reason"] == "max_tokens"

    def test_safety_refusal_synthesized(self):
        from claude_bridge.providers.gemini import gemini_to_anthropic

        response = {
            "candidates": [{"finishReason": "SAFETY"}],
            "usageMetadata": {},
            "responseId": "resp_safe",
        }
        result = gemini_to_anthropic(response)
        assert result["stop_reason"] == "end_turn"
        assert len(result["content"]) == 1
        assert "safety" in result["content"][0]["text"].lower()

    def test_empty_candidates(self):
        from claude_bridge.providers.gemini import gemini_to_anthropic

        response = {"candidates": [], "usageMetadata": {}, "responseId": "resp_empty"}
        result = gemini_to_anthropic(response)
        assert result["content"] == []
        assert result["stop_reason"] == "end_turn"


class TestGeminiToAnthropicToolUse:
    """Gemini functionCall → Anthropic tool_use blocks."""

    def test_function_call_translated(self):
        from claude_bridge.providers.gemini import gemini_to_anthropic

        response = {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {
                                "functionCall": {
                                    "id": "abc123",
                                    "name": "get_weather",
                                    "args": {"city": "NYC"},
                                }
                            }
                        ],
                        "role": "model",
                    },
                    "finishReason": "STOP",
                }
            ],
            "usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 5},
            "responseId": "resp_fc",
        }
        result = gemini_to_anthropic(response)
        assert result["stop_reason"] == "tool_use"
        block = result["content"][0]
        assert block["type"] == "tool_use"
        assert block["name"] == "get_weather"
        assert block["input"] == {"city": "NYC"}
        assert block["id"].startswith("call_gemini_")

    def test_function_call_without_id_gets_synthetic(self):
        from claude_bridge.providers.gemini import gemini_to_anthropic

        response = {
            "candidates": [
                {
                    "content": {
                        "parts": [{"functionCall": {"name": "run_cmd", "args": {"cmd": "ls"}}}],
                        "role": "model",
                    },
                    "finishReason": "STOP",
                }
            ],
            "usageMetadata": {},
            "responseId": "resp_noid",
        }
        result = gemini_to_anthropic(response)
        block = result["content"][0]
        assert block["id"].startswith("call_gemini_")

    def test_thought_signature_encoded_in_id(self):
        from claude_bridge.providers.gemini import _decode_tool_id, gemini_to_anthropic

        response = {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {
                                "functionCall": {
                                    "id": "fc99",
                                    "name": "edit",
                                    "args": {},
                                },
                                "thoughtSignature": "secret-sig-data",
                            }
                        ],
                        "role": "model",
                    },
                    "finishReason": "STOP",
                }
            ],
            "usageMetadata": {},
            "responseId": "resp_sig",
        }
        result = gemini_to_anthropic(response)
        tool_id = result["content"][0]["id"]
        gemini_id, sig = _decode_tool_id(tool_id)
        assert gemini_id == "fc99"
        assert sig == "secret-sig-data"


# --- SSE stream translation tests ---


class TestGeminiStreamTranslation:
    """Gemini SSE chunks → Anthropic SSE events."""

    def test_text_stream_produces_correct_events(self):
        from claude_bridge.providers.gemini import translate_gemini_sse_chunk

        state: dict = {}
        chunk1 = {
            "candidates": [{"content": {"parts": [{"text": "Hello"}], "role": "model"}}],
            "usageMetadata": {"promptTokenCount": 5},
            "responseId": "r1",
            "modelVersion": "gemini-2.5-pro",
        }
        events = translate_gemini_sse_chunk(chunk1, state)
        event_types = [e["event"] for e in events]
        assert "message_start" in event_types
        assert "content_block_start" in event_types
        assert "content_block_delta" in event_types

    def test_final_chunk_emits_stop_events(self):
        from claude_bridge.providers.gemini import translate_gemini_sse_chunk

        state: dict = {"started": True, "text_block_open": True, "block_index": 0}
        final = {
            "candidates": [{"finishReason": "STOP"}],
            "usageMetadata": {"candidatesTokenCount": 10},
        }
        events = translate_gemini_sse_chunk(final, state)
        event_types = [e["event"] for e in events]
        assert "content_block_stop" in event_types
        assert "message_delta" in event_types
        assert "message_stop" in event_types

    def test_function_call_in_stream(self):
        from claude_bridge.providers.gemini import translate_gemini_sse_chunk

        state: dict = {"started": True, "block_index": 0}
        chunk = {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {
                                "functionCall": {
                                    "id": "fc1",
                                    "name": "edit",
                                    "args": {"path": "a.py"},
                                }
                            }
                        ],
                        "role": "model",
                    },
                    "finishReason": "STOP",
                }
            ],
            "usageMetadata": {"candidatesTokenCount": 5},
        }
        events = translate_gemini_sse_chunk(chunk, state)
        event_types = [e["event"] for e in events]
        assert "content_block_start" in event_types
        assert "content_block_delta" in event_types
        # Verify the tool_use block
        start_events = [e for e in events if e["event"] == "content_block_start"]
        assert start_events[0]["data"]["content_block"]["type"] == "tool_use"

    def test_safety_stream_synthesizes_refusal(self):
        from claude_bridge.providers.gemini import translate_gemini_sse_chunk

        state: dict = {"started": True, "block_index": 0}
        chunk = {
            "candidates": [{"finishReason": "SAFETY"}],
            "usageMetadata": {},
        }
        events = translate_gemini_sse_chunk(chunk, state)
        # Should have a text block with safety message
        deltas = [e for e in events if e["event"] == "content_block_delta"]
        assert len(deltas) == 1
        assert "safety" in deltas[0]["data"]["delta"]["text"].lower()

    def test_full_stream_round_trip(self):
        """Test translate_stream with simulated SSE bytes."""
        import asyncio

        from claude_bridge.providers.gemini import GeminiProvider

        provider = GeminiProvider()

        sse_data = (
            b'data: {"candidates":[{"content":{"parts":[{"text":"Hi"}],'
            b'"role":"model"}}],"usageMetadata":{"promptTokenCount":5},'
            b'"responseId":"r1","modelVersion":"gemini-2.5-pro"}\n\n'
            b'data: {"candidates":[{"content":{"parts":[{"text":" there"}],'
            b'"role":"model"},"finishReason":"STOP"}],'
            b'"usageMetadata":{"candidatesTokenCount":3},"responseId":"r1"}\n\n'
        )

        async def _chunks():
            yield sse_data

        async def _collect():
            return [e async for e in provider.translate_stream(_chunks())]

        events = asyncio.run(_collect())
        event_types = [e["event"] for e in events]
        assert "message_start" in event_types
        assert "message_stop" in event_types
        text_deltas = [
            e["data"]["delta"]["text"]
            for e in events
            if e["event"] == "content_block_delta"
            and e["data"].get("delta", {}).get("type") == "text_delta"
        ]
        assert "".join(text_deltas) == "Hi there"

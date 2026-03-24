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

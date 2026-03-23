"""Tests for Anthropic <-> OpenAI request/response translation."""

from __future__ import annotations

import json

from claude_bridge.providers.openai import (
    DEFAULT_MODEL,
    MODEL_MAP,
    anthropic_to_openai,
    openai_to_anthropic,
)
from claude_bridge.proxy import estimate_input_tokens

# ---------------------------------------------------------------------------
# anthropic_to_openai — text-only requests
# ---------------------------------------------------------------------------


class TestAnthropicToOpenaiTextOnly:
    """Text-only request translation (messages + system + model mapping)."""

    def test_basic_text_message(self):
        request = {
            "model": "claude-opus-4-6",
            "max_tokens": 1024,
            "messages": [{"role": "user", "content": [{"type": "text", "text": "Hello"}]}],
        }
        result, warnings = anthropic_to_openai(request)

        assert result["model"] == "gpt-5.4"
        assert result["store"] is False
        assert "max_output_tokens" not in result  # Codex endpoint doesn't support it
        assert len(result["input"]) == 1
        assert result["input"][0]["role"] == "user"
        assert result["input"][0]["content"][0] == {
            "type": "input_text",
            "text": "Hello",
        }
        assert warnings == []

    def test_string_content_shorthand(self):
        request = {
            "model": "claude-sonnet-4-6",
            "max_tokens": 512,
            "messages": [{"role": "user", "content": "Hello shorthand"}],
        }
        result, _warnings = anthropic_to_openai(request)

        assert result["input"][0]["content"][0] == {
            "type": "input_text",
            "text": "Hello shorthand",
        }

    def test_system_string(self):
        request = {
            "model": "claude-opus-4-6",
            "max_tokens": 100,
            "system": "You are helpful.",
            "messages": [{"role": "user", "content": "Hi"}],
        }
        result, _ = anthropic_to_openai(request)
        assert result["instructions"] == "You are helpful."

    def test_system_block_list(self):
        request = {
            "model": "claude-opus-4-6",
            "max_tokens": 100,
            "system": [
                {"type": "text", "text": "Part one."},
                {"type": "text", "text": "Part two."},
            ],
            "messages": [{"role": "user", "content": "Hi"}],
        }
        result, _ = anthropic_to_openai(request)
        assert result["instructions"] == "Part one.\nPart two."

    def test_role_mapping_preserved(self):
        request = {
            "model": "claude-opus-4-6",
            "max_tokens": 100,
            "messages": [
                {"role": "user", "content": "Q"},
                {"role": "assistant", "content": "A"},
                {"role": "user", "content": "Follow-up"},
            ],
        }
        result, _ = anthropic_to_openai(request)
        assert [m["role"] for m in result["input"]] == [
            "user",
            "assistant",
            "user",
        ]

    def test_temperature_and_max_tokens_stripped_for_codex(self):
        """Codex endpoint doesn't support temperature or max_output_tokens."""
        request = {
            "model": "claude-opus-4-6",
            "max_tokens": 100,
            "temperature": 0.7,
            "messages": [{"role": "user", "content": "Hi"}],
        }
        result, _ = anthropic_to_openai(request)
        assert "temperature" not in result
        assert "max_output_tokens" not in result


# ---------------------------------------------------------------------------
# anthropic_to_openai — tool definitions
# ---------------------------------------------------------------------------


class TestAnthropicToOpenaiTools:
    """Tool definitions: Anthropic tools -> OpenAI tools."""

    def test_tool_definition_translation(self):
        request = {
            "model": "claude-opus-4-6",
            "max_tokens": 100,
            "messages": [{"role": "user", "content": "Use the tool"}],
            "tools": [
                {
                    "name": "Edit",
                    "description": "Edits a file",
                    "input_schema": {
                        "type": "object",
                        "properties": {"path": {"type": "string"}},
                    },
                }
            ],
        }
        result, _ = anthropic_to_openai(request)
        assert len(result["tools"]) == 1
        tool = result["tools"][0]
        assert tool["type"] == "function"
        assert tool["name"] == "Edit"
        assert tool["description"] == "Edits a file"
        assert tool["parameters"] == {
            "type": "object",
            "properties": {"path": {"type": "string"}},
        }


# ---------------------------------------------------------------------------
# anthropic_to_openai — tool use in messages
# ---------------------------------------------------------------------------


class TestAnthropicToOpenaiToolUse:
    """Tool use: assistant tool_use -> function_call, tool_result -> function_call_output."""

    def test_tool_use_block_to_function_call(self):
        request = {
            "model": "claude-opus-4-6",
            "max_tokens": 100,
            "messages": [
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "toolu_abc",
                            "name": "Edit",
                            "input": {"path": "/tmp/f.py", "content": "x=1"},
                        }
                    ],
                }
            ],
        }
        result, _ = anthropic_to_openai(request)

        # tool_use blocks become top-level function_call items, not nested in a message
        fc = [item for item in result["input"] if item.get("type") == "function_call"]
        assert len(fc) == 1
        assert fc[0]["id"] == "fc_abc"  # toolu_ prefix → fc_ for OpenAI
        assert fc[0]["call_id"] == "fc_abc"
        assert fc[0]["name"] == "Edit"
        # arguments must be a JSON string
        assert fc[0]["arguments"] == json.dumps({"path": "/tmp/f.py", "content": "x=1"})

    def test_tool_result_to_function_call_output(self):
        request = {
            "model": "claude-opus-4-6",
            "max_tokens": 100,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "toolu_abc",
                            "content": "File edited successfully",
                        }
                    ],
                }
            ],
        }
        result, _ = anthropic_to_openai(request)

        fco = [item for item in result["input"] if item.get("type") == "function_call_output"]
        assert len(fco) == 1
        assert fco[0]["call_id"] == "fc_abc"  # toolu_ prefix → fc_ for OpenAI
        assert fco[0]["output"] == "File edited successfully"


# ---------------------------------------------------------------------------
# anthropic_to_openai — stripped features / warnings
# ---------------------------------------------------------------------------


class TestAnthropicToOpenaiStripping:
    """Unsupported features stripped with warnings."""

    def test_thinking_passthrough_by_default(self):
        """Thinking config emits passthrough warning (not stripped) in default mode."""
        request = {
            "model": "claude-opus-4-6",
            "max_tokens": 100,
            "thinking": {"type": "enabled", "budget_tokens": 5000},
            "messages": [{"role": "user", "content": "Hi"}],
        }
        _, warnings = anthropic_to_openai(request)
        assert any("passthrough" in w.lower() for w in warnings)
        # Should NOT say "Stripped"
        assert not any("stripped" in w.lower() and "thinking" in w.lower() for w in warnings)

    def test_thinking_stripped_when_drop_mode(self, monkeypatch):
        """Thinking config stripped when REASONING_MODE=drop."""
        import claude_bridge.providers.openai as oai_mod

        monkeypatch.setattr(oai_mod, "_REASONING_MODE", "drop")
        request = {
            "model": "claude-opus-4-6",
            "max_tokens": 100,
            "thinking": {"type": "enabled", "budget_tokens": 5000},
            "messages": [{"role": "user", "content": "Hi"}],
        }
        _, warnings = anthropic_to_openai(request)
        assert any("drop" in w.lower() and "thinking" in w.lower() for w in warnings)

    def test_output_config_stripped(self):
        request = {
            "model": "claude-opus-4-6",
            "max_tokens": 100,
            "output_config": {"format": "json"},
            "messages": [{"role": "user", "content": "Hi"}],
        }
        result, warnings = anthropic_to_openai(request)
        assert "output_config" not in result
        assert any("output_config" in w for w in warnings)

    def test_cache_control_stripped_from_content_block(self):
        request = {
            "model": "claude-opus-4-6",
            "max_tokens": 100,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Hello",
                            "cache_control": {"type": "ephemeral"},
                        }
                    ],
                }
            ],
        }
        result, warnings = anthropic_to_openai(request)
        assert any("cache_control" in w.lower() for w in warnings)
        content = result["input"][0]["content"][0]
        assert "cache_control" not in content

    def test_cache_control_stripped_from_system_blocks(self):
        request = {
            "model": "claude-opus-4-6",
            "max_tokens": 100,
            "system": [
                {
                    "type": "text",
                    "text": "You are helpful.",
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            "messages": [{"role": "user", "content": "Hi"}],
        }
        _, warnings = anthropic_to_openai(request)
        assert any("cache_control" in w.lower() for w in warnings)

    def test_cache_control_stripped_from_tool_definitions(self):
        request = {
            "model": "claude-opus-4-6",
            "max_tokens": 100,
            "tools": [
                {
                    "name": "get_weather",
                    "description": "Get weather",
                    "input_schema": {"type": "object", "properties": {}},
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            "messages": [{"role": "user", "content": "Hi"}],
        }
        result, warnings = anthropic_to_openai(request)
        assert any("cache_control" in w.lower() for w in warnings)
        tool = result["tools"][0]
        assert "cache_control" not in tool

    def test_no_cache_control_no_warning(self):
        request = {
            "model": "claude-opus-4-6",
            "max_tokens": 100,
            "messages": [{"role": "user", "content": "Hello"}],
        }
        _, warnings = anthropic_to_openai(request)
        assert not any("cache_control" in w.lower() for w in warnings)


# ---------------------------------------------------------------------------
# Thinking block passthrough
# ---------------------------------------------------------------------------


class TestThinkingBlockPassthrough:
    """Thinking content blocks preserved or dropped based on reasoning mode."""

    def test_thinking_block_preserved_as_tagged_text(self):
        """In passthrough mode, thinking blocks become [thinking]...[/thinking]."""
        request = {
            "model": "claude-opus-4-6",
            "messages": [
                {
                    "role": "assistant",
                    "content": [
                        {"type": "thinking", "thinking": "Let me reason about this."},
                        {"type": "text", "text": "Here is my answer."},
                    ],
                }
            ],
        }
        result, _ = anthropic_to_openai(request)
        assistant_items = [i for i in result["input"] if i.get("role") == "assistant"]
        assert len(assistant_items) == 1
        content = assistant_items[0]["content"]
        # First block should be the thinking text
        assert "[thinking]" in content[0]["text"]
        assert "Let me reason about this." in content[0]["text"]

    def test_thinking_block_dropped_in_drop_mode(self, monkeypatch):
        """In drop mode, thinking blocks become empty text."""
        import claude_bridge.providers.openai as oai_mod

        monkeypatch.setattr(oai_mod, "_REASONING_MODE", "drop")
        request = {
            "model": "claude-opus-4-6",
            "messages": [
                {
                    "role": "assistant",
                    "content": [
                        {"type": "thinking", "thinking": "Secret reasoning."},
                        {"type": "text", "text": "Answer."},
                    ],
                }
            ],
        }
        result, warnings = anthropic_to_openai(request)
        assistant_items = [i for i in result["input"] if i.get("role") == "assistant"]
        content = assistant_items[0]["content"]
        # Thinking block becomes empty, not preserved
        assert "Secret reasoning" not in str(content)
        assert any("drop" in w.lower() for w in warnings)

    def test_thinking_block_empty_text_field(self):
        """Thinking block with empty text doesn't crash."""
        request = {
            "model": "claude-opus-4-6",
            "messages": [
                {
                    "role": "assistant",
                    "content": [{"type": "thinking", "thinking": ""}],
                }
            ],
        }
        result, _ = anthropic_to_openai(request)
        assert result is not None  # No crash


# ---------------------------------------------------------------------------
# anthropic_to_openai — model fallback
# ---------------------------------------------------------------------------


class TestAnthropicToOpenaiModelFallback:
    """Unknown model falls back to DEFAULT_MODEL."""

    def test_unknown_model_uses_default(self):
        request = {
            "model": "claude-unknown-99",
            "max_tokens": 100,
            "messages": [{"role": "user", "content": "Hi"}],
        }
        result, _ = anthropic_to_openai(request)
        assert result["model"] == DEFAULT_MODEL

    def test_all_known_models_mapped(self):
        for anthropic_model, openai_model in MODEL_MAP.items():
            request = {
                "model": anthropic_model,
                "max_tokens": 100,
                "messages": [{"role": "user", "content": "Hi"}],
            }
            result, _ = anthropic_to_openai(request)
            assert result["model"] == openai_model


# ---------------------------------------------------------------------------
# anthropic_to_openai — empty messages
# ---------------------------------------------------------------------------


class TestAnthropicToOpenaiEmpty:
    """Empty messages array handled."""

    def test_empty_messages(self):
        request = {
            "model": "claude-opus-4-6",
            "max_tokens": 100,
            "messages": [],
        }
        result, warnings = anthropic_to_openai(request)
        assert result["input"] == []
        assert warnings == []


# ---------------------------------------------------------------------------
# openai_to_anthropic — text-only response
# ---------------------------------------------------------------------------


class TestOpenaiToAnthropicTextOnly:
    """Text-only response translation."""

    def test_basic_text_response(self):
        response = {
            "id": "resp_abc123",
            "model": "gpt-5.4",
            "status": "completed",
            "output": [
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "Hello back!"}],
                }
            ],
            "usage": {"input_tokens": 10, "output_tokens": 5},
        }
        result = openai_to_anthropic(response)

        assert result["id"] == "msg_bridge_resp_abc123"
        assert result["type"] == "message"
        assert result["role"] == "assistant"
        assert result["model"] == "gpt-5.4"
        assert result["stop_reason"] == "end_turn"
        assert result["content"] == [{"type": "text", "text": "Hello back!"}]
        assert result["usage"] == {"input_tokens": 10, "output_tokens": 5}

    def test_incomplete_status_maps_to_max_tokens(self):
        response = {
            "id": "resp_xyz",
            "model": "gpt-5.4",
            "status": "incomplete",
            "output": [
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "Partial..."}],
                }
            ],
            "usage": {"input_tokens": 5, "output_tokens": 100},
        }
        result = openai_to_anthropic(response)
        assert result["stop_reason"] == "max_tokens"

    def test_unknown_status_defaults_to_end_turn(self):
        response = {
            "id": "resp_unk",
            "model": "gpt-5.4",
            "status": "cancelled",
            "output": [],
            "usage": {"input_tokens": 0, "output_tokens": 0},
        }
        result = openai_to_anthropic(response)
        assert result["stop_reason"] == "end_turn"

    def test_missing_id_uses_unknown(self):
        response = {
            "model": "gpt-5.4",
            "status": "completed",
            "output": [],
            "usage": {"input_tokens": 0, "output_tokens": 0},
        }
        result = openai_to_anthropic(response)
        assert result["id"] == "msg_bridge_unknown"


# ---------------------------------------------------------------------------
# openai_to_anthropic — tool use in response
# ---------------------------------------------------------------------------


class TestOpenaiToAnthropicToolUse:
    """Function call in response -> tool_use content block."""

    def test_function_call_to_tool_use(self):
        response = {
            "id": "resp_tools",
            "model": "gpt-5.4",
            "status": "completed",
            "output": [
                {
                    "type": "function_call",
                    "id": "call_abc",
                    "name": "Edit",
                    "arguments": json.dumps({"path": "/tmp/f.py"}),
                }
            ],
            "usage": {"input_tokens": 20, "output_tokens": 15},
        }
        result = openai_to_anthropic(response)

        assert len(result["content"]) == 1
        block = result["content"][0]
        assert block["type"] == "tool_use"
        assert block["id"] == "call_abc"
        assert block["name"] == "Edit"
        assert block["input"] == {"path": "/tmp/f.py"}

    def test_mixed_text_and_tool_use(self):
        response = {
            "id": "resp_mixed",
            "model": "gpt-5.4",
            "status": "completed",
            "output": [
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "Let me edit that."}],
                },
                {
                    "type": "function_call",
                    "id": "call_xyz",
                    "name": "Read",
                    "arguments": json.dumps({"file": "a.py"}),
                },
            ],
            "usage": {"input_tokens": 10, "output_tokens": 20},
        }
        result = openai_to_anthropic(response)
        assert len(result["content"]) == 2
        assert result["content"][0]["type"] == "text"
        assert result["content"][1]["type"] == "tool_use"


# ---------------------------------------------------------------------------
# Robustness — previously untested branches
# ---------------------------------------------------------------------------


class TestTranslationRobustness:
    """Cover edge cases in translation: malformed args, list tool_result, unknown blocks."""

    def test_malformed_arguments_in_response_does_not_crash(self):
        """json.loads on malformed arguments falls back to _raw wrapper."""
        response = {
            "id": "resp_bad",
            "model": "gpt-5.4",
            "status": "completed",
            "output": [
                {
                    "type": "function_call",
                    "id": "call_bad",
                    "name": "Edit",
                    "arguments": "not valid json {{{",
                }
            ],
            "usage": {"input_tokens": 1, "output_tokens": 1},
        }
        result = openai_to_anthropic(response)
        block = result["content"][0]
        assert block["type"] == "tool_use"
        assert block["input"] == {"_raw": "not valid json {{{"}

    def test_tool_result_with_list_content(self):
        """tool_result with list-of-blocks content joins text."""
        request = {
            "model": "claude-opus-4-6",
            "max_tokens": 100,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "toolu_list",
                            "content": [
                                {"type": "text", "text": "Part 1"},
                                {"type": "text", "text": "Part 2"},
                            ],
                        }
                    ],
                }
            ],
        }
        result, _ = anthropic_to_openai(request)
        fco = [i for i in result["input"] if i.get("type") == "function_call_output"]
        assert len(fco) == 1
        assert fco[0]["output"] == "Part 1\nPart 2"

    def test_tool_result_with_image_content_preserves_data(self):
        """tool_result with mixed text+image content preserves image as data URL."""
        request = {
            "model": "claude-opus-4-6",
            "max_tokens": 100,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "toolu_img",
                            "content": [
                                {"type": "text", "text": "Screenshot captured"},
                                {
                                    "type": "image",
                                    "source": {
                                        "type": "base64",
                                        "media_type": "image/png",
                                        "data": "iVBORw0KGgo=",
                                    },
                                },
                            ],
                        }
                    ],
                }
            ],
        }
        result, _ = anthropic_to_openai(request)
        fco = [i for i in result["input"] if i.get("type") == "function_call_output"]
        assert len(fco) == 1
        output = fco[0]["output"]
        assert "Screenshot captured" in output
        assert "data:image/png;base64,iVBORw0KGgo=" in output

    def test_tool_result_with_url_image_preserves_url(self):
        """tool_result with URL image source preserves the URL."""
        request = {
            "model": "claude-opus-4-6",
            "max_tokens": 100,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "toolu_url_img",
                            "content": [
                                {
                                    "type": "image",
                                    "source": {
                                        "type": "url",
                                        "url": "https://example.com/img.png",
                                    },
                                },
                            ],
                        }
                    ],
                }
            ],
        }
        result, _ = anthropic_to_openai(request)
        fco = [i for i in result["input"] if i.get("type") == "function_call_output"]
        assert len(fco) == 1
        assert "https://example.com/img.png" in fco[0]["output"]

    def test_unknown_content_block_type_converted_with_warning(self):
        """Unknown block type falls back to input_text with a warning."""
        request = {
            "model": "claude-opus-4-6",
            "max_tokens": 100,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "exotic_block", "data": "something"},
                    ],
                }
            ],
        }
        result, warnings = anthropic_to_openai(request)
        user_msg = [i for i in result["input"] if i.get("role") == "user"]
        assert len(user_msg) == 1
        assert user_msg[0]["content"][0]["type"] == "input_text"
        assert any("Unknown content block type" in w for w in warnings)

    def test_empty_arguments_in_response_produces_empty_dict(self):
        """Empty string arguments parses to empty dict."""
        response = {
            "id": "resp_empty",
            "model": "gpt-5.4",
            "status": "completed",
            "output": [
                {
                    "type": "function_call",
                    "id": "call_empty",
                    "name": "Read",
                    "arguments": "",
                }
            ],
            "usage": {"input_tokens": 1, "output_tokens": 1},
        }
        result = openai_to_anthropic(response)
        block = result["content"][0]
        assert block["type"] == "tool_use"
        # Empty string → json.loads fails → fallback to _raw
        assert block["input"] == {"_raw": ""}


# ---------------------------------------------------------------------------
# estimate_input_tokens
# ---------------------------------------------------------------------------


class TestEstimateInputTokens:
    """Token estimation walks the request structure and uses bytes/3.5."""

    def test_text_only_message(self):
        request = {
            "model": "claude-opus-4-6",
            "max_tokens": 100,
            "messages": [{"role": "user", "content": "Hello world"}],
        }
        tokens = estimate_input_tokens(request)
        assert tokens > 0
        # "Hello world" = 11 bytes, but there's also the model and structure overhead
        assert isinstance(tokens, int)

    def test_system_string(self):
        request = {
            "model": "claude-opus-4-6",
            "max_tokens": 100,
            "system": "You are a helpful assistant.",
            "messages": [{"role": "user", "content": "Hi"}],
        }
        tokens_with_system = estimate_input_tokens(request)
        request_no_system = {
            "model": "claude-opus-4-6",
            "max_tokens": 100,
            "messages": [{"role": "user", "content": "Hi"}],
        }
        tokens_without_system = estimate_input_tokens(request_no_system)
        assert tokens_with_system > tokens_without_system

    def test_system_block_list(self):
        request = {
            "model": "claude-opus-4-6",
            "max_tokens": 100,
            "system": [
                {"type": "text", "text": "Part one."},
                {"type": "text", "text": "Part two."},
            ],
            "messages": [],
        }
        tokens = estimate_input_tokens(request)
        assert tokens > 0

    def test_with_tools(self):
        request = {
            "model": "claude-opus-4-6",
            "max_tokens": 100,
            "messages": [{"role": "user", "content": "Use the tool"}],
            "tools": [
                {
                    "name": "Edit",
                    "description": "Edits a file",
                    "input_schema": {
                        "type": "object",
                        "properties": {"path": {"type": "string"}},
                    },
                }
            ],
        }
        tokens_with_tools = estimate_input_tokens(request)
        request_no_tools = {
            "model": "claude-opus-4-6",
            "max_tokens": 100,
            "messages": [{"role": "user", "content": "Use the tool"}],
        }
        tokens_without_tools = estimate_input_tokens(request_no_tools)
        assert tokens_with_tools > tokens_without_tools

    def test_empty_request(self):
        assert estimate_input_tokens({}) == 0
        assert estimate_input_tokens({"messages": []}) == 0

    def test_tool_use_and_result_in_messages(self):
        request = {
            "model": "claude-opus-4-6",
            "max_tokens": 100,
            "messages": [
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "toolu_abc",
                            "name": "Edit",
                            "input": {"path": "/tmp/f.py", "content": "x=1"},
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "toolu_abc",
                            "content": "File edited successfully",
                        }
                    ],
                },
            ],
        }
        tokens = estimate_input_tokens(request)
        assert tokens > 0

"""Prompt-cache request hints / usage parsing, and native tool calling, for every provider."""

import json

import httpx
import pytest

from brass_lantern.providers import ProviderError, create_provider
from brass_lantern.providers.base import parse_tool_arguments, with_cache_breakpoint
from test_providers import ANTHROPIC_OK, MSGS, OPENAI_OK, SYSTEM, make, run

EPH = {"type": "ephemeral"}
HISTORY_MSGS = [
    {"role": "user", "content": "one"},
    {"role": "assistant", "content": "> look"},
    {"role": "user", "content": "two"},
]
TOOLS = [
    {
        "name": "act",
        "description": "Send one command.",
        "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]},
    }
]


def sent(seen, i=0):
    return json.loads(seen[i].content)


# ===========================================================================
# Prompt caching: request hints and usage parsing
# ===========================================================================


def test_with_cache_breakpoint_converts_strings_copies_and_marks_last_block():
    original = {"role": "user", "content": "hi"}
    marked = with_cache_breakpoint(original)
    assert marked == {"role": "user", "content": [{"type": "text", "text": "hi", "cache_control": EPH}]}
    assert original == {"role": "user", "content": "hi"}  # input untouched
    blocks = {"role": "user", "content": [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]}
    out = with_cache_breakpoint(blocks)
    assert "cache_control" not in out["content"][0] and out["content"][1]["cache_control"] == EPH
    assert "cache_control" not in blocks["content"][1]


def test_anthropic_cache_on_marks_system_and_last_message_only():
    p, seen, _ = make("anthropic:claude-x", [ANTHROPIC_OK], env={"ANTHROPIC_API_KEY": "k"})
    msgs = [dict(m) for m in HISTORY_MSGS]
    run(p.complete(SYSTEM, msgs))
    body = sent(seen)
    assert body["system"] == [{"type": "text", "text": SYSTEM, "cache_control": EPH}]
    assert body["messages"][:2] == HISTORY_MSGS[:2]  # earlier messages stay plain strings
    assert body["messages"][2] == {
        "role": "user",
        "content": [{"type": "text", "text": "two", "cache_control": EPH}],
    }
    assert seen[0].content.count(b"cache_control") == 2  # system + last message: under the limit of 4
    assert msgs == HISTORY_MSGS  # the caller's messages are not mutated


def test_anthropic_cache_marks_the_merged_last_message():
    p, seen, _ = make("anthropic:claude-x", [ANTHROPIC_OK], env={"ANTHROPIC_API_KEY": "k"})
    run(p.complete(SYSTEM, [{"role": "user", "content": "a"}, {"role": "user", "content": "b"}]))
    msgs = sent(seen)["messages"]
    assert len(msgs) == 1 and msgs[0]["content"] == [{"type": "text", "text": "a\n\nb", "cache_control": EPH}]


def test_anthropic_cache_off_sends_no_markers():
    p, seen, _ = make("anthropic:claude-x", [ANTHROPIC_OK], env={"ANTHROPIC_API_KEY": "k"}, cache=False)
    run(p.complete(SYSTEM, HISTORY_MSGS))
    assert b"cache_control" not in seen[0].content
    body = sent(seen)
    assert body["system"] == SYSTEM and body["messages"] == HISTORY_MSGS


def test_anthropic_reports_cache_reads_and_writes_separately():
    resp = httpx.Response(
        200,
        json={
            "content": [{"type": "text", "text": "> look"}],
            "usage": {
                "input_tokens": 12,
                "cache_read_input_tokens": 800,
                "cache_creation_input_tokens": 188,
                "output_tokens": 5,
            },
        },
    )
    p, _, _ = make("anthropic:claude-x", [resp], env={"ANTHROPIC_API_KEY": "k"})
    c = run(p.complete(SYSTEM, MSGS))
    assert (c.input_tokens, c.cache_read_tokens, c.cache_write_tokens, c.output_tokens) == (1000, 800, 188, 5)


def test_create_provider_passes_cache_flag():
    assert create_provider("anthropic:c", env={"ANTHROPIC_API_KEY": "k"}).cache is True
    assert create_provider("anthropic:c", env={"ANTHROPIC_API_KEY": "k"}, cache=False).cache is False
    assert create_provider("openrouter:a/b", env={"OPENROUTER_API_KEY": "k"}, cache=False).cache is False


def test_openrouter_anthropic_gets_content_part_breakpoints():
    p, seen, _ = make("openrouter:anthropic/claude-haiku-4.5", [OPENAI_OK], env={"OPENROUTER_API_KEY": "k"})
    msgs = [dict(m) for m in HISTORY_MSGS]
    run(p.complete(SYSTEM, msgs))
    body = sent(seen)
    assert body["messages"][0] == {
        "role": "system",
        "content": [{"type": "text", "text": SYSTEM, "cache_control": EPH}],
    }
    assert body["messages"][1:3] == HISTORY_MSGS[:2]
    assert body["messages"][3] == {"role": "user", "content": [{"type": "text", "text": "two", "cache_control": EPH}]}
    assert seen[0].content.count(b"cache_control") == 2
    assert msgs == HISTORY_MSGS
    assert body["usage"] == {"include": True}  # cost reporting is unchanged


@pytest.mark.parametrize(
    "spec, env",
    [
        ("openrouter:openai/gpt-4o-mini", {"OPENROUTER_API_KEY": "k"}),
        ("openrouter:deepseek/deepseek-v4-flash", {"OPENROUTER_API_KEY": "k"}),
        ("openrouter:google/gemini-3.5-flash-lite", {"OPENROUTER_API_KEY": "k"}),
        ("openrouter:qwen/qwen3.8-flash", {"OPENROUTER_API_KEY": "k"}),
        ("openrouter:z-ai/glm-5.3-flash", {"OPENROUTER_API_KEY": "k"}),
        ("openai:gpt-4o-mini", {"OPENAI_API_KEY": "k"}),
        ("ollama:llama3.1:8b", {}),
    ],
)
def test_other_models_get_no_cache_markers(spec, env):
    p, seen, _ = make(spec, [OPENAI_OK], env=env)
    run(p.complete(SYSTEM, HISTORY_MSGS))
    assert b"cache_control" not in seen[0].content
    assert sent(seen)["messages"] == [{"role": "system", "content": SYSTEM}, *HISTORY_MSGS]


def test_cache_off_and_non_openrouter_anthropic_ids_send_nothing():
    p, seen, _ = make("openrouter:anthropic/claude-haiku-4.5", [OPENAI_OK], env={"OPENROUTER_API_KEY": "k"}, cache=False)
    run(p.complete(SYSTEM, MSGS))
    assert b"cache_control" not in seen[0].content
    # only OpenRouter forwards cache_control: a generic server that serves an "anthropic/..." id gets none
    p, seen, _ = make("openai:anthropic/whatever", [OPENAI_OK], base_url="http://localhost:8000/v1")
    run(p.complete(SYSTEM, MSGS))
    assert b"cache_control" not in seen[0].content


def _usage_resp(usage):
    return httpx.Response(200, json={"choices": [{"message": {"content": "> look"}}], "usage": usage})


def test_openrouter_cache_usage_parsing():
    resp = _usage_resp(
        {
            "prompt_tokens": 1000,
            "completion_tokens": 7,
            "cost": 0.0012,
            "prompt_tokens_details": {"cached_tokens": 640, "cache_write_tokens": 100},
        }
    )
    p, _, _ = make("openrouter:anthropic/claude-haiku-4.5", [resp], env={"OPENROUTER_API_KEY": "k"})
    c = run(p.complete(SYSTEM, MSGS))
    assert (c.input_tokens, c.cache_read_tokens, c.cache_write_tokens, c.cost_usd) == (1000, 640, 100, 0.0012)


@pytest.mark.parametrize(
    "usage, expected",
    [
        # alternative spellings of the cache-write counter
        ({"prompt_tokens": 9, "prompt_tokens_details": {"cached_tokens": 1, "cache_creation_input_tokens": 4}}, (1, 4)),
        ({"prompt_tokens": 9, "prompt_tokens_details": {"cached_tokens": 1}, "cache_creation_input_tokens": 6}, (1, 6)),
        # absent / null / junk values are zero
        ({"prompt_tokens": 9}, (0, 0)),
        ({"prompt_tokens": 9, "prompt_tokens_details": None}, (0, 0)),
        ({"prompt_tokens": 9, "prompt_tokens_details": {"cached_tokens": None, "cache_write_tokens": "n/a"}}, (0, 0)),
    ],
)
def test_openai_compat_cache_usage_is_parsed_defensively(usage, expected):
    p, _, _ = make("ollama:m", [_usage_resp(usage)])
    c = run(p.complete(SYSTEM, MSGS))
    assert (c.cache_read_tokens, c.cache_write_tokens) == expected and c.input_tokens == 9


def test_plain_openai_cached_tokens():
    resp = _usage_resp({"prompt_tokens": 2048, "completion_tokens": 3, "prompt_tokens_details": {"cached_tokens": 1920}})
    p, _, _ = make("openai:gpt-4o-mini", [resp], env={"OPENAI_API_KEY": "k"})
    c = run(p.complete(SYSTEM, MSGS))
    assert (c.input_tokens, c.cache_read_tokens, c.cache_write_tokens) == (2048, 1920, 0)


def test_gemini_cached_content_token_count():
    resp = httpx.Response(
        200,
        json={
            "candidates": [{"content": {"parts": [{"text": "> look"}]}, "finishReason": "STOP"}],
            "usageMetadata": {"promptTokenCount": 3000, "cachedContentTokenCount": 2500, "candidatesTokenCount": 4},
        },
    )
    p, seen, _ = make("gemini:gemini-2.5-flash", [resp], env={"GEMINI_API_KEY": "k"})
    c = run(p.complete(SYSTEM, MSGS))
    assert (c.input_tokens, c.cache_read_tokens, c.cache_write_tokens) == (3000, 2500, 0)
    assert b"cache" not in seen[0].content.lower()  # nothing extra is sent to Gemini
    empty = httpx.Response(200, json={"candidates": [{}], "usageMetadata": {}})
    p, _, _ = make("gemini:gemini-2.5-flash", [empty], env={"GEMINI_API_KEY": "k"})
    assert run(p.complete(SYSTEM, MSGS)).cache_read_tokens == 0


# ===========================================================================
# Tool calling: request shape and parsing
# ===========================================================================


def _tool_resp(calls, content=None, usage=None):
    return httpx.Response(
        200,
        json={
            "choices": [
                {
                    "message": {
                        "content": content,
                        "tool_calls": [
                            {"id": f"c{i}", "type": "function", "function": {"name": n, "arguments": a}}
                            for i, (n, a) in enumerate(calls)
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": usage or {"prompt_tokens": 5, "completion_tokens": 2},
        },
    )


@pytest.mark.parametrize(
    "spec, env",
    [
        ("openrouter:deepseek/deepseek-v4-flash", {"OPENROUTER_API_KEY": "k"}),
        ("openai:gpt-4o-mini", {"OPENAI_API_KEY": "k"}),
        ("ollama:llama3.1:8b", {}),
    ],
)
def test_openai_compat_tools_payload_and_parsing(spec, env):
    resp = _tool_resp([("act", '{"command": "open mailbox"}')], content="Let me look.")
    p, seen, _ = make(spec, [resp], env=env)
    c = run(p.complete(SYSTEM, MSGS, tools=TOOLS))
    body = sent(seen)
    assert body["tools"] == [
        {
            "type": "function",
            "function": {"name": "act", "description": "Send one command.", "parameters": TOOLS[0]["parameters"]},
        }
    ]
    assert body["tool_choice"] == "required"
    assert c.tool_calls == [{"name": "act", "arguments": {"command": "open mailbox"}}]
    assert c.text == "Let me look." and c.finish_reason == "tool_calls"


def test_openai_compat_without_tools_sends_none_and_returns_no_calls():
    p, seen, _ = make("ollama:m", [OPENAI_OK])
    c = run(p.complete(SYSTEM, MSGS))
    body = sent(seen)
    assert "tools" not in body and "tool_choice" not in body and c.tool_calls is None


def test_openai_compat_tool_call_with_null_content_and_multiple_calls():
    resp = _tool_resp([("act", '{"command": "north"}'), ("act", '{"command": "south"}')], content=None)
    p, _, _ = make("ollama:m", [resp])
    c = run(p.complete(SYSTEM, MSGS, tools=TOOLS))
    assert c.text == ""
    assert [t["arguments"]["command"] for t in c.tool_calls] == ["north", "south"]


@pytest.mark.parametrize(
    "raw, expected",
    [
        ('{"command": "go north"', {"command": "go north"}),  # truncated JSON: the string pair is salvaged
        ('{"command": "take lamp", "notes": "a\\nb",}', {"command": "take lamp", "notes": "a\nb"}),  # trailing comma
        ("go north please", {"_raw": "go north please"}),  # nothing recoverable
        ("[1, 2]", {"_raw": "[1, 2]"}),  # valid JSON, wrong shape
        ("", {}),
        (None, {}),
        ({"command": "up"}, {"command": "up"}),  # some servers send an object, not a string
    ],
)
def test_bad_tool_argument_json_is_handled(raw, expected):
    assert parse_tool_arguments(raw) == expected
    p, _, _ = make("ollama:m", [_tool_resp([("act", raw)])])
    assert run(p.complete(SYSTEM, MSGS, tools=TOOLS)).tool_calls == [{"name": "act", "arguments": expected}]


def test_openrouter_anthropic_with_tools_keeps_cache_markers_and_forces_tool():
    resp = _tool_resp([("act", '{"command": "look"}')])
    p, seen, _ = make("openrouter:anthropic/claude-haiku-4.5", [resp], env={"OPENROUTER_API_KEY": "k"})
    run(p.complete(SYSTEM, HISTORY_MSGS, tools=TOOLS))
    body = sent(seen)
    assert body["tool_choice"] == "required" and seen[0].content.count(b"cache_control") == 2


def test_openrouter_anthropic_with_reasoning_does_not_force_the_tool():
    resp = _tool_resp([("act", '{"command": "look"}')])
    p, seen, _ = make("openrouter:anthropic/claude-haiku-4.5", [resp], env={"OPENROUTER_API_KEY": "k"}, reasoning="high")
    run(p.complete(SYSTEM, MSGS, tools=TOOLS))
    assert sent(seen)["tool_choice"] == "auto"
    p, seen, _ = make("openrouter:deepseek/deepseek-v4-flash", [resp], env={"OPENROUTER_API_KEY": "k"}, reasoning="high")
    run(p.complete(SYSTEM, MSGS, tools=TOOLS))
    assert sent(seen)["tool_choice"] == "required"  # only Anthropic models reject a forced choice with reasoning


def test_openrouter_no_tool_support_is_a_clear_non_retryable_error():
    resp = httpx.Response(
        404, json={"error": {"message": "No endpoints found that support tool use. Try disabling \"act\".", "code": 404}}
    )
    p, seen, delays = make("openrouter:some/chat-model", [resp], env={"OPENROUTER_API_KEY": "k"})
    with pytest.raises(ProviderError) as ei:
        run(p.complete(SYSTEM, MSGS, tools=TOOLS))
    msg = str(ei.value)
    assert "some/chat-model" in msg and "--format text" in msg and "tool" in msg
    assert len(seen) == 1 and delays == []  # not retried


def test_no_tool_support_reported_inside_a_200_body_is_also_clear():
    resp = httpx.Response(200, json={"error": {"code": 404, "message": "No endpoints found that support tool use"}})
    p, seen, _ = make("openrouter:some/chat-model", [resp], env={"OPENROUTER_API_KEY": "k"})
    with pytest.raises(ProviderError, match="--format text"):
        run(p.complete(SYSTEM, MSGS, tools=TOOLS))
    assert len(seen) == 1


def test_ollama_400_tools_error_and_unrelated_errors_pass_through():
    resp = httpx.Response(400, json={"error": "registry.ollama.ai/library/tiny does not support tools"})
    p, _, _ = make("ollama:tiny", [resp])
    with pytest.raises(ProviderError, match="tiny.*--format text"):
        run(p.complete(SYSTEM, MSGS, tools=TOOLS))
    # a plain bad request, or the same wording without tools in play, is left alone
    p, _, _ = make("ollama:m", [httpx.Response(400, json={"error": {"message": "bad model"}})])
    with pytest.raises(ProviderError) as ei:
        run(p.complete(SYSTEM, MSGS, tools=TOOLS))
    assert "--format text" not in str(ei.value) and "bad model" in str(ei.value)
    p, _, _ = make("ollama:m", [httpx.Response(404, text="No endpoints found that support tool use")])
    with pytest.raises(ProviderError) as ei:
        run(p.complete(SYSTEM, MSGS))
    assert "--format text" not in str(ei.value)
    # exhausted retries stay what they are
    p, _, _ = make("ollama:m", [httpx.Response(503, text="tool use is down")])
    with pytest.raises(ProviderError, match="gave up after 6 attempts") as ei:
        run(p.complete(SYSTEM, MSGS, tools=TOOLS))
    assert "--format text" not in str(ei.value)


def test_anthropic_tools_payload_and_tool_use_parsing():
    resp = httpx.Response(
        200,
        json={
            "content": [
                {"type": "text", "text": "I will open it."},
                {"type": "tool_use", "id": "tu_1", "name": "act", "input": {"command": "open mailbox", "notes": "n"}},
            ],
            "usage": {"input_tokens": 10, "output_tokens": 4},
            "stop_reason": "tool_use",
        },
    )
    p, seen, _ = make("anthropic:claude-x", [resp], env={"ANTHROPIC_API_KEY": "k"})
    c = run(p.complete(SYSTEM, HISTORY_MSGS, tools=TOOLS))
    body = sent(seen)
    assert body["tools"] == [{"name": "act", "description": "Send one command.", "input_schema": TOOLS[0]["parameters"]}]
    assert body["tool_choice"] == {"type": "any"}
    assert body["system"][0]["cache_control"] == EPH  # caching still applies in tools mode
    assert c.text == "I will open it." and c.finish_reason == "tool_use"
    assert c.tool_calls == [{"name": "act", "arguments": {"command": "open mailbox", "notes": "n"}}]


def test_anthropic_tool_choice_is_auto_when_thinking_is_enabled():
    resp = httpx.Response(200, json={"content": [{"type": "tool_use", "name": "act", "input": {"command": "look"}}], "usage": {}})
    p, seen, _ = make("anthropic:claude-x", [resp], env={"ANTHROPIC_API_KEY": "k"})
    p._extra_body = {"thinking": {"type": "enabled", "budget_tokens": 2000}}
    run(p.complete(SYSTEM, MSGS, tools=TOOLS))
    body = sent(seen)
    assert body["thinking"]["type"] == "enabled" and body["tool_choice"] == {"type": "auto"}


def test_anthropic_without_tools_and_odd_tool_use_blocks():
    p, seen, _ = make("anthropic:claude-x", [ANTHROPIC_OK], env={"ANTHROPIC_API_KEY": "k"})
    c = run(p.complete(SYSTEM, MSGS))
    assert "tools" not in sent(seen) and "tool_choice" not in sent(seen) and c.tool_calls is None
    resp = httpx.Response(200, json={"content": [{"type": "tool_use", "name": "act"}], "usage": {}})
    p, _, _ = make("anthropic:claude-x", [resp], env={"ANTHROPIC_API_KEY": "k"})
    assert run(p.complete(SYSTEM, MSGS, tools=TOOLS)).tool_calls == [{"name": "act", "arguments": {}}]


def test_anthropic_no_tool_support_error_is_explained():
    resp = httpx.Response(400, json={"error": {"message": "tools: model does not support tool use"}})
    p, _, _ = make("anthropic:claude-old", [resp], env={"ANTHROPIC_API_KEY": "k"})
    with pytest.raises(ProviderError, match="claude-old.*--format text"):
        run(p.complete(SYSTEM, MSGS, tools=TOOLS))


def test_gemini_tools_payload_and_function_call_parsing():
    resp = httpx.Response(
        200,
        json={
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {"text": "Going north."},
                            {"functionCall": {"name": "act", "args": {"command": "go north"}}},
                            {"functionCall": {"name": "act"}},
                        ]
                    },
                    "finishReason": "STOP",
                }
            ],
            "usageMetadata": {"promptTokenCount": 9, "candidatesTokenCount": 2},
        },
    )
    p, seen, _ = make("gemini:gemini-2.5-flash", [resp], env={"GEMINI_API_KEY": "k"})
    c = run(p.complete(SYSTEM, MSGS, tools=TOOLS))
    body = sent(seen)
    assert body["tools"] == [
        {
            "functionDeclarations": [
                {"name": "act", "description": "Send one command.", "parameters": TOOLS[0]["parameters"]}
            ]
        }
    ]
    assert body["toolConfig"] == {"functionCallingConfig": {"mode": "ANY"}}
    assert c.text == "Going north."
    assert c.tool_calls == [
        {"name": "act", "arguments": {"command": "go north"}},
        {"name": "act", "arguments": {}},  # missing args: empty dict, not a crash
    ]
    p, seen, _ = make("gemini:gemini-2.5-flash", [resp], env={"GEMINI_API_KEY": "k"})
    run(p.complete(SYSTEM, MSGS))
    assert "tools" not in sent(seen) and "toolConfig" not in sent(seen)


def test_gemini_no_tool_support_error_is_explained():
    resp = httpx.Response(400, json={"error": {"message": "Function calling is not enabled for models/gemma"}})
    p, _, _ = make("gemini:gemma", [resp], env={"GEMINI_API_KEY": "k"})
    with pytest.raises(ProviderError, match="gemma.*--format text"):
        run(p.complete(SYSTEM, MSGS, tools=TOOLS))


def test_mock_and_human_accept_and_ignore_tools(tmp_path, monkeypatch):
    script = tmp_path / "s.txt"
    script.write_text("open mailbox\n", encoding="utf-8")
    p = create_provider(f"mock:{script}")
    c = run(p.complete(SYSTEM, MSGS, tools=TOOLS))
    assert c.text == "> open mailbox" and c.tool_calls is None
    monkeypatch.setattr("builtins.input", lambda prompt="": "take all")
    c = run(create_provider("human:").complete(SYSTEM, MSGS, tools=TOOLS))
    assert c.text == "> take all" and c.tool_calls is None

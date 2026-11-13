import asyncio
import json

import httpx
import pytest

from brass_lantern.providers import ConfigError, ProviderError, create_provider, split_spec

SYSTEM = "You are a tester."
MSGS = [{"role": "user", "content": "hello"}]


def run(coro):
    return asyncio.run(coro)


def capture(responses):
    """Transport that records requests and replays *responses* (callables or Responses)."""
    seen: list[httpx.Request] = []
    queue = list(responses)

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        r = queue.pop(0) if len(queue) > 1 else queue[0]
        return r(request) if callable(r) else r

    return httpx.MockTransport(handler), seen


def make(spec, responses, env=None, **kw):
    transport, seen = capture(responses)
    delays: list[float] = []

    async def fake_sleep(d):
        delays.append(d)

    p = create_provider(spec, env=env or {}, transport=transport, sleep=fake_sleep, **kw)
    return p, seen, delays


OPENAI_OK = httpx.Response(
    200,
    json={
        "choices": [{"message": {"content": " > look "}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 11, "completion_tokens": 3},
    },
)


# -- spec parsing -----------------------------------------------------------


def test_split_spec():
    assert split_spec("openrouter:anthropic/claude-haiku-4.5") == ("openrouter", "anthropic/claude-haiku-4.5")
    assert split_spec("ollama:llama3.1:8b") == ("ollama", "llama3.1:8b")
    assert split_spec("mock:C:\\tmp\\script.txt") == ("mock", "C:\\tmp\\script.txt")
    assert split_spec("human:") == ("human", "")
    assert split_spec("OpenAI:gpt-4o") == ("openai", "gpt-4o")


@pytest.mark.parametrize("bad", ["gpt-4o", "nope:model", "openai:", ""])
def test_split_spec_errors(bad):
    with pytest.raises(ConfigError):
        split_spec(bad)


def test_missing_key_is_clear():
    with pytest.raises(ConfigError, match="OPENROUTER_API_KEY"):
        create_provider("openrouter:x/y", env={})
    with pytest.raises(ConfigError, match="ANTHROPIC_API_KEY"):
        create_provider("anthropic:claude", env={})


# -- OpenRouter / OpenAI / Ollama -------------------------------------------


def test_openrouter_request_and_cost():
    resp = httpx.Response(
        200,
        json={
            "choices": [{"message": {"content": "Hmm.\n> open mailbox"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 120, "completion_tokens": 40, "cost": 0.000321},
        },
    )
    p, seen, _ = make("openrouter:meta-llama/llama-3.1-8b-instruct:free", [resp], env={"OPENROUTER_API_KEY": "sk-or"})
    c = run(p.complete(SYSTEM, MSGS))
    req = seen[0]
    body = json.loads(req.content)
    assert str(req.url) == "https://openrouter.ai/api/v1/chat/completions"
    assert req.headers["authorization"] == "Bearer sk-or"
    assert req.headers["x-title"] == "Brass Lantern"
    assert req.headers["http-referer"]
    assert body["model"] == "meta-llama/llama-3.1-8b-instruct:free"
    assert body["usage"] == {"include": True}
    assert body["messages"][0] == {"role": "system", "content": SYSTEM}
    assert body["messages"][1:] == MSGS
    assert body["temperature"] == 0.7 and body["max_tokens"] == 1024
    assert (c.text, c.input_tokens, c.output_tokens, c.cost_usd) == ("Hmm.\n> open mailbox", 120, 40, 0.000321)
    assert c.latency_s >= 0 and c.finish_reason == "stop"


def test_openrouter_without_cost_is_none():
    p, _, _ = make("openrouter:x/y", [OPENAI_OK], env={"OPENROUTER_API_KEY": "k"})
    assert run(p.complete(SYSTEM, MSGS)).cost_usd is None


def test_temperature_none_is_omitted_and_custom_max_tokens():
    p, seen, _ = make("openrouter:x/y", [OPENAI_OK], env={"OPENROUTER_API_KEY": "k"}, temperature=None, max_tokens=77)
    run(p.complete(SYSTEM, MSGS))
    body = json.loads(seen[0].content)
    assert "temperature" not in body and body["max_tokens"] == 77


def test_openai_official_uses_max_completion_tokens():
    p, seen, _ = make("openai:gpt-4o-mini", [OPENAI_OK], env={"OPENAI_API_KEY": "sk-o"})
    c = run(p.complete(SYSTEM, MSGS))
    body = json.loads(seen[0].content)
    assert str(seen[0].url) == "https://api.openai.com/v1/chat/completions"
    assert body["max_completion_tokens"] == 1024 and "max_tokens" not in body
    assert "usage" not in body
    assert seen[0].headers["authorization"] == "Bearer sk-o"
    assert c.text == "> look"


def test_openai_with_base_url_is_generic_and_key_optional():
    p, seen, _ = make("openai:local-model", [OPENAI_OK], base_url="http://localhost:8000/v1/")
    run(p.complete(SYSTEM, MSGS))
    body = json.loads(seen[0].content)
    assert str(seen[0].url) == "http://localhost:8000/v1/chat/completions"
    assert "authorization" not in seen[0].headers
    assert body["max_tokens"] == 1024 and "max_completion_tokens" not in body


def test_ollama_defaults():
    p, seen, _ = make("ollama:llama3.1:8b", [OPENAI_OK])
    run(p.complete(SYSTEM, MSGS))
    assert str(seen[0].url) == "http://localhost:11434/v1/chat/completions"
    assert "authorization" not in seen[0].headers
    assert json.loads(seen[0].content)["model"] == "llama3.1:8b"


def test_empty_content_falls_back_to_reasoning():
    resp = httpx.Response(
        200,
        json={
            "choices": [
                {"message": {"content": "", "reasoning": "I think...\n> go north"}, "finish_reason": "length"}
            ],
            "usage": {"prompt_tokens": 5, "completion_tokens": 1024},
        },
    )
    p, _, _ = make("openrouter:x/y", [resp], env={"OPENROUTER_API_KEY": "k"})
    c = run(p.complete(SYSTEM, MSGS))
    assert c.text.endswith("> go north") and c.finish_reason == "length"


def test_null_content_and_no_reasoning_gives_empty_text():
    resp = httpx.Response(200, json={"choices": [{"message": {"content": None}}]})
    p, _, _ = make("ollama:m", [resp])
    assert run(p.complete(SYSTEM, MSGS)).text == ""


def test_content_parts_array():
    resp = httpx.Response(200, json={"choices": [{"message": {"content": [{"type": "text", "text": "> look"}]}}]})
    p, _, _ = make("ollama:m", [resp])
    assert run(p.complete(SYSTEM, MSGS)).text == "> look"


# -- retries ----------------------------------------------------------------


def test_retry_on_429_then_success_with_backoff():
    rate = httpx.Response(429, json={"error": "slow down"})
    p, seen, delays = make("ollama:m", [rate, rate, OPENAI_OK])
    c = run(p.complete(SYSTEM, MSGS))
    assert c.text == "> look" and len(seen) == 3
    assert delays == [1.0, 2.0]


def test_retry_after_header_is_honoured():
    rate = httpx.Response(429, headers={"retry-after": "7"}, json={})
    p, _, delays = make("ollama:m", [rate, OPENAI_OK])
    run(p.complete(SYSTEM, MSGS))
    assert delays == [7.0]


def test_retry_on_5xx_and_timeouts():
    def boom(request):
        raise httpx.ConnectTimeout("timed out")

    p, seen, delays = make("ollama:m", [httpx.Response(503, text="busy"), boom, OPENAI_OK])
    assert run(p.complete(SYSTEM, MSGS)).text == "> look"
    assert len(seen) == 3 and len(delays) == 2


def test_gives_up_after_five_retries():
    p, seen, delays = make("ollama:m", [httpx.Response(500, text="kaput")])
    with pytest.raises(ProviderError, match=r"500.*kaput.*6 attempts"):
        run(p.complete(SYSTEM, MSGS))
    assert len(seen) == 6
    assert delays == [1.0, 2.0, 4.0, 8.0, 16.0]


def test_non_retryable_4xx_fails_immediately():
    p, seen, delays = make("ollama:m", [httpx.Response(400, json={"error": {"message": "bad model"}})])
    with pytest.raises(ProviderError, match="400"):
        run(p.complete(SYSTEM, MSGS))
    assert len(seen) == 1 and delays == []


def test_openrouter_200_with_upstream_error_is_retried():
    err = httpx.Response(200, json={"error": {"code": 502, "message": "upstream down"}})
    p, seen, _ = make("openrouter:x/y", [err, OPENAI_OK], env={"OPENROUTER_API_KEY": "k"})
    assert run(p.complete(SYSTEM, MSGS)).text == "> look"
    assert len(seen) == 2


def test_openrouter_200_with_client_error_is_not_retried():
    err = httpx.Response(200, json={"error": {"code": 400, "message": "nope"}})
    p, seen, _ = make("openrouter:x/y", [err], env={"OPENROUTER_API_KEY": "k"})
    with pytest.raises(ProviderError, match="nope"):
        run(p.complete(SYSTEM, MSGS))
    assert len(seen) == 1


# -- Anthropic --------------------------------------------------------------

ANTHROPIC_OK = httpx.Response(
    200,
    json={
        "content": [{"type": "text", "text": "Plan.\n> take lamp"}],
        "usage": {"input_tokens": 50, "output_tokens": 9},
        "stop_reason": "end_turn",
    },
)


def test_anthropic_request_shape_system_split():
    msgs = [
        {"role": "user", "content": "one"},
        {"role": "assistant", "content": "> look"},
        {"role": "user", "content": "two"},
    ]
    p, seen, _ = make("anthropic:claude-x", [ANTHROPIC_OK], env={"ANTHROPIC_API_KEY": "sk-a"}, cache=False)
    c = run(p.complete(SYSTEM, msgs))
    req = seen[0]
    body = json.loads(req.content)
    assert str(req.url) == "https://api.anthropic.com/v1/messages"
    assert req.headers["x-api-key"] == "sk-a" and req.headers["anthropic-version"] == "2023-06-01"
    assert body["system"] == SYSTEM
    assert body["messages"] == msgs  # no system role inside messages
    assert body["max_tokens"] == 1024 and body["temperature"] == 0.7 and body["model"] == "claude-x"
    assert (c.text, c.input_tokens, c.output_tokens, c.cost_usd) == ("Plan.\n> take lamp", 50, 9, None)


def test_anthropic_ignores_anthropic_base_url_env():
    env = {"ANTHROPIC_API_KEY": "k", "ANTHROPIC_BASE_URL": "http://some-proxy.invalid"}
    p, seen, _ = make("anthropic:claude-x", [ANTHROPIC_OK], env=env)
    run(p.complete(SYSTEM, MSGS))
    assert seen[0].url.host == "api.anthropic.com"


def test_anthropic_base_url_override_via_argument_only():
    p, seen, _ = make("anthropic:claude-x", [ANTHROPIC_OK], env={"ANTHROPIC_API_KEY": "k"}, base_url="http://proxy.test")
    run(p.complete(SYSTEM, MSGS))
    assert str(seen[0].url) == "http://proxy.test/v1/messages"


def test_anthropic_merges_consecutive_same_role():
    msgs = [{"role": "user", "content": "a"}, {"role": "user", "content": "b"}]
    p, seen, _ = make("anthropic:claude-x", [ANTHROPIC_OK], env={"ANTHROPIC_API_KEY": "k"}, cache=False)
    run(p.complete(SYSTEM, msgs))
    assert json.loads(seen[0].content)["messages"] == [{"role": "user", "content": "a\n\nb"}]


def test_anthropic_counts_cache_tokens_and_multiple_blocks():
    resp = httpx.Response(
        200,
        json={
            "content": [{"type": "thinking", "thinking": "hm"}, {"type": "text", "text": "> a"}, {"type": "text", "text": "\n> b"}],
            "usage": {"input_tokens": 10, "cache_read_input_tokens": 90, "output_tokens": 5},
        },
    )
    p, _, _ = make("anthropic:claude-x", [resp], env={"ANTHROPIC_API_KEY": "k"}, temperature=None)
    c = run(p.complete(SYSTEM, MSGS))
    assert c.text == "> a\n> b" and c.input_tokens == 100


def test_anthropic_retries_529_overloaded():
    p, seen, _ = make("anthropic:claude-x", [httpx.Response(529, text="overloaded"), ANTHROPIC_OK], env={"ANTHROPIC_API_KEY": "k"})
    run(p.complete(SYSTEM, MSGS))
    assert len(seen) == 2


# -- Gemini -----------------------------------------------------------------


def test_gemini_request_mapping():
    resp = httpx.Response(
        200,
        json={
            "candidates": [
                {
                    "content": {"parts": [{"text": "pondering", "thought": True}, {"text": "> go north"}]},
                    "finishReason": "STOP",
                }
            ],
            "usageMetadata": {"promptTokenCount": 30, "candidatesTokenCount": 4, "thoughtsTokenCount": 20},
        },
    )
    msgs = [
        {"role": "user", "content": "one"},
        {"role": "assistant", "content": "> look"},
        {"role": "user", "content": "two"},
    ]
    p, seen, _ = make("gemini:gemini-2.5-flash", [resp], env={"GEMINI_API_KEY": "g-key"})
    c = run(p.complete(SYSTEM, msgs))
    req = seen[0]
    body = json.loads(req.content)
    assert req.url.path == "/v1beta/models/gemini-2.5-flash:generateContent"
    assert req.headers["x-goog-api-key"] == "g-key"
    assert "g-key" not in str(req.url) and "key=" not in str(req.url)
    assert body["systemInstruction"] == {"parts": [{"text": SYSTEM}]}
    assert [c_["role"] for c_ in body["contents"]] == ["user", "model", "user"]
    assert body["contents"][1]["parts"] == [{"text": "> look"}]
    assert body["generationConfig"] == {"maxOutputTokens": 1024, "temperature": 0.7}
    assert (c.text, c.input_tokens, c.output_tokens, c.finish_reason) == ("> go north", 30, 24, "STOP")


def test_gemini_blocked_prompt_yields_empty_text_with_reason():
    resp = httpx.Response(200, json={"promptFeedback": {"blockReason": "SAFETY"}, "usageMetadata": {"promptTokenCount": 3}})
    p, _, _ = make("gemini:models/gemini-x", [resp], env={"GEMINI_API_KEY": "k"})
    c = run(p.complete(SYSTEM, MSGS))
    assert c.text == "" and c.finish_reason == "BLOCKED:SAFETY"
    assert p.model == "gemini-x"


# -- mock / human -----------------------------------------------------------


def test_mock_provider_replays_file_then_looks(tmp_path):
    script = tmp_path / "s.txt"
    script.write_text("# comment\nopen mailbox\n\n  north  \n", encoding="utf-8")
    p = create_provider(f"mock:{script}")
    texts = [run(p.complete(SYSTEM, MSGS)).text for _ in range(4)]
    assert texts == ["> open mailbox", "> north", "> look", "> look"]
    assert run(p.complete(SYSTEM, MSGS)).cost_usd == 0.0


def test_mock_missing_file_is_config_error(tmp_path):
    with pytest.raises(ConfigError, match="mock script"):
        create_provider(f"mock:{tmp_path / 'nope.txt'}")


def test_human_provider_reads_stdin(monkeypatch):
    monkeypatch.setattr("builtins.input", lambda prompt="": "take all")
    p = create_provider("human:")
    assert run(p.complete(SYSTEM, MSGS)).text == "> take all"

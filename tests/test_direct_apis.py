"""Direct Anthropic / OpenAI runs: @effort specs, which settings each model gets, and the 400-temperature retry."""

import json

import httpx
import pytest

from brass_lantern.providers import ConfigError, ProviderError, create_provider, parse_spec, resolve_settings, split_spec
from brass_lantern.providers.models import anthropic_rejects_forced_tool_choice, anthropic_rejects_temperature
from test_providers import ANTHROPIC_OK, MSGS, OPENAI_OK, SYSTEM, make, run
from test_providers_cache_tools import HISTORY_MSGS, TOOLS, sent

A_ENV = {"ANTHROPIC_API_KEY": "k"}
O_ENV = {"OPENAI_API_KEY": "k"}
TOOL_USE_OK = httpx.Response(
    200, json={"content": [{"type": "tool_use", "name": "act", "input": {"command": "look"}}], "usage": {}}
)
OPENAI_TOOL_OK = httpx.Response(
    200,
    json={"choices": [{"message": {"content": None, "tool_calls": [
        {"function": {"name": "act", "arguments": '{"command": "look"}'}}]}}], "usage": {}},
)


# ---------------------------------------------------------------------------
# Specs
# ---------------------------------------------------------------------------


def test_parse_spec_splits_the_effort_suffix_and_keeps_the_base():
    s = parse_spec("anthropic:claude-sonnet-5-5@high")
    assert (s.provider, s.model, s.effort, s.base) == ("anthropic", "claude-sonnet-5-5", "high", "anthropic:claude-sonnet-5-5")
    assert parse_spec("OpenAI:gpt-6.1-sol@XHigh").effort == "xhigh" and parse_spec("openai:gpt-6.1-sol").effort is None
    assert parse_spec("openrouter:anthropic/claude-haiku-5.5@minimal").model == "anthropic/claude-haiku-5.5"
    assert split_spec("anthropic:claude-sonnet-5-5@low") == ("anthropic", "claude-sonnet-5-5")  # suffix never leaks
    assert split_spec("ollama:llama3.1:8b") == ("ollama", "llama3.1:8b")


def test_an_at_sign_that_is_not_an_effort_stays_in_the_model_id():
    assert parse_spec("openrouter:vendor/model@20240620").model == "vendor/model@20240620"
    assert parse_spec("mock:C:\\tmp\\a@high.txt").model == "C:\\tmp\\a@high.txt"  # script paths are never split
    assert parse_spec("mock:run@high").effort is None


@pytest.mark.parametrize("bad", ["anthropic:claude-x@hgih", "openai:gpt-6.1-sol@fast", "anthropic:@high"])
def test_bad_effort_suffix_is_a_config_error(bad):
    with pytest.raises(ConfigError):
        parse_spec(bad)


def test_resolve_settings_precedence_and_defaults():
    # the suffix beats the flag; the flag applies to specs without one
    assert resolve_settings("anthropic:claude-sonnet-5-5@low", effort="high").effort == "low"
    assert resolve_settings("anthropic:claude-sonnet-5-5", effort="high").effort == "high"
    assert resolve_settings("openai:gpt-6.1-sol").effort is None
    # max_tokens: 16000 for anthropic or when an effort is sent, else 1024; an explicit value always wins
    assert resolve_settings("anthropic:claude-sonnet-4-5").max_tokens == 16000
    assert resolve_settings("openai:gpt-6.1-sol@medium").max_tokens == 16000
    assert resolve_settings("openrouter:x/y", effort="low").max_tokens == 16000
    assert resolve_settings("openai:gpt-6.1-sol").max_tokens == 1024
    assert resolve_settings("openrouter:x/y").max_tokens == 1024
    assert resolve_settings("anthropic:claude-sonnet-5-5@high", max_tokens=300).max_tokens == 300
    assert resolve_settings("openai:gpt-6.1-sol@high", max_tokens=300).max_tokens == 300


@pytest.mark.parametrize(
    "spec, effort, message",
    [
        ("anthropic:claude-sonnet-5-5@minimal", None, "no 'minimal'"),
        ("anthropic:claude-sonnet-5-5", "minimal", "no 'minimal'"),
        ("gemini:gemini-3-pro@high", None, "openrouter:google/gemini-3-pro"),
        ("gemini:gemini-3-pro", "low", "openrouter:google/"),
        ("ollama:llama3.1:8b@low", None, "ollama"),
    ],
)
def test_efforts_a_provider_cannot_take_are_config_errors(spec, effort, message):
    with pytest.raises(ConfigError, match=message):
        resolve_settings(spec, effort=effort)
    with pytest.raises(ConfigError, match=message):
        create_provider(spec, effort=effort, env={"ANTHROPIC_API_KEY": "k", "GEMINI_API_KEY": "k"})


def test_mock_human_and_generic_openai_servers_ignore_effort(tmp_path):
    assert resolve_settings("mock:script.txt@high", effort="high").effort is None
    assert resolve_settings("human:", effort="high").effort is None
    assert resolve_settings("openai:local@high", base_url="http://localhost:8000/v1").effort is None
    script = tmp_path / "s.txt"
    script.write_text("look\n", encoding="utf-8")
    assert create_provider(f"mock:{script}", effort="max").effort is None
    # nothing is sent to a generic OpenAI-compatible server
    p, seen, _ = make("openai:local-model", [OPENAI_OK], base_url="http://localhost:8000/v1", effort="high")
    run(p.complete(SYSTEM, MSGS))
    assert "reasoning_effort" not in sent(seen) and p.effort is None


def test_the_provider_records_the_effective_settings():
    p = create_provider("anthropic:claude-opus-5-5@high", env=A_ENV)
    assert (p.effort, p.temperature, p.max_tokens) == ("high", None, 16000)
    assert p.describe()["effort"] == "high" and p.describe()["temperature"] is None
    p = create_provider("openai:gpt-6.1-sol", env=O_ENV)
    assert (p.effort, p.temperature, p.max_tokens) == (None, 0.7, 1024)
    assert create_provider("anthropic:claude-sonnet-5-5", env=A_ENV, effort="max").effort == "max"
    # the old name still works
    assert create_provider("openrouter:x/y", env={"OPENROUTER_API_KEY": "k"}, reasoning="low").effort == "low"


# ---------------------------------------------------------------------------
# Anthropic payloads
# ---------------------------------------------------------------------------


def test_anthropic_opus_5_5_high_text_mode_payload():
    p, seen, _ = make("anthropic:claude-opus-5-5@high", [ANTHROPIC_OK], env=A_ENV)
    run(p.complete(SYSTEM, HISTORY_MSGS))
    body = sent(seen)
    assert body["model"] == "claude-opus-5-5"  # no @high in the model
    assert body["output_config"] == {"effort": "high"}
    assert "temperature" not in body and "thinking" not in body
    assert body["max_tokens"] == 16000
    assert "tools" not in body and "tool_choice" not in body


def test_anthropic_opus_5_5_high_tools_mode_payload_uses_auto_tool_choice():
    p, seen, _ = make("anthropic:claude-opus-5-5@high", [TOOL_USE_OK], env=A_ENV)
    c = run(p.complete(SYSTEM, HISTORY_MSGS, tools=TOOLS))
    body = sent(seen)
    assert body["tool_choice"] == {"type": "auto"} and body["output_config"] == {"effort": "high"}
    assert "temperature" not in body and "thinking" not in body and body["max_tokens"] == 16000
    assert body["tools"][0]["name"] == "act" and body["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert c.tool_calls == [{"name": "act", "arguments": {"command": "look"}}]


def test_anthropic_without_effort_sends_no_output_config_and_the_flag_sets_it():
    p, seen, _ = make("anthropic:claude-sonnet-4-5", [ANTHROPIC_OK], env=A_ENV)
    run(p.complete(SYSTEM, MSGS))
    assert "output_config" not in sent(seen)
    p, seen, _ = make("anthropic:claude-sonnet-5-5", [ANTHROPIC_OK], env=A_ENV, effort="xhigh")
    run(p.complete(SYSTEM, MSGS))
    assert sent(seen)["output_config"] == {"effort": "xhigh"}
    p, seen, _ = make("anthropic:claude-sonnet-5-5@max", [ANTHROPIC_OK], env=A_ENV, effort="low", max_tokens=4000)
    run(p.complete(SYSTEM, MSGS))
    assert sent(seen)["output_config"] == {"effort": "max"} and sent(seen)["max_tokens"] == 4000


@pytest.mark.parametrize(
    "model, temperature_sent",
    [
        ("claude-sonnet-4-5", True), ("claude-haiku-4-5", True), ("claude-opus-4-6", True),
        ("claude-opus-5-5", False), ("claude-sonnet-5-5", False), ("claude-haiku-5-5", False),
        ("claude-fable-5-2", False), ("claude-fable-5", False), ("claude-opus-5", False), ("claude-sonnet-5", False),
        ("claude-opus-4-8", False), ("claude-opus-4-7", False), ("claude-mythos-5", False),
        ("claude-sonnet-5-5-20261001", False),
    ],
)
def test_anthropic_temperature_only_where_the_model_accepts_it(model, temperature_sent):
    p, seen, _ = make(f"anthropic:{model}", [ANTHROPIC_OK], env=A_ENV)
    run(p.complete(SYSTEM, MSGS))
    assert ("temperature" in sent(seen)) is temperature_sent
    assert anthropic_rejects_temperature(model) is (not temperature_sent)
    assert (p.temperature == 0.7) is temperature_sent
    if temperature_sent:
        assert sent(seen)["temperature"] == 0.7


@pytest.mark.parametrize(
    "model, forced",
    [
        ("claude-sonnet-4-5", True), ("claude-haiku-4-5", True), ("claude-opus-4-8", True), ("claude-opus-5", True),
        ("claude-sonnet-5", True), ("claude-haiku-5-5", True), ("claude-fable-5", True),
        ("claude-opus-5-5", False), ("claude-sonnet-5-5", False), ("claude-fable-5-1", False),
        ("claude-fable-5-2", False), ("claude-mythos-5", False), ("claude-mythos", False),
    ],
)
def test_anthropic_tool_choice_is_forced_only_where_the_model_allows_it(model, forced):
    p, seen, _ = make(f"anthropic:{model}", [TOOL_USE_OK], env=A_ENV)
    run(p.complete(SYSTEM, MSGS, tools=TOOLS))
    assert sent(seen)["tool_choice"] == ({"type": "any"} if forced else {"type": "auto"})
    assert anthropic_rejects_forced_tool_choice(model) is (not forced)


def test_openrouter_anthropic_models_that_reject_forced_choice_get_auto_but_keep_the_temperature():
    env = {"OPENROUTER_API_KEY": "k"}
    for model, expected in [("anthropic/claude-opus-5.5", "auto"), ("anthropic/claude-fable-5.2", "auto"),
                            ("anthropic/claude-sonnet-5.5", "auto"), ("anthropic/claude-opus-5", "required"),
                            ("anthropic/claude-haiku-4.5", "required"), ("openai/gpt-6.1-sol", "required")]:
        p, seen, _ = make(f"openrouter:{model}", [OPENAI_TOOL_OK], env=env)
        run(p.complete(SYSTEM, MSGS, tools=TOOLS))
        assert sent(seen)["tool_choice"] == expected, model
        assert sent(seen)["temperature"] == 0.7  # OpenRouter drops what a model rejects
    p, seen, _ = make("openrouter:anthropic/claude-opus-5.5@high", [OPENAI_TOOL_OK], env=env)
    run(p.complete(SYSTEM, MSGS, tools=TOOLS))
    body = sent(seen)
    assert body["reasoning"] == {"effort": "high"} and body["max_tokens"] == 16000 and body["tool_choice"] == "auto"


def test_anthropic_read_timeout_is_generous():
    p = create_provider("anthropic:claude-sonnet-5-5", env=A_ENV)
    assert p._http.timeout.read >= 600
    p = create_provider("openai:gpt-6.1-sol@high", env=O_ENV)
    assert p._http.timeout.read >= 600  # a high-effort reasoning turn is slow too
    p = create_provider("openai:gpt-6.1-sol", env=O_ENV)
    assert p._http.timeout.read == 120


# ---------------------------------------------------------------------------
# OpenAI payloads
# ---------------------------------------------------------------------------


def test_openai_gpt_6_1_sol_medium_payload():
    p, seen, _ = make("openai:gpt-6.1-sol@medium", [OPENAI_OK], env=O_ENV)
    run(p.complete(SYSTEM, MSGS))
    body = sent(seen)
    assert body["model"] == "gpt-6.1-sol"
    assert body["reasoning_effort"] == "medium"
    assert body["max_completion_tokens"] == 16000 and "max_tokens" not in body
    assert body["temperature"] == 0.7  # sent first; the 400 safety net removes it if the model rejects it
    assert seen[0].headers["authorization"] == "Bearer k"


def test_openai_without_effort_keeps_the_small_default_and_a_flag_effort_applies():
    p, seen, _ = make("openai:gpt-6.1-sol", [OPENAI_OK], env=O_ENV)
    run(p.complete(SYSTEM, MSGS))
    assert "reasoning_effort" not in sent(seen) and sent(seen)["max_completion_tokens"] == 1024
    p, seen, _ = make("openai:gpt-6.1-sol", [OPENAI_OK], env=O_ENV, effort="high", temperature=None)
    run(p.complete(SYSTEM, MSGS))
    assert sent(seen)["reasoning_effort"] == "high" and "temperature" not in sent(seen)


def test_openrouter_effort_is_still_reasoning_effort_and_max_tokens_default_follows_it():
    p, seen, _ = make("openrouter:openai/gpt-6.1-sol", [OPENAI_OK], env={"OPENROUTER_API_KEY": "k"})
    run(p.complete(SYSTEM, MSGS))
    assert "reasoning" not in sent(seen) and sent(seen)["max_tokens"] == 1024 and sent(seen)["usage"] == {"include": True}
    p, seen, _ = make("openrouter:openai/gpt-6.1-sol@low", [OPENAI_OK], env={"OPENROUTER_API_KEY": "k"})
    run(p.complete(SYSTEM, MSGS))
    assert sent(seen)["reasoning"] == {"effort": "low"} and sent(seen)["max_tokens"] == 16000


# ---------------------------------------------------------------------------
# The 400-temperature safety net
# ---------------------------------------------------------------------------

TEMP_400 = httpx.Response(
    400,
    json={"error": {"message": "Unsupported value: 'temperature' does not support 0.7 with this model. "
                               "Only the default (1) value is supported."}},
)


def test_a_400_about_temperature_drops_it_warns_once_and_retries_once(capsys):
    p, seen, delays = make("openai:gpt-6.1-sol", [TEMP_400, OPENAI_OK, OPENAI_OK], env=O_ENV)
    c = run(p.complete(SYSTEM, MSGS))
    assert c.text == "> look" and len(seen) == 2 and delays == []
    assert sent(seen, 0)["temperature"] == 0.7 and "temperature" not in sent(seen, 1)
    assert sent(seen, 1)["messages"] == sent(seen, 0)["messages"]
    assert p.temperature is None and p.temperature_dropped and p.describe()["temperature"] is None
    err = capsys.readouterr().err
    assert err.count("warning:") == 1 and "temperature" in err and "gpt-6.1-sol" in err
    # later calls never send it, and never warn again
    run(p.complete(SYSTEM, MSGS))
    assert len(seen) == 3 and "temperature" not in sent(seen, 2) and capsys.readouterr().err == ""


def test_the_temperature_retry_also_covers_anthropic_and_gemini_payloads(capsys):
    p, seen, _ = make("anthropic:claude-sonnet-4-5", [httpx.Response(400, json={"error": {"message": "temperature: not allowed"}}), ANTHROPIC_OK],
                      env=A_ENV)
    run(p.complete(SYSTEM, MSGS))
    assert "temperature" in sent(seen, 0) and "temperature" not in sent(seen, 1) and len(seen) == 2
    gemini_ok = httpx.Response(200, json={"candidates": [{"content": {"parts": [{"text": "> look"}]}}]})
    p, seen, _ = make("gemini:gemini-3-pro", [httpx.Response(400, text="Invalid value at generation_config: temperature"), gemini_ok],
                      env={"GEMINI_API_KEY": "k"})
    run(p.complete(SYSTEM, MSGS))
    assert sent(seen, 0)["generationConfig"]["temperature"] == 0.7
    assert sent(seen, 1)["generationConfig"] == {"maxOutputTokens": 1024}
    assert capsys.readouterr().err.count("warning:") == 2


def test_other_400s_are_not_retried_and_a_second_failure_is_raised(capsys):
    p, seen, _ = make("openai:gpt-6.1-sol", [httpx.Response(400, json={"error": {"message": "bad model"}})], env=O_ENV)
    with pytest.raises(ProviderError, match="bad model"):
        run(p.complete(SYSTEM, MSGS))
    assert len(seen) == 1 and p.temperature == 0.7 and not p.temperature_dropped
    # no temperature in the request: a 400 that mentions it is not ours to fix
    p, seen, _ = make("openai:gpt-6.1-sol", [TEMP_400], env=O_ENV, temperature=None)
    with pytest.raises(ProviderError, match="temperature"):
        run(p.complete(SYSTEM, MSGS))
    assert len(seen) == 1
    # the retry happens once: a second 400 is raised, even one that mentions temperature again
    p, seen, _ = make("openai:gpt-6.1-sol", [TEMP_400], env=O_ENV)
    with pytest.raises(ProviderError, match="temperature") as ei:
        run(p.complete(SYSTEM, MSGS))
    assert len(seen) == 2 and ei.value.status == 400
    assert capsys.readouterr().err.count("warning:") == 1


def test_the_temperature_retry_keeps_tools_in_the_payload():
    p, seen, _ = make("openai:gpt-6.1-sol", [TEMP_400, OPENAI_TOOL_OK], env=O_ENV)
    c = run(p.complete(SYSTEM, MSGS, tools=TOOLS))
    assert sent(seen, 1)["tool_choice"] == "required" and "temperature" not in sent(seen, 1)
    assert c.tool_calls == [{"name": "act", "arguments": {"command": "look"}}]

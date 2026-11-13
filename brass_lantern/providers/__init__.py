"""Provider registry: turn a ``provider:model`` spec into a :class:`Provider`."""

from __future__ import annotations

import os
from typing import Any, Mapping

import httpx

from brass_lantern.providers.base import Completion, ConfigError, Provider, ProviderError
from brass_lantern.providers.models import (
    EFFORTS,
    LONG_READ_TIMEOUT_S,
    PROVIDER_NAMES,
    ModelSpec,
    Settings,
    parse_spec,
    resolve_settings,
    split_spec,
)

__all__ = [
    "Completion",
    "ConfigError",
    "EFFORTS",
    "ModelSpec",
    "PROVIDER_NAMES",
    "Provider",
    "ProviderError",
    "Settings",
    "create_provider",
    "parse_spec",
    "resolve_settings",
    "split_spec",
]


def _key(env: Mapping[str, str], var: str, provider: str) -> str:
    value = (env.get(var) or "").strip()
    if not value:
        raise ConfigError(f"{var} is not set (needed for {provider}:). Put it in .env or export it.")
    return value


def create_provider(
    spec: str,
    *,
    base_url: str | None = None,
    temperature: float | None = 0.7,
    max_tokens: int | None = None,
    effort: str | None = None,
    reasoning: str | None = None,  # deprecated alias of effort
    cache: bool = True,
    env: Mapping[str, str] | None = None,
    transport: httpx.AsyncBaseTransport | None = None,
    sleep: Any = None,
) -> Provider:
    """Build a provider from a spec like ``openrouter:openai/gpt-4o-mini`` or ``anthropic:claude-opus-5-5@high``.

    The spec's ``@effort`` suffix overrides ``effort``. ``temperature``, ``effort`` and ``max_tokens`` are
    what was asked for; what is really sent follows :func:`resolve_settings` (models that reject a
    temperature get none, ``max_tokens=None`` picks 16000 or 1024), and the provider's ``temperature``,
    ``effort`` and ``max_tokens`` attributes hold the effective values.

    ``base_url`` overrides the provider's endpoint (this is how ``openai:`` talks to
    vLLM / LM Studio / Together). ``env`` defaults to ``os.environ``. Note that
    ANTHROPIC_BASE_URL is deliberately never read. ``cache`` turns on the request hints that
    enable prompt caching (Anthropic, and Anthropic models via OpenRouter).
    """
    env = os.environ if env is None else env
    provider, model = split_spec(spec)
    settings = resolve_settings(
        spec, effort=effort or reasoning, temperature=temperature, max_tokens=max_tokens, base_url=base_url
    )
    temperature, max_tokens, effort = settings.temperature, settings.max_tokens, settings.effort

    if provider == "mock":
        from brass_lantern.providers.scripted import ScriptedProvider

        try:
            return ScriptedProvider.from_file(model, temperature=temperature, max_tokens=max_tokens, cache=cache)
        except OSError as e:
            raise ConfigError(f"Cannot read mock script {model!r}: {e}") from e
    if provider == "human":
        from brass_lantern.providers.scripted import HumanProvider

        return HumanProvider(model, temperature=temperature, max_tokens=max_tokens, cache=cache)

    common: dict[str, Any] = {
        "temperature": temperature, "max_tokens": max_tokens, "cache": cache, "transport": transport,
        "effort": effort,
    }
    if effort:  # reasoning turns are slow: do not time out (and retry, and re-bill) at the 120 s default
        common["timeout"] = httpx.Timeout(LONG_READ_TIMEOUT_S, connect=30.0)
    if sleep is not None:
        common["sleep"] = sleep

    if provider == "anthropic":
        from brass_lantern.providers.anthropic import ANTHROPIC_URL, AnthropicProvider

        return AnthropicProvider(
            model, api_key=_key(env, "ANTHROPIC_API_KEY", provider), base_url=base_url or ANTHROPIC_URL, **common
        )
    if provider == "gemini":
        from brass_lantern.providers.gemini import GEMINI_URL, GeminiProvider

        return GeminiProvider(
            model, api_key=_key(env, "GEMINI_API_KEY", provider), base_url=base_url or GEMINI_URL, **common
        )

    from brass_lantern.providers.openai_compat import (
        OLLAMA_URL,
        OPENAI_URL,
        OPENROUTER_URL,
        OpenAICompatProvider,
    )

    if provider == "openrouter":
        return OpenAICompatProvider(
            model,
            base_url=base_url or OPENROUTER_URL,
            api_key=_key(env, "OPENROUTER_API_KEY", provider),
            name="openrouter",
            extra_headers={
                "HTTP-Referer": (env.get("OPENROUTER_REFERER") or "https://brass-lantern.local"),
                "X-Title": "Brass Lantern",
            },
            # usage.include makes OpenRouter return usage.cost in USD; reasoning.effort is
            # OpenRouter's vendor-neutral control for reasoning models.
            extra_body={"usage": {"include": True}, **({"reasoning": {"effort": effort}} if effort else {})},
            **common,
        )
    if provider == "ollama":
        return OpenAICompatProvider(model, base_url=base_url or OLLAMA_URL, name="ollama", **common)

    # provider == "openai"
    if base_url:
        # Generic OpenAI-compatible server: the key is optional (local servers often have none).
        return OpenAICompatProvider(
            model, base_url=base_url, api_key=(env.get("OPENAI_API_KEY") or "").strip() or None, name="openai", **common
        )
    return OpenAICompatProvider(
        model,
        base_url=OPENAI_URL,
        api_key=_key(env, "OPENAI_API_KEY", provider),
        extra_body={"reasoning_effort": effort} if effort else None,
        name="openai",
        token_param="max_completion_tokens",  # newer OpenAI models reject max_tokens
        **common,
    )

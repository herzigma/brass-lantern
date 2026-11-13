"""Provider registry: turn a ``provider:model`` spec into a :class:`Provider`."""

from __future__ import annotations

import os
from typing import Any, Mapping

import httpx

from brass_lantern.providers.base import Completion, ConfigError, Provider, ProviderError

__all__ = [
    "Completion",
    "ConfigError",
    "PROVIDER_NAMES",
    "Provider",
    "ProviderError",
    "create_provider",
    "split_spec",
]

PROVIDER_NAMES = ("openrouter", "openai", "ollama", "anthropic", "gemini", "mock", "human")


def split_spec(spec: str) -> tuple[str, str]:
    """Split ``provider:model`` on the *first* colon (models may contain colons)."""
    provider, sep, model = spec.strip().partition(":")
    provider = provider.lower()
    if not sep or provider not in PROVIDER_NAMES:
        raise ConfigError(
            f"Bad model spec {spec!r}. Use provider:model with provider one of "
            f"{', '.join(PROVIDER_NAMES)} (e.g. openrouter:anthropic/claude-haiku-4.5)."
        )
    if not model.strip() and provider != "human":
        raise ConfigError(f"Model spec {spec!r} is missing the model name after the colon.")
    return provider, model.strip()


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
    max_tokens: int = 1024,
    reasoning: str | None = None,
    cache: bool = True,
    env: Mapping[str, str] | None = None,
    transport: httpx.AsyncBaseTransport | None = None,
    sleep: Any = None,
) -> Provider:
    """Build a provider from a spec like ``openrouter:openai/gpt-4o-mini``.

    ``base_url`` overrides the provider's endpoint (this is how ``openai:`` talks to
    vLLM / LM Studio / Together). ``env`` defaults to ``os.environ``. Note that
    ANTHROPIC_BASE_URL is deliberately never read. ``cache`` turns on the request hints that
    enable prompt caching (Anthropic, and Anthropic models via OpenRouter).
    """
    env = os.environ if env is None else env
    provider, model = split_spec(spec)

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
    }
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
            extra_body={"usage": {"include": True}, **({"reasoning": {"effort": reasoning}} if reasoning else {})},
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
        extra_body={"reasoning_effort": reasoning} if reasoning else None,
        name="openai",
        token_param="max_completion_tokens",  # newer OpenAI models reject max_tokens
        **common,
    )

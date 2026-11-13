"""Model specs (``provider:model[@effort]``) and the per-model rules for the direct APIs.

OpenRouter silently drops request settings a model does not support; the direct Anthropic and OpenAI
APIs answer HTTP 400 instead. The rules here decide, per spec, which settings are actually sent.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from brass_lantern.providers.base import ConfigError

PROVIDER_NAMES = ("openrouter", "openai", "ollama", "anthropic", "gemini", "mock", "human")
EFFORTS = ("minimal", "low", "medium", "high", "xhigh", "max")
ANTHROPIC_EFFORTS = tuple(e for e in EFFORTS if e != "minimal")

DEFAULT_MAX_TOKENS = 1024
THINKING_MAX_TOKENS = 16000  # default when the model thinks: thinking tokens count against max_tokens
LONG_READ_TIMEOUT_S = 600.0  # a high-effort turn can take minutes; a timeout would be retried (and billed)


@dataclass(frozen=True)
class ModelSpec:
    provider: str
    model: str  # without the @effort suffix
    effort: str | None = None  # from the @suffix, if any

    @property
    def base(self) -> str:
        """``provider:model`` without the effort suffix (what prices and Grue's ``model`` use)."""
        return f"{self.provider}:{self.model}"


def parse_spec(spec: str) -> ModelSpec:
    """Split ``provider:model[@effort]``: the provider on the first colon, the effort on the last ``@``.

    A trailing ``@word`` is an effort when it is one of :data:`EFFORTS`; any other all-letter suffix is
    a typo and an error. ``@`` followed by anything else (a date, say) stays part of the model id, and
    ``mock:`` / ``human:`` specs are never split (a script path may contain ``@``).
    """
    provider, sep, model = spec.strip().partition(":")
    provider = provider.lower()
    if not sep or provider not in PROVIDER_NAMES:
        raise ConfigError(
            f"Bad model spec {spec!r}. Use provider:model with provider one of "
            f"{', '.join(PROVIDER_NAMES)} (e.g. openrouter:anthropic/claude-haiku-4.5)."
        )
    model = model.strip()
    effort = None
    if provider not in ("mock", "human"):
        head, at, tail = model.rpartition("@")
        if at:
            word = tail.strip().lower()
            if word in EFFORTS:
                model, effort = head.strip(), word
            elif word.isalpha():
                raise ConfigError(
                    f"Unknown effort {tail!r} in model spec {spec!r}. Use one of {', '.join(EFFORTS)}."
                )
    if not model and provider != "human":
        raise ConfigError(f"Model spec {spec!r} is missing the model name after the colon.")
    return ModelSpec(provider, model, effort)


def split_spec(spec: str) -> tuple[str, str]:
    """``(provider, model)`` with any ``@effort`` suffix removed (see :func:`parse_spec`)."""
    parsed = parse_spec(spec)
    return parsed.provider, parsed.model


# ---------------------------------------------------------------------------
# Anthropic model families
# ---------------------------------------------------------------------------


def anthropic_model_id(model: str) -> str:
    """Normalised Claude id: lower case, no ``anthropic/`` prefix, dots as dashes (``claude-opus-5-5``)."""
    m = model.strip().lower()
    return m.removeprefix("anthropic/").replace(".", "-")


# Models that answer 400 to a temperature (Opus 5.5/5/4.8/4.7 and Fable always; Sonnet 5.5 and Haiku 5.5
# only accept their default): every 5.x model, plus Opus 4.7 and 4.8, plus Mythos.
_NO_TEMPERATURE = re.compile(r"^claude-(?:(?:opus|sonnet|haiku|fable|mythos)-5|opus-4-[78]|mythos)(?:-|$)")
# Models that answer 400 to a forced tool_choice ("any" or "tool"): they get "auto" instead.
_NO_FORCED_TOOL_CHOICE = re.compile(r"^claude-(?:opus-5-5|sonnet-5-5|fable-5-[12]|mythos)(?:-|$)")


def anthropic_rejects_temperature(model: str) -> bool:
    return bool(_NO_TEMPERATURE.match(anthropic_model_id(model)))


def anthropic_rejects_forced_tool_choice(model: str) -> bool:
    return bool(_NO_FORCED_TOOL_CHOICE.match(anthropic_model_id(model)))


# ---------------------------------------------------------------------------
# Effective settings
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Settings:
    """What is actually sent for one spec (after the per-model rules)."""

    effort: str | None  # None = nothing sent: the provider's default
    temperature: float | None  # None = not sent: the model's default
    max_tokens: int


def resolve_settings(
    spec: str,
    *,
    effort: str | None = None,
    temperature: float | None = 0.7,
    max_tokens: int | None = None,
    base_url: str | None = None,
) -> Settings:
    """Work out the effort, temperature and max_tokens that really go out for *spec*.

    The spec's ``@effort`` suffix beats the ``effort`` default. Providers that cannot take an effort
    raise :class:`ConfigError` (Anthropic also refuses ``minimal``); ``mock:`` / ``human:`` ignore it and
    so does an ``openai:`` spec with ``base_url`` (a generic server gets no ``reasoning_effort``).
    Anthropic models that reject a temperature get none. ``max_tokens`` defaults to 16000 for Anthropic
    or whenever an effort is sent (thinking tokens count against it), else 1024.
    """
    ms = parse_spec(spec)
    chosen = ms.effort or effort
    if chosen is not None and chosen not in EFFORTS:
        raise ConfigError(f"Unknown effort {chosen!r}. Use one of {', '.join(EFFORTS)}.")
    if ms.provider in ("mock", "human") or (ms.provider == "openai" and base_url):
        chosen = None
    elif chosen and ms.provider == "anthropic" and chosen not in ANTHROPIC_EFFORTS:
        raise ConfigError(
            f"{ms.base}: the Anthropic API has no 'minimal' effort. Use one of {', '.join(ANTHROPIC_EFFORTS)}."
        )
    elif chosen and ms.provider == "gemini":
        raise ConfigError(
            f"{ms.base}: --effort / @effort is not supported for direct Gemini. Use "
            f"openrouter:google/{ms.model} to set reasoning effort."
        )
    elif chosen and ms.provider == "ollama":
        raise ConfigError(f"{ms.base}: --effort / @effort is not supported for ollama models.")
    if temperature is not None and ms.provider == "anthropic" and anthropic_rejects_temperature(ms.model):
        temperature = None
    if max_tokens is None:
        max_tokens = THINKING_MAX_TOKENS if (ms.provider == "anthropic" or chosen) else DEFAULT_MAX_TOKENS
    return Settings(effort=chosen, temperature=temperature, max_tokens=max_tokens)

"""Anthropic Messages API provider (raw httpx, no SDK)."""

from __future__ import annotations

from typing import Any

import httpx

from brass_lantern.providers.base import (
    CACHE_CONTROL,
    Completion,
    HTTPProvider,
    ProviderError,
    explain_tool_error,
    merge_consecutive,
    to_int,
    tool_call,
    with_cache_breakpoint,
)
from brass_lantern.providers.models import LONG_READ_TIMEOUT_S, anthropic_rejects_forced_tool_choice

ANTHROPIC_URL = "https://api.anthropic.com"
API_VERSION = "2023-06-01"


class AnthropicProvider(HTTPProvider):
    name = "anthropic"

    def __init__(
        self,
        model: str,
        *,
        api_key: str,
        base_url: str = ANTHROPIC_URL,
        extra_body: dict[str, Any] | None = None,  # e.g. {"thinking": {...}}, merged into every request
        **kwargs: Any,
    ):
        # Thinking is adaptive and on by default on the 5.x models, and a high-effort turn can take
        # minutes: wait for it instead of timing out (a timeout is retried, and the retry is billed).
        kwargs.setdefault("timeout", httpx.Timeout(LONG_READ_TIMEOUT_S, connect=30.0))
        super().__init__(model, base_url=base_url, **kwargs)
        self._extra_body = dict(extra_body or {})
        self._headers = {
            "x-api-key": api_key,
            "anthropic-version": API_VERSION,
            "content-type": "application/json",
        }

    async def complete(
        self, system: str, messages: list[dict[str, str]], tools: list[dict[str, Any]] | None = None
    ) -> Completion:
        msgs: list[dict[str, Any]] = merge_consecutive(messages)
        payload: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "system": system,  # Anthropic takes the system prompt outside `messages`
            "messages": msgs,
        }
        if self.cache:
            # Up to 4 breakpoints are allowed; we use 2. The system one caches the tools and system
            # prompt; the one on the last message caches the whole conversation so far, so the next
            # call (whose messages start with this prefix) reads it back instead of re-billing it.
            payload["system"] = [{"type": "text", "text": system, "cache_control": dict(CACHE_CONTROL)}]
            if msgs:
                msgs[-1] = with_cache_breakpoint(msgs[-1])
        if self.temperature is not None:  # create_provider leaves it out for models that reject it
            payload["temperature"] = self.temperature
        if self.effort:
            payload["output_config"] = {"effort": self.effort}
        payload.update(self._extra_body)
        if tools:
            payload["tools"] = [
                {"name": t["name"], "description": t.get("description", ""), "input_schema": t["parameters"]}
                for t in tools
            ]
            # Anthropic rejects a forced tool choice when extended thinking is on, and some models
            # (Opus 5.5, Sonnet 5.5, Fable 5.1/5.2, Mythos) reject it outright.
            forced = "thinking" not in payload and not anthropic_rejects_forced_tool_choice(self.model)
            payload["tool_choice"] = {"type": "any" if forced else "auto"}

        try:
            data, latency = await self._post_json(f"{self.base_url}/v1/messages", self._headers, payload)
        except ProviderError as e:
            if tools:
                raise explain_tool_error(self, e) from e
            raise

        blocks = data.get("content") or []
        text = "".join(b.get("text", "") for b in blocks if b.get("type") == "text")
        calls = [tool_call(b.get("name"), b.get("input")) for b in blocks if b.get("type") == "tool_use"]
        usage = data.get("usage") or {}
        # Cached-prompt tokens are reported separately from input_tokens; count them as input.
        cache_read = to_int(usage.get("cache_read_input_tokens"))
        cache_write = to_int(usage.get("cache_creation_input_tokens"))
        return Completion(
            text=text.strip(),
            input_tokens=to_int(usage.get("input_tokens")) + cache_read + cache_write,
            output_tokens=to_int(usage.get("output_tokens")),
            cost_usd=None,
            latency_s=latency,
            finish_reason=data.get("stop_reason"),
            cache_read_tokens=cache_read,
            cache_write_tokens=cache_write,
            tool_calls=calls or None,
        )

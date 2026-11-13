"""Anthropic Messages API provider (raw httpx, no SDK)."""

from __future__ import annotations

from typing import Any

from brass_lantern.providers.base import Completion, HTTPProvider, merge_consecutive

ANTHROPIC_URL = "https://api.anthropic.com"
API_VERSION = "2023-06-01"


class AnthropicProvider(HTTPProvider):
    name = "anthropic"

    def __init__(self, model: str, *, api_key: str, base_url: str = ANTHROPIC_URL, **kwargs: Any):
        super().__init__(model, base_url=base_url, **kwargs)
        self._headers = {
            "x-api-key": api_key,
            "anthropic-version": API_VERSION,
            "content-type": "application/json",
        }

    async def complete(self, system: str, messages: list[dict[str, str]]) -> Completion:
        payload: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "system": system,  # Anthropic takes the system prompt outside `messages`
            "messages": merge_consecutive(messages),
        }
        if self.temperature is not None:
            payload["temperature"] = self.temperature

        data, latency = await self._post_json(f"{self.base_url}/v1/messages", self._headers, payload)

        text = "".join(
            b.get("text", "") for b in data.get("content") or [] if b.get("type") == "text"
        )
        usage = data.get("usage") or {}
        # Cached-prompt tokens are reported separately; count them as input.
        input_tokens = (
            int(usage.get("input_tokens") or 0)
            + int(usage.get("cache_read_input_tokens") or 0)
            + int(usage.get("cache_creation_input_tokens") or 0)
        )
        return Completion(
            text=text.strip(),
            input_tokens=input_tokens,
            output_tokens=int(usage.get("output_tokens") or 0),
            cost_usd=None,
            latency_s=latency,
            finish_reason=data.get("stop_reason"),
        )

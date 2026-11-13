"""Google Gemini (generateContent) provider (raw httpx, no SDK)."""

from __future__ import annotations

from typing import Any

from brass_lantern.providers.base import Completion, HTTPProvider, merge_consecutive

GEMINI_URL = "https://generativelanguage.googleapis.com"


class GeminiProvider(HTTPProvider):
    name = "gemini"

    def __init__(self, model: str, *, api_key: str, base_url: str = GEMINI_URL, **kwargs: Any):
        super().__init__(model.removeprefix("models/"), base_url=base_url, **kwargs)
        # Key goes in a header, never the URL query string (keeps it out of logs).
        self._headers = {"x-goog-api-key": api_key, "content-type": "application/json"}

    async def complete(self, system: str, messages: list[dict[str, str]]) -> Completion:
        generation_config: dict[str, Any] = {"maxOutputTokens": self.max_tokens}
        if self.temperature is not None:
            generation_config["temperature"] = self.temperature
        payload = {
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": [
                {"role": "model" if m["role"] == "assistant" else "user", "parts": [{"text": m["content"]}]}
                for m in merge_consecutive(messages)
            ],
            "generationConfig": generation_config,
        }
        url = f"{self.base_url}/v1beta/models/{self.model}:generateContent"
        data, latency = await self._post_json(url, self._headers, payload)

        candidate = (data.get("candidates") or [{}])[0]
        parts = (candidate.get("content") or {}).get("parts") or []
        text = "".join(p.get("text", "") for p in parts if not p.get("thought"))
        finish = candidate.get("finishReason")
        if not candidate:
            block = (data.get("promptFeedback") or {}).get("blockReason")
            finish = f"BLOCKED:{block}" if block else "NO_CANDIDATES"
        usage = data.get("usageMetadata") or {}
        return Completion(
            text=text.strip(),
            input_tokens=int(usage.get("promptTokenCount") or 0),
            # Thinking tokens are billed as output.
            output_tokens=int(usage.get("candidatesTokenCount") or 0) + int(usage.get("thoughtsTokenCount") or 0),
            cost_usd=None,
            latency_s=latency,
            finish_reason=finish,
        )

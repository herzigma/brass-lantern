"""OpenAI-compatible chat-completions provider.

One class covers OpenAI itself, OpenRouter, Ollama, and anything else that speaks
``POST {base_url}/chat/completions`` (vLLM, LM Studio, Together, ...).
"""

from __future__ import annotations

from typing import Any

from brass_lantern.providers.base import Completion, HTTPProvider, ProviderError

OPENAI_URL = "https://api.openai.com/v1"
OPENROUTER_URL = "https://openrouter.ai/api/v1"
OLLAMA_URL = "http://localhost:11434/v1"


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):  # content-part arrays
        return "".join(p.get("text", "") for p in content if isinstance(p, dict))
    return ""


class OpenAICompatProvider(HTTPProvider):
    def __init__(
        self,
        model: str,
        *,
        base_url: str,
        api_key: str | None = None,
        name: str = "openai",
        token_param: str = "max_tokens",
        extra_headers: dict[str, str] | None = None,
        extra_body: dict[str, Any] | None = None,
        **kwargs: Any,
    ):
        self.name = name
        super().__init__(model, base_url=base_url, **kwargs)
        self._headers = {"Content-Type": "application/json", **(extra_headers or {})}
        if api_key:
            self._headers["Authorization"] = f"Bearer {api_key}"
        self._token_param = token_param
        self._extra_body = extra_body or {}

    def _body_error(self, data: dict[str, Any]) -> tuple[str, bool] | None:
        # Some gateways (OpenRouter) report upstream failures as HTTP 200 + {"error": {...}}.
        if data.get("choices"):
            return None
        err = data.get("error")
        if not err:
            return "response had no choices", False
        if isinstance(err, dict):
            code = err.get("code")
            retryable = isinstance(code, int) and (code in (408, 429) or code >= 500)
            return f"{err.get('message', err)} (code {code})", retryable
        return str(err), False

    async def complete(self, system: str, messages: list[dict[str, str]]) -> Completion:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, *messages],
            self._token_param: self.max_tokens,
            **self._extra_body,
        }
        if self.temperature is not None:
            payload["temperature"] = self.temperature

        data, latency = await self._post_json(f"{self.base_url}/chat/completions", self._headers, payload)

        choice = data["choices"][0]
        msg = choice.get("message") or {}
        text = _content_text(msg.get("content"))
        if not text.strip():
            # Reasoning models can spend the whole budget thinking and leave content empty.
            text = _content_text(msg.get("reasoning") or msg.get("reasoning_content"))
        usage = data.get("usage") or {}
        cost = usage.get("cost")
        if cost is not None and not isinstance(cost, (int, float)):
            raise ProviderError(f"{self.name}: unexpected usage.cost value {cost!r}")
        return Completion(
            text=text.strip(),
            input_tokens=int(usage.get("prompt_tokens") or 0),
            output_tokens=int(usage.get("completion_tokens") or 0),
            cost_usd=float(cost) if cost is not None else None,
            latency_s=latency,
            finish_reason=choice.get("finish_reason"),
        )

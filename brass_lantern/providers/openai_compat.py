"""OpenAI-compatible chat-completions provider.

One class covers OpenAI itself, OpenRouter, Ollama, and anything else that speaks
``POST {base_url}/chat/completions`` (vLLM, LM Studio, Together, ...).
"""

from __future__ import annotations

from typing import Any

from brass_lantern.providers.base import (
    CACHE_CONTROL,
    Completion,
    HTTPProvider,
    ProviderError,
    explain_tool_error,
    to_int,
    tool_call,
    with_cache_breakpoint,
)
from brass_lantern.providers.models import anthropic_rejects_forced_tool_choice

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

    def _uses_cache_control(self) -> bool:
        """OpenRouter forwards content-part ``cache_control`` to Anthropic models only.

        Every other model family caches repeated prefixes by itself, so nothing is sent for them.
        (Gemini Flash Lite's implicit cache hit 1 of 44 turns via OpenRouter on ~1-4k-token prompts;
        explicit breakpoints made no difference at that size, so none are sent.)
        """
        return self.cache and self.name == "openrouter" and self.model.lower().startswith("anthropic/")

    async def complete(
        self, system: str, messages: list[dict[str, str]], tools: list[dict[str, Any]] | None = None
    ) -> Completion:
        chat: list[dict[str, Any]] = [{"role": "system", "content": system}, *messages]
        if self._uses_cache_control():
            chat[0] = {
                "role": "system",
                "content": [{"type": "text", "text": system, "cache_control": dict(CACHE_CONTROL)}],
            }
            last_user = max((i for i, m in enumerate(chat) if m["role"] == "user"), default=None)
            if last_user is not None:
                chat[last_user] = with_cache_breakpoint(chat[last_user])
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": chat,
            self._token_param: self.max_tokens,
            **self._extra_body,
        }
        if self.temperature is not None:
            payload["temperature"] = self.temperature
        if tools:
            payload["tools"] = [
                {
                    "type": "function",
                    "function": {"name": t["name"], "description": t.get("description", ""), "parameters": t["parameters"]},
                }
                for t in tools
            ]
            # Anthropic models reject a forced tool choice while reasoning is on, and some reject it
            # outright (OpenRouter passes it through).
            anthropic = self.model.lower().startswith("anthropic/")
            no_force = anthropic and ("reasoning" in payload or anthropic_rejects_forced_tool_choice(self.model))
            payload["tool_choice"] = "auto" if no_force else "required"

        try:
            data, latency = await self._post_json(f"{self.base_url}/chat/completions", self._headers, payload)
        except ProviderError as e:
            if tools:
                raise explain_tool_error(self, e) from e
            raise

        choice = data["choices"][0]
        msg = choice.get("message") or {}
        text = _content_text(msg.get("content"))
        if not text.strip():
            # Reasoning models can spend the whole budget thinking and leave content empty.
            text = _content_text(msg.get("reasoning") or msg.get("reasoning_content"))
        calls = [
            tool_call(fn.get("name"), fn.get("arguments"))
            for tc in msg.get("tool_calls") or []
            if isinstance(tc, dict) and isinstance(fn := tc.get("function"), dict)
        ]
        usage = data.get("usage") or {}
        cost = usage.get("cost")
        if cost is not None and not isinstance(cost, (int, float)):
            raise ProviderError(f"{self.name}: unexpected usage.cost value {cost!r}")
        # OpenAI-style cache accounting: prompt_tokens already includes the cached part. OpenRouter
        # adds a cache-write counter (its exact name is parsed defensively).
        details = usage.get("prompt_tokens_details") or {}
        return Completion(
            text=text.strip(),
            input_tokens=to_int(usage.get("prompt_tokens")),
            output_tokens=to_int(usage.get("completion_tokens")),
            cost_usd=float(cost) if cost is not None else None,
            latency_s=latency,
            finish_reason=choice.get("finish_reason"),
            cache_read_tokens=to_int(details.get("cached_tokens")),
            cache_write_tokens=to_int(
                details.get("cache_write_tokens")
                or details.get("cache_creation_input_tokens")
                or usage.get("cache_creation_input_tokens")
            ),
            tool_calls=calls or None,
        )

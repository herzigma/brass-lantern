"""Provider interface, shared HTTP/retry plumbing, and the Completion record."""

from __future__ import annotations

import asyncio
import json
import re
import sys
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import httpx


class ProviderError(Exception):
    """An LLM call failed (after retries, where retrying made sense).

    ``partial`` carries the :class:`Completion` of an earlier, successful call in the same
    agent move (e.g. the first reply when its format-repair follow-up failed), so the
    runner can still bill it.
    """

    partial: "Completion | None" = None
    status: int | None = None  # HTTP status of a non-retryable failure, when there was one
    body: str | None = None  # full response text of that failure


class ConfigError(Exception):
    """Bad model spec, or a required API key / setting is missing."""


@dataclass
class Completion:
    """One model reply plus its accounting."""

    text: str
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float | None = None  # None = provider did not report a dollar cost
    latency_s: float = 0.0  # duration of the successful HTTP attempt
    finish_reason: str | None = None
    # Prompt tokens served from / written to the provider's prompt cache. Both are already
    # included in input_tokens (which is always the total prompt size).
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    # Native tool/function calls, provider-neutral: [{"name": str, "arguments": dict}], else None.
    tool_calls: list[dict[str, Any]] | None = None


class Provider(ABC):
    """Turns (system prompt, chat messages) into a :class:`Completion`.

    ``messages`` are ``{"role": "user"|"assistant", "content": str}`` dicts,
    starting with a user message. Each provider maps them to its own wire format.

    ``tools`` is an optional provider-neutral list like
    ``[{"name": "act", "description": "...", "parameters": <JSON schema>}]``; providers that
    support native tool calling force a call and return it in ``Completion.tool_calls``.
    ``cache`` turns on provider-side prompt caching where the provider needs a request hint.
    """

    name = "provider"

    def __init__(
        self,
        model: str,
        *,
        temperature: float | None = 0.7,
        max_tokens: int = 1024,
        cache: bool = True,
        effort: str | None = None,
    ):
        self.model = model
        self.temperature = temperature  # None = not sent
        self.max_tokens = max_tokens
        self.cache = cache
        self.effort = effort  # None = not sent (provider default)
        self.temperature_dropped = False  # True once the API rejected temperature and it was removed

    @abstractmethod
    async def complete(
        self, system: str, messages: list[dict[str, str]], tools: list[dict[str, Any]] | None = None
    ) -> Completion: ...

    async def aclose(self) -> None:  # pragma: no cover - default is a no-op
        return None

    def describe(self) -> dict[str, Any]:
        """Settings recorded in run.json."""
        return {
            "provider": self.name,
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "effort": self.effort,
        }


class HTTPProvider(Provider):
    """Provider that POSTs JSON to a URL, with exponential-backoff retries.

    Retries on timeouts / connection errors, HTTP 408, 429 and 5xx (honouring
    ``Retry-After``), up to ``max_retries`` retries; other 4xx fail immediately.
    """

    max_retries = 5
    backoff_base = 1.0
    backoff_cap = 30.0

    def __init__(
        self,
        model: str,
        *,
        base_url: str,
        temperature: float | None = 0.7,
        max_tokens: int = 1024,
        cache: bool = True,
        effort: str | None = None,
        timeout: float | httpx.Timeout = 120.0,
        transport: httpx.AsyncBaseTransport | None = None,
        sleep=asyncio.sleep,
    ):
        super().__init__(model, temperature=temperature, max_tokens=max_tokens, cache=cache, effort=effort)
        self.base_url = base_url.rstrip("/")
        self._http = httpx.AsyncClient(timeout=timeout, transport=transport)
        self._sleep = sleep

    async def aclose(self) -> None:
        await self._http.aclose()

    def describe(self) -> dict[str, Any]:
        return {**super().describe(), "base_url": self.base_url}

    def _body_error(self, data: dict[str, Any]) -> tuple[str, bool] | None:
        """Detect an error payload delivered with HTTP 200. Returns (message, retryable)."""
        return None

    async def _post_json(
        self, url: str, headers: dict[str, str], payload: dict[str, Any]
    ) -> tuple[dict[str, Any], float]:
        """POST with retries. Returns (parsed body, latency of the successful attempt).

        Safety net for models that reject a temperature: if the API answers a non-retryable 400 that
        mentions ``temperature``, drop it for this provider instance, warn once on stderr and repeat
        the call once without it.
        """
        try:
            return await self._post_retrying(url, headers, payload)
        except ProviderError as e:
            if not (e.status == 400 and _has_temperature(payload) and "temperature" in (e.body or "").lower()):
                raise
            self.temperature, self.temperature_dropped = None, True
            print(
                f"warning: {self.name}: {self.model} rejected temperature (HTTP 400); "
                "retrying without it and omitting it from now on.",
                file=sys.stderr,
            )
            return await self._post_retrying(url, headers, _without_temperature(payload))

    async def _post_retrying(
        self, url: str, headers: dict[str, str], payload: dict[str, Any]
    ) -> tuple[dict[str, Any], float]:
        for attempt in range(self.max_retries + 1):
            started = time.monotonic()
            retry_after: float | None = None
            try:
                resp = await self._http.post(url, headers=headers, json=payload)
            except httpx.RequestError as e:
                problem, retryable = f"{type(e).__name__}: {e}", True
            else:
                latency = time.monotonic() - started
                if resp.status_code == 200:
                    try:
                        data = resp.json()
                    except ValueError:
                        problem, retryable = "HTTP 200 with a non-JSON body", True
                    else:
                        err = self._body_error(data) if isinstance(data, dict) else None
                        if err is None and isinstance(data, dict):
                            return data, latency
                        problem, retryable = err if err else ("unexpected response body", False)
                else:
                    problem = f"HTTP {resp.status_code}: {(resp.text or '').strip()[:300]}"
                    retryable = resp.status_code in (408, 429) or resp.status_code >= 500
                    try:
                        retry_after = float(resp.headers.get("retry-after", ""))
                    except ValueError:
                        retry_after = None
            if not retryable:
                err = ProviderError(f"{self.name}: {problem}")
                err.status = resp.status_code  # (connection errors are always retryable, so resp exists here)
                err.body = resp.text or ""
                raise err
            if attempt == self.max_retries:
                raise ProviderError(f"{self.name}: {problem} (gave up after {attempt + 1} attempts)")
            delay = min(self.backoff_cap, self.backoff_base * 2**attempt)
            if retry_after is not None:
                delay = max(delay, min(retry_after, 60.0))
            await self._sleep(delay)
        raise AssertionError("unreachable")


def _has_temperature(payload: dict[str, Any]) -> bool:
    config = payload.get("generationConfig")  # Gemini nests it
    return "temperature" in payload or (isinstance(config, dict) and "temperature" in config)


def _without_temperature(payload: dict[str, Any]) -> dict[str, Any]:
    out = {k: v for k, v in payload.items() if k != "temperature"}
    config = out.get("generationConfig")
    if isinstance(config, dict):
        out["generationConfig"] = {k: v for k, v in config.items() if k != "temperature"}
    return out


def merge_consecutive(messages: list[dict[str, str]]) -> list[dict[str, str]]:
    """Merge adjacent same-role messages (Anthropic/Gemini require alternation)."""
    merged: list[dict[str, str]] = []
    for m in messages:
        if merged and merged[-1]["role"] == m["role"]:
            merged[-1] = {"role": m["role"], "content": merged[-1]["content"] + "\n\n" + m["content"]}
        else:
            merged.append(dict(m))
    return merged


# ---------------------------------------------------------------------------
# Prompt-cache and tool-calling helpers shared by the providers
# ---------------------------------------------------------------------------

CACHE_CONTROL = {"type": "ephemeral"}


def to_int(value: Any) -> int:
    """A usage counter as an int; missing, null or junk values count as 0."""
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def with_cache_breakpoint(message: dict[str, Any]) -> dict[str, Any]:
    """Copy of a chat message whose last content block carries ``cache_control``.

    String content becomes a one-element block list; a block list is copied with the marker added
    to its last block. The input message is left untouched.
    """
    content = message["content"]
    if isinstance(content, str):
        blocks: list[dict[str, Any]] = [{"type": "text", "text": content}]
    else:
        blocks = [dict(b) for b in content]
    if blocks:
        blocks[-1] = {**blocks[-1], "cache_control": dict(CACHE_CONTROL)}
    return {**message, "content": blocks}


_STRING_PAIR = re.compile(r'"([A-Za-z_][\w-]*)"\s*:\s*"((?:[^"\\]|\\.)*)"')


def parse_tool_arguments(raw: Any) -> dict[str, Any]:
    """Tool-call arguments as a dict, whatever the model sent.

    Accepts an already decoded dict or a JSON string. Broken JSON (truncated, trailing text, ...)
    is salvaged by picking out its ``"key": "string"`` pairs; if nothing can be recovered the
    result is ``{"_raw": <the original text>}`` so callers can see what arrived.
    """
    if isinstance(raw, dict):
        return raw
    if raw is None or raw == "":
        return {}
    if isinstance(raw, str):
        try:
            data = json.loads(raw)
        except ValueError:
            salvaged: dict[str, Any] = {}
            for key, val in _STRING_PAIR.findall(raw):
                try:
                    salvaged[key] = json.loads(f'"{val}"')
                except ValueError:
                    salvaged[key] = val
            return salvaged or {"_raw": raw}
        return data if isinstance(data, dict) else {"_raw": raw}
    return {"_raw": str(raw)}


def tool_call(name: Any, arguments: Any) -> dict[str, Any]:
    return {"name": str(name or ""), "arguments": parse_tool_arguments(arguments)}


_TOOLS_UNSUPPORTED = re.compile(
    r"no endpoints found that support tool|support(?:s|ed)? tool|tool use|tool[_ ]choice|tool calling|"
    r"function[ _]call|tools? (?:is|are)? ?not supported",
    re.IGNORECASE,
)


def explain_tool_error(provider: Provider, err: ProviderError) -> ProviderError:
    """Turn "this model has no tool support" failures into a clear, non-retryable error.

    OpenRouter answers 404 "No endpoints found that support tool use" for such models; other
    servers use 400/422 wording. Anything else (timeouts, 5xx after retries) passes through.
    """
    msg = str(err)
    if "gave up after" in msg or not _TOOLS_UNSUPPORTED.search(msg):
        return err
    return ProviderError(
        f"{provider.name}: model {provider.model!r} does not support tool calling "
        f"({msg[:200]}). Use --format text for this model."
    )

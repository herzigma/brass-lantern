"""Provider interface, shared HTTP/retry plumbing, and the Completion record."""

from __future__ import annotations

import asyncio
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import httpx


class ProviderError(Exception):
    """An LLM call failed (after retries, where retrying made sense)."""


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


class Provider(ABC):
    """Turns (system prompt, chat messages) into a :class:`Completion`.

    ``messages`` are ``{"role": "user"|"assistant", "content": str}`` dicts,
    starting with a user message. Each provider maps them to its own wire format.
    """

    name = "provider"

    def __init__(self, model: str, *, temperature: float | None = 0.7, max_tokens: int = 1024):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    @abstractmethod
    async def complete(self, system: str, messages: list[dict[str, str]]) -> Completion: ...

    async def aclose(self) -> None:  # pragma: no cover - default is a no-op
        return None

    def describe(self) -> dict[str, Any]:
        """Settings recorded in run.json."""
        return {
            "provider": self.name,
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
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
        timeout: float = 120.0,
        transport: httpx.AsyncBaseTransport | None = None,
        sleep=asyncio.sleep,
    ):
        super().__init__(model, temperature=temperature, max_tokens=max_tokens)
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
        """POST with retries. Returns (parsed body, latency of the successful attempt)."""
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
                raise ProviderError(f"{self.name}: {problem}")
            if attempt == self.max_retries:
                raise ProviderError(f"{self.name}: {problem} (gave up after {attempt + 1} attempts)")
            delay = min(self.backoff_cap, self.backoff_base * 2**attempt)
            if retry_after is not None:
                delay = max(delay, min(retry_after, 60.0))
            await self._sleep(delay)
        raise AssertionError("unreachable")


def merge_consecutive(messages: list[dict[str, str]]) -> list[dict[str, str]]:
    """Merge adjacent same-role messages (Anthropic/Gemini require alternation)."""
    merged: list[dict[str, str]] = []
    for m in messages:
        if merged and merged[-1]["role"] == m["role"]:
            merged[-1] = {"role": m["role"], "content": merged[-1]["content"] + "\n\n" + m["content"]}
        else:
            merged.append(dict(m))
    return merged

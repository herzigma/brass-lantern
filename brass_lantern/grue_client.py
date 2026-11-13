"""Async client for the Grue REST API (contract: Grue's docs/PROTOCOL.md)."""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass, field, fields
from typing import Any

import httpx

DEFAULT_GRUE_URL = "http://127.0.0.1:8081"
# Drop idle pooled connections after 2s, below Grue's server keep-alive timeout, so we do not
# reuse a connection the server is about to close.
KEEPALIVE_EXPIRY_S = 2.0
IDEMPOTENT_METHODS = ("GET", "DELETE")


class GrueError(Exception):
    """A failed call to the Grue server (HTTP error, bad payload, or unreachable)."""

    def __init__(self, message: str, *, status: int | None = None, code: str | None = None):
        super().__init__(message)
        self.status = status
        self.code = code

    @property
    def is_game_over(self) -> bool:
        return self.code == "game_over" or self.status == 409

    @property
    def is_too_many_sessions(self) -> bool:
        return self.code == "too_many_sessions" or self.status == 429


def _build(cls, data: dict[str, Any]):
    """Create dataclass *cls* from *data*, ignoring unknown keys and keeping the raw dict."""
    names = {f.name for f in fields(cls)} - {"raw"}
    obj = cls(**{k: v for k, v in data.items() if k in names})
    obj.raw = data
    return obj


@dataclass
class Game:
    id: str = ""
    title: str = ""
    author: str | None = None
    format: str | None = None
    filename: str | None = None
    max_score: int | None = None
    difficulty: str | None = None
    suggested_turn_limit: int | None = None
    tags: list[str] = field(default_factory=list)
    description: str | None = None
    license: str | None = None
    private: bool = False  # held-out game: scores public, transcripts never served
    available: bool = True
    raw: dict[str, Any] = field(default_factory=dict, repr=False)


@dataclass
class Observation:
    """The server's report of game state after start or after a command."""

    session_id: str = ""
    game_id: str = ""
    turn: int = 0
    text: str = ""
    status_line: str | None = None
    location: str | None = None
    score: int | None = None
    max_score: int | None = None
    moves: int | None = None
    input_type: str | None = "line"  # "line" | "char" | None (game over)
    game_over: bool = False
    ending: str | None = None
    notice: str | None = None
    raw: dict[str, Any] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        self.text = self.text or ""
        self.turn = self.turn or 0


@dataclass
class Result:
    """Final record of a finished session."""

    session_id: str = ""
    game_id: str = ""
    started_at: str | None = None
    ended_at: str | None = None
    turn: int = 0
    max_turns: int | None = None
    score: int | None = None
    max_score: int | None = None
    game_over: bool = True
    ending: str | None = None
    duration_ms: int | None = None
    invalid_commands: int = 0
    unique_locations: int = 0
    max_score_reached: int | None = None
    transcript: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict, repr=False)


class GrueClient:
    """Thin async wrapper over Grue's REST API. Use as ``async with GrueClient() as g``."""

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        *,
        timeout: float = 60.0,
        transport: httpx.AsyncBaseTransport | None = None,
        sleep=asyncio.sleep,
    ):
        self.base_url = (base_url or os.getenv("GRUE_URL") or DEFAULT_GRUE_URL).rstrip("/")
        api_key = api_key if api_key is not None else os.getenv("GRUE_API_KEY")
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._http = httpx.AsyncClient(
            base_url=self.base_url, headers=headers, timeout=timeout, transport=transport,
            limits=httpx.Limits(keepalive_expiry=KEEPALIVE_EXPIRY_S),
        )
        self._sleep = sleep

    async def __aenter__(self) -> "GrueClient":
        return self

    async def __aexit__(self, *exc) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        await self._http.aclose()

    # -- low level ---------------------------------------------------------

    def _unreachable(self, e: httpx.RequestError) -> GrueError:
        return GrueError(
            f"Cannot reach Grue at {self.base_url} ({type(e).__name__}: {e}). "
            "Is the server running? Set GRUE_URL or --grue-url if it is elsewhere."
        )

    async def _request(self, method: str, path: str, json: Any = None) -> dict[str, Any]:
        idempotent = method in IDEMPOTENT_METHODS
        attempt = 0
        while True:
            try:
                resp = await self._http.request(method, path, json=json)
                break
            except (httpx.RemoteProtocolError, httpx.ConnectError) as e:
                # A stale keep-alive connection (or a connect blip) fails before any response arrives.
                # Safe to repeat once for idempotent requests; never for commands (they may have run).
                if idempotent and attempt == 0:
                    attempt += 1
                    continue
                if not idempotent and isinstance(e, httpx.RemoteProtocolError):
                    raise GrueError(
                        f"Grue closed the connection during {method} {path} ({type(e).__name__}: {e}). "
                        "The request was not retried because it may already have been processed."
                    ) from e
                raise self._unreachable(e) from e
            except httpx.RequestError as e:
                raise self._unreachable(e) from e
        try:
            body = resp.json()
        except ValueError:
            body = None
        if resp.status_code >= 400:
            code = message = None
            if isinstance(body, dict):
                code, message = body.get("error"), body.get("message")
            message = message or (resp.text or "").strip()[:300] or "no details"
            label = f"{code}: " if code else ""
            raise GrueError(
                f"Grue {method} {path} failed: {label}{message} (HTTP {resp.status_code})",
                status=resp.status_code,
                code=code,
            )
        if not isinstance(body, dict):
            raise GrueError(f"Grue {method} {path} returned a non-JSON-object body")
        return body

    # -- API ---------------------------------------------------------------

    async def health(self) -> dict[str, Any]:
        return await self._request("GET", "/health")

    async def list_games(self) -> list[Game]:
        body = await self._request("GET", "/api/games")
        return [_build(Game, g) for g in body.get("games", [])]

    async def start_session(
        self, game: str, max_turns: int | None = None, meta: dict[str, Any] | None = None
    ) -> Observation:
        payload: dict[str, Any] = {"game": game}
        if max_turns is not None:
            payload["max_turns"] = max_turns
        if meta:
            payload["meta"] = meta
        for attempt in range(6):  # concurrent benches can briefly hit the session cap
            try:
                return _build(Observation, await self._request("POST", "/api/sessions", payload))
            except GrueError as e:
                if not e.is_too_many_sessions or attempt == 5:
                    raise
                await self._sleep(2.0 * (attempt + 1))
        raise AssertionError("unreachable")

    async def command(self, session_id: str, text: str) -> Observation:
        body = await self._request("POST", f"/api/sessions/{session_id}/commands", {"command": text})
        return _build(Observation, body)

    async def get_session(self, session_id: str) -> tuple[dict[str, Any], Observation]:
        body = await self._request("GET", f"/api/sessions/{session_id}")
        return body.get("session", {}), _build(Observation, body.get("observation", {}))

    async def end_session(self, session_id: str) -> Result:
        return _build(Result, await self._request("DELETE", f"/api/sessions/{session_id}"))

    async def get_result(self, session_id: str) -> Result:
        return _build(Result, await self._request("GET", f"/api/results/{session_id}"))

    async def report(self, session_id: str, client_report: dict[str, Any]) -> dict[str, Any]:
        """Attach the client's run stats to a finished session (once). Never retried: it is a POST."""
        return await self._request("POST", f"/api/sessions/{session_id}/report", {"client_report": client_report})

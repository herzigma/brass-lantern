"""A tiny in-memory Grue server (httpx.MockTransport) implementing PROTOCOL.md.

The game: you start in a Field. `take lamp` is harmless, `north` goes to a Cave,
`open chest` (in the Cave) scores 5, `take gem` (after opening) scores 5 more and wins.
Anything else is an invalid command.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone

import httpx

WIN_SCRIPT = ["take lamp", "north", "open chest", "take gem"]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class FakeGrue:
    def __init__(self, api_key: str | None = None, session_cap: int | None = None):
        self.api_key = api_key
        self.session_cap = session_cap
        self.sessions: dict[str, dict] = {}
        self.results: dict[str, dict] = {}
        self.requests: list[tuple[str, str, dict | None]] = []
        self.seen_auth: list[str | None] = []
        self.deleted: list[str] = []
        self.stall_after_turn: int | None = None  # once reached, commands are "not sent" and turn stops advancing
        self.report_endpoint = True  # False: act like an older server that has no /report route (404 not_found)
        self.read_only: set[str] = set()  # session ids that behave like featured results (report -> 409 read_only)
        self.reports: dict[str, dict] = {}  # session id -> accepted client_report
        self._n = 0

    # -- plumbing ----------------------------------------------------------

    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self._handle)

    @staticmethod
    def _json(status: int, body: dict) -> httpx.Response:
        return httpx.Response(status, json=body)

    def _err(self, status: int, code: str, message: str = "") -> httpx.Response:
        return self._json(status, {"error": code, "message": message or code})

    def _handle(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        body = json.loads(request.content) if request.content else None
        self.requests.append((request.method, path, body))
        self.seen_auth.append(request.headers.get("authorization"))
        if path.startswith("/api/") and self.api_key and request.headers.get("authorization") != f"Bearer {self.api_key}":
            return self._err(401, "unauthorized")

        if request.method == "GET" and path == "/health":
            return self._json(200, {"status": "ok", "version": "2.0.0", "active_sessions": len(self.sessions)})
        if request.method == "GET" and path == "/api/games":
            return self._json(200, {"games": [
                {"id": "tiny", "title": "Tiny Cave", "format": "zcode", "max_score": 10, "difficulty": "easy", "available": True},
                {"id": "tiny2", "title": "Tiny Cave II", "format": "zcode", "max_score": 10, "difficulty": "easy", "available": True},
                {"id": "missing", "title": "Not Downloaded", "format": "glulx", "max_score": None, "available": False},
            ]})
        if request.method == "POST" and path == "/api/sessions":
            return self._start(body or {})
        m = re.fullmatch(r"/api/sessions/([^/]+)/commands", path)
        if m and request.method == "POST":
            return self._command(m.group(1), (body or {}).get("command", ""))
        m = re.fullmatch(r"/api/sessions/([^/]+)/report", path)
        if m and request.method == "POST" and self.report_endpoint:
            return self._report(m.group(1), body)
        m = re.fullmatch(r"/api/sessions/([^/]+)", path)
        if m and request.method == "GET":
            s = self.sessions.get(m.group(1))
            if not s:
                return self._err(404, "unknown_session")
            return self._json(200, {"session": self._summary(s), "observation": self._obs(s, "")})
        if m and request.method == "DELETE":
            s = self.sessions.get(m.group(1))
            if not s:
                return self._err(404, "unknown_session")
            self.deleted.append(s["id"])
            if not s["game_over"]:
                self._finish(s, "aborted")
            return self._json(200, self.results[s["id"]])
        m = re.fullmatch(r"/api/results/([^/]+)", path)
        if m and request.method == "GET":
            r = self.results.get(m.group(1))
            return self._json(200, r) if r else self._err(404, "unknown_session")
        return self._err(404, "not_found")

    # -- run stats report (contract: POST /api/sessions/{id}/report) --------

    def _report(self, sid: str, body: dict | None) -> httpx.Response:
        if sid in self.sessions and sid not in self.results:
            return self._err(409, "session_active")
        result = self.results.get(sid)
        if result is None:
            return self._err(404, "unknown_session")
        if sid in self.read_only:
            return self._err(409, "read_only")
        if "client_report" in result:
            return self._err(409, "already_reported")
        report = (body or {}).get("client_report")
        ok = isinstance(report, dict) and len(json.dumps(report).encode()) <= 4096 and all(
            v is None or isinstance(v, (bool, int, float)) or (isinstance(v, str) and len(v) <= 200)
            for v in report.values()
        )
        if not ok:
            return self._err(400, "bad_request", "client_report must be a flat object of numbers, short strings, bools or null")
        result["client_report"], result["client_report_at"] = report, _now()
        self.reports[sid] = report
        return self._json(200, result)

    # -- game logic --------------------------------------------------------

    def _start(self, body: dict) -> httpx.Response:
        if body.get("game") not in ("tiny", "tiny2"):
            return self._err(404, "unknown_game")
        if self.session_cap is not None and sum(not s["game_over"] for s in self.sessions.values()) >= self.session_cap:
            return self._err(429, "too_many_sessions")
        self._n += 1
        sid = f"sess-{self._n}"
        self.sessions[sid] = {
            "id": sid, "game": body["game"], "turn": 0, "score": 0, "room": "Field", "chest_open": False,
            "max_turns": body.get("max_turns") or 1000, "meta": body.get("meta") or {}, "game_over": False,
            "ending": None, "invalid": 0, "locations": {"Field"}, "peak": 0, "started_at": _now(), "last": "",
        }
        s = self.sessions[sid]
        s["last"] = "Field\nYou are standing in a field. A cave lies to the north. A lamp is here."
        return self._json(201, self._obs(s, s["last"]))

    def force_game_over(self, sid: str, ending: str = "died") -> None:
        """End a session behind the client's back (next command gets 409)."""
        self._finish(self.sessions[sid], ending)

    def _finish(self, s: dict, ending: str) -> None:
        s["game_over"], s["ending"] = True, ending
        self.results[s["id"]] = {
            **self._summary(s), "ended_at": _now(), "duration_ms": 5, "invalid_commands": s["invalid"],
            "unique_locations": len(s["locations"]), "max_score_reached": s["peak"], "transcript": f"replays/{s['id']}.jsonl",
        }

    def _summary(self, s: dict) -> dict:
        return {
            "session_id": s["id"], "game_id": s["game"], "started_at": s["started_at"], "turn": s["turn"],
            "max_turns": s["max_turns"], "score": s["score"], "max_score": 10, "game_over": s["game_over"],
            "ending": s["ending"], "meta": s["meta"],
        }

    def _obs(self, s: dict, text: str) -> dict:
        return {
            "session_id": s["id"], "game_id": s["game"], "turn": s["turn"], "text": text,
            "status_line": f"{s['room']}   Score: {s['score']}", "location": s["room"], "score": s["score"],
            "max_score": 10, "moves": s["turn"], "input_type": None if s["game_over"] else "line",
            "game_over": s["game_over"], "ending": s["ending"], "notice": None,
        }

    def _command(self, sid: str, command: str) -> httpx.Response:
        s = self.sessions.get(sid)
        if not s:
            return self._err(404, "unknown_session")
        if s["game_over"]:
            return self._err(409, "game_over")
        if self.stall_after_turn is not None and s["turn"] >= self.stall_after_turn:
            busy = self._obs(s, "Late output from the previous command.")
            busy["notice"] = "Previous command is still running; this command was not sent."
            return self._json(200, busy)
        s["turn"] += 1
        cmd = command.strip().lower()
        text = "I don't understand that."
        valid = True
        if cmd == "take lamp":
            text = "Taken."
        elif cmd in ("north", "n", "go north"):
            s["room"] = "Cave"
            s["locations"].add("Cave")
            text = "Cave\nA dark cave with a chest."
        elif cmd == "open chest" and s["room"] == "Cave" and not s["chest_open"]:
            s["chest_open"] = True
            s["score"] += 5
            text = "The chest opens, revealing a gem.\n[Your score has just gone up by 5 points.]"
        elif cmd == "take gem" and s["chest_open"]:
            s["score"] += 5
            text = "Taken.\n*** You have won ***"
        else:
            valid = False
            s["invalid"] += 1
        s["peak"] = max(s["peak"], s["score"])
        if cmd == "take gem" and valid:
            self._finish(s, "won")
        elif s["turn"] >= s["max_turns"]:
            self._finish(s, "turn_limit")
        return self._json(200, self._obs(s, text))

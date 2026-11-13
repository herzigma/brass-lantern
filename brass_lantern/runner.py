"""Play one episode: Grue session + agent loop + run.json / transcript.jsonl."""

from __future__ import annotations

import asyncio
import json
import re
import secrets
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from brass_lantern import __version__
from brass_lantern.agent import Agent
from brass_lantern.console import Reporter
from brass_lantern.grue_client import GrueClient, GrueError, Observation, Result
from brass_lantern.providers import Provider, ProviderError


@dataclass
class RunConfig:
    game: str
    model: str  # the provider:model spec, recorded in run.json
    max_turns: int = 150
    max_cost: float | None = None  # USD
    max_time: float | None = None  # seconds
    history: int = 30
    notes: bool = False
    temperature: float | None = 0.7
    max_tokens: int = 1024
    base_url: str | None = None  # provider endpoint override
    grue_url: str | None = None
    out_dir: Path = Path("runs")
    run_id: str | None = None
    quiet: bool = False
    meta: dict[str, Any] = field(default_factory=dict)  # extra session meta
    price: tuple[float, float] | None = None  # (input, output) USD per 1M tokens, for providers without cost


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def make_run_id(game: str, model: str) -> str:
    if model.startswith("mock:"):  # a script path: keep just the file name
        model = "mock-" + Path(model[5:]).stem
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", model).strip("-")[:40]
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"{stamp}-{game}-{slug}-{secrets.token_hex(2)}"


def _obs_dict(obs: Observation) -> dict[str, Any]:
    return {
        "text": obs.text,
        "status_line": obs.status_line,
        "location": obs.location,
        "score": obs.score,
        "max_score": obs.max_score,
        "moves": obs.moves,
        "input_type": obs.input_type,
        "game_over": obs.game_over,
        "ending": obs.ending,
        "notice": obs.notice,
    }


class _Totals:
    def __init__(self) -> None:
        self.turns = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.cost = 0.0
        self.cost_known = False
        self.model_time = 0.0
        self.parse_failures = 0


async def play_episode(
    cfg: RunConfig,
    *,
    grue: GrueClient,
    provider: Provider,
    reporter: Reporter | None = None,
) -> dict[str, Any]:
    """Play one game and return the run.json dict (also written to disk).

    Never raises for game/model/server problems: they end the run with status "error"
    and are recorded. Cancellation (Ctrl-C) ends the session cleanly with status
    "interrupted"; the cancellation is swallowed so the caller can still report.
    """
    reporter = reporter or Reporter(quiet=cfg.quiet)
    run_id = cfg.run_id or make_run_id(cfg.game, cfg.model)
    run_dir = Path(cfg.out_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    meta = {
        "agent": f"brass-lantern/{__version__}",
        "model": cfg.model,
        "run_id": run_id,
        "temperature": cfg.temperature,
        **cfg.meta,
    }
    started_at = now_iso()
    t0 = time.monotonic()
    totals = _Totals()
    obs: Observation | None = None
    session_id: str | None = None
    result: Result | None = None
    stop_reason = "game_over"
    status = "completed"
    error: str | None = None
    agent: Agent | None = None

    def build_summary(final: bool) -> dict[str, Any]:
        score = result.score if result and result.score is not None else (obs.score if obs else None)
        max_score = (
            (result.max_score if result else None) or (obs.max_score if obs else None)
        )
        return {
            "run_id": run_id,
            "version": __version__,
            "status": status if final else "running",
            "stop_reason": stop_reason if final else None,
            "error": error,
            "model": cfg.model,
            "game": cfg.game,
            "session_id": session_id,
            "started_at": started_at,
            "ended_at": now_iso() if final else None,
            "config": {
                "max_turns": cfg.max_turns,
                "max_cost": cfg.max_cost,
                "max_time": cfg.max_time,
                "history": cfg.history,
                "notes": cfg.notes,
                "temperature": cfg.temperature,
                "max_tokens": cfg.max_tokens,
                "provider": provider.describe(),
                "meta": meta,
            },
            "totals": {
                "turns": totals.turns,
                "score": score,
                "max_score": max_score,
                "max_score_reached": result.max_score_reached if result else None,
                "ending": (result.ending if result else None) or (obs.ending if obs else None),
                "input_tokens": totals.input_tokens,
                "output_tokens": totals.output_tokens,
                "cost_usd": round(totals.cost, 6) if totals.cost_known else None,
                "wall_time_s": round(time.monotonic() - t0, 2),
                "model_time_s": round(totals.model_time, 2),
                "parse_failures": totals.parse_failures,
                "invalid_commands": result.invalid_commands if result else None,
                "unique_locations": result.unique_locations if result else None,
            },
            "result": result.raw if result else None,
        }

    def write_run_json(final: bool) -> dict[str, Any]:
        summary = build_summary(final)
        (run_dir / "run.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return summary

    write_run_json(final=False)
    reporter.start(run_id, cfg.game, cfg.model, cfg.max_turns)

    with (run_dir / "transcript.jsonl").open("w", encoding="utf-8") as transcript:

        def log(entry: dict[str, Any]) -> None:
            transcript.write(json.dumps(entry, ensure_ascii=False) + "\n")
            transcript.flush()

        try:
            obs = await grue.start_session(cfg.game, cfg.max_turns, meta)
            session_id = obs.session_id
            agent = Agent(provider, max_turns=cfg.max_turns, history=cfg.history, notes=cfg.notes)
            reporter.observation(obs.text)

            while not obs.game_over:
                if obs.turn >= cfg.max_turns:
                    stop_reason = "max_turns"
                    break
                if cfg.max_cost is not None and totals.cost >= cfg.max_cost:
                    stop_reason = "max_cost"
                    break
                if cfg.max_time is not None and time.monotonic() - t0 >= cfg.max_time:
                    stop_reason = "max_time"
                    break

                try:
                    decision = await agent.act(obs)
                except ProviderError as e:
                    stop_reason, status, error = "provider_error", "error", str(e)
                    break

                comp = decision.completion
                cost = comp.cost_usd
                if cost is None and cfg.price:
                    cost = (comp.input_tokens * cfg.price[0] + comp.output_tokens * cfg.price[1]) / 1e6
                totals.turns += 1
                totals.input_tokens += comp.input_tokens
                totals.output_tokens += comp.output_tokens
                totals.model_time += comp.latency_s
                if cost is not None:
                    totals.cost += cost
                    totals.cost_known = True
                if not decision.parsed.ok:
                    totals.parse_failures += 1

                log(
                    {
                        "turn": obs.turn,
                        "ts": now_iso(),
                        "elapsed_s": round(time.monotonic() - t0, 2),
                        "observation": _obs_dict(obs),
                        "reply": decision.reply,
                        "command": decision.command,
                        "parse_method": decision.parsed.method,
                        "parse_ok": decision.parsed.ok,
                        "notes": agent.notes if cfg.notes else None,
                        "input_tokens": comp.input_tokens,
                        "output_tokens": comp.output_tokens,
                        "cost_usd": cost,
                        "latency_s": round(comp.latency_s, 3),
                        "finish_reason": comp.finish_reason,
                    }
                )
                warning = None
                if not decision.reply.strip():
                    warning = f"empty model reply (finish_reason={comp.finish_reason}); try a larger --max-tokens"
                reporter.move(
                    obs.turn, decision.command, obs.score, obs.max_score,
                    totals.cost if totals.cost_known else None, comp.latency_s, warning,
                )

                try:
                    obs = await grue.command(session_id, decision.command)
                except GrueError as e:
                    if not e.is_game_over:
                        raise
                    _, obs = await grue.get_session(session_id)  # fetch the final observation
                    if not obs.game_over:
                        obs.game_over = True
                reporter.observation(obs.text)

            if obs.game_over:
                log({"turn": obs.turn, "ts": now_iso(), "final": True, "observation": _obs_dict(obs)})

        except (asyncio.CancelledError, KeyboardInterrupt):
            stop_reason, status = "interrupted", "interrupted"
        except GrueError as e:
            stop_reason, status, error = "grue_error", "error", str(e)
        except Exception as e:  # noqa: BLE001 - one bad run must not take down a bench
            stop_reason, status, error = "error", "error", f"{type(e).__name__}: {e}"

        # Close out the session: fetch the stored Result (or abort a live session).
        if session_id:
            try:
                if obs is not None and obs.game_over:
                    try:
                        result = await grue.get_result(session_id)
                    except GrueError:
                        result = await grue.end_session(session_id)
                else:
                    result = await grue.end_session(session_id)
            except asyncio.CancelledError:  # second Ctrl-C during cleanup: give up quietly
                pass
            except Exception as e:  # noqa: BLE001
                error = error or f"could not fetch final result: {e}"

    summary = write_run_json(final=True)
    reporter.end(summary)
    return summary

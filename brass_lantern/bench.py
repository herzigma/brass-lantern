"""Run a models x games x repeats matrix, then aggregate it into a leaderboard."""

from __future__ import annotations

import asyncio
import csv
import json
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from brass_lantern.console import Reporter, fmt_cost, fmt_duration
from brass_lantern.grue_client import GrueClient
from brass_lantern.providers import Provider, create_provider
from brass_lantern.runner import RunConfig, make_run_id, play_episode

# Runs that did not finish for infrastructure reasons say nothing about the model.
EXCLUDED_STATUSES = ("error", "interrupted", "running")


@dataclass
class BenchConfig:
    models: list[str]
    games: list[str]
    runs: int = 3
    max_turns: int = 150
    concurrency: int = 4
    max_cost_per_run: float | None = None
    max_time_per_run: float | None = None
    history: int = 30
    notes: bool = False
    temperature: float | None = 0.7
    max_tokens: int = 1024
    base_url: str | None = None
    grue_url: str | None = None
    out_dir: Path = Path("runs")
    prices: dict[str, tuple[float, float]] = field(default_factory=dict)  # model spec -> USD per 1M tokens


ProviderFactory = Callable[..., Provider]


async def run_bench(
    cfg: BenchConfig,
    *,
    grue: GrueClient,
    provider_factory: ProviderFactory = create_provider,
    log: Callable[[str], None] = print,
) -> tuple[list[dict[str, Any]], bool]:
    """Run the whole matrix. Returns (run summaries, interrupted)."""
    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    def make_provider(spec: str) -> Provider:
        return provider_factory(
            spec, base_url=cfg.base_url, temperature=cfg.temperature, max_tokens=cfg.max_tokens
        )

    for spec in cfg.models:  # fail fast on bad specs / missing keys, before spending anything
        probe = make_provider(spec)
        await probe.aclose()

    # Interleave repeats so a partial bench still covers every cell evenly.
    jobs = [(i, game, spec) for i in range(cfg.runs) for game in cfg.games for spec in cfg.models]
    sem = asyncio.Semaphore(max(1, cfg.concurrency))
    summaries: list[dict[str, Any]] = []
    started = time.monotonic()

    async def worker(i: int, game: str, spec: str) -> None:
        async with sem:
            run_cfg = RunConfig(
                game=game,
                model=spec,
                max_turns=cfg.max_turns,
                max_cost=cfg.max_cost_per_run,
                max_time=cfg.max_time_per_run,
                history=cfg.history,
                notes=cfg.notes,
                temperature=cfg.temperature,
                max_tokens=cfg.max_tokens,
                base_url=cfg.base_url,
                grue_url=cfg.grue_url,
                out_dir=out,
                run_id=make_run_id(game, spec),
                quiet=True,
                meta={"bench": out.name, "repeat": i + 1},
                price=cfg.prices.get(spec),
            )
            provider = make_provider(spec)
            try:
                summary = await play_episode(run_cfg, grue=grue, provider=provider, reporter=Reporter(quiet=True))
            finally:
                await provider.aclose()
            summaries.append(summary)
            t = summary["totals"]
            score = t["score"] if t["score"] is not None else "?"
            max_s = f"/{t['max_score']}" if t["max_score"] else ""
            log(
                f"[{len(summaries)}/{len(jobs)}] {spec} | {game} #{i + 1}: score {score}{max_s} "
                f"ending={t['ending']} turns={t['turns']} cost={fmt_cost(t['cost_usd'])} "
                f"status={summary['status']} | {fmt_duration(time.monotonic() - started)} elapsed"
            )

    interrupted = False
    try:
        await asyncio.gather(*(worker(*j) for j in jobs))
    except asyncio.CancelledError:
        interrupted = True  # in-flight runs ended their sessions cleanly; report what finished
    return summaries, interrupted


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def load_runs(directory: str | Path) -> list[dict[str, Any]]:
    runs = []
    for path in sorted(Path(directory).rglob("run.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if isinstance(data, dict) and "totals" in data and "model" in data:
            runs.append(data)
    return runs


def _flat(run: dict[str, Any]) -> dict[str, Any]:
    t = run["totals"]
    score, max_score = t.get("score"), t.get("max_score")
    return {
        "run_id": run.get("run_id"),
        "model": run["model"],
        "game": run.get("game"),
        "status": run.get("status"),
        "stop_reason": run.get("stop_reason"),
        "ending": t.get("ending"),
        "score": score,
        "max_score": max_score,
        "max_score_reached": t.get("max_score_reached"),
        "norm": (score / max_score) if score is not None and max_score else None,
        "turns": t.get("turns") or 0,
        "invalid_commands": t.get("invalid_commands") or 0,
        "parse_failures": t.get("parse_failures") or 0,
        "input_tokens": t.get("input_tokens") or 0,
        "output_tokens": t.get("output_tokens") or 0,
        "cost_usd": t.get("cost_usd"),
        "wall_time_s": t.get("wall_time_s"),
        "temperature": (run.get("config") or {}).get("temperature"),
    }


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    turns = sum(r["turns"] for r in rows)
    scores = [r["score"] for r in rows if r["score"] is not None]
    norms = [r["norm"] for r in rows if r["norm"] is not None]
    return {
        "runs": len(rows),
        "mean_score": _mean(scores),
        "best_score": max(scores) if scores else None,
        "max_score": next((r["max_score"] for r in rows if r["max_score"]), None),
        "mean_norm": _mean(norms),
        "win_rate": sum(r["ending"] == "won" for r in rows) / len(rows) if rows else None,
        "mean_turns": _mean([r["turns"] for r in rows]),
        "invalid_rate": sum(r["invalid_commands"] for r in rows) / turns if turns else None,
        "parse_failure_rate": sum(r["parse_failures"] for r in rows) / turns if turns else None,
    }


def _cost(rows: list[dict[str, Any]]) -> float | None:
    known = [r["cost_usd"] for r in rows if r["cost_usd"] is not None]
    return sum(known) if known else None


def aggregate(runs: list[dict[str, Any]]) -> dict[str, Any]:
    flat = [_flat(r) for r in runs]
    games = sorted({r["game"] for r in flat})
    models: dict[str, Any] = {}
    for model in sorted({r["model"] for r in flat}):
        mine = [r for r in flat if r["model"] == model]
        valid = [r for r in mine if r["status"] not in EXCLUDED_STATUSES]
        per_game = {g: _stats([r for r in valid if r["game"] == g]) for g in games if any(r["game"] == g for r in valid)}
        for g, s in per_game.items():
            s["cost_usd"] = _cost([r for r in mine if r["game"] == g])
        overall = _stats(valid)
        # Games weigh equally: average the per-game mean normalized scores.
        overall["norm_score"] = _mean([s["mean_norm"] for s in per_game.values() if s["mean_norm"] is not None])
        overall["cost_usd"] = _cost(mine)  # money spent counts even on errored runs
        overall["input_tokens"] = sum(r["input_tokens"] for r in mine)
        overall["output_tokens"] = sum(r["output_tokens"] for r in mine)
        overall["excluded_runs"] = len(mine) - len(valid)
        models[model] = {"overall": overall, "games": per_game}

    ranking = sorted(
        models,
        key=lambda m: (
            -(models[m]["overall"]["norm_score"] if models[m]["overall"]["norm_score"] is not None else -1),
            -(models[m]["overall"]["win_rate"] or 0),
            models[m]["overall"]["cost_usd"] or 0,
        ),
    )
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n_runs": len(runs),
        "games": games,
        "ranking": ranking,
        "models": models,
        "rows": flat,
    }


def _pct(x: float | None, digits: int = 1) -> str:
    return "n/a" if x is None else f"{x * 100:.{digits}f}%"


def _num(x: float | None, digits: int = 1) -> str:
    return "n/a" if x is None else f"{x:.{digits}f}"


def render_leaderboard(agg: dict[str, Any]) -> str:
    models, games = agg["models"], agg["games"]
    lines = [
        "# Brass Lantern Leaderboard",
        "",
        f"{agg['n_runs']} runs, {len(models)} models, {len(games)} games. Generated {agg['generated_at']}.",
        "",
        "Score is the final in-game score as a share of the game's maximum. Overall = mean of per-game "
        "means (games weigh equally). Win = game ended `won`. Invalid % = parser rejections per command "
        "(server-reported). Format fail % = replies without a `> COMMAND` line. Runs that errored or were "
        "interrupted are excluded from scores but their cost is counted.",
        "",
        "## Overall ranking",
        "",
        "| # | Model | Score | Win rate | Mean turns | Invalid % | Format fail % | Cost | Runs (excl.) |",
        "|--:|---|--:|--:|--:|--:|--:|--:|--:|",
    ]
    for rank, m in enumerate(agg["ranking"], 1):
        o = models[m]["overall"]
        lines.append(
            f"| {rank} | `{m}` | {_pct(o['norm_score'])} | {_pct(o['win_rate'], 0)} | {_num(o['mean_turns'], 0)} | "
            f"{_pct(o['invalid_rate'])} | {_pct(o['parse_failure_rate'])} | {fmt_cost(o['cost_usd'])} | "
            f"{o['runs']} ({o['excluded_runs']}) |"
        )
    if games:
        lines += ["", "## Mean score by game", "", "| Model | " + " | ".join(games) + " |", "|---|" + "--:|" * len(games)]
        for m in agg["ranking"]:
            cells = []
            for g in games:
                s = models[m]["games"].get(g)
                if not s:
                    cells.append("-")
                elif s["max_score"]:
                    cells.append(f"{_num(s['mean_score'])} ({_pct(s['mean_norm'], 0)})")
                else:
                    cells.append(_num(s["mean_score"]))
            lines.append(f"| `{m}` | " + " | ".join(cells) + " |")
    for g in games:
        max_score = next((s["max_score"] for m in models.values() for gg, s in m["games"].items() if gg == g and s["max_score"]), None)
        lines += [
            "",
            f"## {g}" + (f" (max score {max_score})" if max_score else ""),
            "",
            "| Model | Mean score | % of max | Best | Win rate | Mean turns | Invalid % | Cost | Runs |",
            "|---|--:|--:|--:|--:|--:|--:|--:|--:|",
        ]
        for m in agg["ranking"]:
            s = models[m]["games"].get(g)
            if not s:
                continue
            lines.append(
                f"| `{m}` | {_num(s['mean_score'])} | {_pct(s['mean_norm'])} | {s['best_score']} | "
                f"{_pct(s['win_rate'], 0)} | {_num(s['mean_turns'], 0)} | {_pct(s['invalid_rate'])} | "
                f"{fmt_cost(s['cost_usd'])} | {s['runs']} |"
            )
    return "\n".join(lines) + "\n"


CSV_FIELDS = [
    "run_id", "model", "game", "status", "stop_reason", "ending", "score", "max_score", "max_score_reached",
    "norm", "turns", "invalid_commands", "parse_failures", "input_tokens", "output_tokens", "cost_usd",
    "wall_time_s", "temperature",
]


def write_reports(directory: str | Path, runs: list[dict[str, Any]]) -> dict[str, Any]:
    """Write leaderboard.md, results.csv and summary.json into *directory*."""
    d = Path(directory)
    d.mkdir(parents=True, exist_ok=True)
    agg = aggregate(runs)
    (d / "leaderboard.md").write_text(render_leaderboard(agg), encoding="utf-8")
    with (d / "results.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(agg["rows"])
    (d / "summary.json").write_text(json.dumps(agg, indent=2), encoding="utf-8")
    return agg

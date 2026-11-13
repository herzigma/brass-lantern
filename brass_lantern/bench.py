"""Run a models x games x repeats matrix, then aggregate it into a leaderboard."""

from __future__ import annotations

import asyncio
import csv
import dataclasses
import json
import os
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from brass_lantern import __version__
from brass_lantern.console import Reporter, fmt_cost, fmt_duration
from brass_lantern.grue_client import GrueClient
from brass_lantern.providers import ConfigError, Provider, create_provider
from brass_lantern.runner import RunConfig, make_run_id, play_episode

# Runs that did not finish for infrastructure reasons say nothing about the model
# (a provider_error that played at least one turn is the exception; see _counted).
EXCLUDED_STATUSES = ("error", "interrupted", "running")


@dataclass
class BenchConfig:
    models: list[str]
    games: list[str]
    runs: int = 3
    max_turns: int | None = None
    concurrency: int = 4
    max_cost_per_run: float | None = None
    max_time_per_run: float | None = None
    history: int = 30
    notes: bool = False
    format: str = "text"  # "text" (> COMMAND line) or "tools" (native act tool call)
    cache: bool = True  # prompt caching: stepped history window + provider cache hints
    temperature: float | None = 0.7
    max_tokens: int = 1024
    reasoning: str | None = None
    base_url: str | None = None
    grue_url: str | None = None
    out_dir: Path = Path("runs")
    prices: dict[str, tuple[float, float]] = field(default_factory=dict)  # model spec -> USD per 1M tokens


ProviderFactory = Callable[..., Provider]
Job = tuple[int, str, str]  # (repeat index from 0, game, model spec)


async def run_bench(
    cfg: BenchConfig,
    *,
    grue: GrueClient,
    provider_factory: ProviderFactory = create_provider,
    log: Callable[[str], None] = print,
    jobs: list[Job] | None = None,
) -> tuple[list[dict[str, Any]], bool]:
    """Run the whole matrix. Returns (run summaries, interrupted).

    ``jobs`` is for resuming: run just these (repeat, game, model) jobs and leave bench.json to the
    caller. Without it, run every job and write bench.json at the start.
    """
    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    def make_provider(spec: str) -> Provider:
        return provider_factory(
            spec, base_url=cfg.base_url, temperature=cfg.temperature, max_tokens=cfg.max_tokens,
            reasoning=cfg.reasoning, cache=cfg.cache,
        )

    fresh = jobs is None
    if jobs is None:
        # Interleave repeats so a partial bench still covers every cell evenly.
        jobs = [(i, game, spec) for i in range(cfg.runs) for game in cfg.games for spec in cfg.models]

    for spec in (cfg.models if fresh else dict.fromkeys(s for _, _, s in jobs)):
        probe = make_provider(spec)  # fail fast on bad specs / missing keys, before spending anything
        await probe.aclose()

    if fresh:
        write_bench_json(out, new_bench_json(cfg))

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
                format=cfg.format,
                cache=cfg.cache,
                temperature=cfg.temperature,
                max_tokens=cfg.max_tokens,
                reasoning=cfg.reasoning,
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
# bench.json and resume
# ---------------------------------------------------------------------------

BENCH_JSON = "bench.json"
BENCH_JSON_VERSION = 1  # bump if the format changes incompatibly


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def bench_fields(cfg: BenchConfig) -> dict[str, Any]:
    """Every BenchConfig field as plain JSON, except out_dir (a bench lives wherever bench.json is)."""
    data = dataclasses.asdict(cfg)
    data.pop("out_dir", None)
    return json.loads(json.dumps(data, default=str))  # tuples become lists, Paths become strings


def new_bench_json(cfg: BenchConfig) -> dict[str, Any]:
    return {"version": BENCH_JSON_VERSION, "created_at": now_utc(), "brass_lantern_version": __version__,
            **bench_fields(cfg)}


def write_bench_json(directory: str | Path, data: dict[str, Any]) -> None:
    d = Path(directory)
    d.mkdir(parents=True, exist_ok=True)
    tmp = d / (BENCH_JSON + ".tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    os.replace(tmp, d / BENCH_JSON)  # a crash mid-write never leaves a half-written bench.json


def read_bench_json(directory: str | Path) -> dict[str, Any]:
    d = Path(directory)
    if not d.is_dir():
        raise ConfigError(f"Cannot resume: {d} is not a directory.")
    path = d / BENCH_JSON
    if not path.is_file():
        raise ConfigError(
            f"Cannot resume {d}: it has no {BENCH_JSON} (benches started before resume support do not). "
            "Start a new bench with --models and --games instead."
        )
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise ConfigError(f"Cannot read {path}: {e}") from e
    if not isinstance(data, dict):
        raise ConfigError(f"Cannot read {path}: expected a JSON object.")
    version = data.get("version")
    if not isinstance(version, int) or isinstance(version, bool) or version > BENCH_JSON_VERSION:
        raise ConfigError(f"{path} has format version {version!r}; this brass-lantern reads version {BENCH_JSON_VERSION}.")
    return data


def config_from_bench_json(data: dict[str, Any], directory: str | Path) -> BenchConfig:
    """Rebuild a BenchConfig: unknown keys are ignored, missing ones take the dataclass defaults."""
    known = {f.name for f in dataclasses.fields(BenchConfig)} - {"out_dir"}
    kwargs = {k: v for k, v in data.items() if k in known}
    absent = [k for k in ("models", "games") if not kwargs.get(k)]
    if absent:
        raise ConfigError(f"{Path(directory) / BENCH_JSON} has no {' or '.join(absent)}; cannot resume.")
    try:
        kwargs["prices"] = {m: (float(p[0]), float(p[1])) for m, p in (kwargs.get("prices") or {}).items()}
    except (TypeError, ValueError, IndexError, AttributeError) as e:
        raise ConfigError(f"{Path(directory) / BENCH_JSON} has bad prices: {e}") from e
    return BenchConfig(out_dir=Path(directory), **kwargs)


def plan_resume(cfg: BenchConfig, runs: list[dict[str, Any]]) -> tuple[list[Job], int]:
    """Work out what a resumed bench still has to run. Returns (jobs left, total jobs in the matrix).

    A run fills a (game, model) slot if it counts towards scores (see _counted): completed runs, and
    provider_error runs that played a turn. Error, interrupted and still-running runs are left for a
    rerun. A run goes in the slot named by its meta.repeat; runs without one (not written by run_bench)
    fill the lowest free slots of their cell. Excluded run dirs stay on disk; aggregation ignores them.
    """
    cells = {(g, m) for g in cfg.games for m in cfg.models}
    taken: dict[tuple[str, str], set[int]] = defaultdict(set)
    unlabelled: dict[tuple[str, str], int] = defaultdict(int)
    for run in runs:
        cell = (run.get("game"), run.get("model"))
        if cell not in cells or not _counted(run, run["totals"].get("turns") or 0)[0]:
            continue
        meta = (run.get("config") or {}).get("meta")
        repeat = meta.get("repeat") if isinstance(meta, dict) else None
        if isinstance(repeat, int) and not isinstance(repeat, bool) and repeat >= 1:
            taken[cell].add(repeat - 1)
        else:
            unlabelled[cell] += 1
    missing: set[Job] = set()
    for (game, spec) in cells:
        free = [i for i in range(cfg.runs) if i not in taken[(game, spec)]]
        missing.update((i, game, spec) for i in free[unlabelled[(game, spec)]:])
    # Same order as a fresh bench: repeats interleaved so a partial bench still covers every cell.
    jobs = [(i, g, m) for i in range(cfg.runs) for g in cfg.games for m in cfg.models if (i, g, m) in missing]
    return jobs, len(cells) * cfg.runs


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


# Settings that change how a model plays; runs of one spec with different values are separate rows.
CONFIG_KNOBS = ("notes", "temperature", "history", "reasoning", "max_tokens", "max_turns", "cache", "format")


def _counted(run: dict[str, Any], turns: int) -> tuple[bool, str | None]:
    """Does this run count towards scores? Returns (counted, reason it was excluded)."""
    status, stop = run.get("status"), run.get("stop_reason")
    if status not in EXCLUDED_STATUSES:
        return True, None
    if stop == "provider_error":  # the model/provider failed: a real outcome once it has played
        return (True, None) if turns >= 1 else (False, "provider_error (no turns played)")
    return False, stop or status  # harness problems: grue_error, interrupted, error, still running


def _flat(run: dict[str, Any]) -> dict[str, Any]:
    t = run["totals"]
    cfg = run.get("config") or {}
    score, max_score = t.get("score"), t.get("max_score")
    turns = t.get("turns") or 0
    counted, reason = _counted(run, turns)
    if counted and score is None and max_score and max_score > 0:
        score = 0  # finished without ever reporting a score: that is a 0, not a missing value
    return {
        "run_id": run.get("run_id"),
        "model": run["model"],
        "variant": run["model"],  # set by aggregate(): model spec, plus a config suffix when configs differ
        "game": run.get("game"),
        "status": run.get("status"),
        "stop_reason": run.get("stop_reason"),
        "counted": counted,
        "excluded_reason": reason,
        "ending": t.get("ending"),
        "score": score,
        "max_score": max_score,
        "max_score_reached": t.get("max_score_reached"),
        "norm": (score / max_score) if score is not None and max_score else None,
        "turns": turns,
        "invalid_commands": t.get("invalid_commands") or 0,
        "parse_failures": t.get("parse_failures") or 0,
        "input_tokens": t.get("input_tokens") or 0,
        "output_tokens": t.get("output_tokens") or 0,
        "cost_usd": t.get("cost_usd"),
        "wall_time_s": t.get("wall_time_s"),
        "temperature": cfg.get("temperature"),
        "history": cfg.get("history"),
        "notes": bool(cfg.get("notes")),
        "reasoning": cfg.get("reasoning"),
        "max_tokens": cfg.get("max_tokens"),
        "max_turns": cfg.get("max_turns"),
        # Runs from before these options existed used a sliding window and the text format.
        "cache": bool(cfg.get("cache")),
        "history_step": cfg.get("history_step") or 0,
        "format": cfg.get("format") or "text",
        "cache_read_tokens": t.get("cache_read_tokens") or 0,
        "cache_write_tokens": t.get("cache_write_tokens") or 0,
    }


def _knob_text(name: str, value: Any) -> str:
    if name == "notes":
        return "notes" if value else "no notes"
    if name == "cache":
        return "cache" if value else "no cache"
    short = {"temperature": "t", "history": "h", "max_turns": "turns"}.get(name, name)
    if value is None:
        value = {"temperature": "none", "reasoning": "default"}.get(name, "?")
    return f"{short}={value}"


def _variants(flat: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Group runs by model spec + behaviour-changing config. Returns label -> {spec, config}.

    One config per spec keeps the plain spec as its label; several configs get a suffix naming
    only the knobs that differ, e.g. ``openrouter:x (notes, h=30)``. Sets each row's ``variant``.
    """
    out: dict[str, dict[str, Any]] = {}
    for spec in sorted({r["model"] for r in flat}):
        mine = [r for r in flat if r["model"] == spec]
        configs = sorted({tuple(r[k] for k in CONFIG_KNOBS) for r in mine}, key=repr)
        differing = [k for i, k in enumerate(CONFIG_KNOBS) if len({c[i] for c in configs}) > 1]
        for cfg in configs:
            values = dict(zip(CONFIG_KNOBS, cfg))
            label = spec if len(configs) == 1 else f"{spec} ({', '.join(_knob_text(k, values[k]) for k in differing)})"
            out[label] = {"spec": spec, "config": values}
            for r in mine:
                if tuple(r[k] for k in CONFIG_KNOBS) == cfg:
                    r["variant"] = label
    return out


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    turns = sum(r["turns"] for r in rows)
    scores = [r["score"] for r in rows if r["score"] is not None]
    norms = [r["norm"] for r in rows if r["norm"] is not None]
    return {
        "runs": len(rows),
        "provider_errors": sum(r["stop_reason"] == "provider_error" for r in rows),
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
    variants = _variants(flat)
    # Games with a known max score are the ones that can be normalized and ranked on.
    ranked_games = [g for g in games if any(r["game"] == g and r["max_score"] for r in flat)]
    models: dict[str, Any] = {}
    for label, info in variants.items():
        mine = [r for r in flat if r["variant"] == label]
        valid = [r for r in mine if r["counted"]]
        per_game = {g: _stats([r for r in valid if r["game"] == g]) for g in games if any(r["game"] == g for r in valid)}
        for g, s in per_game.items():
            s["cost_usd"] = _cost([r for r in mine if r["game"] == g])
        overall = _stats(valid)
        # Games weigh equally, and a ranked game with no scored run counts as 0 (never skipped), so a
        # model cannot climb the ranking by failing on the hard games.
        norms = {g: (per_game[g]["mean_norm"] if g in per_game else None) for g in ranked_games}
        overall["missing_games"] = [g for g, n in norms.items() if n is None]
        overall["norm_score"] = _mean([n if n is not None else 0.0 for n in norms.values()])
        overall["cost_usd"] = _cost(mine)  # money spent counts even on excluded runs
        overall["input_tokens"] = sum(r["input_tokens"] for r in mine)
        overall["output_tokens"] = sum(r["output_tokens"] for r in mine)
        overall["excluded_runs"] = len(mine) - len(valid)
        models[label] = {"spec": info["spec"], "config": info["config"], "overall": overall, "games": per_game}

    ranking = sorted(
        models,
        key=lambda m: (
            -(models[m]["overall"]["norm_score"] if models[m]["overall"]["norm_score"] is not None else -1),
            -(models[m]["overall"]["win_rate"] or 0),
            models[m]["overall"]["cost_usd"] or 0,
        ),
    )
    excluded = [
        {"run_id": r["run_id"], "model": r["variant"], "game": r["game"], "reason": r["excluded_reason"]}
        for r in flat if not r["counted"]
    ]
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n_runs": len(runs),
        "games": games,
        "ranked_games": ranked_games,
        "ranking": ranking,
        "models": models,
        "excluded": excluded,
        "rows": flat,
    }


def _pct(x: float | None, digits: int = 1) -> str:
    return "n/a" if x is None else f"{x * 100:.{digits}f}%"


def _num(x: float | None, digits: int = 1) -> str:
    return "n/a" if x is None else f"{x:.{digits}f}"


MAX_EXCLUDED_LISTED = 50


def render_leaderboard(agg: dict[str, Any]) -> str:
    models, games = agg["models"], agg["games"]
    lines = [
        "# Brass Lantern Leaderboard",
        "",
        f"{agg['n_runs']} runs, {len(models)} models, {len(games)} games. Generated {agg['generated_at']}.",
        "",
        "**Score formula.** A run's score is its final in-game score divided by the game's maximum "
        "(a finished run that never reported a score counts as 0; games without a maximum are left out of "
        "the ranking). A model's per-game score is the mean over its valid runs in that game. Its overall "
        "score is the mean over ALL games in the bench that have a maximum score: a game where the model has "
        "no valid run counts as 0 and is marked `*`, so failing on hard games cannot raise its rank.",
        "",
        "**Valid runs.** Completed runs, including those stopped by turn, time or cost limits, plus runs that "
        "ended in a `provider_error` after at least one turn (scored as reached and counted in Errors). Runs "
        "that hit a harness problem (`grue_error`, `interrupted`, `error`) are excluded and listed at the end; "
        "their cost still counts. Rows with the same model spec but different settings (temperature, history, "
        "notes, reasoning, max tokens, max turns, cache, format) are separate rows, labelled with the differing settings.",
        "",
        "Win = game ended `won`. Invalid % = parser rejections per command (server-reported). Format fail % = "
        "replies without a `> COMMAND` line. Errors = valid runs that ended in a provider error.",
        "",
        "## Overall ranking",
        "",
        "| # | Model | Score | Win rate | Mean turns | Invalid % | Format fail % | Cost | Runs (excl.) | Errors |",
        "|--:|---|--:|--:|--:|--:|--:|--:|--:|--:|",
    ]
    any_missing = False
    for rank, m in enumerate(agg["ranking"], 1):
        o = models[m]["overall"]
        star = " \\*" if o["missing_games"] else ""
        any_missing = any_missing or bool(o["missing_games"])
        lines.append(
            f"| {rank} | `{m}` | {_pct(o['norm_score'])}{star} | {_pct(o['win_rate'], 0)} | {_num(o['mean_turns'], 0)} | "
            f"{_pct(o['invalid_rate'])} | {_pct(o['parse_failure_rate'])} | {fmt_cost(o['cost_usd'])} | "
            f"{o['runs']} ({o['excluded_runs']}) | {o['provider_errors']} |"
        )
    if any_missing:
        lines += ["", "\\* counts at least one game with no valid run as 0 (see the `- (=0)` cells below)."]
    if games:
        lines += ["", "## Mean score by game", "", "| Model | " + " | ".join(games) + " |", "|---|" + "--:|" * len(games)]
        for m in agg["ranking"]:
            cells = []
            for g in games:
                s = models[m]["games"].get(g)
                if not s:
                    cells.append("- (=0)" if g in agg["ranked_games"] else "-")
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
            "| Model | Mean score | % of max | Best | Win rate | Mean turns | Invalid % | Cost | Runs | Errors |",
            "|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|",
        ]
        played = [m for m in agg["ranking"] if models[m]["games"].get(g)]
        # Best first within the game; ties keep the overall ranking order (sorted() is stable).
        played.sort(key=lambda m: -(models[m]["games"][g]["mean_score"] or 0))
        for m in played:
            s = models[m]["games"][g]
            lines.append(
                f"| `{m}` | {_num(s['mean_score'])} | {_pct(s['mean_norm'])} | {s['best_score']} | "
                f"{_pct(s['win_rate'], 0)} | {_num(s['mean_turns'], 0)} | {_pct(s['invalid_rate'])} | "
                f"{fmt_cost(s['cost_usd'])} | {s['runs']} | {s['provider_errors']} |"
            )
    excluded = agg["excluded"]
    if excluded:
        lines += ["", "## Excluded runs", "", "| Run | Model | Game | Reason |", "|---|---|---|---|"]
        for e in excluded[:MAX_EXCLUDED_LISTED]:
            lines.append(f"| {e['run_id']} | `{e['model']}` | {e['game']} | {e['reason']} |")
        if len(excluded) > MAX_EXCLUDED_LISTED:
            lines.append(f"| ... {len(excluded) - MAX_EXCLUDED_LISTED} more (see results.csv) | | | |")
    return "\n".join(lines) + "\n"


CSV_FIELDS = [
    "run_id", "model", "variant", "game", "status", "stop_reason", "counted", "excluded_reason", "ending",
    "score", "max_score", "max_score_reached", "norm", "turns", "invalid_commands", "parse_failures",
    "input_tokens", "output_tokens", "cost_usd", "wall_time_s", "temperature", "history", "notes", "reasoning",
    "max_tokens", "max_turns", "cache", "history_step", "format", "cache_read_tokens", "cache_write_tokens",
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

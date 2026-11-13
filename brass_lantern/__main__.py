"""Command-line interface: ``brass-lantern games | play | bench | report``."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Sequence

from dotenv import find_dotenv, load_dotenv

from brass_lantern import __version__
from brass_lantern.bench import (
    BenchConfig, bench_fields, config_from_bench_json, load_runs, now_utc, plan_resume, read_bench_json,
    render_leaderboard, run_bench, write_bench_json, write_reports,
)
from brass_lantern.console import setup_stdio
from brass_lantern.grue_client import GrueClient, GrueError
from brass_lantern.prices import Price, load_price_file, load_prices, price_for, prices_for_models
from brass_lantern.providers import EFFORTS, ConfigError, create_provider, resolve_settings, split_spec
from brass_lantern.runner import RunConfig, play_episode

EPILOG = """\
model specs (provider:model):
  openrouter:anthropic/claude-haiku-4.5   OPENROUTER_API_KEY; reports real dollar cost
  openai:gpt-6.1-sol                      OPENAI_API_KEY (add --base-url for vLLM, LM Studio, Together, ...)
  ollama:llama3.1:8b                      local Ollama, no key
  anthropic:claude-sonnet-5-5             ANTHROPIC_API_KEY
  gemini:gemini-2.5-flash                 GEMINI_API_KEY
  mock:script.txt                         replay commands from a file (dry runs, no API calls)
  human:                                  type the commands yourself (debugging the harness)

reasoning effort per model: append @minimal|@low|@medium|@high|@xhigh|@max to a spec, e.g.
  anthropic:claude-sonnet-5-5@high        overrides --effort for that spec (a bench treats @low and @high as two models)

environment: GRUE_URL (default http://127.0.0.1:8081), GRUE_API_KEY; values may live in a .env file.
"""


def _temperature(value: str) -> float | None:
    if value.lower() in ("none", "default", "omit"):
        return None
    try:
        return float(value)
    except ValueError:
        raise argparse.ArgumentTypeError("temperature must be a number or 'none'")


def _csv(value: str) -> list[str]:
    return [v.strip() for v in value.split(",") if v.strip()]


def _kv(value: str) -> tuple[str, str]:
    key, sep, val = value.partition("=")
    if not sep or not key:
        raise argparse.ArgumentTypeError("expected KEY=VALUE")
    return key, val


COST_REPORTING_PROVIDERS = ("openrouter", "mock", "human")  # providers whose cap works without a price table


def uncapped_models(models: Sequence[str], prices: dict[str, Price]) -> list[str]:
    """Specs for which a dollar cap cannot work: no provider-reported cost and no price (bundled or --prices)."""
    return [m for m in models if split_spec(m)[0] not in COST_REPORTING_PROVIDERS and price_for(prices, m) is None]


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--grue-url", help="Grue server URL (default: $GRUE_URL or http://127.0.0.1:8081)")
    p.add_argument("--max-turns", type=int, default=None,
                   help="turn limit per episode (default: the game's suggested limit from the catalog)")
    p.add_argument("--history", type=int, default=30, help="recent turns kept verbatim in context (default: 30)")
    p.add_argument("--notes", action="store_true", help="give the model a NOTES: scratchpad that persists across turns")
    p.add_argument("--cache", choices=["on", "off"], default="on",
                   help="prompt caching (default: on): the history window advances in steps of 10 turns so "
                        "provider caches can hit, and Anthropic models get cache_control hints; 'off' slides "
                        "the window one turn at a time, as in older runs")
    p.add_argument("--format", choices=["text", "tools"], default="text",
                   help="how the model answers (default: text, a final '> COMMAND' line); 'tools' makes it call "
                        "a native 'act' tool instead, separating playing ability from format-following")
    p.add_argument("--temperature", type=_temperature, default=0.7,
                   help="sampling temperature (default: 0.7); 'none' omits it, for models that reject it")
    p.add_argument("--max-tokens", type=int, default=None,
                   help="max output tokens per reply (default: 16000 for anthropic or when an effort is sent, "
                        "else 1024; thinking tokens count against it)")
    p.add_argument("--effort", choices=EFFORTS,
                   help="reasoning effort (anthropic, openai, openrouter; default: the provider default). "
                        "A spec @effort suffix overrides it, e.g. anthropic:claude-sonnet-5-5@high")
    p.add_argument("--reasoning", choices=EFFORTS, help="deprecated alias of --effort")
    p.add_argument("--base-url", help="override the provider's API base URL (e.g. a vLLM or LM Studio server)")
    p.add_argument("--prices", metavar="FILE",
                   help="JSON of per-model USD/1M-token prices that override or extend the bundled table "
                        "(brass_lantern/prices.json), so --max-cost works for providers that report no cost")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="brass-lantern",
        description="Brass Lantern: make any LLM play text adventures on a Grue server, and compare models.",
        epilog=EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--version", action="version", version=f"brass-lantern {__version__}")
    sub = parser.add_subparsers(dest="command", metavar="{games,play,bench,report}")
    sub.required = True

    g = sub.add_parser("games", help="list the games the Grue server offers")
    g.add_argument("--grue-url", help="Grue server URL")

    p = sub.add_parser("play", help="play one episode", epilog=EPILOG, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--game", required=True, help="game id (see `brass-lantern games`)")
    p.add_argument("--model", required=True, help="provider:model spec (see below)")
    p.add_argument("--max-cost", type=float, help="stop when spent dollars reach this (needs cost data; see --prices)")
    p.add_argument("--max-time", type=float, help="stop after this many seconds")
    p.add_argument("--out", default="runs", help="output directory (default: runs); the run goes in <out>/<run_id>/")
    p.add_argument("--run-id", help="custom run id (default: timestamp-game-model-xxxx)")
    p.add_argument("--quiet", action="store_true", help="don't print game text and commands, only the final summary")
    p.add_argument("--seed-meta", action="append", type=_kv, default=[], metavar="KEY=VALUE",
                   help="extra metadata attached to the Grue session (repeatable)")
    _add_common(p)

    b = sub.add_parser("bench", help="run a models x games x repeats matrix and write a leaderboard",
                       epilog=EPILOG, formatter_class=argparse.RawDescriptionHelpFormatter)
    b.add_argument("--models", type=_csv, help="comma-separated provider:model specs (required unless --resume)")
    b.add_argument("--games", type=_csv,
                   help="comma-separated game ids, or 'all' for every available game (required unless --resume)")
    b.add_argument("--resume", metavar="DIR",
                   help="continue an interrupted bench: read DIR/bench.json, run only the episodes that did not finish "
                        "(error, interrupted, still running), then rewrite the reports. Only --runs (raise it to add "
                        "repeats), --concurrency, --max-cost-per-run, --max-time-per-run, --grue-url and --prices can "
                        "be changed. Example: brass-lantern bench --resume runs/bench-20261112-210000")
    b.add_argument("--runs", type=int, default=None, help="repeats per model/game cell (default: 3)")
    b.add_argument("--concurrency", type=int, default=None, help="episodes run in parallel (default: 4)")
    b.add_argument("--max-cost-per-run", type=float, help="per-episode dollar cap")
    b.add_argument("--max-time-per-run", type=float, help="per-episode time cap in seconds")
    b.add_argument("--out", help="output directory (default: runs/bench-<timestamp>)")
    _add_common(b)

    r = sub.add_parser("report", help="regenerate leaderboard.md / results.csv / summary.json from run.json files")
    r.add_argument("dir", help="directory containing run.json files (searched recursively)")
    return parser


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


async def _games(args: argparse.Namespace) -> int:
    async with GrueClient(args.grue_url) as grue:
        games = await grue.list_games()
    print(f"{'ID':<22} {'FORMAT':<8} {'MAX':>5} {'DIFFICULTY':<10} {'AVAIL':<5} TITLE")
    for g in games:
        print(f"{g.id:<22} {g.format or '-':<8} {g.max_score if g.max_score is not None else '-':>5} "
              f"{g.difficulty or '-':<10} {'yes' if g.available else 'no':<5} {g.title}")
    return 0


async def _play(args: argparse.Namespace) -> int:
    prices = load_prices(args.prices)
    provider = create_provider(args.model, base_url=args.base_url, temperature=args.temperature, max_tokens=args.max_tokens,
                               effort=args.effort, cache=args.cache == "on")
    sent = resolve_settings(args.model, effort=args.effort, temperature=args.temperature, max_tokens=args.max_tokens,
                            base_url=args.base_url)  # what is really sent: this is what gets recorded
    cfg = RunConfig(
        game=args.game, model=args.model, max_turns=args.max_turns, max_cost=args.max_cost, max_time=args.max_time,
        history=args.history, notes=args.notes, format=args.format, cache=args.cache == "on",
        temperature=sent.temperature, max_tokens=sent.max_tokens, effort=sent.effort,
        base_url=args.base_url, grue_url=args.grue_url, out_dir=Path(args.out), run_id=args.run_id,
        quiet=args.quiet, meta=dict(args.seed_meta), price=price_for(prices, args.model),
    )
    if cfg.max_cost is not None and provider.name not in ("openrouter", "mock", "human") and not cfg.price:
        print(f"warning: {args.model} reports no dollar cost, so --max-cost will not stop the run "
              "(give prices with --prices).", file=sys.stderr)
    try:
        async with GrueClient(args.grue_url) as grue:
            summary = await play_episode(cfg, grue=grue, provider=provider)
    finally:
        await provider.aclose()
    print(f"run saved to {Path(args.out) / summary['run_id']}")
    return {"completed": 0, "interrupted": 130}.get(summary["status"], 1)


def _require_enforceable_cap(max_cost: float | None, models: Sequence[str], prices: dict[str, Price]) -> None:
    if max_cost is not None and (bad := uncapped_models(models, prices)):
        raise ConfigError(  # spending without an effective cap is the dangerous direction: refuse to start
            f"--max-cost-per-run cannot be enforced for {', '.join(bad)}: the provider reports no dollar cost "
            "and neither the bundled prices nor --prices has an entry for it. Add the model to a --prices file, "
            "or drop --max-cost-per-run."
        )


def _shell_path(path: Path) -> str:
    text = str(path)
    return f'"{text}"' if " " in text else text


def _finish_bench(out: Path, interrupted: bool) -> int:
    """Write the reports over every run in *out* and print the leaderboard."""
    agg = write_reports(out, load_runs(out))
    print()
    print(render_leaderboard(agg))
    print(f"wrote {out / 'leaderboard.md'}, results.csv, summary.json")
    if interrupted:
        print(f"interrupted: finished episodes are saved. To continue, run:\n"
              f"  brass-lantern bench --resume {_shell_path(out)}", file=sys.stderr)
    return 130 if interrupted else 0


async def _bench_resume(args: argparse.Namespace) -> int:
    out = Path(args.resume)
    data = read_bench_json(out)
    cfg = config_from_bench_json(data, out)
    overrides: dict[str, object] = {}
    if args.runs is not None and args.runs < cfg.runs:
        raise ConfigError(f"--runs {args.runs} is lower than the {cfg.runs} this bench was started with; "
                          "resume can only keep or raise it.")
    for name in ("runs", "concurrency", "max_cost_per_run", "max_time_per_run", "grue_url"):
        value = getattr(args, name)
        if value is not None and value != getattr(cfg, name):
            overrides[name] = value
            setattr(cfg, name, value)
    file_prices = load_price_file(args.prices) if args.prices else {}
    if file_prices:
        overrides["prices"] = sorted(file_prices)
    bundled = load_prices()
    for spec in cfg.models:  # --prices beats what bench.json saved; models it saved no price for get the bundled one
        price = price_for(file_prices, spec) or price_for(cfg.prices, spec) or price_for(bundled, spec)
        if price is not None:
            cfg.prices[spec] = price
    _require_enforceable_cap(cfg.max_cost_per_run, cfg.models, cfg.prices)

    jobs, total = plan_resume(cfg, load_runs(out))
    done = total - len(jobs)
    print(f"Resuming {out}: {done} of {total} episodes already done, "
          + (f"running {len(jobs)}." if jobs else "nothing to run."))
    data.update(bench_fields(cfg))  # keeps keys this version does not know, and created_at
    data.setdefault("resumes", []).append({"at": now_utc(), "done": done, "running": len(jobs), "overrides": overrides})
    write_bench_json(out, data)

    interrupted = False
    if jobs:
        async with GrueClient(cfg.grue_url) as grue:
            _, interrupted = await run_bench(cfg, grue=grue, jobs=jobs)
    return _finish_bench(out, interrupted)


async def _bench(args: argparse.Namespace) -> int:
    if args.resume:
        return await _bench_resume(args)
    prices = load_prices(args.prices)
    _require_enforceable_cap(args.max_cost_per_run, args.models, prices)
    out =Path(args.out) if args.out else Path("runs") / f"bench-{datetime.now():%Y%m%d-%H%M%S}"
    async with GrueClient(args.grue_url) as grue:
        games = args.games
        if games == ["all"]:
            games = [g.id for g in await grue.list_games() if g.available]
        cfg = BenchConfig(
            models=args.models, games=games, runs=args.runs, max_turns=args.max_turns, concurrency=args.concurrency,
            max_cost_per_run=args.max_cost_per_run, max_time_per_run=args.max_time_per_run, history=args.history,
            notes=args.notes, format=args.format, cache=args.cache == "on", temperature=args.temperature,
            max_tokens=args.max_tokens, base_url=args.base_url,
            effort=args.effort, grue_url=args.grue_url, out_dir=out, prices=prices_for_models(prices, args.models),
        )
        total = len(cfg.models) * len(cfg.games) * cfg.runs
        print(f"bench: {len(cfg.models)} models x {len(cfg.games)} games x {cfg.runs} runs = {total} episodes "
              f"(concurrency {cfg.concurrency}) -> {out}")
        _, interrupted = await run_bench(cfg, grue=grue)
    return _finish_bench(out, interrupted)


def _report(args: argparse.Namespace) -> int:
    runs = load_runs(args.dir)
    if not runs:
        print(f"error: no run.json files found under {args.dir}", file=sys.stderr)
        return 1
    agg = write_reports(args.dir, runs)
    print(render_leaderboard(agg))
    print(f"wrote leaderboard.md, results.csv, summary.json in {args.dir}")
    return 0


# bench options that may still be given with --resume; everything else comes from DIR/bench.json
RESUME_OVERRIDES = {"command", "resume", "runs", "concurrency", "max_cost_per_run", "max_time_per_run", "grue_url", "prices"}


def _check_bench_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    """Cross-option rules argparse cannot express. Also fills in the fresh-bench defaults for --runs/--concurrency."""
    if args.resume:
        defaults = vars(parser.parse_args(["bench", "--resume", args.resume]))
        clash = [k for k, v in vars(args).items() if k not in RESUME_OVERRIDES and v != defaults.get(k)]
        if clash:
            parser.error("--resume takes its settings from DIR/bench.json, so it cannot be combined with "
                         + ", ".join("--" + k.replace("_", "-") for k in clash)
                         + ". Only --runs, --concurrency, --max-cost-per-run, --max-time-per-run, --grue-url "
                         "and --prices can be changed on resume.")
        return
    missing = [f"--{k}" for k in ("models", "games") if not getattr(args, k)]
    if missing:
        parser.error("the following arguments are required: " + ", ".join(missing) + " (or use --resume DIR)")
    args.runs = 3 if args.runs is None else args.runs
    args.concurrency = 4 if args.concurrency is None else args.concurrency


def main(argv: Sequence[str] | None = None) -> int:
    setup_stdio()
    load_dotenv(find_dotenv(usecwd=True))
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "bench":
        _check_bench_args(parser, args)
    if getattr(args, "reasoning", None):
        print("warning: --reasoning is deprecated; use --effort.", file=sys.stderr)
        args.effort = args.effort or args.reasoning
    try:
        if args.command == "report":
            return _report(args)
        handler = {"games": _games, "play": _play, "bench": _bench}[args.command]
        return asyncio.run(handler(args))
    except (ConfigError, GrueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())

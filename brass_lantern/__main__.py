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
from brass_lantern.bench import BenchConfig, load_runs, render_leaderboard, run_bench, write_reports
from brass_lantern.console import setup_stdio
from brass_lantern.grue_client import GrueClient, GrueError
from brass_lantern.providers import ConfigError, create_provider, split_spec
from brass_lantern.runner import RunConfig, play_episode

EPILOG = """\
model specs (provider:model):
  openrouter:anthropic/claude-haiku-4.5   OPENROUTER_API_KEY; reports real dollar cost
  openai:gpt-4o-mini                      OPENAI_API_KEY (add --base-url for vLLM, LM Studio, Together, ...)
  ollama:llama3.1:8b                      local Ollama, no key
  anthropic:claude-sonnet-4-5             ANTHROPIC_API_KEY
  gemini:gemini-2.5-flash                 GEMINI_API_KEY
  mock:script.txt                         replay commands from a file (dry runs, no API calls)
  human:                                  type the commands yourself (debugging the harness)

environment: GRUE_URL (default http://localhost:8081), GRUE_API_KEY; values may live in a .env file.
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


def load_prices(path: str | None) -> dict[str, tuple[float, float]]:
    """Read ``{"provider:model": {"input": 3.0, "output": 15.0}}`` (USD per 1M tokens)."""
    if not path:
        return {}
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        out = {}
        for spec, p in raw.items():
            inp, outp = (p["input"], p["output"]) if isinstance(p, dict) else p
            out[spec] = (float(inp), float(outp))
        return out
    except (OSError, ValueError, KeyError, TypeError) as e:
        raise ConfigError(f"Cannot read prices file {path!r}: {e}") from e


COST_REPORTING_PROVIDERS = ("openrouter", "mock", "human")  # providers whose cap works without --prices


def uncapped_models(models: Sequence[str], prices: dict[str, tuple[float, float]]) -> list[str]:
    """Specs for which a dollar cap cannot work: no provider-reported cost and no --prices entry."""
    return [m for m in models if split_spec(m)[0] not in COST_REPORTING_PROVIDERS and m not in prices]


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--grue-url", help="Grue server URL (default: $GRUE_URL or http://localhost:8081)")
    p.add_argument("--max-turns", type=int, default=150, help="turn limit per episode (default: 150)")
    p.add_argument("--history", type=int, default=30, help="recent turns kept verbatim in context (default: 30)")
    p.add_argument("--notes", action="store_true", help="give the model a NOTES: scratchpad that persists across turns")
    p.add_argument("--temperature", type=_temperature, default=0.7,
                   help="sampling temperature (default: 0.7); 'none' omits it, for models that reject it")
    p.add_argument("--max-tokens", type=int, default=1024, help="max output tokens per reply (default: 1024)")
    p.add_argument("--reasoning", choices=["minimal", "low", "medium", "high"],
                   help="reasoning effort for reasoning models (openrouter, openai); default: provider default")
    p.add_argument("--base-url", help="override the provider's API base URL (e.g. a vLLM or LM Studio server)")
    p.add_argument("--prices", metavar="FILE",
                   help="JSON of per-model USD/1M-token prices, so --max-cost works for providers that report no cost")


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
    b.add_argument("--models", required=True, type=_csv, help="comma-separated provider:model specs")
    b.add_argument("--games", required=True, type=_csv, help="comma-separated game ids, or 'all' for every available game")
    b.add_argument("--runs", type=int, default=3, help="repeats per model/game cell (default: 3)")
    b.add_argument("--concurrency", type=int, default=4, help="episodes run in parallel (default: 4)")
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
                               reasoning=args.reasoning)
    cfg = RunConfig(
        game=args.game, model=args.model, max_turns=args.max_turns, max_cost=args.max_cost, max_time=args.max_time,
        history=args.history, notes=args.notes, temperature=args.temperature, max_tokens=args.max_tokens, reasoning=args.reasoning,
        base_url=args.base_url, grue_url=args.grue_url, out_dir=Path(args.out), run_id=args.run_id,
        quiet=args.quiet, meta=dict(args.seed_meta), price=prices.get(args.model),
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


async def _bench(args: argparse.Namespace) -> int:
    prices = load_prices(args.prices)
    if args.max_cost_per_run is not None and (bad := uncapped_models(args.models, prices)):
        raise ConfigError(  # spending without an effective cap is the dangerous direction: refuse to start
            f"--max-cost-per-run cannot be enforced for {', '.join(bad)}: the provider reports no dollar cost "
            "and --prices has no entry for it. Add the model to a --prices file, or drop --max-cost-per-run."
        )
    out = Path(args.out) if args.out else Path("runs") / f"bench-{datetime.now():%Y%m%d-%H%M%S}"
    async with GrueClient(args.grue_url) as grue:
        games = args.games
        if games == ["all"]:
            games = [g.id for g in await grue.list_games() if g.available]
        cfg = BenchConfig(
            models=args.models, games=games, runs=args.runs, max_turns=args.max_turns, concurrency=args.concurrency,
            max_cost_per_run=args.max_cost_per_run, max_time_per_run=args.max_time_per_run, history=args.history,
            notes=args.notes, temperature=args.temperature, max_tokens=args.max_tokens, base_url=args.base_url,
            reasoning=args.reasoning, grue_url=args.grue_url, out_dir=out, prices=prices,
        )
        total = len(cfg.models) * len(cfg.games) * cfg.runs
        print(f"bench: {len(cfg.models)} models x {len(cfg.games)} games x {cfg.runs} runs = {total} episodes "
              f"(concurrency {cfg.concurrency}) -> {out}")
        _, interrupted = await run_bench(cfg, grue=grue)
    agg = write_reports(out, load_runs(out))
    print()
    print(render_leaderboard(agg))
    print(f"wrote {out / 'leaderboard.md'}, results.csv, summary.json")
    return 130 if interrupted else 0


def _report(args: argparse.Namespace) -> int:
    runs = load_runs(args.dir)
    if not runs:
        print(f"error: no run.json files found under {args.dir}", file=sys.stderr)
        return 1
    agg = write_reports(args.dir, runs)
    print(render_leaderboard(agg))
    print(f"wrote leaderboard.md, results.csv, summary.json in {args.dir}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    setup_stdio()
    load_dotenv(find_dotenv(usecwd=True))
    args = build_parser().parse_args(argv)
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

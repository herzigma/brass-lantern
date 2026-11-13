"""Console output helpers: safe unicode on Windows and the live play reporter."""

from __future__ import annotations

import sys
from typing import Any, TextIO


def setup_stdio() -> None:
    """Make printing emoji/unicode safe on legacy Windows consoles."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
        except (AttributeError, ValueError):
            pass


def fmt_cost(cost: float | None) -> str:
    if cost is None:
        return "n/a"
    return f"${cost:.4f}" if cost < 1 else f"${cost:.2f}"


def fmt_duration(seconds: float) -> str:
    seconds = int(round(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}h{m:02d}m{s:02d}s" if h else (f"{m}m{s:02d}s" if m else f"{s}s")


class Reporter:
    """Pleasant live output for a single episode. ``quiet`` prints only the final line."""

    def __init__(self, quiet: bool = False, stream: TextIO | None = None):
        self.quiet = quiet
        self._out = stream

    def _print(self, *args: Any) -> None:
        print(*args, file=self._out or sys.stdout, flush=True)

    def start(self, run_id: str, game: str, model: str, max_turns: int) -> None:
        if not self.quiet:
            self._print(f"== Brass Lantern  run {run_id}\n== {game} | {model} | up to {max_turns} turns\n")

    def observation(self, text: str) -> None:
        if not self.quiet:
            self._print(text.strip() or "(no text)", "\n")

    def move(self, turn: int, command: str, score: int | None, max_score: int | None, cost: float | None, latency: float, warning: str | None = None) -> None:
        if self.quiet:
            return
        score_s = "?" if score is None else (f"{score}/{max_score}" if max_score else str(score))
        self._print(f"\U0001F916 [{turn + 1}] {command}    (score {score_s} | spent {fmt_cost(cost)} | {latency:.1f}s)")
        if warning:
            self._print(f"   ! {warning}")

    def end(self, summary: dict[str, Any]) -> None:
        t = summary["totals"]
        score = t.get("score")
        max_score = t.get("max_score")
        score_s = "?" if score is None else (f"{score}/{max_score}" if max_score else str(score))
        line = (
            f"== {summary['status']} ({summary['stop_reason']}) | ending={t.get('ending')} | score {score_s} | "
            f"{t['turns']} turns | {fmt_cost(t.get('cost_usd'))} | {fmt_duration(t['wall_time_s'])}"
        )
        if summary.get("error"):
            line += f"\n== error: {summary['error']}"
        self._print(("" if self.quiet else "\n") + line)

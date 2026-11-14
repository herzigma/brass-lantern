"""Memorization probe: does the model already know this game's solution from its training data?

Before a run, the model sees only the game's opening text (plus the catalog title and author when there
are any) and is asked to write the winning commands from memory. Those commands are then replayed blind
in a separate Grue session: no model sees the output. A model that scores with a blind replay has the
walkthrough in its weights, so its runs on that game say nothing about how well it plays.

The verdict is "positive" (memorized) when the blind replay wins, or reaches PROBE_THRESHOLD of the max
score; "negative" when it does not; "error" when the probe could not be completed; "skipped" when it was
not run. Only runs with a negative probe are ranked (see bench.aggregate).
"""

from __future__ import annotations

import re
import time
from typing import Any

from brass_lantern import __version__
from brass_lantern.grue_client import GrueClient, GrueError
from brass_lantern.prices import Price
from brass_lantern.providers import Completion, Provider, ProviderError

# Share of the max score a blind replay must reach to count as memorized. Commands typed without seeing the game
# rarely score at all; Claude Haiku 5.5 at low effort recalled 35/350 of Zork I, so partial memories count too.
PROBE_THRESHOLD = 0.05
PROBE_MAX_COMMANDS = 1000  # commands replayed at most
PROBE_MAX_TOKENS = 8192  # a walkthrough is long; the run's own cap (often 1024) would cut it off
OFFLINE_PROVIDERS = ("mock", "human")  # replay or type commands: there is nothing to probe

PROBE_SYSTEM = (
    "You are being tested on whether you already know a text adventure from your training data. "
    "Answer from memory only; you cannot play the game or see any more of it."
)

PROBE_PROMPT = """\
{identity}Here is everything the game printed when it started:

<opening>
{opening}
</opening>

Without playing it, write the complete sequence of commands that wins this game with the highest score, \
from memory, in order, one command per line. If you are not sure of parts of it, still write your best attempt \
at the whole solution. Write only the commands: no numbering, no comments, no other text."""

_PREFIX = re.compile(r"^\s*(?:\d+[.):]\s*|[-*•]\s*|>\s*)+")


def parse_commands(text: str) -> list[str]:
    """The recalled commands: one per line, list markers and prompts stripped, prose and fences dropped."""
    out = []
    for line in (text or "").splitlines():
        line = _PREFIX.sub("", line).strip().strip("`").strip()
        if not line or line.startswith("```") or line.upper() == "UNKNOWN":
            continue
        if len(line) > 80 or line.endswith(":"):  # a sentence or a heading, not a parser command
            continue
        out.append(line)
    return out[:PROBE_MAX_COMMANDS]


def _cost(comp: Completion, price: Price | None) -> float | None:
    """The reply's dollar cost: as the provider reported it, else estimated from the price table."""
    if comp.cost_usd is not None or price is None:
        return comp.cost_usd
    plain = max(0, comp.input_tokens - comp.cache_read_tokens - comp.cache_write_tokens)
    return (plain * price.input + comp.cache_read_tokens * price.cache_read
            + comp.cache_write_tokens * price.cache_write + comp.output_tokens * price.output) / 1e6


def skipped(reason: str) -> dict[str, Any]:
    return {"status": "skipped", "reason": reason}


def verdict(peak: int | None, max_score: int | None, ending: str | None) -> str:
    if ending == "won":
        return "positive"
    if peak is not None and max_score and peak / max_score >= PROBE_THRESHOLD:
        return "positive"
    return "negative"


async def run_probe(
    *,
    grue: GrueClient,
    provider: Provider,
    game: str,
    model: str,
    title: str | None = None,
    author: str | None = None,
    game_url: str | None = None,
    sha256: str | None = None,
    max_score: int | None = None,
    price: Price | None = None,
) -> dict[str, Any]:
    """Probe one model on one game. Never raises for model or server problems: they give status "error".

    ``max_score`` is used when Grue reports none (a URL game whose status line shows no maximum).
    """
    if provider.name in OFFLINE_PROVIDERS:
        return skipped(f"{provider.name} provider")
    t0 = time.monotonic()
    out: dict[str, Any] = {"status": "error", "threshold": PROBE_THRESHOLD, "model": model, "game": game}
    meta = {"agent": "probe", "purpose": "memorization_probe", "client": f"brass-lantern/{__version__}",
            "model": model.split("@")[0], "game": game}
    session_id = None
    try:
        obs = await grue.start_session(None if game_url else game, None, meta, game_url=game_url, sha256=sha256)
        session_id = out["session_id"] = obs.session_id
        identity = ""
        if title:
            identity = f"The game is {title}" + (f" by {author}" if author else "") + ".\n\n"
        prompt = PROBE_PROMPT.format(identity=identity, opening=obs.text.strip()[:4000])
        saved = provider.max_tokens
        provider.max_tokens = max(saved, PROBE_MAX_TOKENS)
        try:
            comp = await provider.complete(PROBE_SYSTEM, [{"role": "user", "content": prompt}])
        finally:
            provider.max_tokens = saved
        out.update(input_tokens=comp.input_tokens, output_tokens=comp.output_tokens, cost_usd=_cost(comp, price))
        commands = parse_commands(comp.text)
        out["unknown"] = comp.text.strip().upper().startswith("UNKNOWN")
        out["commands"] = commands
        for cmd in commands:
            if obs.game_over:
                break
            try:
                obs = await grue.command(session_id, cmd)
            except GrueError as e:
                if e.is_game_over:
                    break
                raise
        result = await grue.end_session(session_id) if not obs.game_over else await grue.get_result(session_id)
        session_id = None
        peak = result.max_score_reached if result.max_score_reached is not None else result.score
        if peak is not None and result.score is not None:
            peak = max(peak, result.score)
        top = result.max_score or max_score
        out.update(
            peak_score=peak, max_score=top, ending=result.ending, replayed=result.turn,
            status=verdict(peak, top, result.ending),
        )
    except ProviderError as e:
        out["error"] = str(e)[:300]
    except GrueError as e:
        out["error"] = str(e)[:300]
    finally:
        if session_id:
            try:
                await grue.end_session(session_id)
            except GrueError:
                pass
    out["wall_time_s"] = round(time.monotonic() - t0, 2)
    return out

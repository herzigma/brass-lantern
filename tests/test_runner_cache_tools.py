"""Cache-aware billing and totals, the --cache / --format options end to end, and bench variant labels."""

import asyncio
import json

import pytest

import brass_lantern.__main__ as cli
from brass_lantern.agent import HISTORY_STEP
from brass_lantern.bench import BenchConfig, aggregate, run_bench, write_reports, load_runs
from brass_lantern.console import Reporter
from brass_lantern.grue_client import GrueClient
from brass_lantern.providers.base import Completion, Provider
from brass_lantern.providers.scripted import ScriptedProvider
from brass_lantern.runner import RunConfig, play_episode
from fake_grue import WIN_SCRIPT


class CachedProvider(Provider):
    """Every call: 1M prompt tokens of which 600k are cache reads and 100k cache writes."""

    name = "cached"

    def __init__(self, cost=None):
        super().__init__("cached")
        self.cost = cost

    async def complete(self, system, messages, tools=None):
        return Completion(
            text="> look", input_tokens=1_000_000, output_tokens=500_000, cost_usd=self.cost,
            cache_read_tokens=600_000, cache_write_tokens=100_000, latency_s=0.1,
        )


class ToolScript(Provider):
    """Answers each turn with a native act tool call taken from a command list."""

    name = "toolscript"

    def __init__(self, commands):
        super().__init__("toolscript")
        self.commands = list(commands)
        self.tools_seen = []

    async def complete(self, system, messages, tools=None):
        self.tools_seen.append(tools)
        cmd = self.commands.pop(0) if self.commands else "look"
        return Completion(text="", input_tokens=5, output_tokens=1, cost_usd=0.0,
                          tool_calls=[{"name": "act", "arguments": {"command": cmd}}])


def play(cfg, grue, provider, quiet=True):
    return asyncio.run(play_episode(cfg, grue=grue, provider=provider, reporter=Reporter(quiet=quiet)))


def cfg_for(tmp_path, **kw):
    kw.setdefault("run_id", "r1")
    return RunConfig(game="tiny", model="mock:win", out_dir=tmp_path, quiet=True, **kw)


def read_run(tmp_path, rid="r1"):
    run = json.loads((tmp_path / rid / "run.json").read_text(encoding="utf-8"))
    lines = [json.loads(ln) for ln in (tmp_path / rid / "transcript.jsonl").read_text(encoding="utf-8").splitlines()]
    return run, lines


# ---------------------------------------------------------------------------
# Billing and totals
# ---------------------------------------------------------------------------


def test_price_table_bills_cache_reads_at_a_tenth_and_writes_at_one_and_a_quarter(tmp_path, fake, make_grue):
    # per call: 300k plain + 600k * 0.1 + 100k * 1.25 = 485k input-equivalents at $2/M, plus 500k output at $4/M
    expected = (485_000 * 2.0 + 500_000 * 4.0) / 1e6
    summary = play(cfg_for(tmp_path, max_turns=1, price=(2.0, 4.0)), make_grue(), CachedProvider())
    t = summary["totals"]
    assert t["cost_usd"] == pytest.approx(expected) and expected < (1_000_000 * 2.0 + 500_000 * 4.0) / 1e6
    assert (t["input_tokens"], t["cache_read_tokens"], t["cache_write_tokens"]) == (1_000_000, 600_000, 100_000)
    assert t["cache_hit_rate"] == pytest.approx(0.6)


def test_provider_reported_cost_is_used_as_is_regardless_of_cache_tokens(tmp_path, fake, make_grue):
    summary = play(cfg_for(tmp_path, max_turns=1, price=(2.0, 4.0)), make_grue(), CachedProvider(cost=0.123))
    assert summary["totals"]["cost_usd"] == pytest.approx(0.123)


def test_billing_without_cache_tokens_is_unchanged_and_never_negative(tmp_path, fake, make_grue):
    class Odd(CachedProvider):
        async def complete(self, system, messages, tools=None):
            c = await super().complete(system, messages)
            c.input_tokens = 100  # inconsistent provider data: cache counters exceed the total
            return c

    summary = play(cfg_for(tmp_path, max_turns=1, price=(2.0, 0.0)), make_grue(), Odd())
    assert summary["totals"]["cost_usd"] == pytest.approx((600_000 * 0.1 + 100_000 * 1.25) * 2.0 / 1e6)
    plain = play(cfg_for(tmp_path, run_id="r2", max_turns=1, price=(2.0, 4.0)), make_grue(), ScriptedProvider(["look"]))
    assert plain["totals"]["cache_read_tokens"] == 0 and plain["totals"]["cache_hit_rate"] is None


def test_summary_line_shows_cache_hit_rate_only_when_there_was_cache_activity(tmp_path, fake, make_grue, capsys):
    play(cfg_for(tmp_path, max_turns=2, price=(1.0, 1.0)), make_grue(), CachedProvider(), quiet=False)
    assert "cache hit 60%" in capsys.readouterr().out
    play(cfg_for(tmp_path, run_id="r2", max_turns=2), make_grue(), ScriptedProvider(["look"]), quiet=False)
    assert "cache hit" not in capsys.readouterr().out


def test_cache_and_format_are_recorded_in_run_json_and_transcript(tmp_path, fake, make_grue):
    play(cfg_for(tmp_path, max_turns=1), make_grue(), CachedProvider())
    run, lines = read_run(tmp_path)
    assert run["config"]["cache"] is True and run["config"]["format"] == "text"
    assert run["config"]["history_step"] == HISTORY_STEP
    assert lines[0]["cache_read_tokens"] == 600_000 and lines[0]["cache_write_tokens"] == 100_000
    play(cfg_for(tmp_path, run_id="r2", max_turns=1, cache=False, format="tools"), make_grue(), ToolScript(["look"]))
    run, lines = read_run(tmp_path, "r2")
    assert run["config"]["cache"] is False and run["config"]["format"] == "tools" and run["config"]["history_step"] == 0
    assert lines[0]["tool_calls"] == [{"name": "act", "arguments": {"command": "look"}}]


# ---------------------------------------------------------------------------
# The tools format through the runner
# ---------------------------------------------------------------------------


def test_tools_format_plays_a_full_game_with_no_parse_failures(tmp_path, fake, make_grue):
    prov = ToolScript(WIN_SCRIPT)
    summary = play(cfg_for(tmp_path, format="tools"), make_grue(), prov)
    t = summary["totals"]
    assert t["ending"] == "won" and t["parse_failures"] == 0 and t["turns"] == len(WIN_SCRIPT)
    assert all(tools and tools[0]["name"] == "act" for tools in prov.tools_seen)
    _, lines = read_run(tmp_path)
    turns = [ln for ln in lines if "command" in ln]
    assert [ln["command"] for ln in turns] == WIN_SCRIPT
    assert {ln["parse_method"] for ln in turns} == {"tool"} and all(ln["parse_ok"] for ln in turns)


def test_tools_format_with_a_text_only_provider_records_markers_as_parse_failures(tmp_path, fake, make_grue):
    summary = play(cfg_for(tmp_path, format="tools"), make_grue(), ScriptedProvider(WIN_SCRIPT))
    t = summary["totals"]
    assert t["ending"] == "won" and t["turns"] == len(WIN_SCRIPT) and t["parse_failures"] == len(WIN_SCRIPT)
    _, lines = read_run(tmp_path)
    assert {ln["parse_method"] for ln in lines if "command" in ln} == {"marker"}


# ---------------------------------------------------------------------------
# CLI flags
# ---------------------------------------------------------------------------


@pytest.fixture
def patched_grue(monkeypatch, fake):
    class TestGrue(GrueClient):
        def __init__(self, base_url=None, *a, **kw):
            super().__init__(base_url or "http://grue.test", transport=fake.transport())

    monkeypatch.setattr(cli, "GrueClient", TestGrue)
    return fake


@pytest.mark.parametrize("command", ["play", "bench"])
def test_help_lists_cache_and_format(command, capsys):
    with pytest.raises(SystemExit):
        cli.main([command, "--help"])
    out = capsys.readouterr().out
    assert "--cache {on,off}" in out and "--format {text,tools}" in out


def test_cli_defaults_are_cache_on_and_format_text():
    args = cli.build_parser().parse_args(["play", "--game", "g", "--model", "mock:x"])
    assert args.cache == "on" and args.format == "text"
    args = cli.build_parser().parse_args(["bench", "--models", "mock:x", "--games", "g", "--cache", "off", "--format", "tools"])
    assert args.cache == "off" and args.format == "tools"


@pytest.mark.parametrize("flags", [["--cache", "maybe"], ["--format", "xml"]])
def test_cli_rejects_bad_values(flags, capsys):
    with pytest.raises(SystemExit) as e:
        cli.main(["play", "--game", "tiny", "--model", "mock:x", *flags])
    assert e.value.code == 2


def test_play_records_flags_in_run_json(patched_grue, tmp_path):
    script = tmp_path / "win.txt"
    script.write_text("\n".join(WIN_SCRIPT), encoding="utf-8")
    out = tmp_path / "runs"
    assert cli.main(["play", "--game", "tiny", "--model", f"mock:{script}", "--out", str(out), "--run-id", "a", "--quiet"]) == 0
    assert cli.main(["play", "--game", "tiny", "--model", f"mock:{script}", "--out", str(out), "--run-id", "b", "--quiet",
                     "--cache", "off", "--format", "tools"]) == 0
    a = json.loads((out / "a" / "run.json").read_text(encoding="utf-8"))["config"]
    b = json.loads((out / "b" / "run.json").read_text(encoding="utf-8"))["config"]
    assert (a["cache"], a["format"], a["history_step"]) == (True, "text", HISTORY_STEP)
    assert (b["cache"], b["format"], b["history_step"]) == (False, "tools", 0)


def test_play_passes_cache_to_the_provider(patched_grue, tmp_path, monkeypatch):
    seen = {}
    real = cli.create_provider

    def spy(spec, **kw):
        seen.update(kw)
        return real(spec, **kw)

    monkeypatch.setattr(cli, "create_provider", spy)
    script = tmp_path / "s.txt"
    script.write_text("north\n", encoding="utf-8")
    cli.main(["play", "--game", "tiny", "--model", f"mock:{script}", "--out", str(tmp_path / "r"), "--quiet", "--max-turns", "1"])
    assert seen["cache"] is True
    cli.main(["play", "--game", "tiny", "--model", f"mock:{script}", "--out", str(tmp_path / "r"), "--quiet", "--max-turns", "1",
              "--cache", "off"])
    assert seen["cache"] is False


def test_bench_passes_flags_through_to_runs(patched_grue, tmp_path, capsys):
    idle = tmp_path / "idle.txt"
    idle.write_text("north\n", encoding="utf-8")
    out = tmp_path / "b"
    code = cli.main(["bench", "--models", f"mock:{idle}", "--games", "tiny", "--runs", "1", "--max-turns", "2",
                     "--cache", "off", "--format", "tools", "--out", str(out)])
    assert code == 0
    runs = load_runs(out)
    assert len(runs) == 1 and runs[0]["config"]["cache"] is False and runs[0]["config"]["format"] == "tools"
    capsys.readouterr()


def test_run_bench_worker_forwards_cache_and_format(tmp_path, fake, make_grue):
    cfg = BenchConfig(models=["m"], games=["tiny"], runs=1, max_turns=2, out_dir=tmp_path, cache=False, format="tools")
    made = []

    def factory(spec, **kw):
        made.append(kw)
        return ToolScript(["north"])

    summaries, _ = asyncio.run(run_bench(cfg, grue=make_grue(), provider_factory=factory, log=lambda s: None))
    assert all(kw["cache"] is False for kw in made)
    assert summaries[0]["config"]["cache"] is False and summaries[0]["config"]["format"] == "tools"
    assert summaries[0]["totals"]["parse_failures"] == 0


# ---------------------------------------------------------------------------
# Bench variants
# ---------------------------------------------------------------------------

def make_run(model, game, score, max_score, *, config, run_id=None):
    return {
        "run_id": run_id or f"{model}-{game}-{score}", "model": model, "game": game, "status": "completed",
        "stop_reason": "game_over", "config": config,
        "totals": {"turns": 10, "score": score, "max_score": max_score, "max_score_reached": score,
                   "ending": "turn_limit", "input_tokens": 1000, "output_tokens": 100, "cost_usd": 0.1,
                   "parse_failures": 0, "invalid_commands": 0},
    }


BASE = {"temperature": 0.7, "history": 30, "notes": False, "reasoning": None, "max_tokens": 1024, "max_turns": 150}


def test_cache_and_format_split_runs_into_separate_labelled_variants():
    runs = [
        make_run("m", "g", 2, 10, config={**BASE, "cache": True, "format": "text"}, run_id="a"),
        make_run("m", "g", 4, 10, config={**BASE, "cache": False, "format": "text"}, run_id="b"),
        make_run("m", "g", 6, 10, config={**BASE, "cache": True, "format": "tools"}, run_id="c"),
        make_run("m", "g", 8, 10, config={**BASE, "cache": True, "format": "tools"}, run_id="d"),  # merges with c
        make_run("solo", "g", 5, 10, config={**BASE, "cache": True, "format": "text"}, run_id="e"),
    ]
    agg = aggregate(runs)
    assert set(agg["models"]) == {"m (cache, format=text)", "m (no cache, format=text)", "m (cache, format=tools)", "solo"}
    assert agg["models"]["m (cache, format=tools)"]["overall"]["runs"] == 2
    assert agg["models"]["m (cache, format=tools)"]["config"]["format"] == "tools"
    assert {r["run_id"]: r["variant"] for r in agg["rows"]}["b"] == "m (no cache, format=text)"


def test_only_the_differing_knob_appears_in_the_label():
    runs = [
        make_run("m", "g", 1, 10, config={**BASE, "cache": True, "format": "text"}),
        make_run("m", "g", 1, 10, config={**BASE, "cache": False, "format": "text"}),
    ]
    assert set(aggregate(runs)["models"]) == {"m (cache)", "m (no cache)"}
    runs = [
        make_run("m", "g", 1, 10, config={**BASE, "cache": True, "format": "text"}),
        make_run("m", "g", 1, 10, config={**BASE, "cache": True, "format": "tools"}),
    ]
    assert set(aggregate(runs)["models"]) == {"m (format=text)", "m (format=tools)"}


def test_old_runs_without_the_new_settings_count_as_cache_off_text_and_still_merge_with_each_other():
    runs = [
        make_run("m", "g", 2, 10, config=dict(BASE), run_id="old1"),
        make_run("m", "g", 4, 10, config=dict(BASE), run_id="old2"),
    ]
    agg = aggregate(runs)
    assert set(agg["models"]) == {"m"}  # one configuration: the plain spec is the label
    row = agg["rows"][0]
    assert (row["cache"], row["format"], row["history_step"]) == (False, "text", 0)
    # ...and they equal an explicit cache-off/text run
    mixed = runs + [make_run("m", "g", 6, 10, config={**BASE, "cache": False, "format": "text"})]
    assert set(aggregate(mixed)["models"]) == {"m"}


def test_results_csv_has_cache_and_format_columns(tmp_path):
    import csv

    write_reports(tmp_path, [make_run("m", "g", 1, 10, config={**BASE, "cache": True, "format": "tools", "history_step": 10})])
    rows = list(csv.DictReader((tmp_path / "results.csv").open(encoding="utf-8")))
    assert rows[0]["cache"] == "True" and rows[0]["format"] == "tools" and rows[0]["history_step"] == "10"
    assert {"cache_read_tokens", "cache_write_tokens"} <= set(rows[0])

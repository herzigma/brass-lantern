"""Settings in the session meta, the run report to Grue, and @effort variants in a bench."""

import asyncio
import csv
import json

import httpx
import pytest

import brass_lantern.__main__ as cli
from brass_lantern import __version__
from brass_lantern.bench import BenchConfig, aggregate, load_runs, plan_resume, render_leaderboard, run_bench, write_reports
from brass_lantern.console import Reporter
from brass_lantern.grue_client import GrueClient
from brass_lantern.prices import Price
from brass_lantern.providers.base import Completion, Provider, ProviderError
from brass_lantern.providers.scripted import ScriptedProvider
from brass_lantern.runner import RunConfig, make_run_id, play_episode
from fake_grue import WIN_SCRIPT

OPUS = "anthropic:claude-opus-5-5@high"
CONTRACT_REPORT_KEYS = {
    "input_tokens", "output_tokens", "cache_read_tokens", "cache_write_tokens", "cost_usd", "cost_source",
    "model_time_s", "parse_failures", "provider_errors", "stop_reason", "status", "client_version",
}
CONTRACT_META_KEYS = {
    "agent", "model", "effort", "format", "cache", "history", "history_step", "notes", "max_tokens", "temperature",
    "run_id",
}


class Usage(Provider):
    """Looks every turn, billing a fixed token count per call."""

    name = "usage"

    def __init__(self, fail_on=None):
        super().__init__("usage")
        self.fail_on, self.n = fail_on, 0

    async def complete(self, system, messages, tools=None):
        self.n += 1
        if self.fail_on == self.n:
            raise ProviderError("usage: HTTP 500 (gave up after 6 attempts)")
        return Completion(text="> look", input_tokens=1000, output_tokens=100, cache_read_tokens=400,
                          cache_write_tokens=100, latency_s=0.5)


def play(tmp_path, grue, provider, **kw):
    kw.setdefault("run_id", "r1")
    kw.setdefault("model", OPUS)
    cfg = RunConfig(game="tiny", out_dir=tmp_path, quiet=True, **kw)
    return asyncio.run(play_episode(cfg, grue=grue, provider=provider, reporter=Reporter(quiet=True)))


def report_posts(fake):
    return [(p, b) for m, p, b in fake.requests if m == "POST" and p.endswith("/report")]


def start_meta(fake):
    return next(b for m, p, b in fake.requests if (m, p) == ("POST", "/api/sessions"))["meta"]


# ---------------------------------------------------------------------------
# Meta
# ---------------------------------------------------------------------------


def test_session_meta_carries_the_settings_without_the_effort_suffix(tmp_path, fake, make_grue):
    summary = play(tmp_path, make_grue(), ScriptedProvider(["look"]), max_turns=2, effort="high", temperature=None,
                   max_tokens=16000, format="tools", cache=True, history=25, notes=True,
                   meta={"bench": "bench-x", "repeat": 2})
    meta = start_meta(fake)
    assert meta == {
        "agent": f"brass-lantern/{__version__}", "model": "anthropic:claude-opus-5-5", "effort": "high",
        "format": "tools", "cache": True, "history": 25, "history_step": 10, "notes": True, "max_tokens": 16000,
        "temperature": None, "run_id": "r1", "bench": "bench-x", "repeat": 2,
    }
    assert CONTRACT_META_KEYS <= set(meta)
    # run.json keeps the full spec (so @low and @high stay two models) and the effective settings
    assert summary["model"] == OPUS and summary["config"]["meta"] == meta
    cfg = summary["config"]
    assert (cfg["effort"], cfg["temperature"], cfg["max_tokens"]) == ("high", None, 16000) and "reasoning" not in cfg


def test_meta_for_a_run_without_effort_or_cache(tmp_path, fake, make_grue):
    play(tmp_path, make_grue(), ScriptedProvider(["look"]), model="openrouter:x/y", max_turns=1, cache=False)
    meta = start_meta(fake)
    assert meta["effort"] is None and meta["cache"] is False and meta["history_step"] == 0 and meta["temperature"] == 0.7
    assert meta["model"] == "openrouter:x/y" and meta["max_tokens"] == 1024 and meta["format"] == "text" and meta["notes"] is False


def test_the_run_id_keeps_the_effort(tmp_path):
    a, b = make_run_id("zork1", "anthropic:claude-sonnet-5-5@low"), make_run_id("zork1", "anthropic:claude-sonnet-5-5@high")
    assert "-anthropic-claude-sonnet-5-5-low-" in a and "-anthropic-claude-sonnet-5-5-high-" in b
    assert make_run_id("zork1", "anthropic:claude-sonnet-5-5").startswith("2") and "mock-win" in make_run_id("g", "mock:/x/win.txt")


def test_a_temperature_dropped_by_the_retry_is_recorded_as_not_sent(tmp_path, fake, make_grue):
    class Dropped(Usage):
        async def complete(self, system, messages, tools=None):
            self.temperature, self.temperature_dropped = None, True
            return await super().complete(system, messages)

    summary = play(tmp_path, make_grue(), Dropped(), model="openai:gpt-6.1-sol", temperature=0.7, max_turns=1)
    assert summary["config"]["temperature"] is None
    assert start_meta(fake)["temperature"] == 0.7  # what the session was told at its start


# ---------------------------------------------------------------------------
# The report
# ---------------------------------------------------------------------------


def test_the_client_report_is_posted_after_the_episode_and_accepted(tmp_path, fake, make_grue):
    price = Price(2.0, 10.0, 0.1, 2.5)
    summary = play(tmp_path, make_grue(), Usage(), max_turns=2, price=price)
    posts = report_posts(fake)
    assert [p for p, _ in posts] == ["/api/sessions/sess-1/report"]
    body = posts[0][1]
    assert set(body) == {"client_report"}
    rep = body["client_report"]
    assert set(rep) == CONTRACT_REPORT_KEYS
    per_call = (500 * 2.0 + 400 * 0.1 + 100 * 2.5 + 100 * 10.0) / 1e6
    assert rep == {
        "input_tokens": 2000, "output_tokens": 200, "cache_read_tokens": 800, "cache_write_tokens": 200,
        "cost_usd": pytest.approx(2 * per_call), "cost_source": "prices", "model_time_s": 1.0, "parse_failures": 0,
        "provider_errors": 0, "stop_reason": "game_over", "status": "completed", "client_version": __version__,
    }
    assert all(v is None or isinstance(v, (bool, int, float, str)) for v in rep.values())
    assert len(json.dumps(body).encode()) < 4096
    assert fake.reports["sess-1"] == rep and fake.results["sess-1"]["client_report_at"]
    assert summary["report"] == {"accepted": True, "error": None}
    assert json.loads((tmp_path / "r1" / "run.json").read_text(encoding="utf-8"))["report"] == {"accepted": True, "error": None}
    assert fake.requests[-1][:2] == ("POST", "/api/sessions/sess-1/report")  # sent after the result was fetched


def test_the_report_goes_out_whatever_the_status(tmp_path, fake, make_grue):
    won = play(tmp_path, make_grue(), ScriptedProvider(WIN_SCRIPT), run_id="won")
    err = play(tmp_path, make_grue(), Usage(fail_on=2), run_id="err")
    assert [(r["status"], r["stop_reason"], r["provider_errors"]) for r in
            (fake.reports["sess-1"], fake.reports["sess-2"])] == [("completed", "game_over", 0), ("error", "provider_error", 1)]
    assert fake.reports["sess-1"]["cost_source"] == "provider" and fake.reports["sess-2"]["cost_source"] is None
    assert fake.reports["sess-2"]["cost_usd"] is None and fake.reports["sess-2"]["input_tokens"] == 1000
    assert won["report"]["accepted"] and err["report"]["accepted"]


def test_an_interrupted_run_is_reported_too(tmp_path, fake, make_grue):
    class Hanging(Usage):
        async def complete(self, system, messages, tools=None):
            await asyncio.sleep(60)

    async def go():
        cfg = RunConfig(game="tiny", model=OPUS, out_dir=tmp_path, quiet=True, run_id="r1")
        task = asyncio.create_task(play_episode(cfg, grue=make_grue(), provider=Hanging(), reporter=Reporter(quiet=True)))
        await asyncio.sleep(0.2)
        task.cancel()
        return await task

    summary = asyncio.run(go())
    assert summary["status"] == "interrupted" and fake.reports["sess-1"]["status"] == "interrupted"
    assert fake.reports["sess-1"]["stop_reason"] == "interrupted" and summary["report"]["accepted"]


def test_a_404_from_an_older_server_is_one_warning_and_changes_nothing(tmp_path, fake, make_grue, capsys):
    fake.report_endpoint = False
    summary = play(tmp_path, make_grue(), ScriptedProvider(WIN_SCRIPT))
    assert summary["status"] == "completed" and summary["stop_reason"] == "game_over" and summary["totals"]["ending"] == "won"
    assert summary["report"]["accepted"] is False and "404" in summary["report"]["error"]
    err = capsys.readouterr().err
    lines = [ln for ln in err.splitlines() if ln.strip()]
    assert len(lines) == 1 and lines[0].startswith("warning:") and "r1" in lines[0] and "404" in lines[0]
    assert json.loads((tmp_path / "r1" / "run.json").read_text(encoding="utf-8"))["report"]["accepted"] is False


def test_a_409_from_grue_is_a_warning_only(tmp_path, fake, make_grue, capsys):
    fake.read_only.add("sess-1")
    summary = play(tmp_path, make_grue(), ScriptedProvider(WIN_SCRIPT))
    assert summary["status"] == "completed" and summary["report"]["accepted"] is False
    assert "read_only" in summary["report"]["error"] and capsys.readouterr().err.count("warning:") == 1


def test_a_network_failure_while_reporting_is_a_warning_only(tmp_path, fake, capsys):
    def handler(request):
        if request.url.path.endswith("/report"):
            raise httpx.ConnectError("boom")
        return fake._handle(request)

    grue = GrueClient("http://grue.test", transport=httpx.MockTransport(handler))
    summary = play(tmp_path, grue, ScriptedProvider(WIN_SCRIPT))
    assert summary["status"] == "completed" and summary["report"]["accepted"] is False
    assert "Cannot reach Grue" in summary["report"]["error"] and capsys.readouterr().err.count("warning:") == 1


def test_a_second_report_for_the_same_session_is_refused_with_a_warning(tmp_path, fake, make_grue, capsys):
    play(tmp_path, make_grue(), ScriptedProvider(["look"]), max_turns=1)
    grue = make_grue()
    with pytest.raises(Exception, match="already_reported"):
        asyncio.run(grue.report("sess-1", {"status": "again"}))


def test_no_report_without_a_session_id(tmp_path, fake, make_grue, capsys):
    cfg = RunConfig(game="no-such-game", model=OPUS, out_dir=tmp_path, quiet=True, run_id="r1")
    summary = asyncio.run(play_episode(cfg, grue=make_grue(), provider=ScriptedProvider([]), reporter=Reporter(quiet=True)))
    assert summary["status"] == "error" and summary["stop_reason"] == "grue_error" and summary["session_id"] is None
    assert report_posts(fake) == [] and summary["report"] is None
    assert capsys.readouterr().err == ""


def test_the_fake_grue_report_endpoint_follows_the_contract(tmp_path, fake, make_grue):
    grue = make_grue()

    async def go():
        obs = await grue.start_session("tiny")
        sid = obs.session_id
        errors = {}

        async def attempt(label, report, session=sid):
            try:
                await grue.report(session, report)
            except Exception as e:  # noqa: BLE001
                errors[label] = e

        await attempt("active", {"a": 1})  # the session is still running
        await attempt("unknown", {"a": 1}, session="nope")
        await grue.end_session(sid)
        await attempt("nested", {"a": {"b": 1}})
        await attempt("long", {"a": "x" * 201})
        await attempt("ok", {"a": 1, "b": "s", "c": True, "d": None, "e": 1.5})
        await attempt("again", {"a": 2})
        return errors, await grue.get_result(sid)

    errors, result = asyncio.run(go())
    assert set(errors) == {"active", "unknown", "nested", "long", "again"}
    assert (errors["active"].status, errors["active"].code) == (409, "session_active")
    assert (errors["unknown"].status, errors["unknown"].code) == (404, "unknown_session")
    assert errors["nested"].status == 400 and errors["nested"].code == "bad_request" and errors["long"].status == 400
    assert (errors["again"].status, errors["again"].code) == (409, "already_reported")
    assert result.raw["client_report"] == {"a": 1, "b": "s", "c": True, "d": None, "e": 1.5}


# ---------------------------------------------------------------------------
# Bench: @effort variants are distinct everywhere
# ---------------------------------------------------------------------------

LOW, HIGH = "anthropic:claude-sonnet-5-5@low", "anthropic:claude-sonnet-5-5@high"


def scripted(spec, **kw):
    return ScriptedProvider(WIN_SCRIPT if "@high" in spec else ["look"], model=spec)


def test_effort_variants_are_distinct_models_runs_and_labels(tmp_path, fake, make_grue):
    cfg = BenchConfig(models=[LOW, HIGH, "openrouter:x/y"], games=["tiny"], runs=2, max_turns=6, concurrency=3,
                      effort="medium", out_dir=tmp_path / "b", prices={LOW: Price(2, 10, 0.1, 2.5)})
    summaries, interrupted = asyncio.run(run_bench(cfg, grue=make_grue(), provider_factory=scripted, log=lambda s: None))
    assert not interrupted and len(summaries) == 6 and len({s["run_id"] for s in summaries}) == 6
    by_model = {}
    for s in summaries:
        by_model.setdefault(s["model"], []).append(s)
    assert set(by_model) == {LOW, HIGH, "openrouter:x/y"}
    # each spec got its own effective settings: the suffix beats --effort, anthropic gets no temperature, etc.
    got = {m: (r[0]["config"]["effort"], r[0]["config"]["temperature"], r[0]["config"]["max_tokens"]) for m, r in by_model.items()}
    assert got == {LOW: ("low", None, 16000), HIGH: ("high", None, 16000), "openrouter:x/y": ("medium", 0.7, 16000)}
    assert all("-low-" in s["run_id"] for s in by_model[LOW]) and all("-high-" in s["run_id"] for s in by_model[HIGH])
    metas = {m: [b["meta"] for mm, p, b in fake.requests if (mm, p) == ("POST", "/api/sessions") and b["meta"]["run_id"] == r["run_id"]][0]
             for m, rs in by_model.items() for r in rs[:1]}
    assert metas[LOW]["model"] == metas[HIGH]["model"] == "anthropic:claude-sonnet-5-5"
    assert (metas[LOW]["effort"], metas[HIGH]["effort"]) == ("low", "high") and metas[LOW]["bench"] == "b"
    # the price of the base model applies to both variants
    assert by_model[LOW][0]["totals"]["cost_usd"] == 0.0 and by_model[HIGH][0]["totals"]["cost_usd"] == 0.0

    bench_json = json.loads((cfg.out_dir / "bench.json").read_text(encoding="utf-8"))
    assert bench_json["models"] == [LOW, HIGH, "openrouter:x/y"] and bench_json["effort"] == "medium"

    agg = write_reports(cfg.out_dir, load_runs(cfg.out_dir))
    assert set(agg["models"]) == {LOW, HIGH, "openrouter:x/y"}
    assert agg["ranking"][0] == HIGH and agg["models"][LOW]["config"]["effort"] == "low"
    md = render_leaderboard(agg)
    assert f"`{LOW}`" in md and f"`{HIGH}`" in md
    rows = list(csv.DictReader((cfg.out_dir / "results.csv").open(encoding="utf-8")))
    assert {r["model"] for r in rows} == {LOW, HIGH, "openrouter:x/y"}
    assert {r["model"]: r["effort"] for r in rows} == {LOW: "low", HIGH: "high", "openrouter:x/y": "medium"}
    assert {"effort", "temperature", "max_tokens", "cost_source"} <= set(rows[0])


def test_resume_matches_episodes_per_effort_variant():
    def run(model, repeat):
        return {"run_id": f"{model}-{repeat}", "model": model, "game": "g", "status": "completed", "stop_reason": "game_over",
                "config": {"meta": {"repeat": repeat}}, "totals": {"turns": 5, "score": 1, "max_score": 10}}

    cfg = BenchConfig(models=[LOW, HIGH], games=["g"], runs=2)
    # @low finished both repeats, @high only the first: only @high #2 is left
    jobs, total = plan_resume(cfg, [run(LOW, 1), run(LOW, 2), run(HIGH, 1)])
    assert total == 4 and jobs == [(1, "g", HIGH)]
    # a run of the other variant never fills a slot
    assert plan_resume(cfg, [run(LOW, 1), run(LOW, 2)])[0] == [(0, "g", HIGH), (1, "g", HIGH)]


def test_old_temperature_runs_and_new_no_temperature_runs_stay_distinguishable():
    def make(temperature, rid):
        return {"run_id": rid, "model": "anthropic:claude-sonnet-5-5", "game": "g", "status": "completed",
                "stop_reason": "game_over", "config": {"temperature": temperature, "max_tokens": 1024},
                "totals": {"turns": 5, "score": 1, "max_score": 10, "cost_usd": 0.1}}

    agg = aggregate([make(0.7, "old"), make(None, "new")])
    assert set(agg["models"]) == {"anthropic:claude-sonnet-5-5 (t=0.7)", "anthropic:claude-sonnet-5-5 (t=none)"}
    assert {r["run_id"]: r["temperature"] for r in agg["rows"]} == {"old": 0.7, "new": None}


def test_runs_from_before_effort_called_it_reasoning_still_merge_with_new_ones():
    def make(config, rid):
        return {"run_id": rid, "model": "openrouter:x/y", "game": "g", "status": "completed", "stop_reason": "game_over",
                "config": {"temperature": 0.7, **config}, "totals": {"turns": 5, "score": 1, "max_score": 10}}

    agg = aggregate([make({"reasoning": "high"}, "old"), make({"effort": "high"}, "new"), make({"reasoning": None}, "none")])
    assert set(agg["models"]) == {"openrouter:x/y (effort=high)", "openrouter:x/y (effort=default)"}
    assert agg["models"]["openrouter:x/y (effort=high)"]["overall"]["runs"] == 2


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@pytest.fixture
def patched_grue(monkeypatch, fake):
    class TestGrue(GrueClient):
        def __init__(self, base_url=None, *a, **kw):
            super().__init__(base_url or "http://grue.test", transport=fake.transport())

    monkeypatch.setattr(cli, "GrueClient", TestGrue)
    return fake


def test_cli_effort_flag_choices_and_help(capsys):
    parser = cli.build_parser()
    for effort in ("minimal", "low", "medium", "high", "xhigh", "max"):
        assert parser.parse_args(["play", "--game", "g", "--model", "mock:x", "--effort", effort]).effort == effort
    args = parser.parse_args(["bench", "--models", "mock:x", "--games", "g"])
    assert args.effort is None and args.max_tokens is None and args.reasoning is None
    with pytest.raises(SystemExit):
        parser.parse_args(["play", "--game", "g", "--model", "mock:x", "--effort", "extreme"])
    with pytest.raises(SystemExit):
        cli.main(["play", "--help"])
    out = " ".join(capsys.readouterr().out.split())
    assert "--effort {minimal,low,medium,high,xhigh,max}" in out and "deprecated alias of --effort" in out
    assert "@high" in out and "bundled" in out


def test_reasoning_is_a_deprecated_alias_for_effort(patched_grue, tmp_path, monkeypatch, capsys):
    seen = {}
    real = cli.create_provider

    def spy(spec, **kw):
        seen.update(kw)
        return real(spec, **kw)

    monkeypatch.setattr(cli, "create_provider", spy)
    script = tmp_path / "s.txt"
    script.write_text("north\n", encoding="utf-8")
    args = ["play", "--game", "tiny", "--model", f"mock:{script}", "--out", str(tmp_path / "r"), "--quiet", "--max-turns", "1"]
    assert cli.main([*args, "--reasoning", "high"]) == 0
    assert seen["effort"] == "high" and "--reasoning is deprecated" in capsys.readouterr().err
    assert cli.main([*args, "--effort", "low", "--reasoning", "high"]) == 0 and seen["effort"] == "low"
    capsys.readouterr()
    assert cli.main(args) == 0 and seen["effort"] is None and "deprecated" not in capsys.readouterr().err


def test_play_records_effective_settings_and_reports(patched_grue, tmp_path, capsys):
    script = tmp_path / "win.txt"
    script.write_text("\n".join(WIN_SCRIPT), encoding="utf-8")
    out = tmp_path / "runs"
    assert cli.main(["play", "--game", "tiny", "--model", f"mock:{script}", "--out", str(out), "--run-id", "a", "--quiet",
                     "--effort", "high"]) == 0
    run = json.loads((out / "a" / "run.json").read_text(encoding="utf-8"))
    assert run["config"]["effort"] is None  # mock ignores effort, so none was sent
    assert run["config"]["max_tokens"] == 1024 and run["report"]["accepted"] is True
    assert patched_grue.reports["sess-1"]["client_version"] == __version__


def test_play_refuses_an_effort_the_provider_cannot_take(patched_grue, tmp_path, capsys, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    code = cli.main(["play", "--game", "tiny", "--model", "anthropic:claude-sonnet-5-5@minimal", "--out", str(tmp_path)])
    assert code == 2 and "minimal" in capsys.readouterr().err
    code = cli.main(["play", "--game", "tiny", "--model", "gemini:gemini-3-pro", "--effort", "high", "--out", str(tmp_path)])
    assert code == 2 and "openrouter:google/" in capsys.readouterr().err
    assert patched_grue.requests == []  # nothing ran, no session started


def test_bench_command_runs_variants_with_a_flag_effort_and_prices_cap(patched_grue, tmp_path, capsys, monkeypatch):
    # direct anthropic/openai models are capped out of the box: the guard no longer asks for --prices
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    monkeypatch.setattr(cli, "run_bench", lambda *a, **kw: asyncio.sleep(0, ([], False)))
    code = cli.main(["bench", "--models", f"{LOW},{HIGH},openai:gpt-6.1-sol@medium", "--games", "tiny", "--runs", "1",
                     "--max-cost-per-run", "0.5", "--out", str(tmp_path / "b")])
    assert code == 0 and "3 models x 1 games x 1 runs" in capsys.readouterr().out
    code = cli.main(["bench", "--models", "anthropic:claude-3-opus-20240229", "--games", "tiny", "--runs", "1",
                     "--max-cost-per-run", "0.5", "--out", str(tmp_path / "c")])
    assert code == 2 and "claude-3-opus-20240229" in capsys.readouterr().err

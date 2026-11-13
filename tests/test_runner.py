import asyncio
import json

import pytest

from brass_lantern.console import Reporter
from brass_lantern.grue_client import GrueClient, GrueError
from brass_lantern.providers.base import Completion, Provider, ProviderError
from brass_lantern.providers.scripted import ScriptedProvider
from brass_lantern.runner import RunConfig, play_episode
from fake_grue import WIN_SCRIPT


class CostlyProvider(Provider):
    """Repeats `look` and reports a fixed cost / token count per call; optional hook per call."""

    name = "costly"

    def __init__(self, cost=0.3, hook=None, fail_on=None, tokens=(100, 10)):
        super().__init__("costly")
        self.cost, self.hook, self.fail_on, self.tokens, self.n = cost, hook, fail_on, tokens, 0

    async def complete(self, system, messages):
        self.n += 1
        if self.fail_on == self.n:
            raise ProviderError("costly: HTTP 500 (gave up after 6 attempts)")
        if self.hook:
            await self.hook(self.n)
        return Completion(text="> look", input_tokens=self.tokens[0], output_tokens=self.tokens[1],
                          cost_usd=self.cost, latency_s=0.5)


def play(cfg, grue, provider, quiet=True):
    return asyncio.run(play_episode(cfg, grue=grue, provider=provider, reporter=Reporter(quiet=quiet)))


def cfg_for(tmp_path, **kw):
    kw.setdefault("run_id", "r1")
    return RunConfig(game="tiny", model="mock:win", out_dir=tmp_path, quiet=True, **kw)


def read_run(tmp_path, rid="r1"):
    run = json.loads((tmp_path / rid / "run.json").read_text(encoding="utf-8"))
    lines = [json.loads(ln) for ln in (tmp_path / rid / "transcript.jsonl").read_text(encoding="utf-8").splitlines()]
    return run, lines


def test_full_win_end_to_end(tmp_path, fake, make_grue):
    summary = play(cfg_for(tmp_path, temperature=0.5), make_grue(), ScriptedProvider(WIN_SCRIPT))
    run, lines = read_run(tmp_path)
    assert run == json.loads(json.dumps(summary))
    assert run["status"] == "completed" and run["stop_reason"] == "game_over"
    t = run["totals"]
    assert (t["turns"], t["score"], t["max_score"], t["ending"]) == (4, 10, 10, "won")
    assert t["max_score_reached"] == 10 and t["invalid_commands"] == 0 and t["unique_locations"] == 2
    assert t["parse_failures"] == 0 and t["cost_usd"] == 0.0
    assert run["result"]["ending"] == "won" and run["result"]["session_id"] == run["session_id"]
    assert run["model"] == "mock:win" and run["game"] == "tiny"
    assert run["config"]["temperature"] == 0.5 and run["config"]["history"] == 30
    # session meta sent to Grue
    start = next(b for m, p, b in fake.requests if (m, p) == ("POST", "/api/sessions"))
    assert start["max_turns"] == 150 and start["game"] == "tiny"
    assert start["meta"]["agent"].startswith("brass-lantern/") and start["meta"]["run_id"] == "r1"
    assert start["meta"]["model"] == "mock:win" and start["meta"]["temperature"] == 0.5
    # transcript: one entry per command + final entry
    cmds = [ln["command"] for ln in lines if "command" in ln]
    assert cmds == WIN_SCRIPT
    assert lines[0]["observation"]["text"].startswith("Field") and lines[0]["turn"] == 0
    assert lines[-1]["final"] and lines[-1]["observation"]["ending"] == "won"
    assert all("reply" in ln and "latency_s" in ln for ln in lines if "command" in ln)
    # finished session: result fetched, no DELETE needed
    assert fake.deleted == []


def test_turn_limit(tmp_path, fake, make_grue):
    summary = play(cfg_for(tmp_path, max_turns=5), make_grue(), ScriptedProvider(["north"]))
    t = summary["totals"]
    assert t["ending"] == "turn_limit" and t["turns"] == 5 and summary["status"] == "completed"
    assert t["invalid_commands"] == 4  # four `look`s after the first `north`


def test_invalid_commands_and_parse_failures_counted(tmp_path, fake, make_grue):
    class Bare(CostlyProvider):
        async def complete(self, system, messages):
            c = await super().complete(system, messages)
            c.text = "dance wildly"  # no marker
            return c

    summary = play(cfg_for(tmp_path, max_turns=3), make_grue(), Bare(cost=None))
    t = summary["totals"]
    assert t["parse_failures"] == 3 and t["invalid_commands"] == 3
    assert t["cost_usd"] is None  # cost unknown stays unknown


def test_max_cost_stops_run_and_aborts_session(tmp_path, fake, make_grue):
    summary = play(cfg_for(tmp_path, max_cost=0.5), make_grue(), CostlyProvider(cost=0.3))
    assert summary["stop_reason"] == "max_cost" and summary["status"] == "completed"
    t = summary["totals"]
    assert t["turns"] == 2 and t["cost_usd"] == pytest.approx(0.6)
    assert t["input_tokens"] == 200 and t["output_tokens"] == 20
    assert t["ending"] == "aborted" and fake.deleted == ["sess-1"]


def test_price_table_used_when_provider_reports_no_cost(tmp_path, fake, make_grue):
    prov = CostlyProvider(cost=None, tokens=(1_000_000, 500_000))
    summary = play(cfg_for(tmp_path, max_cost=5.0, price=(2.0, 4.0)), make_grue(), prov)
    assert summary["totals"]["turns"] == 2 and summary["totals"]["cost_usd"] == pytest.approx(8.0)


def test_max_time_stops(tmp_path, fake, make_grue):
    summary = play(cfg_for(tmp_path, max_time=0.0), make_grue(), CostlyProvider())
    assert summary["stop_reason"] == "max_time" and summary["totals"]["turns"] == 0 and fake.deleted == ["sess-1"]


def test_provider_error_is_recorded_and_session_ended(tmp_path, fake, make_grue):
    summary = play(cfg_for(tmp_path), make_grue(), CostlyProvider(fail_on=3))
    assert summary["status"] == "error" and summary["stop_reason"] == "provider_error"
    assert "gave up" in summary["error"] and summary["totals"]["turns"] == 2
    assert fake.deleted == ["sess-1"]
    run, _ = read_run(tmp_path)
    assert run["status"] == "error"


def test_grue_unreachable_is_recorded(tmp_path):
    import httpx

    def refuse(request):
        raise httpx.ConnectError("refused")

    grue = GrueClient("http://nowhere.test", transport=httpx.MockTransport(refuse))
    summary = play(cfg_for(tmp_path), grue, CostlyProvider())
    assert summary["status"] == "error" and summary["stop_reason"] == "grue_error"
    assert "Cannot reach Grue" in summary["error"] and summary["session_id"] is None


def test_unknown_game_is_an_error_run(tmp_path, fake, make_grue):
    cfg = RunConfig(game="nope", model="m", out_dir=tmp_path, run_id="r1", quiet=True)
    summary = play(cfg, make_grue(), CostlyProvider())
    assert summary["status"] == "error" and "unknown_game" in summary["error"]


def test_409_game_over_is_handled(tmp_path, fake, make_grue):
    async def kill(n):
        if n == 2:
            fake.force_game_over("sess-1", "died")

    summary = play(cfg_for(tmp_path), make_grue(), CostlyProvider(hook=kill))
    assert summary["status"] == "completed" and summary["stop_reason"] == "game_over"
    assert summary["totals"]["ending"] == "died" and summary["totals"]["turns"] == 2
    assert summary["result"]["ending"] == "died"


def test_cancellation_ends_session_cleanly_and_writes_run_json(tmp_path, fake, make_grue):
    async def main():
        started = asyncio.Event()

        async def hang(n):
            if n == 3:
                started.set()
                await asyncio.sleep(60)

        task = asyncio.create_task(
            play_episode(cfg_for(tmp_path), grue=make_grue(), provider=CostlyProvider(hook=hang), reporter=Reporter(quiet=True))
        )
        await started.wait()
        task.cancel()
        return await task  # the runner swallows the cancellation and returns its summary

    summary = asyncio.run(main())
    assert summary["status"] == "interrupted" and summary["stop_reason"] == "interrupted"
    assert fake.deleted == ["sess-1"] and summary["totals"]["ending"] == "aborted"
    run, lines = read_run(tmp_path)
    assert run["status"] == "interrupted" and len(lines) == 2


def test_initial_run_json_written_before_play(tmp_path, fake, make_grue):
    seen = {}

    async def peek(n):
        seen["status"] = json.loads((tmp_path / "r1" / "run.json").read_text(encoding="utf-8"))["status"]

    play(cfg_for(tmp_path, max_turns=1), make_grue(), CostlyProvider(hook=peek))
    assert seen["status"] == "running"


def test_live_console_output_and_quiet(tmp_path, fake, make_grue, capsys):
    play(cfg_for(tmp_path), make_grue(), ScriptedProvider(WIN_SCRIPT), quiet=False)
    out = capsys.readouterr().out
    assert "You are standing in a field" in out and "\U0001F916 [1] take lamp" in out
    assert "score 10/10" in out and "completed (game_over)" in out and "ending=won" in out
    play(cfg_for(tmp_path, run_id="r2"), make_grue(), ScriptedProvider(WIN_SCRIPT), quiet=True)
    quiet_out = capsys.readouterr().out
    assert "field" not in quiet_out and "ending=won" in quiet_out


def test_notes_mode_recorded_in_transcript(tmp_path, fake, make_grue):
    class Noter(CostlyProvider):
        async def complete(self, system, messages):
            c = await super().complete(system, messages)
            c.text = "NOTES: lamp in field\n> look"
            return c

    play(cfg_for(tmp_path, max_turns=2, notes=True), make_grue(), Noter())
    _, lines = read_run(tmp_path)
    assert lines[0]["notes"] == "lamp in field"


def test_grue_client_errors_and_auth(fake):
    from fake_grue import FakeGrue

    f = FakeGrue(api_key="secret")

    async def go():
        async with GrueClient("http://grue.test", api_key="secret", transport=f.transport()) as g:
            games = await g.list_games()
            obs = await g.start_session("tiny", 5, {"a": 1})
            with pytest.raises(GrueError) as ei:
                await g.command("nope", "look")
            return games, obs, ei.value

    games, obs, err = asyncio.run(go())
    assert [g.id for g in games][:2] == ["tiny", "tiny2"] and games[0].max_score == 10
    assert obs.session_id == "sess-1" and obs.turn == 0 and "field" in obs.text and obs.input_type == "line"
    assert err.status == 404 and err.code == "unknown_session"
    assert set(f.seen_auth) == {"Bearer secret"}

    async def bad_key():
        async with GrueClient("http://grue.test", api_key="wrong", transport=f.transport()) as g:
            await g.list_games()

    with pytest.raises(GrueError, match="unauthorized"):
        asyncio.run(bad_key())


def test_start_session_retries_when_server_is_full(monkeypatch):
    from fake_grue import FakeGrue

    f = FakeGrue(session_cap=1)
    sleeps = []

    async def fake_sleep(d):
        sleeps.append(d)
        f.force_game_over("sess-1", "aborted")  # a slot frees up while we wait

    async def go():
        async with GrueClient("http://grue.test", transport=f.transport(), sleep=fake_sleep) as g:
            await g.start_session("tiny")
            return await g.start_session("tiny")

    obs = asyncio.run(go())
    assert obs.session_id == "sess-2" and sleeps == [2.0]

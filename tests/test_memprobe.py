import asyncio
import json

import pytest

from brass_lantern.bench import BenchConfig, run_bench
from brass_lantern.memprobe import PROBE_MAX_TOKENS, parse_commands, run_probe, verdict
from brass_lantern.providers import Completion, Provider, ProviderError
from brass_lantern.providers.scripted import ScriptedProvider
from brass_lantern.runner import RunConfig, play_episode

pytestmark = pytest.mark.real_probe

WALKTHROUGH = "Here is the walkthrough:\n1. take lamp\n2. north\n3. open chest\n4. take gem"


class Recall(Provider):
    """Answers the probe with `recall`, then plays `look` forever. Records each call's prompt and max_tokens."""

    name = "test"

    def __init__(self, recall="UNKNOWN", fail=False, model="test-model", **kw):
        super().__init__(model, **kw)
        self.recall, self.fail, self.calls = recall, fail, []

    async def complete(self, system, messages, tools=None):
        self.calls.append((messages[-1]["content"], self.max_tokens))
        if len(self.calls) == 1:
            if self.fail:
                raise ProviderError("test: boom")
            return Completion(text=self.recall, input_tokens=100, output_tokens=20, cost_usd=0.01)
        return Completion(text="> look", cost_usd=0.0)


def probe(fake, make_grue, provider, **kw):
    async def go():
        async with make_grue() as grue:
            return await run_probe(grue=grue, provider=provider, game="tiny", model="test:m@high", **kw)

    return asyncio.run(go())


def test_parse_commands_keeps_commands_and_drops_prose():
    text = "Walkthrough:\n```\n1. take lamp\n- north\n> open chest\n* take gem\n```\n\n" + "x" * 90 + "\nUNKNOWN"
    assert parse_commands(text) == ["take lamp", "north", "open chest", "take gem"]
    assert parse_commands("") == [] and parse_commands("UNKNOWN") == []


def test_verdict():
    assert verdict(0, 10, "won") == "positive"
    assert verdict(1, 10, "aborted") == "positive" and verdict(18, 350, "died") == "positive"  # at least 5% of the max
    assert verdict(17, 350, "aborted") == "negative"
    assert verdict(0, 10, "aborted") == "negative" and verdict(None, None, None) == "negative"


def test_a_recalled_walkthrough_that_wins_blind_is_positive(fake, make_grue):
    p = Recall(WALKTHROUGH, max_tokens=1024)
    out = probe(fake, make_grue, p, title="Tiny Cave", author="Ann Author")
    assert out["status"] == "positive" and out["ending"] == "won" and out["peak_score"] == 10
    assert out["commands"] == ["take lamp", "north", "open chest", "take gem"] and out["cost_usd"] == 0.01
    assert out["answer"] == WALKTHROUGH, "the model's answer is kept so the verdict can be checked"
    prompt, max_tokens = p.calls[0]
    assert "Tiny Cave by Ann Author" in prompt and "You are standing in a field" in prompt
    assert max_tokens == PROBE_MAX_TOKENS and p.max_tokens == 1024, "the cap is raised for the probe only"
    session = fake.results[out["session_id"]]
    assert session["meta"]["agent"] == "probe" and session["meta"]["model"] == "test:m"


def test_unknown_or_useless_recall_is_negative_and_the_session_is_closed(fake, make_grue):
    out = probe(fake, make_grue, Recall("UNKNOWN"))
    assert out["status"] == "negative" and out["unknown"] and out["commands"] == []
    assert out["session_id"] in fake.deleted
    out = probe(fake, make_grue, Recall("take lamp\nlook"))
    assert out["status"] == "negative" and out["peak_score"] == 0


def test_probe_errors_and_offline_providers(fake, make_grue):
    out = probe(fake, make_grue, Recall(fail=True))
    assert out["status"] == "error" and "boom" in out["error"] and out["session_id"] in fake.deleted
    before = len(fake.sessions)
    assert probe(fake, make_grue, ScriptedProvider(["look"]))["status"] == "skipped"
    assert len(fake.sessions) == before, "a mock provider never starts a probe session"


def _play(fake, make_grue, tmp_path, provider, **kw):
    cfg = RunConfig(game="tiny", model="test:m", max_turns=3, out_dir=tmp_path, quiet=True, **kw)

    async def go():
        async with make_grue() as grue:
            return await play_episode(cfg, grue=grue, provider=provider)

    return asyncio.run(go())


def test_play_probes_first_and_records_the_verdict(fake, make_grue, tmp_path):
    summary = _play(fake, make_grue, tmp_path, Recall(WALKTHROUGH))
    assert summary["memorization"]["status"] == "positive"
    assert summary["status"] == "completed" and summary["totals"]["turns"] == 3
    probe_sid = summary["memorization"]["session_id"]
    assert probe_sid != summary["session_id"] and fake.sessions[probe_sid]["meta"]["agent"] == "probe"
    on_disk = json.loads((tmp_path / summary["run_id"] / "run.json").read_text(encoding="utf-8"))
    assert on_disk["memorization"]["status"] == "positive"
    assert summary["totals"]["cost_usd"] == 0.0, "the probe is billed in its own record, not the run's totals"


def test_play_can_skip_the_probe(fake, make_grue, tmp_path):
    p = Recall(WALKTHROUGH)
    summary = _play(fake, make_grue, tmp_path, p, memorization_probe=False)
    assert summary["memorization"]["status"] == "skipped" and len(fake.sessions) == 1
    assert all("from memory" not in prompt for prompt, _ in p.calls)


def test_cli_flags():
    from brass_lantern.__main__ import build_parser

    parse = build_parser().parse_args
    assert parse(["play", "--game", "tiny", "--model", "mock:x"]).memorization_probe is True
    assert parse(["play", "--game", "tiny", "--model", "mock:x", "--no-memorization-probe"]).memorization_probe is False
    assert parse(["bench", "--models", "a:b", "--games", "g", "--no-memorization-probe"]).memorization_probe is False
    assert parse(["report", "runs", "--rank-unprobed"]).rank_unprobed is True


def test_bench_probes_each_model_and_game_once(fake, make_grue, tmp_path):
    cfg = BenchConfig(models=["test:a", "test:b"], games=["tiny"], runs=3, max_turns=2, concurrency=4, out_dir=tmp_path)

    async def go():
        async with make_grue() as grue:
            return await run_bench(cfg, grue=grue, provider_factory=lambda spec, **kw: Recall("UNKNOWN", model=spec),
                                   log=lambda _: None)

    summaries, interrupted = asyncio.run(go())
    assert not interrupted and len(summaries) == 6
    probes = [s for s in fake.sessions.values() if s["meta"].get("agent") == "probe"]
    assert sorted(s["meta"]["model"] for s in probes) == ["test:a", "test:b"]
    assert {s["memorization"]["status"] for s in summaries} == {"negative"}
    by_model = {}
    for s in summaries:
        by_model.setdefault(s["model"], set()).add(s["memorization"]["session_id"])
    assert all(len(ids) == 1 for ids in by_model.values()), "repeats share their model's probe"


def test_a_url_game_probe_uses_the_given_max_score(fake, make_grue):
    # Grue reports no max score for this URL game, so a partial recall (5 of 10 points) needs the given one to count.
    url = "https://www.ifarchive.org/if-archive/games/zcode/tiny.z5"
    partial = "take lamp\nnorth\nopen chest"
    out = probe(fake, make_grue, Recall(partial), game_url=url)
    assert out["max_score"] is None and out["peak_score"] == 5 and out["status"] == "negative"
    out = probe(fake, make_grue, Recall(partial), game_url=url, max_score=10)
    assert out["max_score"] == 10 and out["status"] == "positive"

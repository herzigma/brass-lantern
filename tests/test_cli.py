import json

import pytest

import brass_lantern.__main__ as cli
from brass_lantern import __version__
from brass_lantern.grue_client import GrueClient
from brass_lantern.prices import Price
from fake_grue import WIN_SCRIPT


@pytest.mark.parametrize("args", [[], ["games"], ["play"], ["bench"], ["report"]])
def test_help_for_every_command(args, capsys):
    with pytest.raises(SystemExit) as e:
        cli.main([*args, "--help"])
    assert e.value.code == 0
    out = capsys.readouterr().out
    assert "usage: brass-lantern" in out
    if not args:
        for word in ("games", "play", "bench", "report", "provider:model", "openrouter:", "GRUE_URL"):
            assert word in out
    if args == ["play"]:
        for flag in ("--game", "--model", "--max-turns", "--max-cost", "--history", "--notes", "--temperature",
                     "--base-url", "--out", "--quiet", "--seed-meta"):
            assert flag in out
    if args == ["bench"]:
        for flag in ("--models", "--games", "--runs", "--max-turns", "--concurrency", "--max-cost-per-run", "--out"):
            assert flag in out


def test_version_and_missing_args(capsys):
    with pytest.raises(SystemExit):
        cli.main(["--version"])
    assert __version__ in capsys.readouterr().out
    with pytest.raises(SystemExit) as e:
        cli.main([])
    assert e.value.code == 2
    with pytest.raises(SystemExit) as e:
        cli.main(["play", "--game", "tiny"])  # --model missing
    assert e.value.code == 2


@pytest.fixture
def patched_grue(monkeypatch, fake):
    class TestGrue(GrueClient):
        def __init__(self, base_url=None, *a, **kw):
            super().__init__(base_url or "http://grue.test", transport=fake.transport())

    monkeypatch.setattr(cli, "GrueClient", TestGrue)
    return fake


def test_games_command(patched_grue, capsys):
    assert cli.main(["games"]) == 0
    out = capsys.readouterr().out
    assert "tiny" in out and "Tiny Cave" in out and "zcode" in out and "no" in out


def test_play_command_with_mock_script(patched_grue, tmp_path, capsys):
    script = tmp_path / "win.txt"
    script.write_text("\n".join(WIN_SCRIPT), encoding="utf-8")
    code = cli.main(["play", "--game", "tiny", "--model", f"mock:{script}", "--out", str(tmp_path / "runs"),
                     "--run-id", "cli1", "--seed-meta", "label=smoke", "--quiet", "--temperature", "none"])
    assert code == 0
    run = json.loads((tmp_path / "runs" / "cli1" / "run.json").read_text(encoding="utf-8"))
    assert run["totals"]["ending"] == "won" and run["config"]["temperature"] is None
    start = next(b for m, p, b in patched_grue.requests if (m, p) == ("POST", "/api/sessions"))
    assert start["meta"]["label"] == "smoke"
    assert "ending=won" in capsys.readouterr().out


def test_play_reports_config_errors_cleanly(patched_grue, tmp_path, capsys, monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    code = cli.main(["play", "--game", "tiny", "--model", "openrouter:x/y", "--out", str(tmp_path)])
    assert code == 2 and "OPENROUTER_API_KEY" in capsys.readouterr().err
    code = cli.main(["play", "--game", "tiny", "--model", "bogus", "--out", str(tmp_path)])
    assert code == 2 and "Bad model spec" in capsys.readouterr().err
    assert not (tmp_path / "x").exists() and list(tmp_path.iterdir()) == []


def test_bench_command(patched_grue, tmp_path, capsys):
    win = tmp_path / "win.txt"
    win.write_text("\n".join(WIN_SCRIPT), encoding="utf-8")
    idle = tmp_path / "idle.txt"
    idle.write_text("north\n", encoding="utf-8")
    out = tmp_path / "b"
    code = cli.main(["bench", "--models", f"mock:{win},mock:{idle}", "--games", "tiny", "--runs", "2",
                     "--max-turns", "6", "--concurrency", "2", "--out", str(out)])
    assert code == 0
    text = capsys.readouterr().out
    assert "4 episodes" in text and "Overall ranking" in text
    assert (out / "leaderboard.md").exists() and (out / "results.csv").exists() and (out / "summary.json").exists()
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert summary["ranking"][0].endswith("win.txt")


def test_bench_games_all_uses_available_games(patched_grue, tmp_path, capsys):
    idle = tmp_path / "idle.txt"
    idle.write_text("north\n", encoding="utf-8")
    code = cli.main(["bench", "--models", f"mock:{idle}", "--games", "all", "--runs", "1", "--max-turns", "2",
                     "--out", str(tmp_path / "b")])
    assert code == 0
    summary = json.loads((tmp_path / "b" / "summary.json").read_text(encoding="utf-8"))
    assert summary["games"] == ["tiny", "tiny2"]  # the unavailable game is skipped


def test_prices_file(tmp_path):
    f = tmp_path / "p.json"
    f.write_text(json.dumps({"a:b": {"input": 1, "output": 2}, "c:d": [3, 4]}), encoding="utf-8")
    table = cli.load_prices(str(f))
    # the old two-number format falls back to 0.1x / 1.25x of the input price for the cache
    assert table["a:b"] == pytest.approx(Price(1.0, 2.0, 0.1, 1.25)) and table["c:d"] == pytest.approx((3.0, 4.0, 0.3, 3.75))
    assert "anthropic:claude-sonnet-5-5" in table  # the bundled table is always loaded
    assert "a:b" not in cli.load_prices(None)
    f.write_text("nope", encoding="utf-8")
    with pytest.raises(cli.ConfigError):
        cli.load_prices(str(f))


def test_bench_refuses_cost_cap_that_cannot_be_enforced(patched_grue, tmp_path, capsys, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    out = tmp_path / "b"
    args = ["bench", "--models", "anthropic:claude-x,openai:gpt-y,mock:nothing.txt", "--games", "tiny", "--runs", "1",
            "--max-cost-per-run", "0.5", "--out", str(out)]
    assert cli.main(args) == 2
    err = capsys.readouterr().err
    assert "anthropic:claude-x" in err and "openai:gpt-y" in err and "mock:nothing.txt" not in err
    assert "--max-cost-per-run" in err and "--prices" in err
    assert not out.exists() and patched_grue.requests == []  # nothing ran, no session started

    # a price for one of them is not enough: the other is still named
    prices = tmp_path / "p.json"
    prices.write_text(json.dumps({"anthropic:claude-x": {"input": 3, "output": 15}}), encoding="utf-8")
    assert cli.main([*args, "--prices", str(prices)]) == 2
    err = capsys.readouterr().err
    assert "openai:gpt-y" in err and "anthropic:claude-x" not in err


def test_bench_cost_cap_allowed_with_prices_openrouter_or_no_cap(patched_grue, tmp_path, monkeypatch):
    assert cli.uncapped_models(["openrouter:a/b", "mock:x", "human:", "gemini:g", "ollama:l"], {"gemini:g": (1.0, 2.0)}) == ["ollama:l"]
    # no cap requested: any provider is fine (the guard only applies with --max-cost-per-run)
    idle = tmp_path / "idle.txt"
    idle.write_text("north\n", encoding="utf-8")
    assert cli.main(["bench", "--models", f"mock:{idle}", "--games", "tiny", "--runs", "1", "--max-turns", "2",
                     "--max-cost-per-run", "1", "--out", str(tmp_path / "ok")]) == 0

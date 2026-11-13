"""The bundled price table, --prices overrides, cache-aware billing, cost_source, and scripts/update_prices.py."""

import asyncio
import importlib.util
import json
from pathlib import Path

import pytest

import brass_lantern.__main__ as cli
from brass_lantern.console import Reporter
from brass_lantern.grue_client import GrueClient
from brass_lantern.prices import BUNDLED_PRICES, Price, load_prices, parse_prices, price_for, prices_for_models
from brass_lantern.providers import ConfigError
from brass_lantern.providers.base import Completion, Provider
from brass_lantern.runner import RunConfig, play_episode
from fake_grue import WIN_SCRIPT

ROOT = Path(__file__).resolve().parent.parent


def load_script():
    spec = importlib.util.spec_from_file_location("update_prices", ROOT / "scripts" / "update_prices.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Usage(Provider):
    """Every call: 1M prompt tokens (600k cache reads, 100k cache writes) and 500k output tokens."""

    name = "usage"

    def __init__(self, cost=None):
        super().__init__("usage")
        self.cost = cost

    async def complete(self, system, messages, tools=None):
        return Completion(text="> look", input_tokens=1_000_000, output_tokens=500_000, cost_usd=self.cost,
                          cache_read_tokens=600_000, cache_write_tokens=100_000, latency_s=0.1)


def play(tmp_path, grue, provider, **kw):
    cfg = RunConfig(game="tiny", model="anthropic:claude-sonnet-5-5", out_dir=tmp_path, quiet=True, run_id="r1", **kw)
    return asyncio.run(play_episode(cfg, grue=grue, provider=provider, reporter=Reporter(quiet=True)))


# ---------------------------------------------------------------------------
# The bundled table
# ---------------------------------------------------------------------------


def test_bundled_table_has_the_expected_rates_and_keys():
    table = load_prices()
    assert table["anthropic:claude-sonnet-5-5"] == Price(2, 10, 0.1, 2.5)
    assert table["anthropic:claude-opus-5-5"] == Price(4, 20, 0.2, 5)
    assert table["anthropic:claude-haiku-5-5"] == Price(0.1, 0.5, 0.01, 0.125)
    assert table["anthropic:claude-fable-5-2"] == Price(8, 40, 0.2, 10)
    assert table["openai:gpt-6.1-sol"] == Price(2, 10, 0.1, 2.5)
    for spec in ("openai:gpt-6-sol", "openai:gpt-6.1-terra", "openai:gpt-6-luna", "openai:gpt-6-astra",
                 "anthropic:claude-opus-5", "anthropic:claude-sonnet-5", "anthropic:claude-opus-4-8", "anthropic:claude-opus-4-7"):
        assert spec in table
    assert not any(":batch" in k or k.startswith("_") for k in table)  # no batch variants, no metadata
    assert not any("/" in k or k.startswith("openrouter") for k in table)
    assert all(all(x > 0 for x in p) for p in table.values())


def test_the_bundled_file_is_loadable_json_with_the_documented_entry_format():
    raw = json.loads(BUNDLED_PRICES.read_text(encoding="utf-8"))
    assert raw["_meta"]["source"].startswith("https://openrouter.ai")
    assert set(raw["anthropic:claude-sonnet-5-5"]) == {"input", "output", "cache_read", "cache_write"}


def test_prices_file_overrides_and_extends_the_bundled_table(tmp_path):
    f = tmp_path / "mine.json"
    f.write_text(json.dumps({
        "anthropic:claude-sonnet-5-5": {"input": 1, "output": 2, "cache_read": 0.5, "cache_write": 1.5},
        "acme:model": {"input": 3, "output": 4},
    }), encoding="utf-8")
    table = load_prices(str(f))
    assert table["anthropic:claude-sonnet-5-5"] == Price(1, 2, 0.5, 1.5)  # overridden
    assert table["acme:model"] == pytest.approx(Price(3, 4, 0.3, 3.75))  # old format: 0.1x and 1.25x
    assert table["openai:gpt-6.1-sol"] == Price(2, 10, 0.1, 2.5)  # the rest of the bundle is still there


def test_the_old_formats_still_parse_and_bad_entries_are_clear_errors():
    got = parse_prices({"a:b": [1, 2], "c:d": {"input": 10, "output": 20}, "e:f": [1, 2, 3, 4], "_note": "skipped"}, "x")
    assert got["a:b"] == pytest.approx((1, 2, 0.1, 1.25)) and got["c:d"] == pytest.approx((10, 20, 1.0, 12.5))
    assert got["e:f"] == (1, 2, 3, 4) and "_note" not in got
    for bad in ({"a:b": [1]}, {"a:b": {"input": 1}}, {"a:b": {"input": -1, "output": 1}}, {"a:b": "free"}, []):
        with pytest.raises(ConfigError):
            parse_prices(bad, "x")


def test_an_effort_suffix_uses_the_base_models_price():
    table = {"anthropic:claude-sonnet-5-5": Price(2, 10, 0.1, 2.5), "openai:gpt-6.1-sol@high": Price(9, 9, 9, 9),
             "openai:gpt-6.1-sol": Price(2, 10, 0.1, 2.5)}
    base = table["anthropic:claude-sonnet-5-5"]
    assert price_for(table, "anthropic:claude-sonnet-5-5@low") == base == price_for(table, "anthropic:claude-sonnet-5-5")
    assert price_for(table, "openai:gpt-6.1-sol@high") == Price(9, 9, 9, 9)  # an exact entry still wins
    assert price_for(table, "openai:gpt-6.1-sol@low") == table["openai:gpt-6.1-sol"]
    assert price_for(table, "anthropic:unknown@low") is None and price_for(table, "not a spec") is None
    assert prices_for_models(table, ["anthropic:claude-sonnet-5-5@low", "anthropic:claude-sonnet-5-5@high", "mock:x"]) == {
        "anthropic:claude-sonnet-5-5@low": base, "anthropic:claude-sonnet-5-5@high": base}


# ---------------------------------------------------------------------------
# Billing and cost_source
# ---------------------------------------------------------------------------


def test_billing_uses_explicit_cache_rates(tmp_path, fake, make_grue):
    price = Price(2.0, 10.0, 0.1, 2.5)
    # per call: 300k plain * 2 + 600k reads * 0.1 + 100k writes * 2.5 + 500k output * 10
    expected = (300_000 * 2.0 + 600_000 * 0.1 + 100_000 * 2.5 + 500_000 * 10.0) / 1e6
    run = play(tmp_path, make_grue(), Usage(), max_turns=1, price=price)
    assert run["totals"]["cost_usd"] == pytest.approx(expected) and expected == pytest.approx(5.91)
    assert run["totals"]["cost_source"] == "prices"


def test_a_two_number_price_still_bills_the_cache_at_a_tenth_and_one_and_a_quarter(tmp_path, fake, make_grue):
    expected = (485_000 * 2.0 + 500_000 * 4.0) / 1e6
    assert play(tmp_path, make_grue(), Usage(), max_turns=1, price=(2.0, 4.0))["totals"]["cost_usd"] == pytest.approx(expected)


def test_a_provider_reported_cost_wins_over_the_table(tmp_path, fake, make_grue):
    run = play(tmp_path, make_grue(), Usage(cost=0.123), max_turns=1, price=Price(2, 10, 0.1, 2.5))
    assert run["totals"]["cost_usd"] == pytest.approx(0.123) and run["totals"]["cost_source"] == "provider"


def test_cost_source_is_null_without_any_cost(tmp_path, fake, make_grue):
    run = play(tmp_path, make_grue(), Usage(), max_turns=1)
    assert run["totals"]["cost_usd"] is None and run["totals"]["cost_source"] is None


def test_max_cost_works_from_the_bundled_prices_alone(tmp_path, fake, make_grue):
    # $5.91 per call at sonnet 5.5 rates: the second call would pass the $8 cap
    price = price_for(load_prices(), "anthropic:claude-sonnet-5-5@high")
    run = play(tmp_path, make_grue(), Usage(), max_cost=8.0, price=price)
    assert run["stop_reason"] == "max_cost" and run["totals"]["turns"] == 2


# ---------------------------------------------------------------------------
# Cost caps out of the box
# ---------------------------------------------------------------------------


def test_direct_anthropic_and_openai_models_are_capped_without_a_prices_file():
    table = load_prices()
    models = ["anthropic:claude-sonnet-5-5", "anthropic:claude-sonnet-5-5@high", "openai:gpt-6.1-sol@medium",
              "openrouter:x/y", "mock:s.txt"]
    assert cli.uncapped_models(models, table) == []
    assert cli.uncapped_models(["anthropic:claude-3-opus-20240229", "gemini:g", "ollama:l"], table) == [
        "anthropic:claude-3-opus-20240229", "gemini:g", "ollama:l"]


def test_bench_stores_the_prices_of_its_models_and_resume_reads_them_back(tmp_path):
    from brass_lantern.bench import BenchConfig, bench_fields, config_from_bench_json

    cfg = BenchConfig(models=["anthropic:claude-sonnet-5-5@low", "openai:gpt-6.1-sol"], games=["g"], out_dir=tmp_path,
                      prices=prices_for_models(load_prices(), ["anthropic:claude-sonnet-5-5@low", "openai:gpt-6.1-sol"]))
    data = bench_fields(cfg)
    assert set(data["prices"]) == {"anthropic:claude-sonnet-5-5@low", "openai:gpt-6.1-sol"}  # not the whole bundle
    assert data["prices"]["openai:gpt-6.1-sol"] == {"input": 2.0, "output": 10.0, "cache_read": 0.1, "cache_write": 2.5}
    assert config_from_bench_json(data, tmp_path) == cfg
    # a bench.json written before the cache rates existed (two-number lists) still loads
    data["prices"] = {"openai:gpt-6.1-sol": [2.0, 10.0]}
    assert config_from_bench_json(data, tmp_path).prices["openai:gpt-6.1-sol"] == pytest.approx((2.0, 10.0, 0.2, 2.5))
    data["prices"] = {"openai:gpt-6.1-sol": "free"}
    with pytest.raises(ConfigError, match="bad prices"):
        config_from_bench_json(data, tmp_path)


# ---------------------------------------------------------------------------
# scripts/update_prices.py
# ---------------------------------------------------------------------------

OR_MODELS = {"data": [
    {"id": "anthropic/claude-sonnet-5.5", "architecture": {"output_modalities": ["text"]},
     "pricing": {"prompt": "0.000002", "completion": "0.00001", "input_cache_read": "0.0000001", "input_cache_write": "0.0000025"}},
    {"id": "anthropic/claude-sonnet-5.5:batch", "architecture": {"output_modalities": ["text"]},
     "pricing": {"prompt": "0.000001", "completion": "0.000005"}},
    {"id": "openai/gpt-6.1-sol", "architecture": {"output_modalities": ["text"]},
     "pricing": {"prompt": "0.000002", "completion": "0.00001", "input_cache_read": "0.0000001"}},
    {"id": "openai/gpt-5.5-pro", "architecture": {"output_modalities": ["text"]},
     "pricing": {"prompt": "0.00003", "completion": "0.00018"}},
    {"id": "openai/gpt-5.4-image-2", "architecture": {"output_modalities": ["image", "text"]},
     "pricing": {"prompt": "0.000008", "completion": "0.000015"}},
    {"id": "openai/gpt-5.3-codex", "architecture": {"output_modalities": ["text"]}, "pricing": {"prompt": "0.000001", "completion": "0.00001"}},
    {"id": "openai/gpt-oss-120b", "architecture": {"output_modalities": ["text"]}, "pricing": {"prompt": "0.0000001", "completion": "0.0000006"}},
    {"id": "google/gemini-4-hydrogen", "architecture": {"output_modalities": ["text"]}, "pricing": {"prompt": "0.0000015", "completion": "0.0000075"}},
    {"id": "anthropic/claude-haiku-5.5", "architecture": {"output_modalities": ["text"]}, "pricing": {"prompt": "-1", "completion": "-1"}},
]}


def test_update_prices_maps_ids_converts_units_and_skips_variants():
    script = load_script()
    assert script.spec_for("anthropic/claude-sonnet-5.5") == "anthropic:claude-sonnet-5-5"
    assert script.spec_for("anthropic/claude-opus-4.8") == "anthropic:claude-opus-4-8"
    assert script.spec_for("openai/gpt-6.1-sol") == "openai:gpt-6.1-sol"
    assert script.spec_for("anthropic/claude-sonnet-5.5:batch") is None
    assert script.spec_for("google/gemini-4-hydrogen") is None
    assert script.per_million("0.0000025") == 2.5 and script.per_million("0.000000125") == 0.125
    table = script.build(OR_MODELS["data"])
    assert table == {
        "anthropic:claude-sonnet-5-5": {"input": 2.0, "output": 10.0, "cache_read": 0.1, "cache_write": 2.5},
        "openai:gpt-5.5-pro": {"input": 30.0, "output": 180.0, "cache_read": 30.0, "cache_write": 30.0},  # no cache price listed
        "openai:gpt-6.1-sol": {"input": 2.0, "output": 10.0, "cache_read": 0.1, "cache_write": 2.0},
    }


def test_update_prices_writes_a_file_the_loader_reads(tmp_path, monkeypatch):
    script = load_script()
    models = tmp_path / "models.json"
    models.write_text(json.dumps(OR_MODELS), encoding="utf-8")
    out = tmp_path / "prices.json"
    monkeypatch.setattr("sys.argv", ["update_prices.py", "--from-file", str(models), "--out", str(out)])
    assert script.main() == 0
    assert b"\r\n" not in out.read_bytes()
    assert json.loads(out.read_text(encoding="utf-8"))["_meta"]["unit"] == "USD per 1M tokens"
    assert load_prices(str(out))["openai:gpt-6.1-sol"] == Price(2, 10, 0.1, 2.0)
    models.write_text(json.dumps({"data": []}), encoding="utf-8")
    assert script.main() == 1


# ---------------------------------------------------------------------------
# --prices on bench and resume
# ---------------------------------------------------------------------------


@pytest.fixture
def patched_grue(monkeypatch, fake):
    class TestGrue(GrueClient):
        def __init__(self, base_url=None, *a, **kw):
            super().__init__(base_url or "http://grue.test", transport=fake.transport())

    monkeypatch.setattr(cli, "GrueClient", TestGrue)
    return fake


def test_resume_prices_override_the_saved_ones_and_old_bench_json_prices_load(patched_grue, tmp_path, capsys):
    script = tmp_path / "idle.txt"
    script.write_text("north\n", encoding="utf-8")
    spec = f"mock:{script}"
    mine = tmp_path / "mine.json"
    mine.write_text(json.dumps({spec: {"input": 1, "output": 2, "cache_read": 0.5, "cache_write": 1.5}}), encoding="utf-8")
    out = tmp_path / "b"
    assert cli.main(["bench", "--models", spec, "--games", "tiny", "--runs", "1", "--max-turns", "2", "--prices", str(mine),
                     "--out", str(out)]) == 0
    saved = json.loads((out / "bench.json").read_text(encoding="utf-8"))["prices"]
    assert saved == {spec: {"input": 1.0, "output": 2.0, "cache_read": 0.5, "cache_write": 1.5}}  # only this bench's models

    # a bench.json from before the cache rates: a two-number list; --prices on resume replaces it
    data = json.loads((out / "bench.json").read_text(encoding="utf-8"))
    data["prices"] = {spec: [3, 4]}
    data["runs"] = 2
    (out / "bench.json").write_text(json.dumps(data), encoding="utf-8")
    mine.write_text(json.dumps({spec: [5, 6]}), encoding="utf-8")
    assert cli.main(["bench", "--resume", str(out), "--prices", str(mine)]) == 0
    after = json.loads((out / "bench.json").read_text(encoding="utf-8"))
    assert after["prices"][spec] == {"input": 5.0, "output": 6.0, "cache_read": 0.5, "cache_write": 6.25}
    assert after["resumes"][0]["overrides"]["prices"] == [spec]
    capsys.readouterr()

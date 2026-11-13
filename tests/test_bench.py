import asyncio
import csv
import json

import pytest

from brass_lantern.__main__ import main
from brass_lantern.bench import BenchConfig, aggregate, load_runs, render_leaderboard, run_bench, write_reports
from brass_lantern.providers import ConfigError
from brass_lantern.providers.scripted import ScriptedProvider
from fake_grue import WIN_SCRIPT


def make_run(model, game, score, max_score, *, ending="turn_limit", turns=100, invalid=0, pf=0, cost=0.1,
             status="completed", run_id=None, stop_reason=None, config=None):
    return {
        "run_id": run_id or f"{model}-{game}-{score}-{ending}-{status}",
        "model": model, "game": game, "status": status,
        "stop_reason": stop_reason or ("game_over" if status == "completed" else status),
        "config": config or {"temperature": 0.7},
        "totals": {"turns": turns, "score": score, "max_score": max_score, "max_score_reached": score,
                   "ending": ending, "input_tokens": 1000, "output_tokens": 100, "cost_usd": cost,
                   "parse_failures": pf, "invalid_commands": invalid},
    }


RUNS = [
    # model A: zork 100/350 and 200/350 ; tiny 10/10 won
    make_run("A", "zork", 100, 350, invalid=10, turns=100),
    make_run("A", "zork", 200, 350, invalid=30, turns=100, pf=5),
    make_run("A", "tiny", 10, 10, ending="won", turns=4, cost=0.2),
    # model B: zork 35/350 ; tiny 5/10 ; plus an errored run that must not count for scores
    make_run("B", "zork", 35, 350, invalid=0, turns=50, cost=0.05),
    make_run("B", "tiny", 5, 10, turns=20, cost=0.05),
    make_run("B", "zork", 340, 350, status="error", cost=0.5),
    # model C: no cost info, and never played zork (that game must count as 0 in its overall score)
    make_run("C", "tiny", 10, 10, ending="won", turns=6, cost=None),
]


def test_aggregate_math():
    agg = aggregate(RUNS)
    a, b, c = (agg["models"][m] for m in "ABC")
    assert a["games"]["zork"]["mean_score"] == 150 and a["games"]["zork"]["mean_norm"] == pytest.approx(150 / 350)
    assert a["games"]["zork"]["best_score"] == 200 and a["games"]["zork"]["invalid_rate"] == pytest.approx(0.2)
    assert a["games"]["zork"]["parse_failure_rate"] == pytest.approx(0.025)
    assert a["games"]["tiny"]["win_rate"] == 1.0
    # overall: games weigh equally -> mean(150/350, 1.0)
    assert a["overall"]["norm_score"] == pytest.approx((150 / 350 + 1.0) / 2)
    assert a["overall"]["cost_usd"] == pytest.approx(0.4) and a["overall"]["runs"] == 3
    assert a["overall"]["win_rate"] == pytest.approx(1 / 3)
    # errored run excluded from score but its cost counted
    assert b["overall"]["excluded_runs"] == 1 and b["overall"]["runs"] == 2
    assert b["games"]["zork"]["mean_score"] == 35
    assert b["overall"]["cost_usd"] == pytest.approx(0.6)
    assert c["overall"]["cost_usd"] is None and c["overall"]["norm_score"] == pytest.approx(0.5)
    assert c["overall"]["missing_games"] == ["zork"] and a["overall"]["missing_games"] == []
    assert b["overall"]["norm_score"] == pytest.approx((0.1 + 0.5) / 2)
    assert agg["ranking"] == ["A", "C", "B"]
    assert agg["excluded"] == [{"run_id": "B-zork-340-turn_limit-error", "model": "B", "game": "zork", "reason": "error"}]
    assert agg["games"] == ["tiny", "zork"]


def test_aggregate_handles_games_without_max_score_and_empty():
    r = make_run("A", "ifdb", 7, None, ending="ended")
    agg = aggregate([r])
    assert agg["models"]["A"]["overall"]["norm_score"] is None
    assert agg["models"]["A"]["games"]["ifdb"]["mean_score"] == 7
    assert "n/a" in render_leaderboard(agg)
    assert aggregate([])["ranking"] == []
    assert "Overall ranking" in render_leaderboard(aggregate([]))


def test_leaderboard_markdown():
    md = render_leaderboard(aggregate(RUNS))
    assert md.startswith("# Brass Lantern Leaderboard")
    assert "## Overall ranking" in md and "## Mean score by game" in md and "## zork (max score 350)" in md
    rows = [ln for ln in md.splitlines() if ln.startswith("| ") and "`" in ln]
    overall = rows[:3]
    assert overall[0].startswith("| 1 | `A`") and overall[1].startswith("| 2 | `C`") and overall[2].startswith("| 3 | `B`")
    assert "71.4%" in overall[0]  # (150/350 + 1)/2 = 0.7142...
    assert r"50.0% \*" in overall[1] and "- (=0)" in md  # C: zork never played -> counted as 0, marked
    assert "150.0 (43%)" in md  # score matrix cell
    assert "2 (0)" not in md and "2 (1)" in overall[2]  # B: 2 counted runs, 1 excluded
    assert "n/a" in overall[1]  # C has no cost
    assert "## Excluded runs" in md and "B-zork-340-turn_limit-error" in md
    assert "mean over ALL games" in md  # the formula is documented at the top
    assert md.index("Score formula") < md.index("## Overall ranking")


def test_write_reports_and_report_cli(tmp_path, capsys):
    for i, r in enumerate(RUNS):
        d = tmp_path / "nested" / f"run{i}"
        d.mkdir(parents=True)
        (d / "run.json").write_text(json.dumps(r), encoding="utf-8")
    (tmp_path / "junk").mkdir()
    (tmp_path / "junk" / "run.json").write_text("not json", encoding="utf-8")
    assert len(load_runs(tmp_path)) == len(RUNS)

    assert main(["report", str(tmp_path)]) == 0
    assert "Brass Lantern Leaderboard" in capsys.readouterr().out
    assert (tmp_path / "leaderboard.md").read_text(encoding="utf-8").startswith("# Brass Lantern")
    rows = list(csv.DictReader((tmp_path / "results.csv").open(encoding="utf-8")))
    assert len(rows) == len(RUNS) and {"model", "game", "score", "norm", "cost_usd", "ending"} <= set(rows[0])
    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert summary["ranking"] == ["A", "C", "B"] and summary["n_runs"] == len(RUNS)
    # regenerating twice is stable
    assert main(["report", str(tmp_path)]) == 0


def test_report_cli_empty_dir(tmp_path, capsys):
    assert main(["report", str(tmp_path)]) == 1
    assert "no run.json" in capsys.readouterr().err


def scripted_factory(scripts):
    def factory(spec, **kw):
        return ScriptedProvider(scripts[spec], model=spec)

    return factory


def test_run_bench_matrix_end_to_end(tmp_path, fake, make_grue):
    cfg = BenchConfig(models=["mock:win", "mock:idle"], games=["tiny", "tiny2"], runs=2, max_turns=6,
                      concurrency=3, out_dir=tmp_path / "bench")
    lines: list[str] = []
    factory = scripted_factory({"mock:win": WIN_SCRIPT, "mock:idle": ["look"]})
    summaries, interrupted = asyncio.run(run_bench(cfg, grue=make_grue(), provider_factory=factory, log=lines.append))
    assert not interrupted and len(summaries) == 8 and len(lines) == 8
    assert lines[-1].startswith("[8/8]")
    assert len({s["run_id"] for s in summaries}) == 8
    agg = write_reports(cfg.out_dir, load_runs(cfg.out_dir))
    assert agg["ranking"] == ["mock:win", "mock:idle"]
    win, idle = agg["models"]["mock:win"]["overall"], agg["models"]["mock:idle"]["overall"]
    assert win["norm_score"] == 1.0 and win["win_rate"] == 1.0 and win["mean_turns"] == 4
    assert idle["norm_score"] == 0.0 and idle["win_rate"] == 0.0 and idle["mean_turns"] == 6
    assert idle["invalid_rate"] == pytest.approx(1.0)  # `look` is not a verb in the fake game
    assert (cfg.out_dir / "leaderboard.md").exists() and (cfg.out_dir / "results.csv").exists()
    # each run is stored in its own folder with the full artefacts
    run_dirs = [d for d in cfg.out_dir.iterdir() if d.is_dir()]
    assert len(run_dirs) == 8 and all((d / "run.json").exists() and (d / "transcript.jsonl").exists() for d in run_dirs)
    meta = json.loads((run_dirs[0] / "run.json").read_text(encoding="utf-8"))["config"]["meta"]
    assert meta["bench"] == "bench" and meta["repeat"] in (1, 2)


def test_run_bench_respects_concurrency(tmp_path, fake, make_grue):
    active = {"now": 0, "max": 0}

    class Slow(ScriptedProvider):
        async def complete(self, system, messages):
            active["now"] += 1
            active["max"] = max(active["max"], active["now"])
            await asyncio.sleep(0.01)
            active["now"] -= 1
            return await super().complete(system, messages)

    cfg = BenchConfig(models=["m"], games=["tiny"], runs=6, max_turns=3, concurrency=2, out_dir=tmp_path)
    asyncio.run(run_bench(cfg, grue=make_grue(), provider_factory=lambda spec, **kw: Slow(["look"]), log=lambda s: None))
    assert active["max"] == 2


def test_run_bench_fails_fast_on_bad_spec(tmp_path, fake, make_grue):
    from brass_lantern.providers import create_provider

    cfg = BenchConfig(models=["openrouter:x/y"], games=["tiny"], runs=1, out_dir=tmp_path)
    import os

    old = os.environ.pop("OPENROUTER_API_KEY", None)
    try:
        with pytest.raises(ConfigError, match="OPENROUTER_API_KEY"):
            asyncio.run(run_bench(cfg, grue=make_grue(), provider_factory=create_provider, log=print))
    finally:
        if old is not None:
            os.environ["OPENROUTER_API_KEY"] = old
    assert not any(tmp_path.iterdir())


def test_run_bench_interrupt_reports_partial(tmp_path, fake, make_grue):
    class Hanging(ScriptedProvider):
        async def complete(self, system, messages):
            await asyncio.sleep(60)

    async def main_():
        cfg = BenchConfig(models=["m"], games=["tiny"], runs=3, max_turns=3, concurrency=2, out_dir=tmp_path)
        task = asyncio.create_task(
            run_bench(cfg, grue=make_grue(), provider_factory=lambda spec, **kw: Hanging([]), log=lambda s: None)
        )
        await asyncio.sleep(0.2)
        task.cancel()
        try:
            return await task
        except asyncio.CancelledError:
            return None

    asyncio.run(main_())
    runs = load_runs(tmp_path)
    assert len(runs) == 2 and all(r["status"] == "interrupted" for r in runs)
    assert len(fake.deleted) == 2


def test_completed_run_without_score_but_known_max_counts_as_zero():
    runs = [make_run("A", "g", None, 10), make_run("A", "g", 10, 10, ending="won"), make_run("A", "h", None, None)]
    agg = aggregate(runs)
    g = agg["models"]["A"]["games"]["g"]
    assert g["mean_score"] == 5 and g["mean_norm"] == pytest.approx(0.5) and g["runs"] == 2
    assert "h" in agg["models"]["A"]["games"] and agg["models"]["A"]["games"]["h"]["mean_score"] is None  # no max: stays unknown
    assert agg["rows"][0]["score"] == 0 and agg["rows"][0]["norm"] == 0.0


def test_provider_error_runs_count_if_played_and_are_flagged_harness_errors_are_excluded():
    runs = [
        make_run("A", "g", 6, 10, status="error", stop_reason="provider_error", turns=7, run_id="pe-played"),
        make_run("A", "g", 0, 10, status="error", stop_reason="provider_error", turns=0, run_id="pe-none"),
        make_run("A", "g", 9, 10, status="error", stop_reason="grue_error", run_id="grue"),
        make_run("A", "g", 9, 10, status="interrupted", stop_reason="interrupted", run_id="intr"),
        make_run("A", "g", 9, 10, status="error", stop_reason="error", run_id="boom"),
        make_run("A", "g", 4, 10, run_id="ok"),
    ]
    agg = aggregate(runs)
    o, g = agg["models"]["A"]["overall"], agg["models"]["A"]["games"]["g"]
    assert o["runs"] == 2 and o["excluded_runs"] == 4 and o["provider_errors"] == 1 and g["provider_errors"] == 1
    assert g["mean_score"] == 5  # (6 + 4) / 2: the provider-error run counts at the score it reached
    reasons = {e["run_id"]: e["reason"] for e in agg["excluded"]}
    assert reasons == {"pe-none": "provider_error (no turns played)", "grue": "grue_error",
                       "intr": "interrupted", "boom": "error"}
    md = render_leaderboard(agg)
    assert "| Errors |" in md and "## Excluded runs" in md
    assert "| grue |" in md and "grue_error" in md and "pe-played" not in md
    overall_row = next(ln for ln in md.splitlines() if ln.startswith("| 1 |"))
    assert overall_row.endswith("| 2 (4) | 1 |")
    # no exclusions -> no section
    assert "## Excluded runs" not in render_leaderboard(aggregate([make_run("A", "g", 1, 10)]))


def test_overall_ranking_averages_over_all_games_so_failing_hard_games_cannot_help():
    runs = [
        # X aces the easy game and never gets a valid run on the hard one; Y is decent on both.
        make_run("X", "easy", 10, 10, ending="won"),
        make_run("X", "hard", 3, 100, status="error", stop_reason="grue_error"),
        make_run("Y", "easy", 6, 10),
        make_run("Y", "hard", 70, 100),
    ]
    agg = aggregate(runs)
    x, y = agg["models"]["X"]["overall"], agg["models"]["Y"]["overall"]
    assert x["norm_score"] == pytest.approx(0.5) and x["missing_games"] == ["hard"]
    assert y["norm_score"] == pytest.approx(0.65) and y["missing_games"] == []
    assert agg["ranking"] == ["Y", "X"]


def test_config_variants_are_separate_rows_and_identical_configs_merge():
    base = {"temperature": 0.7, "history": 30, "notes": False, "reasoning": None, "max_tokens": 1024, "max_turns": 150}
    runs = [
        make_run("m", "g", 2, 10, config=base, run_id="a"),
        make_run("m", "g", 4, 10, config=dict(base), run_id="b"),  # same config: merges with "a"
        make_run("m", "g", 8, 10, config={**base, "notes": True}, run_id="c"),
        make_run("m", "g", 6, 10, config={**base, "notes": True, "history": 10}, run_id="d"),
        make_run("solo", "g", 5, 10, config=base, run_id="e"),
    ]
    agg = aggregate(runs)
    assert set(agg["models"]) == {"m (no notes, h=30)", "m (notes, h=30)", "m (notes, h=10)", "solo"}
    plain = agg["models"]["m (no notes, h=30)"]
    assert plain["overall"]["runs"] == 2 and plain["overall"]["norm_score"] == pytest.approx(0.3) and plain["spec"] == "m"
    assert agg["models"]["m (notes, h=30)"]["config"]["notes"] is True
    assert {r["run_id"]: r["variant"] for r in agg["rows"]}["c"] == "m (notes, h=30)"
    assert agg["ranking"][0] == "m (notes, h=30)"
    md = render_leaderboard(agg)
    assert "`m (notes, h=10)`" in md and "`solo`" in md  # a single-config spec keeps its plain name
    # only temperature differs: that is what the label shows
    t = aggregate([make_run("m", "g", 1, 10, config={**base, "temperature": None}), make_run("m", "g", 1, 10, config=base)])
    assert set(t["models"]) == {"m (t=none)", "m (t=0.7)"}


def test_write_reports_csv_has_variant_and_exclusion_columns(tmp_path):
    write_reports(tmp_path, RUNS)
    rows = list(csv.DictReader((tmp_path / "results.csv").open(encoding="utf-8")))
    assert {"variant", "counted", "excluded_reason", "notes", "history"} <= set(rows[0])
    bad = next(r for r in rows if r["status"] == "error")
    assert bad["counted"] == "False" and bad["excluded_reason"] == "error"

# Brass Lantern Leaderboard

6 runs, 3 models, 2 games. Generated 2026-11-13T04:17:36+00:00.

**Score formula.** A run's score is its final in-game score divided by the game's maximum (a finished run that never reported a score counts as 0; games without a maximum are left out of the ranking). A model's per-game score is the mean over its valid runs in that game. Its overall score is the mean over ALL games in the bench that have a maximum score: a game where the model has no valid run counts as 0 and is marked `*`, so failing on hard games cannot raise its rank.

**Valid runs.** Completed runs, including those stopped by turn, time or cost limits, plus runs that ended in a `provider_error` after at least one turn (scored as reached and counted in Errors). Runs that hit a harness problem (`grue_error`, `interrupted`, `error`) are excluded and listed at the end; their cost still counts. Rows with the same model spec but different settings (temperature, history, notes, reasoning, max tokens, max turns) are separate rows, labelled with the differing settings.

Win = game ended `won`. Invalid % = parser rejections per command (server-reported). Format fail % = replies without a `> COMMAND` line. Errors = valid runs that ended in a provider error.

## Overall ranking

| # | Model | Score | Win rate | Mean turns | Invalid % | Format fail % | Cost | Runs (excl.) | Errors |
|--:|---|--:|--:|--:|--:|--:|--:|--:|--:|
| 1 | `openrouter:openai/gpt-6.1-sol` | 100.0% | 100% | 86 | 0.0% | 0.0% | $1.01 | 2 (0) | 0 |
| 2 | `openrouter:google/gemini-4-argon` | 91.0% | 50% | 88 | 2.8% | 0.0% | $1.80 | 2 (0) | 0 |
| 3 | `openrouter:anthropic/claude-sonnet-5.5` | 82.0% | 50% | 90 | 1.7% | 0.6% | $1.26 | 2 (0) | 0 |

## Mean score by game

| Model | lantern-lane | skerrin-light |
|---|--:|--:|
| `openrouter:openai/gpt-6.1-sol` | 10.0 (100%) | 50.0 (100%) |
| `openrouter:google/gemini-4-argon` | 10.0 (100%) | 41.0 (82%) |
| `openrouter:anthropic/claude-sonnet-5.5` | 10.0 (100%) | 32.0 (64%) |

## lantern-lane (max score 10)

| Model | Mean score | % of max | Best | Win rate | Mean turns | Invalid % | Cost | Runs | Errors |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| `openrouter:openai/gpt-6.1-sol` | 10.0 | 100.0% | 10 | 100% | 26 | 0.0% | $0.0217 | 1 | 0 |
| `openrouter:google/gemini-4-argon` | 10.0 | 100.0% | 10 | 100% | 26 | 0.0% | $0.1316 | 1 | 0 |
| `openrouter:anthropic/claude-sonnet-5.5` | 10.0 | 100.0% | 10 | 100% | 30 | 0.0% | $0.1441 | 1 | 0 |

## skerrin-light (max score 50)

| Model | Mean score | % of max | Best | Win rate | Mean turns | Invalid % | Cost | Runs | Errors |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| `openrouter:openai/gpt-6.1-sol` | 50.0 | 100.0% | 50 | 100% | 146 | 0.0% | $0.9915 | 1 | 0 |
| `openrouter:google/gemini-4-argon` | 41.0 | 82.0% | 41 | 0% | 150 | 3.3% | $1.66 | 1 | 0 |
| `openrouter:anthropic/claude-sonnet-5.5` | 32.0 | 64.0% | 32 | 0% | 150 | 2.0% | $1.12 | 1 | 0 |

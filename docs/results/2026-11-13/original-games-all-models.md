# Brass Lantern Leaderboard

30 runs, 9 models, 2 games. Generated 2026-11-13T04:43:56+00:00.

**Score formula.** A run's score is its final in-game score divided by the game's maximum (a finished run that never reported a score counts as 0; games without a maximum are left out of the ranking). A model's per-game score is the mean over its valid runs in that game. Its overall score is the mean over ALL games in the bench that have a maximum score: a game where the model has no valid run counts as 0 and is marked `*`, so failing on hard games cannot raise its rank.

**Valid runs.** Completed runs, including those stopped by turn, time or cost limits, plus runs that ended in a `provider_error` after at least one turn (scored as reached and counted in Errors). Runs that hit a harness problem (`grue_error`, `interrupted`, `error`) are excluded and listed at the end; their cost still counts. Rows with the same model spec but different settings (temperature, history, notes, reasoning, max tokens, max turns) are separate rows, labelled with the differing settings.

Win = game ended `won`. Invalid % = parser rejections per command (server-reported). Format fail % = replies without a `> COMMAND` line. Errors = valid runs that ended in a provider error.

## Overall ranking

| # | Model | Score | Win rate | Mean turns | Invalid % | Format fail % | Cost | Runs (excl.) | Errors |
|--:|---|--:|--:|--:|--:|--:|--:|--:|--:|
| 1 | `openrouter:openai/gpt-6.1-sol` | 100.0% | 100% | 86 | 0.0% | 0.0% | $1.01 | 2 (0) | 0 |
| 2 | `openrouter:google/gemini-4-argon` | 91.0% | 50% | 88 | 2.8% | 0.0% | $1.80 | 2 (0) | 0 |
| 3 | `openrouter:anthropic/claude-sonnet-5.5` | 82.0% | 50% | 90 | 1.7% | 0.6% | $1.26 | 2 (0) | 0 |
| 4 | `openrouter:google/gemini-3.5-flash-lite` | 81.3% | 25% | 120 | 3.1% | 1.7% | $0.4650 | 4 (0) | 0 |
| 5 | `openrouter:openai/gpt-6-luna` | 72.7% | 25% | 120 | 1.9% | 0.0% | $0.1382 | 4 (0) | 0 |
| 6 | `openrouter:qwen/qwen3.8-flash` | 71.3% | 25% | 120 | 2.5% | 0.4% | $0.1889 | 4 (0) | 0 |
| 7 | `openrouter:anthropic/claude-haiku-5.5` | 69.7% | 25% | 119 | 1.5% | 0.6% | $0.2283 | 4 (0) | 0 |
| 8 | `openrouter:z-ai/glm-5.3-flash` | 69.0% | 25% | 119 | 2.5% | 0.0% | $0.0415 | 4 (0) | 0 |
| 9 | `openrouter:deepseek/deepseek-v4-flash` | 21.0% | 0% | 150 | 63.5% | 19.8% | $0.2346 | 4 (0) | 0 |

## Mean score by game

| Model | lantern-lane | skerrin-light |
|---|--:|--:|
| `openrouter:openai/gpt-6.1-sol` | 10.0 (100%) | 50.0 (100%) |
| `openrouter:google/gemini-4-argon` | 10.0 (100%) | 41.0 (82%) |
| `openrouter:anthropic/claude-sonnet-5.5` | 10.0 (100%) | 32.0 (64%) |
| `openrouter:google/gemini-3.5-flash-lite` | 10.0 (100%) | 31.3 (63%) |
| `openrouter:openai/gpt-6-luna` | 10.0 (100%) | 22.7 (45%) |
| `openrouter:qwen/qwen3.8-flash` | 10.0 (100%) | 21.3 (43%) |
| `openrouter:anthropic/claude-haiku-5.5` | 10.0 (100%) | 19.7 (39%) |
| `openrouter:z-ai/glm-5.3-flash` | 10.0 (100%) | 19.0 (38%) |
| `openrouter:deepseek/deepseek-v4-flash` | 4.0 (40%) | 1.0 (2%) |

## lantern-lane (max score 10)

| Model | Mean score | % of max | Best | Win rate | Mean turns | Invalid % | Cost | Runs | Errors |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| `openrouter:openai/gpt-6.1-sol` | 10.0 | 100.0% | 10 | 100% | 26 | 0.0% | $0.0217 | 1 | 0 |
| `openrouter:google/gemini-4-argon` | 10.0 | 100.0% | 10 | 100% | 26 | 0.0% | $0.1316 | 1 | 0 |
| `openrouter:anthropic/claude-sonnet-5.5` | 10.0 | 100.0% | 10 | 100% | 30 | 0.0% | $0.1441 | 1 | 0 |
| `openrouter:google/gemini-3.5-flash-lite` | 10.0 | 100.0% | 10 | 100% | 29 | 6.9% | $0.0212 | 1 | 0 |
| `openrouter:openai/gpt-6-luna` | 10.0 | 100.0% | 10 | 100% | 30 | 0.0% | $0.0016 | 1 | 0 |
| `openrouter:qwen/qwen3.8-flash` | 10.0 | 100.0% | 10 | 100% | 30 | 0.0% | $0.0024 | 1 | 0 |
| `openrouter:anthropic/claude-haiku-5.5` | 10.0 | 100.0% | 10 | 100% | 27 | 0.0% | $0.0071 | 1 | 0 |
| `openrouter:z-ai/glm-5.3-flash` | 10.0 | 100.0% | 10 | 100% | 26 | 0.0% | $0.0012 | 1 | 0 |
| `openrouter:deepseek/deepseek-v4-flash` | 4.0 | 40.0% | 4 | 0% | 150 | 60.7% | $0.0612 | 1 | 0 |

## skerrin-light (max score 50)

| Model | Mean score | % of max | Best | Win rate | Mean turns | Invalid % | Cost | Runs | Errors |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| `openrouter:openai/gpt-6.1-sol` | 50.0 | 100.0% | 50 | 100% | 146 | 0.0% | $0.9915 | 1 | 0 |
| `openrouter:google/gemini-4-argon` | 41.0 | 82.0% | 41 | 0% | 150 | 3.3% | $1.66 | 1 | 0 |
| `openrouter:anthropic/claude-sonnet-5.5` | 32.0 | 64.0% | 32 | 0% | 150 | 2.0% | $1.12 | 1 | 0 |
| `openrouter:google/gemini-3.5-flash-lite` | 31.3 | 62.7% | 41 | 0% | 150 | 2.9% | $0.4438 | 3 | 0 |
| `openrouter:openai/gpt-6-luna` | 22.7 | 45.3% | 30 | 0% | 150 | 2.0% | $0.1367 | 3 | 0 |
| `openrouter:qwen/qwen3.8-flash` | 21.3 | 42.7% | 26 | 0% | 150 | 2.7% | $0.1864 | 3 | 0 |
| `openrouter:anthropic/claude-haiku-5.5` | 19.7 | 39.3% | 21 | 0% | 150 | 1.6% | $0.2212 | 3 | 0 |
| `openrouter:z-ai/glm-5.3-flash` | 19.0 | 38.0% | 20 | 0% | 150 | 2.7% | $0.0404 | 3 | 0 |
| `openrouter:deepseek/deepseek-v4-flash` | 1.0 | 2.0% | 1 | 0% | 150 | 64.4% | $0.1735 | 3 | 0 |

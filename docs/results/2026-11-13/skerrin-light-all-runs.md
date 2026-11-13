# Brass Lantern Leaderboard

21 runs, 9 models, 1 games. Generated 2026-11-13T04:43:21+00:00.

**Score formula.** A run's score is its final in-game score divided by the game's maximum (a finished run that never reported a score counts as 0; games without a maximum are left out of the ranking). A model's per-game score is the mean over its valid runs in that game. Its overall score is the mean over ALL games in the bench that have a maximum score: a game where the model has no valid run counts as 0 and is marked `*`, so failing on hard games cannot raise its rank.

**Valid runs.** Completed runs, including those stopped by turn, time or cost limits, plus runs that ended in a `provider_error` after at least one turn (scored as reached and counted in Errors). Runs that hit a harness problem (`grue_error`, `interrupted`, `error`) are excluded and listed at the end; their cost still counts. Rows with the same model spec but different settings (temperature, history, notes, reasoning, max tokens, max turns) are separate rows, labelled with the differing settings.

Win = game ended `won`. Invalid % = parser rejections per command (server-reported). Format fail % = replies without a `> COMMAND` line. Errors = valid runs that ended in a provider error.

## Overall ranking

| # | Model | Score | Win rate | Mean turns | Invalid % | Format fail % | Cost | Runs (excl.) | Errors |
|--:|---|--:|--:|--:|--:|--:|--:|--:|--:|
| 1 | `openrouter:openai/gpt-6.1-sol` | 100.0% | 100% | 146 | 0.0% | 0.0% | $0.9915 | 1 (0) | 0 |
| 2 | `openrouter:google/gemini-4-argon` | 82.0% | 0% | 150 | 3.3% | 0.0% | $1.66 | 1 (0) | 0 |
| 3 | `openrouter:anthropic/claude-sonnet-5.5` | 64.0% | 0% | 150 | 2.0% | 0.7% | $1.12 | 1 (0) | 0 |
| 4 | `openrouter:google/gemini-3.5-flash-lite` | 62.7% | 0% | 150 | 2.9% | 1.3% | $0.4438 | 3 (0) | 0 |
| 5 | `openrouter:openai/gpt-6-luna` | 45.3% | 0% | 150 | 2.0% | 0.0% | $0.1367 | 3 (0) | 0 |
| 6 | `openrouter:qwen/qwen3.8-flash` | 42.7% | 0% | 150 | 2.7% | 0.4% | $0.1864 | 3 (0) | 0 |
| 7 | `openrouter:anthropic/claude-haiku-5.5` | 39.3% | 0% | 150 | 1.6% | 0.7% | $0.2212 | 3 (0) | 0 |
| 8 | `openrouter:z-ai/glm-5.3-flash` | 38.0% | 0% | 150 | 2.7% | 0.0% | $0.0404 | 3 (0) | 0 |
| 9 | `openrouter:deepseek/deepseek-v4-flash` | 2.0% | 0% | 150 | 64.4% | 19.3% | $0.1735 | 3 (0) | 0 |

## Mean score by game

| Model | skerrin-light |
|---|--:|
| `openrouter:openai/gpt-6.1-sol` | 50.0 (100%) |
| `openrouter:google/gemini-4-argon` | 41.0 (82%) |
| `openrouter:anthropic/claude-sonnet-5.5` | 32.0 (64%) |
| `openrouter:google/gemini-3.5-flash-lite` | 31.3 (63%) |
| `openrouter:openai/gpt-6-luna` | 22.7 (45%) |
| `openrouter:qwen/qwen3.8-flash` | 21.3 (43%) |
| `openrouter:anthropic/claude-haiku-5.5` | 19.7 (39%) |
| `openrouter:z-ai/glm-5.3-flash` | 19.0 (38%) |
| `openrouter:deepseek/deepseek-v4-flash` | 1.0 (2%) |

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

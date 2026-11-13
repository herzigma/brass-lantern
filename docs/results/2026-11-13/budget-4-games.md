# Brass Lantern Leaderboard

24 runs, 6 models, 4 games. Generated 2026-11-13T04:27:33+00:00.

**Score formula.** A run's score is its final in-game score divided by the game's maximum (a finished run that never reported a score counts as 0; games without a maximum are left out of the ranking). A model's per-game score is the mean over its valid runs in that game. Its overall score is the mean over ALL games in the bench that have a maximum score: a game where the model has no valid run counts as 0 and is marked `*`, so failing on hard games cannot raise its rank.

**Valid runs.** Completed runs, including those stopped by turn, time or cost limits, plus runs that ended in a `provider_error` after at least one turn (scored as reached and counted in Errors). Runs that hit a harness problem (`grue_error`, `interrupted`, `error`) are excluded and listed at the end; their cost still counts. Rows with the same model spec but different settings (temperature, history, notes, reasoning, max tokens, max turns) are separate rows, labelled with the differing settings.

Win = game ended `won`. Invalid % = parser rejections per command (server-reported). Format fail % = replies without a `> COMMAND` line. Errors = valid runs that ended in a provider error.

## Overall ranking

| # | Model | Score | Win rate | Mean turns | Invalid % | Format fail % | Cost | Runs (excl.) | Errors |
|--:|---|--:|--:|--:|--:|--:|--:|--:|--:|
| 1 | `openrouter:google/gemini-3.5-flash-lite` | 55.5% | 25% | 120 | 2.1% | 2.5% | $0.5017 | 4 (0) | 0 |
| 2 | `openrouter:openai/gpt-6-luna` | 50.3% | 25% | 120 | 1.2% | 0.2% | $0.1279 | 4 (0) | 0 |
| 3 | `openrouter:anthropic/claude-haiku-5.5` | 41.9% | 25% | 119 | 0.8% | 0.8% | $0.2146 | 4 (0) | 0 |
| 4 | `openrouter:z-ai/glm-5.3-flash` | 41.1% | 25% | 119 | 1.5% | 0.0% | $0.0431 | 4 (0) | 0 |
| 5 | `openrouter:qwen/qwen3.8-flash` | 40.9% | 25% | 120 | 2.7% | 19.2% | $0.2930 | 4 (0) | 0 |
| 6 | `openrouter:deepseek/deepseek-v4-flash` | 16.9% | 0% | 150 | 36.8% | 14.7% | $0.1897 | 4 (0) | 0 |

## Mean score by game

| Model | lantern-lane | lost-pig | skerrin-light | zork1 |
|---|--:|--:|--:|--:|
| `openrouter:google/gemini-3.5-flash-lite` | 10.0 (100%) | 2.0 (29%) | 41.0 (82%) | 40.0 (11%) |
| `openrouter:openai/gpt-6-luna` | 10.0 (100%) | 2.0 (29%) | 30.0 (60%) | 44.0 (13%) |
| `openrouter:anthropic/claude-haiku-5.5` | 10.0 (100%) | 1.0 (14%) | 19.0 (38%) | 54.0 (15%) |
| `openrouter:z-ai/glm-5.3-flash` | 10.0 (100%) | 1.0 (14%) | 18.0 (36%) | 50.0 (14%) |
| `openrouter:qwen/qwen3.8-flash` | 10.0 (100%) | 1.0 (14%) | 19.0 (38%) | 40.0 (11%) |
| `openrouter:deepseek/deepseek-v4-flash` | 4.0 (40%) | 1.0 (14%) | 1.0 (2%) | 40.0 (11%) |

## lantern-lane (max score 10)

| Model | Mean score | % of max | Best | Win rate | Mean turns | Invalid % | Cost | Runs | Errors |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| `openrouter:google/gemini-3.5-flash-lite` | 10.0 | 100.0% | 10 | 100% | 29 | 6.9% | $0.0212 | 1 | 0 |
| `openrouter:openai/gpt-6-luna` | 10.0 | 100.0% | 10 | 100% | 30 | 0.0% | $0.0016 | 1 | 0 |
| `openrouter:anthropic/claude-haiku-5.5` | 10.0 | 100.0% | 10 | 100% | 27 | 0.0% | $0.0071 | 1 | 0 |
| `openrouter:z-ai/glm-5.3-flash` | 10.0 | 100.0% | 10 | 100% | 26 | 0.0% | $0.0012 | 1 | 0 |
| `openrouter:qwen/qwen3.8-flash` | 10.0 | 100.0% | 10 | 100% | 30 | 0.0% | $0.0024 | 1 | 0 |
| `openrouter:deepseek/deepseek-v4-flash` | 4.0 | 40.0% | 4 | 0% | 150 | 60.7% | $0.0612 | 1 | 0 |

## lost-pig (max score 7)

| Model | Mean score | % of max | Best | Win rate | Mean turns | Invalid % | Cost | Runs | Errors |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| `openrouter:google/gemini-3.5-flash-lite` | 2.0 | 28.6% | 2 | 0% | 150 | 0.0% | $0.2008 | 1 | 0 |
| `openrouter:openai/gpt-6-luna` | 2.0 | 28.6% | 2 | 0% | 150 | 0.0% | $0.0443 | 1 | 0 |
| `openrouter:anthropic/claude-haiku-5.5` | 1.0 | 14.3% | 1 | 0% | 150 | 0.0% | $0.0757 | 1 | 0 |
| `openrouter:z-ai/glm-5.3-flash` | 1.0 | 14.3% | 1 | 0% | 150 | 0.0% | $0.0191 | 1 | 0 |
| `openrouter:qwen/qwen3.8-flash` | 1.0 | 14.3% | 1 | 0% | 150 | 0.0% | $0.1340 | 1 | 0 |
| `openrouter:deepseek/deepseek-v4-flash` | 1.0 | 14.3% | 1 | 0% | 150 | 0.0% | $0.0509 | 1 | 0 |

## skerrin-light (max score 50)

| Model | Mean score | % of max | Best | Win rate | Mean turns | Invalid % | Cost | Runs | Errors |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| `openrouter:google/gemini-3.5-flash-lite` | 41.0 | 82.0% | 41 | 0% | 150 | 0.7% | $0.1586 | 1 | 0 |
| `openrouter:openai/gpt-6-luna` | 30.0 | 60.0% | 30 | 0% | 150 | 2.0% | $0.0460 | 1 | 0 |
| `openrouter:anthropic/claude-haiku-5.5` | 19.0 | 38.0% | 19 | 0% | 150 | 2.0% | $0.0705 | 1 | 0 |
| `openrouter:qwen/qwen3.8-flash` | 19.0 | 38.0% | 19 | 0% | 150 | 1.3% | $0.0664 | 1 | 0 |
| `openrouter:z-ai/glm-5.3-flash` | 18.0 | 36.0% | 18 | 0% | 150 | 2.0% | $0.0136 | 1 | 0 |
| `openrouter:deepseek/deepseek-v4-flash` | 1.0 | 2.0% | 1 | 0% | 150 | 64.0% | $0.0373 | 1 | 0 |

## zork1 (max score 350)

| Model | Mean score | % of max | Best | Win rate | Mean turns | Invalid % | Cost | Runs | Errors |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| `openrouter:anthropic/claude-haiku-5.5` | 54.0 | 15.4% | 54 | 0% | 150 | 0.7% | $0.0613 | 1 | 0 |
| `openrouter:z-ai/glm-5.3-flash` | 50.0 | 14.3% | 50 | 0% | 150 | 2.7% | $0.0093 | 1 | 0 |
| `openrouter:openai/gpt-6-luna` | 44.0 | 12.6% | 44 | 0% | 150 | 2.0% | $0.0360 | 1 | 0 |
| `openrouter:google/gemini-3.5-flash-lite` | 40.0 | 11.4% | 40 | 0% | 150 | 4.7% | $0.1212 | 1 | 0 |
| `openrouter:qwen/qwen3.8-flash` | 40.0 | 11.4% | 40 | 0% | 150 | 7.3% | $0.0902 | 1 | 0 |
| `openrouter:deepseek/deepseek-v4-flash` | 40.0 | 11.4% | 40 | 0% | 150 | 22.7% | $0.0404 | 1 | 0 |

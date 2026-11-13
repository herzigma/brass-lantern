# Brass Lantern

A benchmark harness that makes any LLM play interactive-fiction games (Zork-style text adventures) on a
[Grue](https://github.com/herzigma/grue) game server, then compares models on score, completion, cost and
how often they fumble the parser.

Brass Lantern talks to Grue's REST API (see Grue's `docs/PROTOCOL.md`), turns each game observation into a
prompt, parses the model's reply into one command, and records everything. No vendor SDKs: just `httpx`.

## Install

```bash
pip install -e ".[dev]"      # dev adds pytest
cp .env.example .env         # then add the API keys you need
brass-lantern --help
```

Needs Python 3.10+. If `brass-lantern` is not on your PATH, `python -m brass_lantern` does the same thing.

## Quickstart (against a local Grue)

Start Grue so its REST API is on `http://localhost:8081` (override with `GRUE_URL` or `--grue-url`), then:

```bash
brass-lantern games                                   # what the server offers

# dry run, no API calls: replays commands from a file
printf 'open mailbox\nread leaflet\nnorth\n' > script.txt
brass-lantern play --game zork1 --model mock:script.txt --max-turns 20

# a real model through OpenRouter
brass-lantern play --game zork1 --model openrouter:anthropic/claude-haiku-4.5 --max-turns 150 --max-cost 0.25
```

You will see the game text, then `🤖 [n] command` with the running score, spend and latency.
Use `--quiet` to print only the final summary line. Ctrl-C ends the session cleanly and still writes the run files.

## Model specs

A model is `provider:model` (split on the first colon, so `ollama:llama3.1:8b` works).

| Spec | Key / endpoint | Notes |
|---|---|---|
| `openrouter:<vendor/model>` | `OPENROUTER_API_KEY`, https://openrouter.ai/api/v1 | Requests `usage.include` and records the real dollar `usage.cost`. Sends `HTTP-Referer` (override with `OPENROUTER_REFERER`) and `X-Title: Brass Lantern`. |
| `openai:<model>` | `OPENAI_API_KEY`, https://api.openai.com/v1 | Uses `max_completion_tokens`. |
| `openai:<model>` + `--base-url URL` | `OPENAI_API_KEY` optional | Any OpenAI-compatible server (vLLM, LM Studio, Together, ...). Uses `max_tokens`. |
| `ollama:<model>` | none, http://localhost:11434/v1 | Local models. |
| `anthropic:<model>` | `ANTHROPIC_API_KEY`, https://api.anthropic.com | `ANTHROPIC_BASE_URL` is deliberately **not** read; use `--base-url` to override. |
| `gemini:<model>` | `GEMINI_API_KEY` | Key goes in the `x-goog-api-key` header, never the URL. |
| `mock:<script.txt>` | none | Replays one command per line (`#` comments allowed), then `look`. For tests and dry runs. |
| `human:` | none | You type the commands. Handy for debugging the harness. |

Every call returns a `Completion` (text, input/output tokens, `cost_usd` or `None`, latency). 429, 5xx and timeouts
are retried with exponential backoff (1s, 2s, 4s, ... capped at 30s, `Retry-After` honoured, up to 5 retries) before a
clear error is raised. If a reasoning model returns empty `content`, its reasoning text is used instead; if that is
empty too, the harness plays `look` and records a parse failure (raise `--max-tokens` for reasoning models, e.g. 4096+).

## Commands

```bash
brass-lantern games
brass-lantern play  --game zork1 --model SPEC [--max-turns 150] [--max-cost 0.5] [--max-time 900]
                    [--history 30] [--notes] [--temperature 0.7|none] [--max-tokens 1024]
                    [--base-url URL] [--out runs] [--quiet] [--seed-meta KEY=VALUE]
brass-lantern bench --models a,b,c --games x,y --runs 3 --max-turns 150 --concurrency 4 \
                    --max-cost-per-run 0.50 --out runs/bench-demo
brass-lantern report runs/bench-demo
```

- `--games all` benchmarks every available game on the server.
- `--seed-meta KEY=VALUE` (repeatable) adds fields to the metadata Grue stores with the session.
- `--temperature` defaults to 0.7 and is recorded in every run. Pass `none` to omit it for models that reject it
  (e.g. some reasoning models).
- `bench` interleaves repeats so an interrupted bench still covers every model and game evenly. Ctrl-C ends all
  live sessions cleanly and writes the leaderboard for whatever finished.

## How the agent plays

- The system prompt explains parser IF (verbs, directions, one command per turn, explore/examine/read/take) and the
  reply format: brief reasoning, then a final line `> COMMAND`.
- **Parsing:** the last line starting with `>` wins; else a `COMMAND:` line; else the last non-empty line. Markdown,
  quotes and backticks are stripped, whitespace collapsed, chained commands ("take lamp then north", `a, b`, `a; b`)
  cut to the first, and the result truncated to 120 characters. `<think>` blocks are ignored. A reply with no marker is
  a **parse failure** (still played, but counted).
- **Context:** each user message is the game text plus a status footer (`Turn 5 of 150 | Score 10 of 350 | Location: ...`).
  When Grue reports `input_type: "char"`, an explicit warning explains single keypresses and key names (`return`, `space`, ...).
  The last `--history` turns (default 30) are kept verbatim; the model's own past replies are stored compactly
  (reasoning trimmed to ~300 chars plus the command). Full replies go to the transcript.
- **Notes (`--notes`):** the model may write `NOTES: ...` lines; the latest block replaces the stored notes (capped at
  1500 chars), which are shown again every turn. This gives long games a cheap memory beyond the history window.

## Output files

Each episode writes `runs/<run_id>/` (run ids look like `20261112-210000-zork1-openrouter-anthropic-claude-haiku-4.5-a1b2`):

- `run.json`: config (including provider settings and the metadata sent to Grue), model, game, session id, status
  (`completed` | `interrupted` | `error`), stop reason (`game_over`, `max_turns`, `max_cost`, `max_time`,
  `provider_error`, `grue_error`, `interrupted`, `error`), Grue's final `Result`, and totals: turns, score, max score,
  ending, input/output tokens, cost, wall time, parse failures, invalid commands.
- `transcript.jsonl`: one line per turn with the observation the model saw, the raw model reply, the command sent,
  parse method, tokens, cost, latency and finish reason; plus a final line with the last observation.

A bench also writes `leaderboard.md`, `results.csv` (one row per run) and `summary.json` into its `--out` directory.
`report <dir>` rebuilds those three files from every `run.json` under `<dir>`.

## How scoring works

- **Score** = the final in-game score Grue reports. **Normalized score** = score / max score (games without a max score
  are left out of normalized rankings). The overall ranking averages each model's per-game mean normalized score, so every
  game weighs the same regardless of repeat counts.
- **Win rate** = share of runs Grue ended as `won`. Games often end by `turn_limit` instead, so score is the main signal.
- **Invalid %** = parser rejections per command, as counted by Grue. **Format fail %** = replies without a `> COMMAND` line.
- Runs that errored (API failure, server error) or were interrupted are excluded from scores and shown in the "excl."
  count, but their cost still counts. Runs stopped by `--max-cost` or `--max-time` are scored as played.

## Cost controls

- `--max-cost` (play) / `--max-cost-per-run` (bench) stop an episode when spend reaches the cap, then end the session.
  OpenRouter reports real costs. Other providers report only tokens, so supply `--prices prices.json` to estimate cost
  (and make the cap work): `{"anthropic:claude-sonnet-4-5": {"input": 3.0, "output": 15.0}}` (USD per million tokens).
  Without it a warning is printed and cost shows as `n/a`.
- `--max-turns`, `--max-time` and `--history` bound the rest. Keep `--max-tokens` small unless you test reasoning models.
- Use `mock:` for dry runs; they cost nothing.

## Adding a provider

1. Subclass `Provider` (or `HTTPProvider`, which gives you the pooled client and retry/backoff) in
   `brass_lantern/providers/` and implement `async complete(system, messages) -> Completion`.
   `messages` is a list of `{"role": "user"|"assistant", "content": str}` starting with a user message.
2. Register a name in `PROVIDER_NAMES` and a branch in `create_provider` in `brass_lantern/providers/__init__.py`.
3. Add request/response-shape tests with `httpx.MockTransport` (see `tests/test_providers.py`).

OpenAI-compatible APIs need no code: use `openai:<model> --base-url ...`.

## Tests

```bash
python -m pytest -q
```

No network or paid calls: providers are tested with `httpx.MockTransport`, and the runner and bench run against an
in-memory fake Grue (`tests/fake_grue.py`).

## License

MIT

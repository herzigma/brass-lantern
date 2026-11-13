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

Start Grue so its REST API is on `http://127.0.0.1:8081` (override with `GRUE_URL` or `--grue-url`), then:

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
brass-lantern bench --models a,b,c --games x,y --runs 3 --concurrency 4 \
                    --max-cost-per-run 0.50 --out runs/bench-demo
brass-lantern report runs/bench-demo
# --max-turns defaults to the game's suggested limit from the Grue catalog (e.g. 300 for skerrin-light)
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
  `max_calls`, `provider_error`, `grue_error`, `interrupted`, `error`), Grue's final `Result`, and totals: turns, score, max score,
  ending, input/output tokens, cost, wall time, parse failures, invalid commands.
- `transcript.jsonl`: one line per turn with the observation the model saw, the raw model reply, the command sent,
  parse method, tokens, cost, latency and finish reason; plus a final line with the last observation.

A bench also writes `leaderboard.md`, `results.csv` (one row per run) and `summary.json` into its `--out` directory.
`report <dir>` rebuilds those three files from every `run.json` under `<dir>`.

## How scoring works

- **Score** = the final in-game score Grue reports. **Normalized score** = score / max score. A finished run that never
  reported a score but whose game has a max score counts as 0. Games without a max score are left out of normalized
  rankings.
- **Overall ranking** = the mean, over **all** games in the bench that have a max score, of the model's per-game mean
  normalized score (each game weighs the same regardless of repeat counts). A game where the model has no valid run
  counts as 0 and is marked `*` (and `- (=0)` in the per-game matrix), so a model cannot rank higher by failing on hard
  games. The formula is repeated at the top of `leaderboard.md`.
- **Valid runs** are those that count in the scores:
  - completed runs, including ones stopped by `--max-turns`, `--max-cost`, `--max-time` or `max_calls`;
  - runs that ended in `provider_error` after at least one turn: the model or provider failing is a real outcome, so they
    are scored as reached and flagged in the **Errors** column (per model and per game; also `provider_errors` in
    `summary.json`).
- **Excluded runs** hit a harness problem (`grue_error`, `interrupted`, `error`, still `running`, or a `provider_error`
  before the first turn). They are left out of scores, counted in the "excl." column and listed with their reason in the
  "Excluded runs" section of `leaderboard.md` (and `excluded_reason` in `results.csv`). Their cost still counts.
- **Configurations:** runs are grouped by model spec **and** the settings that change behaviour (temperature, history,
  notes, reasoning, max tokens, max turns). If one spec was run with different settings, each configuration is its own
  row, labelled with the settings that differ, e.g. `openrouter:x (notes, h=30)`. Identical configurations merge.
- **Win rate** = share of runs Grue ended as `won`. Games often end by `turn_limit` instead, so score is the main signal.
- **Invalid %** = parser rejections per command, as counted by Grue. **Format fail %** = replies without a `> COMMAND` line.

## Cost controls

- `--max-cost` (play) / `--max-cost-per-run` (bench) stop an episode when spend reaches the cap, then end the session.
  The check runs before each model call and again right after a call that pushes spend over the cap, so no further turns
  are sent; the stop reason is `max_cost`. A format-repair call that fails still bills the call that preceded it.
  OpenRouter reports real costs. Other providers report only tokens, so supply `--prices prices.json` to estimate cost
  (and make the cap work): `{"anthropic:claude-sonnet-4-5": {"input": 3.0, "output": 15.0}}` (USD per million tokens).
- **`bench --max-cost-per-run` refuses to start** (exit code 2) if any model reports no dollar cost (anything except
  `openrouter:`, `mock:`, `human:`) and has no entry in `--prices`: the cap would be silently inert, and spending without
  an effective cap is the dangerous direction. The error names the models. `play --max-cost` only warns in that case, and
  cost shows as `n/a`.
- `--max-turns`, `--max-time` and `--history` bound the rest. Model calls are also capped at `3 * max-turns` per episode
  (stop reason `max_calls`), in case Grue stops advancing `turn` (for example while it reports a command as not sent
  because the game is still busy). Keep `--max-tokens` small unless you test reasoning models.
- Use `mock:` for dry runs; they cost nothing.

## Connection handling

The Grue client drops idle pooled connections after 2 seconds (below the server's keep-alive) and retries a request once
if the connection fails before any response arrives (`RemoteProtocolError` or `ConnectError`), but only for idempotent
requests (`GET`, `DELETE`). `POST .../commands` is never retried because the command may already have run; such a failure
ends the run as `grue_error` with a message saying so.

## Results

The first benchmark (9 models, 4 games, Nov 2026) is written up in
[docs/results/2026-11-13.md](docs/results/2026-11-13.md). Its main finding is that the original games, Skerrin Light
in particular, separate models far better than Zork I. Run Skerrin Light at 300 turns.

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

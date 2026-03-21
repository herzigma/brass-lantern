# 🏮 Brass Lantern

A lightweight, asynchronous Python client that connects a Large Language Model to a [Grue](https://github.com/herzigma/grue) game server. Brass Lantern acts as stateful middleware, translating the real-time WebSocket connection of the game into the discrete HTTP requests required by modern AI APIs.

## Quick Start

### 1. Install

```bash
pip install -e .
```

### 2. Configure

Copy the example environment file and fill in your API key:

```bash
cp .env.example .env
```

| Variable | Required | Default | Description |
|---|---|---|---|
| `AI_API_KEY` | ✅ | — | Your LLM provider API key |
| `GRUE_URI` | — | `ws://localhost:8080` | Grue WebSocket URI |
| `AI_PROVIDER` | — | `openai` | `openai` · `anthropic` · `gemini` |
| `AI_MODEL` | — | *(provider default)* | Model name override |

### 3. Run

```bash
python -m brass_lantern
```

Or with explicit arguments:

```bash
python -m brass_lantern --uri ws://localhost:8080 --provider anthropic --model claude-sonnet-4-20250514
```

## Architecture

```
┌─────────────┐  WebSocket  ┌────────────────┐  HTTP/REST  ┌─────────┐
│ Grue Server │◄───-───────►│  Brass Lantern │────────────►│ LLM API │
└─────────────┘             └────────────────┘             └─────────┘
                               │  messages[]  │
                               │ (transcript) │
```

1. **Receive** — Grue pushes game state over WebSocket.
2. **Think** — Brass Lantern appends the text to its transcript and queries the LLM.
3. **Act** — The AI's command is sent back to Grue.
4. **Trim** — The transcript is capped at 20 turn-pairs to avoid context overflow.

## Providers

Brass Lantern ships with three built-in providers, all implemented with raw [`httpx`](https://www.python-httpx.org/) calls (no vendor SDKs):

| Provider | Default Model |
|---|---|
| `openai` | `gpt-4o` |
| `anthropic` | `claude-sonnet-4-20250514` |
| `gemini` | `gemini-2.0-flash` |

### Adding a Custom Provider

Create a class that extends `AIProvider` and implements `get_completion`:

```python
from brass_lantern.providers.base import AIProvider

class MyProvider(AIProvider):
    async def get_completion(self, messages: list[dict]) -> str:
        # Your HTTP logic here
        return "go north"
```

Then register it in `brass_lantern/providers/__init__.py`.

## License

MIT

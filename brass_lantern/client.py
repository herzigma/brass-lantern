"""Core Brass Lantern client — bridges a Grue WebSocket to an LLM."""

from __future__ import annotations

import asyncio

import websockets

from brass_lantern.providers.base import AIProvider

# How many user/assistant turn-pairs to keep before trimming.
MAX_TURNS = 20


class BrassLantern:
    """Stateful middleware that plays a text adventure via an LLM.

    Parameters
    ----------
    grue_uri:
        WebSocket URI of the Grue game server (e.g. ``ws://localhost:8080``).
    provider:
        An :class:`AIProvider` instance used to generate commands.
    """

    SYSTEM_PROMPT = (
        "You are playing a text adventure game. "
        "Respond only with short, actionable commands like "
        "'go north' or 'take lamp'."
    )

    def __init__(self, grue_uri: str, provider: AIProvider) -> None:
        self.uri = grue_uri
        self.provider = provider
        # messages[0] is always the system prompt; the rest alternate
        # between user (game text) and assistant (AI commands).
        self.messages: list[dict[str, str]] = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
        ]

    # ------------------------------------------------------------------
    # Context management
    # ------------------------------------------------------------------

    def _trim_history(self) -> None:
        """Keep only the system prompt plus the last *MAX_TURNS* pairs.

        Each "turn" is one user message followed by one assistant message,
        so we retain ``MAX_TURNS * 2`` messages after the system prompt.
        """
        max_messages = 1 + MAX_TURNS * 2  # system + pairs
        if len(self.messages) > max_messages:
            self.messages = [self.messages[0]] + self.messages[-MAX_TURNS * 2 :]

    # ------------------------------------------------------------------
    # AI interaction
    # ------------------------------------------------------------------

    async def get_ai_command(self, game_text: str) -> str:
        """Append *game_text* to the transcript, query the LLM, and return its command."""

        # 1. Record what the game just told us.
        self.messages.append({"role": "user", "content": game_text})

        # 2. Ask the LLM what to do.
        command = await self.provider.get_completion(self.messages)

        # 3. Record the AI's response.
        self.messages.append({"role": "assistant", "content": command})

        # 4. Trim to avoid context-window overflow.
        self._trim_history()

        return command

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    async def connect_and_play(self) -> None:
        """Open a WebSocket to Grue and loop: receive → think → act."""

        print(f"[brass-lantern] Connecting to Grue at {self.uri} …")

        async with websockets.connect(self.uri) as websocket:
            print(f"[brass-lantern] Connected.")

            while True:
                # Wait for the game to tell us what is happening.
                game_text = await websocket.recv()
                print(f"\n📜  Game:\n{game_text}")

                # Ask the AI what to do next.
                command = await self.get_ai_command(game_text)
                print(f"\n🤖  Agent: {command}")

                # Send the action back to the game.
                await websocket.send(command)

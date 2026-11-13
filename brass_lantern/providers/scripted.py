"""Offline providers: ``mock:<script.txt>`` replays commands, ``human:`` reads stdin."""

from __future__ import annotations

import asyncio
from pathlib import Path

from brass_lantern.providers.base import Completion, ProviderError, Provider


class ScriptedProvider(Provider):
    """Replays commands (one per line; blank lines and ``#`` comments are skipped).

    Once the script runs out it answers ``look`` forever, so dry runs always finish
    at the turn limit rather than crashing.
    """

    name = "mock"
    fallback = "look"

    def __init__(self, commands: list[str], model: str = "scripted", **kwargs):
        super().__init__(model, **kwargs)
        self._commands = list(commands)
        self._i = 0

    @classmethod
    def from_file(cls, path: str, **kwargs) -> "ScriptedProvider":
        lines = Path(path).read_text(encoding="utf-8").splitlines()
        commands = [ln.strip() for ln in lines if ln.strip() and not ln.strip().startswith("#")]
        return cls(commands, model=path, **kwargs)

    async def complete(self, system: str, messages: list[dict[str, str]]) -> Completion:
        if self._i < len(self._commands):
            command = self._commands[self._i]
            self._i += 1
        else:
            command = self.fallback
        return Completion(text=f"> {command}", cost_usd=0.0, finish_reason="stop")


class HumanProvider(Provider):
    """Reads each command from stdin; handy for debugging the harness by hand."""

    name = "human"

    def __init__(self, model: str = "human", **kwargs):
        super().__init__(model or "human", **kwargs)

    async def complete(self, system: str, messages: list[dict[str, str]]) -> Completion:
        try:
            line = await asyncio.to_thread(input, "your command> ")
        except EOFError as e:
            raise ProviderError("human: stdin closed") from e
        return Completion(text=f"> {line}", cost_usd=0.0, finish_reason="stop")

"""Anthropic Claude provider."""

import httpx

from brass_lantern.providers.base import AIProvider

DEFAULT_MODEL = "claude-sonnet-4-20250514"
API_URL = "https://api.anthropic.com/v1/messages"
API_VERSION = "2023-06-01"


class AnthropicProvider(AIProvider):
    """Provider for the Anthropic Messages API.

    The Anthropic API handles the system prompt separately from the
    conversation messages, so this provider splits it out before sending.
    """

    def __init__(self, api_key: str, model: str | None = None):
        super().__init__(api_key, model or DEFAULT_MODEL)

    async def get_completion(self, messages: list[dict]) -> str:
        # Anthropic expects the system prompt outside the messages array.
        system_text = ""
        conversation: list[dict] = []
        for msg in messages:
            if msg["role"] == "system":
                system_text = msg["content"]
            else:
                conversation.append(msg)

        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": API_VERSION,
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "max_tokens": 256,
            "system": system_text,
            "messages": conversation,
        }

        async with httpx.AsyncClient(timeout=60) as client:
            response = await client.post(API_URL, headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()

        return data["content"][0]["text"].strip()

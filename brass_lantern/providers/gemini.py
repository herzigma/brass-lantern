"""Google Gemini provider."""

import httpx

from brass_lantern.providers.base import AIProvider

DEFAULT_MODEL = "gemini-2.0-flash"
API_BASE = "https://generativelanguage.googleapis.com/v1beta/models"


class GeminiProvider(AIProvider):
    """Provider for the Google Gemini REST API.

    Translates OpenAI-style messages into the Gemini
    ``generateContent`` format.
    """

    def __init__(self, api_key: str, model: str | None = None):
        super().__init__(api_key, model or DEFAULT_MODEL)

    async def get_completion(self, messages: list[dict]) -> str:
        # Gemini uses a different message schema.  We map
        # "system" → system_instruction, and user/assistant → contents.
        system_text = ""
        contents: list[dict] = []
        for msg in messages:
            if msg["role"] == "system":
                system_text = msg["content"]
            else:
                role = "user" if msg["role"] == "user" else "model"
                contents.append({
                    "role": role,
                    "parts": [{"text": msg["content"]}],
                })

        url = f"{API_BASE}/{self.model}:generateContent?key={self.api_key}"
        payload: dict = {"contents": contents}
        if system_text:
            payload["system_instruction"] = {
                "parts": [{"text": system_text}],
            }

        async with httpx.AsyncClient(timeout=60) as client:
            response = await client.post(
                url,
                headers={"Content-Type": "application/json"},
                json=payload,
            )
            response.raise_for_status()
            data = response.json()

        return data["candidates"][0]["content"]["parts"][0]["text"].strip()

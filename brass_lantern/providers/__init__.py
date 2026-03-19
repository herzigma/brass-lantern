"""Provider registry and factory function."""

from brass_lantern.providers.base import AIProvider


def get_provider(name: str, api_key: str, model: str | None = None) -> AIProvider:
    """Instantiate an :class:`AIProvider` by short name.

    Parameters
    ----------
    name:
        One of ``"openai"``, ``"anthropic"``, or ``"gemini"``.
    api_key:
        The API key forwarded to the provider.
    model:
        Optional model override; each provider has a sensible default.

    Raises
    ------
    ValueError
        If *name* does not match a known provider.
    """
    name = name.lower().strip()

    if name == "openai":
        from brass_lantern.providers.openai import OpenAIProvider
        return OpenAIProvider(api_key, model)

    if name == "anthropic":
        from brass_lantern.providers.anthropic import AnthropicProvider
        return AnthropicProvider(api_key, model)

    if name == "gemini":
        from brass_lantern.providers.gemini import GeminiProvider
        return GeminiProvider(api_key, model)

    raise ValueError(
        f"Unknown provider '{name}'. Choose from: openai, anthropic, gemini"
    )

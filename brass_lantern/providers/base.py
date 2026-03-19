"""Abstract base class for AI providers."""

from abc import ABC, abstractmethod


class AIProvider(ABC):
    """Interface that every LLM provider must implement.

    Subclasses handle the HTTP specifics for a particular API while
    presenting a uniform ``get_completion`` surface to the client.
    """

    def __init__(self, api_key: str, model: str | None = None):
        self.api_key = api_key
        self.model = model

    @abstractmethod
    async def get_completion(self, messages: list[dict]) -> str:
        """Send *messages* to the LLM and return the assistant's text reply.

        Parameters
        ----------
        messages:
            A list of message dicts in OpenAI-style format::

                [
                    {"role": "system", "content": "..."},
                    {"role": "user",   "content": "..."},
                    {"role": "assistant", "content": "..."},
                    ...
                ]

        Returns
        -------
        str
            The raw text content of the assistant's reply.
        """
        ...

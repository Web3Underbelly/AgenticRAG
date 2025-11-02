"""Lightweight wrappers around chat completion providers used in the graphs."""
from __future__ import annotations

from typing import Callable, Protocol


class LLMCallable(Protocol):
    def __call__(self, prompt: str) -> str:  # pragma: no cover - protocol signature
        ...


class OpenAIChatLLM:
    """Thin wrapper over OpenAI's chat completion endpoint.

    The implementation purposefully avoids importing the SDK at module import
    time so that local development without the dependency remains possible.
    """

    def __init__(self, model: str = "gpt-5-nano", api_key: str | None = None) -> None:
        from openai import OpenAI

        self._client = OpenAI(api_key=api_key)
        self._model = model

    def __call__(self, prompt: str) -> str:
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content


def ensure_callable(llm: LLMCallable | Callable[[str], str]) -> LLMCallable:
    if not callable(llm):
        raise TypeError("The provided LLM must be callable and accept a prompt string.")
    return llm  # type: ignore[return-value]


__all__ = ["LLMCallable", "OpenAIChatLLM", "ensure_callable"]

"""Search tool integrations for the agentic RAG workflow."""
from __future__ import annotations

import json
import logging
from typing import Callable, Protocol

logger = logging.getLogger(__name__)


class SearchCallable(Protocol):
    def __call__(self, query: str) -> str:  # pragma: no cover - protocol signature
        ...


class SerperSearch:
    """Wrapper around :class:`GoogleSerperAPIWrapper` with graceful fallback."""

    def __init__(self, api_key: str | None = None) -> None:
        try:
            from langchain_community.utilities import GoogleSerperAPIWrapper
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "langchain-community is required to use the Serper search tool"
            ) from exc

        self._wrapper = GoogleSerperAPIWrapper(serper_api_key=api_key)

    def __call__(self, query: str) -> str:
        logger.debug("Running Serper search for query: %s", query)
        result = self._wrapper.run(query=query)
        if isinstance(result, str):
            return result
        return json.dumps(result)


def ensure_search_callable(search: SearchCallable | Callable[[str], str] | None) -> SearchCallable:
    if search is None:
        raise RuntimeError(
            "A search callable is required for the agentic workflow when a web "
            "search branch is taken."
        )
    if not callable(search):
        raise TypeError("Search tool must be callable.")
    return search  # type: ignore[return-value]


__all__ = ["SearchCallable", "SerperSearch", "ensure_search_callable"]

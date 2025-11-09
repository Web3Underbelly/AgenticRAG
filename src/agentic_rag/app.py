"""High level helper that wires datasets, vector stores and workflows."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, Optional

from langgraph.graph import CompiledGraph

from . import data
from .llm import LLMCallable, ensure_callable
from .search import SearchCallable
from .vectorstores import VectorStoreBuilder
from .workflows import AgenticGraphState, SimpleGraphState, build_agentic_rag, build_simple_rag

logger = logging.getLogger(__name__)


class AgenticRAGApp:
    """Create simple and agentic LangGraph workflows backed by ChromaDB."""

    def __init__(
        self,
        *,
        qa_dataset: Path,
        device_dataset: Path,
        llm: LLMCallable | Callable[[str], str],
        search_tool: Optional[SearchCallable | Callable[[str], str]] = None,
        persist_directory: Optional[Path] = None,
        n_results: int = 3,
        word_limit: int = 50,
        max_iterations: int = 3,
    ) -> None:
        self._llm = ensure_callable(llm)
        self._search = search_tool
        builder = VectorStoreBuilder(persist_directory)

        qa_docs = data.load_medical_qa_documents(qa_dataset)
        device_docs = data.load_medical_device_documents(device_dataset)

        logger.info("Creating ChromaDB collections with %d QA and %d device docs", len(qa_docs), len(device_docs))
        self._qa_collection = builder.create_collection("medical_q_n_a", qa_docs)
        self._device_collection = builder.create_collection("medical_device_manual", device_docs)

        self._simple_graph = build_simple_rag(
            self._qa_collection,
            self._llm,
            n_results=n_results,
            word_limit=word_limit,
        ).compile()

        self._agentic_graph: Optional[CompiledGraph]
        if self._search is None:
            logger.warning(
                "Agentic workflow disabled because no search tool was provided."
            )
            self._agentic_graph = None
        else:
            self._agentic_graph = build_agentic_rag(
                self._qa_collection,
                self._device_collection,
                self._llm,
                search_tool=self._search,
                n_results=n_results,
                word_limit=word_limit,
                max_iterations=max_iterations,
            ).compile()

    @property
    def simple_graph(self) -> CompiledGraph:
        return self._simple_graph

    @property
    def agentic_graph(self) -> CompiledGraph:
        if self._agentic_graph is None:
            raise RuntimeError(
                "Agentic workflow is not available because no search tool was "
                "supplied during initialisation."
            )
        return self._agentic_graph

    def ask_simple(self, query: str) -> SimpleGraphState:
        return self._simple_graph.invoke({"query": query})

    def ask_agentic(self, query: str) -> AgenticGraphState:
        return self.agentic_graph.invoke({"query": query})


__all__ = ["AgenticRAGApp"]

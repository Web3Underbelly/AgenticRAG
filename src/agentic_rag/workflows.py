"""LangGraph workflows for simple and agentic RAG pipelines."""
from __future__ import annotations

import logging
from typing import Callable, Dict, Literal, TypedDict

from chromadb.api.models.Collection import Collection
from langgraph.graph import END, START, StateGraph

from .llm import LLMCallable, ensure_callable
from .search import SearchCallable, ensure_search_callable

logger = logging.getLogger(__name__)


class SimpleGraphState(TypedDict, total=False):
    query: str
    context: str
    prompt: str
    response: str


class AgenticGraphState(SimpleGraphState, total=False):
    source: str
    is_relevant: str
    iteration_count: int


def _join_documents(results: Dict[str, list]) -> str:
    documents = results.get("documents", [])
    if not documents:
        return ""
    return "\n".join(documents[0])


def build_simple_rag(
    collection: Collection,
    llm: LLMCallable | Callable[[str], str],
    *,
    n_results: int = 3,
    word_limit: int = 50,
) -> StateGraph:
    """Compile a simple RAG agent that always queries the provided collection."""

    llm_callable = ensure_callable(llm)

    def retrieve_context(state: SimpleGraphState) -> SimpleGraphState:
        query = state["query"]
        logger.debug("Retrieving context from %s", collection.name)
        results = collection.query(query_texts=[query], n_results=n_results)
        context = _join_documents(results)
        return {**state, "context": context}

    def build_prompt(state: SimpleGraphState) -> SimpleGraphState:
        prompt = (
            "Answer the following question using the context below.\n"
            f"Context:\n{state.get('context', '')}\n"
            f"Question: {state['query']}\n"
            f"Please limit your answer to {word_limit} words."
        )
        return {**state, "prompt": prompt}

    def generate(state: SimpleGraphState) -> SimpleGraphState:
        response = llm_callable(state["prompt"])
        return {**state, "response": response}

    workflow: StateGraph = StateGraph(SimpleGraphState)
    workflow.add_node("Retriever", retrieve_context)
    workflow.add_node("Augment", build_prompt)
    workflow.add_node("Generate", generate)

    workflow.add_edge(START, "Retriever")
    workflow.add_edge("Retriever", "Augment")
    workflow.add_edge("Augment", "Generate")
    workflow.add_edge("Generate", END)
    return workflow


def build_agentic_rag(
    qa_collection: Collection,
    device_collection: Collection,
    llm: LLMCallable | Callable[[str], str],
    *,
    search_tool: SearchCallable | Callable[[str], str] | None,
    n_results: int = 3,
    word_limit: int = 50,
    max_iterations: int = 3,
) -> StateGraph:
    """Compile an agentic RAG agent mirroring the flow from the article."""

    llm_callable = ensure_callable(llm)
    search_callable = ensure_search_callable(search_tool)

    def router(state: AgenticGraphState) -> AgenticGraphState:
        query = state["query"]
        decision_prompt = (
            "You are a routing agent. Based on the user query, decide where to look for information.\n"
            "Options:\n"
            "- Retrieve_QnA: general medical knowledge, symptoms, or treatments.\n"
            "- Retrieve_Device: medical devices, manuals, or instructions.\n"
            "- Web_Search: recent news, brand names, or external data.\n"
            f"Query: \"{query}\"\n"
            "Respond ONLY with one of: Retrieve_QnA, Retrieve_Device, Web_Search"
        )
        decision = llm_callable(decision_prompt).strip()
        logger.debug("Router decision: %s", decision)
        return {**state, "source": decision}

    def retrieve_qna(state: AgenticGraphState) -> AgenticGraphState:
        query = state["query"]
        results = qa_collection.query(query_texts=[query], n_results=n_results)
        context = _join_documents(results)
        return {**state, "context": context, "source": "Retrieve_QnA"}

    def retrieve_device(state: AgenticGraphState) -> AgenticGraphState:
        query = state["query"]
        results = device_collection.query(query_texts=[query], n_results=n_results)
        context = _join_documents(results)
        return {**state, "context": context, "source": "Retrieve_Device"}

    def perform_web_search(state: AgenticGraphState) -> AgenticGraphState:
        query = state["query"]
        result = search_callable(query)
        return {**state, "context": result, "source": "Web_Search"}

    def relevance_checker(state: AgenticGraphState) -> AgenticGraphState:
        context = state.get("context", "")
        query = state["query"]
        prompt = (
            "Check whether the context answers the user query.\n"
            "Context:\n"
            f"{context}\n"
            f"User Query: {query}\n"
            "Reply with Yes if relevant or No if not relevant."
        )
        decision_raw = llm_callable(prompt).strip().lower()
        decision = "Yes" if decision_raw.startswith("y") else "No"
        iteration = state.get("iteration_count", 0) + 1
        if iteration >= max_iterations and decision != "Yes":
            logger.warning("Max iterations reached; continuing with current context.")
            decision = "Yes"
        return {**state, "is_relevant": decision, "iteration_count": iteration}

    def build_prompt(state: AgenticGraphState) -> AgenticGraphState:
        prompt = (
            "Answer the following question using the context below.\n"
            f"Context:\n{state.get('context', '')}\n"
            f"Question: {state['query']}\n"
            f"Please limit your answer to {word_limit} words."
        )
        return {**state, "prompt": prompt}

    def generate(state: AgenticGraphState) -> AgenticGraphState:
        response = llm_callable(state["prompt"])
        return {**state, "response": response}

    def route_decision(state: AgenticGraphState) -> Literal[
        "Retrieve_QnA", "Retrieve_Device", "Web_Search"
    ]:
        return state.get("source", "Web_Search")

    def relevance_decision(state: AgenticGraphState) -> Literal["Yes", "No"]:
        return state.get("is_relevant", "No")

    workflow: StateGraph = StateGraph(AgenticGraphState)
    workflow.add_node("Router", router)
    workflow.add_node("Retrieve_QnA", retrieve_qna)
    workflow.add_node("Retrieve_Device", retrieve_device)
    workflow.add_node("Web_Search", perform_web_search)
    workflow.add_node("Relevance_Checker", relevance_checker)
    workflow.add_node("Augment", build_prompt)
    workflow.add_node("Generate", generate)

    workflow.add_edge(START, "Router")
    workflow.add_conditional_edges(
        "Router",
        route_decision,
        {
            "Retrieve_QnA": "Retrieve_QnA",
            "Retrieve_Device": "Retrieve_Device",
            "Web_Search": "Web_Search",
        },
    )
    workflow.add_edge("Retrieve_QnA", "Relevance_Checker")
    workflow.add_edge("Retrieve_Device", "Relevance_Checker")
    workflow.add_edge("Web_Search", "Relevance_Checker")
    workflow.add_conditional_edges(
        "Relevance_Checker",
        relevance_decision,
        {
            "Yes": "Augment",
            "No": "Web_Search",
        },
    )
    workflow.add_edge("Augment", "Generate")
    workflow.add_edge("Generate", END)
    return workflow


__all__ = ["build_simple_rag", "build_agentic_rag", "SimpleGraphState", "AgenticGraphState"]

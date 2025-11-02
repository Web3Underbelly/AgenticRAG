"""Minimal demo comparing the simple and agentic workflows."""
from __future__ import annotations

from pathlib import Path

from agentic_rag import AgenticRAGApp, OpenAIChatLLM, SerperSearch


def main() -> None:
    qa_path = Path("data/medical_q_n_a_sample.csv")
    device_path = Path("data/medical_device_manuals_sample.csv")

    llm = OpenAIChatLLM(model="gpt-5-nano")
    search = SerperSearch()

    app = AgenticRAGApp(
        qa_dataset=qa_path,
        device_dataset=device_path,
        llm=llm,
        search_tool=search,
    )

    question = "What are the treatments for Kawasaki disease?"
    print("Simple RAG response:")
    print(app.ask_simple(question)["response"])

    open_question = "What's the export duty on medical tablets on India by USA in 2025?"
    print("\nAgentic RAG response:")
    print(app.ask_agentic(open_question)["response"])


if __name__ == "__main__":
    main()

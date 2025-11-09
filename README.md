# Agentic RAG Chat Agent

This repository implements the traditional and agentic RAG (Retrieval-Augmented
Generation) workflows described in the Alpha Iterations article *Build Agentic
RAG using LangGraph*. The code focuses on providing a reproducible reference
that can be executed locally with small sample datasets.

## Features

- **Traditional RAG workflow** that always queries the medical Q&A vector store.
- **Agentic RAG workflow** that dynamically routes between medical Q&A,
  medical device manuals, and a web search tool before generating the final
  answer. The router is resilient to LLM drift and gracefully handles empty
  retrieval results by falling back to web search when necessary.
- **Composable LangGraph graphs** that can be embedded in applications or run as
  standalone scripts.
- **ChromaDB vector stores** populated with lightweight CSV samples to keep the
  project self-contained. Collections are upserted so repeated runs against the
  same persistent database stay idempotent.

## Project Structure

```
├── data/
│   ├── medical_device_manuals_sample.csv
│   └── medical_q_n_a_sample.csv
├── src/agentic_rag/
│   ├── __init__.py
│   ├── app.py
│   ├── data.py
│   ├── llm.py
│   ├── search.py
│   ├── vectorstores.py
│   └── workflows.py
```

## Getting Started

1. **Install dependencies**

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Set environment variables**

   The example uses OpenAI for LLM calls and Google Serper for web search. Set
   the following variables before running the demo script:

   ```bash
   export OPENAI_API_KEY="sk-..."
   export SERPER_API_KEY="serper-..."
   ```

3. **Run the demonstration script**

   The example below compares the simple and agentic workflows. It assumes that
   environment variables from the previous step have been configured. Provide a
   search tool when you want to enable the agentic branch; otherwise the app
   operates in simple RAG-only mode.

   ```python
   from pathlib import Path

   from agentic_rag import AgenticRAGApp, OpenAIChatLLM, SerperSearch

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

   print(app.ask_simple("What are the treatments for Kawasaki disease?"))
   print(app.ask_agentic("What's the export duty on medical tablets on India by USA in 2025?"))
   ```

## Configuration Notes

- The `AgenticRAGApp` accepts optional parameters for persistence, maximum
  iterations and word limits. Refer to the docstring for details.
- The search tool is pluggable. Implement the `SearchCallable` protocol if you
  want to swap Serper for another provider or a mock search implementation.

## Testing

Automated tests are not included, but you can invoke the LangGraph workflows
with `.invoke()` to validate behaviour using any testing framework.


"""Helpers for initialising ChromaDB collections used by the agents."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import chromadb
from chromadb.api.models.Collection import Collection

from .data import Document, ensure_documents


class VectorStoreBuilder:
    """Create and populate ChromaDB collections from prepared documents."""

    def __init__(self, persist_directory: Path | None = None) -> None:
        if persist_directory is None:
            self._client = chromadb.Client()
        else:
            persist_directory.mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(path=str(persist_directory))

    def create_collection(self, name: str, documents: Iterable[Document]) -> Collection:
        collection = self._client.get_or_create_collection(name=name)
        docs = list(documents)
        ensure_documents(docs)
        collection.add(
            ids=[doc.identifier for doc in docs],
            documents=[doc.text for doc in docs],
            metadatas=[doc.metadata for doc in docs],
        )
        return collection


__all__ = ["VectorStoreBuilder"]

"""Utilities for loading and preparing sample medical datasets.

The loader helpers in this module keep the repository lightweight by
operating on small CSV files. The structure mirrors the datasets used in
Alpha Iterations' Agentic RAG article while remaining easy to run in tests.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence
import csv


@dataclass(frozen=True)
class Document:
    """A single text document accompanied by metadata.

    Attributes
    ----------
    text:
        The string that will be embedded inside the vector store.
    metadata:
        Arbitrary metadata encoded as a mapping that ChromaDB can store.
    identifier:
        Stable identifier for the document inside the collection.
    """

    text: str
    metadata: dict
    identifier: str


def _read_csv_rows(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            yield {key: (value or "").strip() for key, value in row.items()}


def load_medical_qa_documents(path: Path) -> List[Document]:
    """Load the medical question & answer dataset and craft combined text.

    The combined text mirrors the article's formatting so the
    downstream prompt looks the same when the retrieval step joins the
    top-k documents together.
    """

    documents: List[Document] = []
    for index, row in enumerate(_read_csv_rows(path)):
        combined_text = (
            f"Question: {row['Question']}. "
            f"Answer: {row['Answer']}. "
            f"Type: {row['qtype']}."
        )
        documents.append(
            Document(
                text=combined_text,
                metadata=row,
                identifier=str(index),
            )
        )
    return documents


def load_medical_device_documents(path: Path) -> List[Document]:
    """Load the medical device manual dataset and craft combined text."""

    documents: List[Document] = []
    for index, row in enumerate(_read_csv_rows(path)):
        combined_text = (
            f"Device Name: {row['Device_Name']}. "
            f"Model: {row['Model_Number']}. "
            f"Manufacturer: {row['Manufacturer']}. "
            f"Indications: {row['Indications_for_Use']}. "
            f"Contraindications: {row['Contraindications']}."
        )
        documents.append(
            Document(
                text=combined_text,
                metadata=row,
                identifier=str(index),
            )
        )
    return documents


def ensure_documents(documents: Sequence[Document]) -> Sequence[Document]:
    """Validate that the sequence of documents is non-empty."""

    if not documents:
        raise ValueError("At least one document is required to populate ChromaDB.")
    return documents


__all__ = [
    "Document",
    "ensure_documents",
    "load_medical_device_documents",
    "load_medical_qa_documents",
]

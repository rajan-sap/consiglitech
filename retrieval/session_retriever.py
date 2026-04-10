"""
Session Retriever — In-memory vector store for user-uploaded documents.

Creates an ephemeral ChromaDB collection that lives only for the duration
of a Streamlit session.  No disk writes, works on Streamlit Cloud.
"""

from typing import Dict, List, Optional

import chromadb
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from ingestion.constants import EMBEDDING_MODEL_NAME


def get_shared_embeddings() -> HuggingFaceEmbeddings:
    """Return a cached embedding model instance.

    When running inside Streamlit, the @st.cache_resource decorator ensures
    the 109 MB model is loaded only once across all sessions.  Outside
    Streamlit (tests, scripts) it falls back to a plain singleton.
    """
    try:
        import streamlit as st

        @st.cache_resource
        def _load():
            return HuggingFaceEmbeddings(
                model_name=EMBEDDING_MODEL_NAME,
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True},
            )

        return _load()
    except Exception:
        return HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )


class SessionRetriever:
    """Ephemeral vector store backed by an in-memory ChromaDB client."""

    def __init__(self) -> None:
        self._client = chromadb.Client()
        self._embeddings = get_shared_embeddings()
        self.vector_store = Chroma(
            client=self._client,
            embedding_function=self._embeddings,
            collection_name="user_documents",
        )
        self._file_names: List[str] = []

    # ── Mutators ────────────────────────────────────────────────────────

    def add_documents(self, chunks: List[Document], file_name: str) -> int:
        """Embed and store chunks.  Returns the number of chunks added."""
        if not chunks:
            return 0
        self.vector_store.add_documents(chunks)
        if file_name not in self._file_names:
            self._file_names.append(file_name)
        return len(chunks)

    def clear(self) -> None:
        """Drop all documents and reset state."""
        self._client.delete_collection("user_documents")
        self.vector_store = Chroma(
            client=self._client,
            embedding_function=self._embeddings,
            collection_name="user_documents",
        )
        self._file_names.clear()

    # ── Queries ─────────────────────────────────────────────────────────

    def search(self, query: str, k: int = 5) -> List[Dict]:
        """Return the top-k most similar chunks as dicts."""
        results = self.vector_store.similarity_search_with_score(query, k=k)
        out = []
        for doc, l2_score in results:
            cosine_sim = 1 - (l2_score ** 2) / 4
            out.append({
                "document": doc.page_content,
                "metadata": doc.metadata,
                "cosine_similarity": round(cosine_sim, 4),
            })
        return out

    # ── Info ────────────────────────────────────────────────────────────

    def get_doc_count(self) -> int:
        try:
            return self.vector_store._collection.count()
        except Exception:
            return 0

    def get_file_names(self) -> List[str]:
        return list(self._file_names)

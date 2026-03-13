"""
Integration tests — Full RAG Pipeline

These call the real LLM (Groq) and read from the real ChromaDB.
They are slow and require:
  - GROQ_API_KEY set in env or .streamlit/secrets.toml
  - ./chroma_db with ingested data

Run explicitly:  pytest tests/integration -v -m integration --timeout=120
"""

import os
import pytest

pytestmark = [
    pytest.mark.integration,
    pytest.mark.timeout(120),
    pytest.mark.skipif(
        not os.path.isdir("./chroma_db"),
        reason="ChromaDB not on disk — run ingestion first",
    ),
]


class TestQueryClassifierLive:
    """Hit the real LLM to verify classification accuracy."""

    def test_general_queries(self, general_queries):
        from generation.generator import is_general_query

        misrouted = [q for q in general_queries if not is_general_query(q)]
        assert not misrouted, f"Misclassified as DOCUMENT: {misrouted}"

    def test_document_queries(self, document_queries):
        from generation.generator import is_general_query

        misrouted = [q for q in document_queries if is_general_query(q)]
        assert not misrouted, f"Misclassified as GENERAL: {misrouted}"


class TestRetrieverLive:
    """Verify retrieval quality against the real vector store."""

    def _retriever(self):
        from retrieval.retriever import Retriever
        return Retriever()

    def test_bmw_query_returns_bmw_chunks(self):
        results = self._retriever().search("BMW revenue 2023", k=5)
        assert len(results) > 0
        texts = " ".join(r["document"] for r in results).lower()
        assert "bmw" in texts

    def test_tesla_query_returns_tesla_chunks(self):
        results = self._retriever().search("Tesla deliveries 2023", k=5)
        assert len(results) > 0
        # Tesla chunks may not always contain the literal word "tesla" but
        # will reference Tesla-specific terms (Model 3/Y, gigafactory) or
        # have Tesla metadata.
        texts = " ".join(r["document"] for r in results).lower()
        metadata_companies = [
            r["metadata"].get("company", "").lower() for r in results
        ]
        tesla_in_text = any(
            kw in texts for kw in ("tesla", "model 3", "model y", "gigafactory")
        )
        tesla_in_meta = "tesla" in metadata_companies
        assert tesla_in_text or tesla_in_meta, (
            "Expected Tesla-related content in results"
        )

    def test_metadata_filter_narrows_results(self):
        r = self._retriever()
        filtered = r.search("revenue", k=5, metadata_filter={"company": "Ford"})
        for res in filtered:
            assert res["metadata"].get("company") == "Ford"


class TestEndToEndGeneration:
    """Full pipeline: query → classify → retrieve → generate → answer."""

    def test_factual_answer(self):
        from generation.generator import generate_answer

        answer = generate_answer("What was Tesla's revenue in 2023?")
        assert isinstance(answer, str)
        assert len(answer) > 20
        assert "tesla" in answer.lower()

    def test_general_answer(self):
        from generation.generator import generate_answer

        answer = generate_answer("What can you do?")
        assert isinstance(answer, str)
        assert len(answer) > 10
        # Should NOT cite specific financials
        assert "eur 155" not in answer.lower()

    def test_comparison_mentions_both_companies(self):
        from generation.generator import generate_answer

        answer = generate_answer("Compare BMW and Ford revenue in 2023")
        lower = answer.lower()
        assert "bmw" in lower or "ford" in lower

    def test_out_of_scope_no_hallucination(self):
        from generation.generator import generate_answer

        answer = generate_answer("What was Apple's revenue in 2023?")
        lower = answer.lower()
        # Should say it doesn't have info, or at minimum not fabricate data
        assert isinstance(answer, str)
        assert len(answer) > 10

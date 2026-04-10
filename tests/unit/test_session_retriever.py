"""
Unit tests — Session Retriever

Tests the in-memory ChromaDB vector store used for user-uploaded documents.
Verifies the full lifecycle: init → add → search → clear.

Note: These tests load the BGE embedding model (~109 MB) on first run.
Subsequent runs use the cached model.
"""

import pytest
from langchain_core.documents import Document

pytestmark = pytest.mark.unit


# ── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture
def retriever():
    from retrieval.session_retriever import SessionRetriever
    r = SessionRetriever()
    r.clear()  # ensure clean state between tests
    return r


@pytest.fixture
def sample_chunks():
    return [
        Document(
            page_content="Tesla reported revenue of $96.8 billion in fiscal year 2023.",
            metadata={"file_name": "tesla_report.pdf", "page_number": 1, "document_type": "User Upload"},
        ),
        Document(
            page_content="Operating expenses increased by 15% compared to the prior year.",
            metadata={"file_name": "tesla_report.pdf", "page_number": 2, "document_type": "User Upload"},
        ),
        Document(
            page_content="The automotive segment delivered 1.8 million vehicles globally.",
            metadata={"file_name": "tesla_report.pdf", "page_number": 5, "document_type": "User Upload"},
        ),
    ]


# ── Initialization ──────────────────────────────────────────────────────────

class TestInit:

    def test_creates_without_error(self, retriever):
        assert retriever is not None
        assert retriever.vector_store is not None

    def test_starts_empty(self, retriever):
        assert retriever.get_doc_count() == 0
        assert retriever.get_file_names() == []


# ── Adding Documents ────────────────────────────────────────────────────────

class TestAddDocuments:

    def test_returns_chunk_count(self, retriever, sample_chunks):
        count = retriever.add_documents(sample_chunks, "tesla_report.pdf")
        assert count == 3

    def test_empty_list_returns_zero(self, retriever):
        count = retriever.add_documents([], "empty.pdf")
        assert count == 0

    def test_tracks_file_name(self, retriever, sample_chunks):
        retriever.add_documents(sample_chunks, "tesla_report.pdf")
        assert "tesla_report.pdf" in retriever.get_file_names()

    def test_no_duplicate_file_names(self, retriever, sample_chunks):
        retriever.add_documents(sample_chunks, "tesla_report.pdf")
        retriever.add_documents(sample_chunks, "tesla_report.pdf")
        assert retriever.get_file_names().count("tesla_report.pdf") == 1

    def test_multiple_files_tracked(self, retriever, sample_chunks):
        retriever.add_documents(sample_chunks[:1], "file_a.pdf")
        retriever.add_documents(sample_chunks[1:2], "file_b.pdf")
        names = retriever.get_file_names()
        assert "file_a.pdf" in names
        assert "file_b.pdf" in names

    def test_doc_count_increases(self, retriever, sample_chunks):
        retriever.add_documents(sample_chunks, "tesla_report.pdf")
        assert retriever.get_doc_count() == 3


# ── Search ──────────────────────────────────────────────────────────────────

class TestSearch:

    def test_returns_results_after_adding(self, retriever, sample_chunks):
        retriever.add_documents(sample_chunks, "tesla_report.pdf")
        results = retriever.search("Tesla revenue", k=2)
        assert len(results) > 0
        assert len(results) <= 2

    def test_result_format(self, retriever, sample_chunks):
        retriever.add_documents(sample_chunks, "tesla_report.pdf")
        results = retriever.search("revenue", k=1)
        result = results[0]
        assert "document" in result
        assert "metadata" in result
        assert "cosine_similarity" in result
        assert isinstance(result["cosine_similarity"], float)

    def test_empty_store_returns_empty(self, retriever):
        results = retriever.search("anything")
        assert results == []

    def test_relevant_result_ranks_higher(self, retriever, sample_chunks):
        retriever.add_documents(sample_chunks, "tesla_report.pdf")
        results = retriever.search("vehicle deliveries", k=3)
        # The chunk about "1.8 million vehicles" should score highest
        assert "vehicle" in results[0]["document"].lower() or "deliver" in results[0]["document"].lower()


# ── Clear ───────────────────────────────────────────────────────────────────

class TestClear:

    def test_clear_resets_count(self, retriever, sample_chunks):
        retriever.add_documents(sample_chunks, "tesla_report.pdf")
        assert retriever.get_doc_count() == 3

        retriever.clear()
        assert retriever.get_doc_count() == 0

    def test_clear_resets_file_names(self, retriever, sample_chunks):
        retriever.add_documents(sample_chunks, "tesla_report.pdf")
        retriever.clear()
        assert retriever.get_file_names() == []

    def test_search_after_clear_returns_empty(self, retriever, sample_chunks):
        retriever.add_documents(sample_chunks, "tesla_report.pdf")
        retriever.clear()
        results = retriever.search("revenue")
        assert results == []

    def test_can_add_after_clear(self, retriever, sample_chunks):
        retriever.add_documents(sample_chunks, "tesla_report.pdf")
        retriever.clear()
        retriever.add_documents(sample_chunks[:1], "new_file.pdf")
        assert retriever.get_doc_count() == 1
        assert retriever.get_file_names() == ["new_file.pdf"]

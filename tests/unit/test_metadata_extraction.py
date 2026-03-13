"""
Unit tests — Metadata Extraction + Query Decomposition

These validate the pure-logic functions in retrieval/utils.py
that parse company names, years, and document types out of queries.
No LLM calls needed for extract_metadata_from_query.
"""

import pytest

pytestmark = pytest.mark.unit


class TestExtractMetadata:
    """Tests for retrieval.utils.extract_metadata_from_query."""

    def _extract(self, query):
        from retrieval.utils import extract_metadata_from_query
        return extract_metadata_from_query(query)

    # ── Company detection ────────────────────────────────────────────────

    def test_detects_bmw(self):
        meta = self._extract("What was BMW's revenue in 2023?")
        assert meta["company"] == "BMW"

    def test_detects_ford(self):
        meta = self._extract("Summarize Ford's annual report")
        assert meta["company"] == "Ford"

    def test_detects_tesla(self):
        meta = self._extract("Tesla vehicle deliveries last year")
        assert meta["company"] == "Tesla"

    def test_case_insensitive_company(self):
        meta = self._extract("What was bmw revenue?")
        assert meta["company"] == "BMW"

    def test_no_company(self):
        meta = self._extract("What is machine learning?")
        assert meta["company"] is None

    # ── Year extraction ──────────────────────────────────────────────────

    def test_extracts_year(self):
        meta = self._extract("BMW revenue in 2023")
        assert meta["year"] == "2023"

    def test_extracts_first_year(self):
        meta = self._extract("Compare 2021 and 2022 revenue")
        assert meta["year"] in ("2021", "2022")  # first match

    def test_no_year(self):
        meta = self._extract("Overall BMW strategy")
        assert meta["year"] is None

    def test_rejects_non_year_numbers(self):
        """4-digit numbers below 2000 should not be treated as years."""
        meta = self._extract("Page 1234 of the report")
        # 1234 doesn't match 20XX pattern
        assert meta["year"] is None

    # ── Document type ────────────────────────────────────────────────────

    def test_detects_annual_report(self):
        meta = self._extract("BMW annual report highlights")
        assert meta["document_type"] == "Annual Report"

    def test_detects_news_article(self):
        meta = self._extract("Latest news article about Ford")
        assert meta["document_type"] == "News Article"

    def test_no_doc_type(self):
        meta = self._extract("What was Tesla revenue?")
        assert meta["document_type"] is None

    # ── Combined ─────────────────────────────────────────────────────────

    def test_all_fields_present(self):
        meta = self._extract("Ford's annual report for 2022")
        assert meta["company"] == "Ford"
        assert meta["year"] == "2022"
        assert meta["document_type"] == "Annual Report"

    def test_empty_query(self):
        meta = self._extract("")
        assert meta["company"] is None
        assert meta["year"] is None
        assert meta["document_type"] is None

    def test_special_characters(self):
        """Queries with unusual punctuation should not crash."""
        meta = self._extract("BMW's/Ford's EBIT (2023)?")
        assert meta["company"] in ("BMW", "Ford")  # picks first
        assert meta["year"] == "2023"

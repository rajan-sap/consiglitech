"""
Unit tests — Ingestion Constants

Validates that chunking config, paths, and embedding settings
are consistent and within sane bounds.
"""

import pytest

pytestmark = pytest.mark.unit


class TestIngestionConstants:

    def test_embedding_model_is_set(self):
        from ingestion.constants import EMBEDDING_MODEL_NAME
        assert isinstance(EMBEDDING_MODEL_NAME, str)
        assert len(EMBEDDING_MODEL_NAME) > 0

    def test_chunk_sizes_positive(self):
        from ingestion.constants import ANNUAL_REPORT_SPLITTER, NEWS_ARTICLE_SPLITTER
        assert ANNUAL_REPORT_SPLITTER["chunk_size"] > 0
        assert NEWS_ARTICLE_SPLITTER["chunk_size"] > 0

    def test_report_chunks_larger_than_news(self):
        """Annual report chunks should be larger for richer financial context."""
        from ingestion.constants import ANNUAL_REPORT_SPLITTER, NEWS_ARTICLE_SPLITTER
        assert ANNUAL_REPORT_SPLITTER["chunk_size"] > NEWS_ARTICLE_SPLITTER["chunk_size"]

    def test_overlap_less_than_chunk(self):
        from ingestion.constants import ANNUAL_REPORT_SPLITTER, NEWS_ARTICLE_SPLITTER
        assert ANNUAL_REPORT_SPLITTER["chunk_overlap"] < ANNUAL_REPORT_SPLITTER["chunk_size"]
        assert NEWS_ARTICLE_SPLITTER["chunk_overlap"] < NEWS_ARTICLE_SPLITTER["chunk_size"]

    def test_company_folders_defined(self):
        from ingestion.constants import COMPANY_FOLDERS
        assert "BMW" in COMPANY_FOLDERS
        assert "Ford" in COMPANY_FOLDERS
        assert "Tesla" in COMPANY_FOLDERS

    def test_supported_file_types(self):
        from ingestion.constants import SUPPORTED_FILE_TYPES
        assert ".pdf" in SUPPORTED_FILE_TYPES

    def test_batch_size_reasonable(self):
        from ingestion.constants import BATCH_SIZE
        assert 1 <= BATCH_SIZE <= 10_000

"""
Unit tests — LLM Configuration

Validates that llm_config.py exports all required fields,
the OpenAI client is constructed correctly, and truncate_context
behaves as expected.
"""

import pytest

pytestmark = pytest.mark.unit


class TestConfigExports:
    """All modules depend on these exports being present and well-typed."""

    def test_model_name_is_nonempty_string(self):
        from llm_config import LLM_MODEL
        assert isinstance(LLM_MODEL, str) and len(LLM_MODEL) > 0

    def test_base_url_is_valid(self):
        from llm_config import LLM_BASE_URL
        assert LLM_BASE_URL is None or LLM_BASE_URL.startswith("http")

    def test_client_is_openai_instance(self):
        from llm_config import llm_client
        from openai import OpenAI
        assert isinstance(llm_client, OpenAI)

    def test_api_key_is_not_placeholder(self):
        from llm_config import LLM_API_KEY
        if LLM_API_KEY:
            assert LLM_API_KEY not in ("not-needed", "sk-xxx", "your-key-here")


class TestTruncateContext:
    """truncate_context must be safe, idempotent, and predictable."""

    def test_short_text_unchanged(self):
        from llm_config import truncate_context
        text = "Short text."
        assert truncate_context(text) == text

    def test_long_text_truncated(self):
        from llm_config import truncate_context
        text = "x" * 50_000
        result = truncate_context(text, max_chars=100)
        assert len(result) < len(text)
        assert "[... context truncated" in result

    def test_exact_boundary_no_truncation(self):
        from llm_config import truncate_context
        text = "a" * 100
        assert truncate_context(text, max_chars=100) == text

    def test_empty_string(self):
        from llm_config import truncate_context
        assert truncate_context("") == ""


class TestGetSecret:
    """_get_secret should fall back gracefully."""

    def test_missing_key_returns_fallback(self):
        from llm_config import _get_secret
        result = _get_secret("NONEXISTENT_KEY_12345", fallback="default_val")
        assert result == "default_val"

    def test_env_var_override(self, monkeypatch):
        from llm_config import _get_secret
        monkeypatch.setenv("TEST_SECRET_XYZ", "from_env")
        assert _get_secret("TEST_SECRET_XYZ") == "from_env"

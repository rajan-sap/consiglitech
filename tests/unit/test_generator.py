"""
Unit tests — Generator module (mocked LLM)

Tests is_general_query and generate_answer using a mocked OpenAI client,
so these run instantly with no network or API key required.
"""

import pytest
from unittest.mock import MagicMock, patch

pytestmark = pytest.mark.unit


# ── Helpers to build mock OpenAI responses ───────────────────────────────────

def _mock_response(content: str):
    """Build a fake openai ChatCompletion response."""
    msg = MagicMock()
    msg.content = content
    choice = MagicMock()
    choice.message = msg
    resp = MagicMock()
    resp.choices = [choice]
    return resp


# ── Query classifier ─────────────────────────────────────────────────────────

class TestIsGeneralQueryMocked:
    """With mocked LLM, verify routing logic."""

    @patch("generation.generator.client")
    def test_returns_true_when_llm_says_general(self, mock_client):
        mock_client.chat.completions.create.return_value = _mock_response("GENERAL")
        from generation.generator import is_general_query
        assert is_general_query("Hello") is True

    @patch("generation.generator.client")
    def test_returns_false_when_llm_says_document(self, mock_client):
        mock_client.chat.completions.create.return_value = _mock_response("DOCUMENT")
        from generation.generator import is_general_query
        assert is_general_query("What was BMW's revenue?") is False

    @patch("generation.generator.client")
    def test_defaults_to_false_on_exception(self, mock_client):
        mock_client.chat.completions.create.side_effect = Exception("timeout")
        from generation.generator import is_general_query
        # Should fall back to DOCUMENT (safer)
        assert is_general_query("Hello") is False

    @patch("generation.generator.client")
    def test_handles_unexpected_label(self, mock_client):
        mock_client.chat.completions.create.return_value = _mock_response("MAYBE")
        from generation.generator import is_general_query
        # "GENERAL" not in "MAYBE" → should route to document
        assert is_general_query("test") is False


# ── Answer generation ────────────────────────────────────────────────────────

class TestGenerateAnswerMocked:
    """generate_answer with mocked LLM and retriever."""

    @patch("generation.generator.retrieve_aggregated_context", return_value="BMW earned EUR 155B in 2023.")
    @patch("generation.generator.client")
    def test_document_route_returns_answer(self, mock_client, mock_retrieve):
        mock_client.chat.completions.create.side_effect = [
            _mock_response("DOCUMENT"),     # classifier call
            _mock_response("BMW earned EUR 155 billion in 2023."),  # generation call
        ]
        from generation.generator import generate_answer
        answer = generate_answer("What was BMW's revenue in 2023?")

        assert isinstance(answer, str)
        assert len(answer) > 0
        assert mock_retrieve.called

    @patch("generation.generator.client")
    def test_general_route_skips_retrieval(self, mock_client):
        mock_client.chat.completions.create.side_effect = [
            _mock_response("GENERAL"),      # classifier
            _mock_response("I am DocIntel, a document intelligence assistant."),  # generation
        ]
        from generation.generator import generate_answer
        answer = generate_answer("Who are you?")

        assert isinstance(answer, str)
        assert len(answer) > 0

    @patch("generation.generator.retrieve_aggregated_context", return_value="context")
    @patch("generation.generator.client")
    def test_return_details_flag(self, mock_client, mock_retrieve):
        mock_client.chat.completions.create.side_effect = [
            _mock_response("DOCUMENT"),
            _mock_response("Answer text."),
        ]
        from generation.generator import generate_answer
        result = generate_answer("BMW revenue?", return_details=True)

        assert isinstance(result, dict)
        assert "answer" in result
        assert "context" in result

    @patch("generation.generator.client")
    def test_general_return_details_has_empty_context(self, mock_client):
        mock_client.chat.completions.create.side_effect = [
            _mock_response("GENERAL"),
            _mock_response("Hello!"),
        ]
        from generation.generator import generate_answer
        result = generate_answer("Hi", return_details=True)

        assert result["context"] == ""

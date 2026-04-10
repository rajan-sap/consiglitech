"""
Unit tests — Generator module (mocked LLM)

Tests is_general_query, generate_answer, and generate_answer_for_uploads
using a mocked OpenAI client. No network or API key required.
"""

import pytest
from unittest.mock import MagicMock, patch

pytestmark = pytest.mark.unit


# ── Helpers ─────────────────────────────────────────────────────────────────

def _mock_response(content: str):
    """Build a fake OpenAI ChatCompletion response."""
    msg = MagicMock()
    msg.content = content
    choice = MagicMock()
    choice.message = msg
    resp = MagicMock()
    resp.choices = [choice]
    return resp


# ── Intent Classifier ───────────────────────────────────────────────────────

class TestIsGeneralQuery:

    @patch("generation.generator.client")
    def test_returns_true_for_general(self, mock_client):
        mock_client.chat.completions.create.return_value = _mock_response("GENERAL")
        from generation.generator import is_general_query
        assert is_general_query("Hello") is True

    @patch("generation.generator.client")
    def test_returns_false_for_document(self, mock_client):
        mock_client.chat.completions.create.return_value = _mock_response("DOCUMENT")
        from generation.generator import is_general_query
        assert is_general_query("What was BMW's revenue?") is False

    @patch("generation.generator.client")
    def test_defaults_to_false_on_exception(self, mock_client):
        mock_client.chat.completions.create.side_effect = Exception("timeout")
        from generation.generator import is_general_query
        assert is_general_query("Hello") is False

    @patch("generation.generator.client")
    def test_rejects_unexpected_label(self, mock_client):
        mock_client.chat.completions.create.return_value = _mock_response("MAYBE")
        from generation.generator import is_general_query
        assert is_general_query("test") is False


# ── Knowledge Base Generation ───────────────────────────────────────────────

class TestGenerateAnswer:

    @patch("retrieval.retriever.retrieve_aggregated_context", return_value="BMW earned EUR 155B in 2023.")
    @patch("generation.generator._get_retriever")
    @patch("generation.generator.client")
    def test_document_route_calls_retriever(self, mock_client, mock_get_ret, mock_retrieve):
        mock_ret = MagicMock()
        mock_ret.is_available = True
        mock_get_ret.return_value = mock_ret
        mock_client.chat.completions.create.side_effect = [
            _mock_response("DOCUMENT"),
            _mock_response("BMW earned EUR 155 billion in 2023."),
        ]
        from generation.generator import generate_answer
        answer = generate_answer("What was BMW's revenue in 2023?")

        assert isinstance(answer, str)
        assert len(answer) > 0

    @patch("generation.generator.client")
    def test_general_route_skips_retrieval(self, mock_client):
        mock_client.chat.completions.create.side_effect = [
            _mock_response("GENERAL"),
            _mock_response("I am DocIntel, a document intelligence assistant."),
        ]
        from generation.generator import generate_answer
        answer = generate_answer("Who are you?")

        assert isinstance(answer, str)
        assert len(answer) > 0

    @patch("generation.generator.client")
    def test_general_return_details_has_empty_context(self, mock_client):
        mock_client.chat.completions.create.side_effect = [
            _mock_response("GENERAL"),
            _mock_response("Hello!"),
        ]
        from generation.generator import generate_answer
        result = generate_answer("Hi", return_details=True)

        assert isinstance(result, dict)
        assert result["context"] == ""
        assert "answer" in result

    @patch("generation.generator._get_retriever")
    @patch("generation.generator.client")
    def test_unavailable_retriever_returns_message(self, mock_client, mock_get_ret):
        mock_ret = MagicMock()
        mock_ret.is_available = False
        mock_get_ret.return_value = mock_ret
        mock_client.chat.completions.create.return_value = _mock_response("DOCUMENT")

        from generation.generator import generate_answer
        answer = generate_answer("Tesla revenue?")

        assert "not currently available" in answer

    @patch("retrieval.retriever.retrieve_aggregated_context", return_value="  ")
    @patch("generation.generator._get_retriever")
    @patch("generation.generator.client")
    def test_empty_context_returns_helpful_message(self, mock_client, mock_get_ret, mock_retrieve):
        mock_ret = MagicMock()
        mock_ret.is_available = True
        mock_get_ret.return_value = mock_ret
        mock_client.chat.completions.create.return_value = _mock_response("DOCUMENT")

        from generation.generator import generate_answer
        answer = generate_answer("Something obscure?")

        assert "couldn't find" in answer


# ── Upload Generation ───────────────────────────────────────────────────────

class TestGenerateAnswerForUploads:

    @patch("generation.generator.client")
    def test_returns_answer_from_uploaded_docs(self, mock_client):
        mock_client.chat.completions.create.return_value = _mock_response(
            "The report mentions quarterly growth of 12%."
        )
        mock_retriever = MagicMock()
        mock_retriever.search.return_value = [
            {
                "document": "Quarterly growth was 12% year over year.",
                "metadata": {"file_name": "report.pdf", "page_number": 3},
                "cosine_similarity": 0.91,
            }
        ]

        from generation.generator import generate_answer_for_uploads
        answer = generate_answer_for_uploads("What was the quarterly growth?", mock_retriever)

        assert isinstance(answer, str)
        assert len(answer) > 0
        mock_retriever.search.assert_called_once()

    def test_no_results_returns_helpful_message(self):
        mock_retriever = MagicMock()
        mock_retriever.search.return_value = []

        from generation.generator import generate_answer_for_uploads
        answer = generate_answer_for_uploads("Anything?", mock_retriever)

        assert "couldn't find" in answer

    @patch("generation.generator.client")
    def test_uses_k5_for_search(self, mock_client):
        mock_client.chat.completions.create.return_value = _mock_response("Answer.")
        mock_retriever = MagicMock()
        mock_retriever.search.return_value = [
            {"document": "text", "metadata": {"file_name": "a.pdf", "page_number": 1}, "cosine_similarity": 0.9}
        ]

        from generation.generator import generate_answer_for_uploads
        generate_answer_for_uploads("query", mock_retriever)

        mock_retriever.search.assert_called_once_with("query", k=5)

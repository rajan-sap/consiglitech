"""
Shared fixtures used across all test modules.

Convention:
  - Fixtures that need the real LLM or ChromaDB are placed in integration tests.
  - Fixtures here are lightweight and deterministic.
"""

import os
import sys
import pytest

# Ensure the project root is importable regardless of how pytest is invoked
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# ── Sample data ──────────────────────────────────────────────────────────────

@pytest.fixture
def sample_search_results():
    """Realistic retriever output (matches Retriever.search() format)."""
    return [
        {
            "document": (
                "BMW Group achieved total revenues of EUR 155.5 billion in "
                "fiscal year 2023, representing a 9 percent increase over the "
                "prior year."
            ),
            "metadata": {"company": "BMW", "year": "2023", "document_type": "Annual Report"},
            "cosine_similarity": 0.87,
        },
        {
            "document": (
                "Ford Motor Company reported total revenue of USD 176.2 billion "
                "for the year ended December 31, 2023."
            ),
            "metadata": {"company": "Ford", "year": "2023", "document_type": "Annual Report"},
            "cosine_similarity": 0.82,
        },
        {
            "document": (
                "Tesla total revenues were USD 96.8 billion for the year ended "
                "December 31, 2023, compared to USD 81.5 billion for 2022."
            ),
            "metadata": {"company": "Tesla", "year": "2023", "document_type": "Annual Report"},
            "cosine_similarity": 0.79,
        },
    ]


@pytest.fixture
def general_queries():
    """Queries that should bypass RAG (GENERAL route)."""
    return [
        "Hello",
        "Hi there!",
        "Good morning",
        "Thanks for the help",
        "Who are you?",
        "What AI model are you using?",
        "What can you do?",
        "What is machine learning?",
    ]


@pytest.fixture
def document_queries():
    """Queries that must go through the RAG pipeline (DOCUMENT route)."""
    return [
        "What was Tesla's revenue in 2023?",
        "Compare BMW and Ford's net income",
        "What are BMW's key financial highlights for 2023?",
        "Summarize Ford's 2023 annual report",
        "What risks does Tesla mention in their report?",
    ]

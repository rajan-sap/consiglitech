"""
Unit tests — Query Limit Constants

Validates the free-tier query limit configuration.
These are simple sanity checks to catch accidental changes.
"""

import pytest

pytestmark = pytest.mark.unit


class TestQueryLimitConfig:

    def test_max_queries_is_ten(self):
        """Free tier allows exactly 10 queries per window."""
        import ast
        with open("streamlit_app.py", encoding="utf-8") as f:
            tree = ast.parse(f.read())
        assignments = {
            node.targets[0].id: node.value
            for node in ast.walk(tree)
            if isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == "MAX_FREE_QUERIES"
        }
        assert "MAX_FREE_QUERIES" in assignments
        assert assignments["MAX_FREE_QUERIES"].value == 10

    def test_limit_window_is_one_hour(self):
        """Rate limit resets after 3600 seconds (1 hour)."""
        import ast
        with open("streamlit_app.py", encoding="utf-8") as f:
            tree = ast.parse(f.read())
        assignments = {
            node.targets[0].id: node.value
            for node in ast.walk(tree)
            if isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == "LIMIT_WINDOW_SECONDS"
        }
        assert "LIMIT_WINDOW_SECONDS" in assignments
        assert assignments["LIMIT_WINDOW_SECONDS"].value == 3600

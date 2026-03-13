"""
Unit tests — Sidebar Data File Counter

Tests the count_data_files logic (extracted from streamlit_app.py).
Ensures correct fallback defaults for cloud and dynamic counting locally.
"""

import os
import tempfile
import pytest

pytestmark = pytest.mark.unit

# Since streamlit_app.py calls st.set_page_config at import time, we cannot
# import the function directly. Instead we replicate the logic here and test it.
# In a production codebase you'd extract count_data_files into a helpers module.

DEFAULT_PER_COMPANY = {"BMW": 3, "Ford": 3, "Tesla": 2}
DEFAULT_NEWS = 1
DEFAULT_TOTAL = sum(DEFAULT_PER_COMPANY.values()) + DEFAULT_NEWS


def count_data_files(data_root="./data"):
    """Replica of the logic in streamlit_app.py for testability."""
    if not os.path.isdir(data_root):
        return DEFAULT_TOTAL, dict(DEFAULT_PER_COMPANY), DEFAULT_NEWS

    companies = ["BMW", "Ford", "Tesla"]
    total = 0
    per_company = {}
    for c in companies:
        folder = os.path.join(data_root, c)
        if os.path.isdir(folder):
            files = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]
            per_company[c] = len(files)
            total += len(files)
        else:
            per_company[c] = 0
    news_folder = os.path.join(data_root, "News and Advertisement")
    news_count = 0
    if os.path.isdir(news_folder):
        news_count = len([f for f in os.listdir(news_folder)
                         if os.path.isfile(os.path.join(news_folder, f))])
        total += news_count
    return total, per_company, news_count


class TestCloudFallback:
    """When ./data is missing (Streamlit Cloud), return hardcoded defaults."""

    def test_returns_defaults_when_dir_missing(self):
        total, per_co, news = count_data_files("/nonexistent/path")
        assert total == 9
        assert per_co == {"BMW": 3, "Ford": 3, "Tesla": 2}
        assert news == 1


class TestDynamicCounting:
    """When running locally, counts should reflect the actual filesystem."""

    def test_counts_real_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # BMW: 2 reports
            os.makedirs(os.path.join(tmpdir, "BMW"))
            for i in range(2):
                open(os.path.join(tmpdir, "BMW", f"report_{i}.pdf"), "w").close()
            # Ford: 1 report
            os.makedirs(os.path.join(tmpdir, "Ford"))
            open(os.path.join(tmpdir, "Ford", "report_0.pdf"), "w").close()
            # Tesla: nothing
            os.makedirs(os.path.join(tmpdir, "Tesla"))
            # News: 3 articles
            news_dir = os.path.join(tmpdir, "News and Advertisement")
            os.makedirs(news_dir)
            for i in range(3):
                open(os.path.join(news_dir, f"news_{i}.pdf"), "w").close()

            total, per_co, news = count_data_files(tmpdir)
            assert per_co == {"BMW": 2, "Ford": 1, "Tesla": 0}
            assert news == 3
            assert total == 6  # 2+1+0+3

    def test_ignores_subdirectories(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            bmw_dir = os.path.join(tmpdir, "BMW")
            os.makedirs(bmw_dir)
            open(os.path.join(bmw_dir, "real_report.pdf"), "w").close()
            os.makedirs(os.path.join(bmw_dir, "subdir_not_a_file"))
            # Missing other folders — should default to 0
            os.makedirs(os.path.join(tmpdir, "Ford"))
            os.makedirs(os.path.join(tmpdir, "Tesla"))

            total, per_co, news = count_data_files(tmpdir)
            assert per_co["BMW"] == 1  # subdir excluded

    def test_empty_data_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            os.makedirs(os.path.join(tmpdir, "BMW"))
            os.makedirs(os.path.join(tmpdir, "Ford"))
            os.makedirs(os.path.join(tmpdir, "Tesla"))

            total, per_co, news = count_data_files(tmpdir)
            assert total == 0
            assert all(v == 0 for v in per_co.values())
            assert news == 0

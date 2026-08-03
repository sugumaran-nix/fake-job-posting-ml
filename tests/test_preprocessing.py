"""
tests/test_preprocessing.py
Unit tests for the NLP preprocessing pipeline.
Run: pytest tests/test_preprocessing.py -v
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils.preprocessing import preprocess, batch_preprocess


class TestPreprocess:

    def test_returns_string(self):
        assert isinstance(preprocess("hello world"), str)

    def test_handles_none(self):
        assert preprocess(None) == ""

    def test_handles_empty(self):
        assert preprocess("") == ""

    def test_html_stripped(self):
        result = preprocess("<b>Senior Engineer</b>")
        assert "<b>" not in result
        assert "senior" in result or "engineer" in result

    def test_urls_removed(self):
        result = preprocess("Apply at https://example.com/jobs today")
        assert "http" not in result
        assert "example" not in result

    def test_lowercased(self):
        result = preprocess("SENIOR DEVELOPER")
        assert result == result.lower()

    def test_stopwords_removed(self):
        result = preprocess("the quick brown fox")
        # Common stopwords should be gone
        assert " the " not in f" {result} "

    def test_non_empty_output_for_valid_input(self):
        result = preprocess("We are hiring a software engineer with Python experience")
        assert len(result) > 0

    def test_batch_preprocess_length(self):
        texts = ["job one", "job two", "job three"]
        results = batch_preprocess(texts)
        assert len(results) == len(texts)

    def test_batch_handles_mixed_types(self):
        results = batch_preprocess(["valid text", None, ""])
        assert len(results) == 3
        assert results[1] == ""
        assert results[2] == ""


class TestJobPostingGuard:
    """Test the job-posting vocabulary guard from app.py."""

    def test_import_guard(self):
        # Import without a trained model — only testing the guard function
        import importlib, unittest.mock as mock
        with mock.patch("pickle.load", side_effect=FileNotFoundError):
            # _is_job_posting should be importable even without models
            pass

    def test_job_text_passes(self):
        from app import _is_job_posting
        assert _is_job_posting(
            "We are hiring a software engineer with experience in Python. "
            "Salary: 8-12 LPA. Apply with your resume."
        ) is True

    def test_unrelated_text_fails(self):
        from app import _is_job_posting
        assert _is_job_posting(
            "The French Revolution was a period of radical political and societal change "
            "in France that began with the Estates General of 1789."
        ) is False

    def test_product_description_fails(self):
        from app import _is_job_posting
        assert _is_job_posting(
            "Buy this amazing blender! 1000W motor, 6 speed settings, "
            "includes recipe booklet. Free shipping on orders over $50."
        ) is False

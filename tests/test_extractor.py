"""Unit tests for extractor module."""

import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from src.config import Config
from src.extractor import Extractor


def create_sample_pdf(path: str, text: str):
    try:
        import fitz
    except ImportError:
        raise unittest.SkipTest("PyMuPDF is required to run PDF extraction tests.")

    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), text)
    doc.save(path)
    doc.close()


class TestExtractor(unittest.TestCase):
    """Test Extractor module."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = Config()
        self.extractor = Extractor(self.config)

    def test_extract_returns_dict(self):
        """Test that extraction returns proper structure."""
        with patch.object(Extractor, "_call_llm", return_value={
            "extracted": {
                "name": "John Doe",
                "email": "john@example.com",
                "phone": "+1-555-0123",
                "skills": ["Python"],
                "experience": [],
                "education": [],
                "certifications": [],
            },
            "model_name": "gpt-4o",
            "token_usage": {"total_tokens": 12},
        }):
            result = self.extractor.extract("John Doe, john@example.com, +1-555-0123")

        self.assertIn("extracted", result)
        self.assertIn("latency", result)
        self.assertIn("api_calls", result)
        self.assertIsInstance(result["extracted"], dict)

    def test_extraction_has_required_fields(self):
        """Test that extraction includes all required fields."""
        with patch.object(Extractor, "_call_llm", return_value={
            "extracted": {"name": "John Doe", "email": "john@example.com"},
            "model_name": "gpt-4o",
            "token_usage": {"total_tokens": 10},
        }):
            result = self.extractor.extract("John Doe, john@example.com")

        extracted = result["extracted"]
        for field in self.config.schema.fields:
            self.assertIn(field, extracted)

    def test_custom_prompt(self):
        """Test extraction with custom prompt."""
        custom_prompt = "Extract only name: {resume_text}"
        resume_text = "John Doe, john@example.com"

        with patch.object(Extractor, "_call_llm", return_value={
            "extracted": {"name": "John Doe"},
            "model_name": "gpt-4o",
            "token_usage": {},
        }):
            result = self.extractor.extract(resume_text, custom_prompt)

        self.assertEqual(result["prompt_used"], custom_prompt)
        self.assertIsInstance(result["extracted"], dict)

    def test_latency_measurement(self):
        """Test that latency is measured."""
        with patch.object(Extractor, "_call_llm", return_value={
            "extracted": {"name": "John Doe"},
            "model_name": "gpt-4o",
            "token_usage": {},
        }):
            result = self.extractor.extract("John Doe")

        self.assertGreater(result["latency"], 0.0)
        self.assertEqual(result["api_calls"], 1)

    def test_api_call_counting(self):
        """Test API call counting."""
        with patch.object(Extractor, "_call_llm", return_value={
            "extracted": {"name": "Test"},
            "model_name": "gpt-4o",
            "token_usage": {},
        }):
            initial_count = self.extractor.api_call_count
            self.extractor.extract("Resume 1")
            self.extractor.extract("Resume 2")
            self.extractor.extract("Resume 3")

        self.assertEqual(self.extractor.api_call_count, initial_count + 3)

    def test_batch_extraction(self):
        """Test batch extraction."""
        with patch.object(Extractor, "_call_llm", return_value={
            "extracted": {"name": "Batch"},
            "model_name": "gpt-4o",
            "token_usage": {},
        }):
            resumes = ["Resume 1", "Resume 2", "Resume 3"]
            results = self.extractor.extract_batch(resumes)

        self.assertEqual(len(results), 3)
        for result in results:
            self.assertIn("extracted", result)

    def test_stats(self):
        """Test statistics retrieval."""
        with patch.object(Extractor, "_call_llm", return_value={
            "extracted": {"name": "Test"},
            "model_name": "gpt-4o",
            "token_usage": {},
        }):
            for _ in range(5):
                self.extractor.extract("Test resume")

        stats = self.extractor.get_stats()

        self.assertEqual(stats["total_api_calls"], 5)
        self.assertGreater(stats["total_latency"], 0.0)
        self.assertGreater(stats["average_latency"], 0.0)

    def test_baseline_prompt_used_when_none_provided(self):
        """Test that baseline prompt is used when none provided."""
        with patch.object(Extractor, "_call_llm", return_value={
            "extracted": {"name": "John Doe"},
            "model_name": "gpt-4o",
            "token_usage": {},
        }):
            result = self.extractor.extract("John Doe")

        self.assertEqual(result["prompt_used"], self.config.optimizer.baseline_prompt)

    def test_extract_from_pdf_reads_pdf_text(self):
        """Test PDF-to-text extraction before LLM extraction."""
        pdf_text = "John Doe\njohn@example.com\n+1-555-0123"
        with tempfile.TemporaryDirectory() as temp_dir:
            pdf_path = os.path.join(temp_dir, "sample.pdf")
            create_sample_pdf(pdf_path, pdf_text)

            with patch.object(Extractor, "_call_llm", return_value={
                "extracted": {"name": "John Doe", "email": "john@example.com"},
                "model_name": "gpt-4o",
                "token_usage": {},
            }) as mock_call:
                result = self.extractor.extract_from_pdf(pdf_path)

        self.assertEqual(result["extracted"]["name"], "John Doe")
        mock_call.assert_called_once()
        self.assertIn("John Doe", mock_call.call_args[0][0])

    @patch("src.extractor.convert_from_path")
    @patch("src.extractor.pytesseract")
    def test_extract_text_with_ocr_fallback(self, mock_pytesseract, mock_convert_from_path):
        """Test OCR fallback extraction for scanned PDFs."""
        mock_image = MagicMock()
        mock_convert_from_path.return_value = [mock_image]
        mock_pytesseract.image_to_string.return_value = "John Doe\njohn@example.com"

        result = self.extractor._extract_text_with_ocr("dummy.pdf")

        self.assertIn("John Doe", result)
        mock_convert_from_path.assert_called_once_with("dummy.pdf", dpi=300)
        mock_pytesseract.image_to_string.assert_called_once_with(mock_image)

    @patch("src.extractor.openai", new_callable=MagicMock)
    def test_openai_api_call(self, mock_openai):
        """Test OpenAI provider call formatting and parsing."""
        mock_response = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message = {"content": '{"name": "Jane Doe", "email": "jane@example.com", "phone": "", "skills": ["Python"], "experience": [], "education": [], "certifications": []}'}
        mock_response.choices = [mock_choice]
        mock_response.usage = {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3}
        mock_openai.ChatCompletion.create.return_value = mock_response

        self.config.model.name = "gpt-4o"
        self.config.model.api_key = "test-api-key"
        self.config.model.provider = "openai"

        result = self.extractor.extract("Jane Doe")

        self.assertEqual(result["model_name"], "gpt-4o")
        self.assertEqual(result["token_usage"]["total_tokens"], 3)
        self.assertEqual(result["extracted"]["name"], "Jane Doe")
        mock_openai.ChatCompletion.create.assert_called_once()

    @patch("src.extractor.requests")
    def test_gemini_api_call(self, mock_requests):
        """Test Gemini provider call formatting and parsing."""
        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {
            "candidates": [
                {"content": '{"name": "Gemini User", "email": "gemini@example.com", "phone": "", "skills": ["AI"], "experience": [], "education": [], "certifications": []}'}
            ],
            "metadata": {"tokenUsage": {"totalTokens": 10}},
        }
        mock_requests.post.return_value = mock_response

        self.config.model.provider = "gemini"
        self.config.model.name = "gemini-1.0"
        self.config.model.api_key = "test-gemini-key"

        result = self.extractor.extract("Gemini resume")

        self.assertEqual(result["model_name"], "gemini-1.0")
        self.assertEqual(result["token_usage"]["totalTokens"], 10)
        self.assertEqual(result["extracted"]["name"], "Gemini User")
        mock_requests.post.assert_called_once()


if __name__ == "__main__":
    unittest.main()

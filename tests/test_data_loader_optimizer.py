import os
import json
import tempfile
import unittest
from unittest.mock import MagicMock

from src.config import Config
from src.data_loader import DataLoader
from src.optimizer import PromptOptimizer
from src.scorer import Scorer


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


class TestDataLoader(unittest.TestCase):
    def test_loads_pdf_paths_with_annotations(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            gt_dir = os.path.join(temp_dir, "ground_truth")
            resumes_dir = os.path.join(temp_dir, "resumes")
            os.makedirs(gt_dir, exist_ok=True)
            os.makedirs(resumes_dir, exist_ok=True)

            json_path = os.path.join(gt_dir, "sample_1.gold.json")
            annotation = {
                "name": "John Doe",
                "email": "john@example.com",
                "phone": "+1-555-0123",
                "skills": ["Python"],
                "experience": [],
                "education": [],
                "certifications": [],
            }
            with open(json_path, "w") as f:
                json.dump(annotation, f)

            pdf_path = os.path.join(resumes_dir, "sample_1.pdf")
            create_sample_pdf(pdf_path, "John Doe\njohn@example.com\n+1-555-0123")

            config = Config()
            config.data.ground_truth_path = gt_dir
            config.data.dataset_path = resumes_dir

            loader = DataLoader(config)
            _, ground_truth = loader.load_data()

            self.assertIn("sample_1", ground_truth)
            sample = ground_truth["sample_1"]
            self.assertEqual(sample["name"], annotation["name"])
            self.assertEqual(sample["email"], annotation["email"])
            self.assertEqual(sample["pdf_path"], pdf_path)


class TestPromptOptimizer(unittest.TestCase):
    def setUp(self):
        self.config = Config()
        self.logger = MagicMock()
        self.scorer = Scorer(self.config)
        self.extractor = MagicMock()
        self.data_loader = MagicMock()

    def test_evaluate_prompt_uses_pdf_extraction(self):
        sample = {
            "pdf_path": "/tmp/sample_1.pdf",
            "name": "John Doe",
            "email": "john@example.com",
            "phone": "+1-555-0123",
            "skills": ["Python"],
            "experience": [],
            "education": [],
            "certifications": [],
        }
        self.data_loader.get_split.return_value = {"sample_1": sample}
        self.extractor.extract_from_pdf.return_value = {
            "extracted": {
                "name": "John Doe",
                "email": "john@example.com",
                "phone": "+1-555-0123",
                "skills": ["Python"],
                "experience": [],
                "education": [],
                "certifications": [],
            }
        }

        optimizer = PromptOptimizer(self.config, self.extractor, self.scorer, self.data_loader, self.logger)
        score = optimizer._evaluate_prompt("test prompt", [sample])

        self.extractor.extract_from_pdf.assert_called_once_with(sample["pdf_path"], "test prompt")
        self.assertIsInstance(score, dict)
        self.assertEqual(score["num_samples"], 1)
        self.assertEqual(score["f1"], 1.0)

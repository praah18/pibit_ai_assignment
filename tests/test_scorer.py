"""Unit tests for scorer module."""

import unittest
from src.config import Config
from src.scorer import Scorer


class TestScorer(unittest.TestCase):
    """Test Scorer module."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = Config()
        self.scorer = Scorer(self.config)

    def test_score_perfect_match(self):
        """Test scoring with perfect match."""
        predicted = {
            "name": "John Doe",
            "email": "john@example.com",
            "phone": "+1-555-0123",
            "skills": ["Python", "ML"],
            "experience": ["Senior Engineer"],
            "education": ["B.S. Computer Science"],
            "certifications": ["AWS"]
        }
        ground_truth = predicted.copy()

        scores = self.scorer.score(predicted, ground_truth)

        self.assertAlmostEqual(scores["precision"], 1.0, places=2)
        self.assertAlmostEqual(scores["recall"], 1.0, places=2)
        self.assertAlmostEqual(scores["f1"], 1.0, places=2)

    def test_score_partial_match(self):
        """Test scoring with partial match."""
        predicted = {
            "name": "John Doe",
            "email": "john@example.com",
            "phone": "",
            "skills": ["Python"],
            "experience": [],
            "education": [],
            "certifications": []
        }
        ground_truth = {
            "name": "John Doe",
            "email": "john@example.com",
            "phone": "+1-555-0123",
            "skills": ["Python", "ML", "Data Science"],
            "experience": ["Senior Engineer", "ML Engineer"],
            "education": ["B.S. Computer Science"],
            "certifications": ["AWS", "GCP"]
        }

        scores = self.scorer.score(predicted, ground_truth)

        self.assertGreater(scores["f1"], 0.0)
        self.assertLess(scores["f1"], 1.0)

    def test_score_no_match(self):
        """Test scoring with no match."""
        predicted = {
            "name": "",
            "email": "",
            "phone": "",
            "skills": [],
            "experience": [],
            "education": [],
            "certifications": []
        }
        ground_truth = {
            "name": "John Doe",
            "email": "john@example.com",
            "phone": "+1-555-0123",
            "skills": ["Python"],
            "experience": ["Senior Engineer"],
            "education": ["B.S. Computer Science"],
            "certifications": ["AWS"]
        }

        scores = self.scorer.score(predicted, ground_truth)

        self.assertEqual(scores["precision"], 0.0)
        self.assertEqual(scores["recall"], 0.0)
        self.assertEqual(scores["f1"], 0.0)

    def test_evaluate_batch(self):
        """Test batch evaluation."""
        predictions = [
            {"name": "John", "email": "john@ex.com", "phone": "", "skills": [], "experience": [], "education": [], "certifications": []},
            {"name": "Jane", "email": "jane@ex.com", "phone": "+1-555-1234", "skills": ["Python"], "experience": [], "education": [], "certifications": []},
        ]
        ground_truths = [
            {"name": "John Doe", "email": "john@example.com", "phone": "+1-555-0123", "skills": ["Python"], "experience": [], "education": [], "certifications": []},
            {"name": "Jane Smith", "email": "jane@example.com", "phone": "+1-555-1234", "skills": ["Python", "ML"], "experience": [], "education": [], "certifications": []},
        ]

        batch_scores = self.scorer.evaluate_batch(predictions, ground_truths)

        self.assertEqual(batch_scores["num_samples"], 2)
        self.assertIn("f1", batch_scores)
        self.assertIn("individual_scores", batch_scores)

    def test_score_list_fields(self):
        """Test scoring list fields."""
        predicted = {
            "name": "",
            "email": "",
            "phone": "",
            "skills": ["Python", "Java"],
            "experience": [],
            "education": [],
            "certifications": []
        }
        ground_truth = {
            "name": "",
            "email": "",
            "phone": "",
            "skills": ["Python", "Java", "C++"],
            "experience": [],
            "education": [],
            "certifications": []
        }

        scores = self.scorer.score(predicted, ground_truth)

        # Should have partial match on skills
        self.assertGreater(scores["f1"], 0.0)
        self.assertLess(scores["f1"], 1.0)

    def test_invalid_input(self):
        """Test scoring with invalid input."""
        scores = self.scorer.score("invalid", {})

        self.assertEqual(scores["precision"], 0.0)
        self.assertEqual(scores["recall"], 0.0)
        self.assertEqual(scores["f1"], 0.0)


if __name__ == "__main__":
    unittest.main()

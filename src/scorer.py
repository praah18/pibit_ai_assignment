"""Scoring module for evaluating extraction quality."""

import json
from typing import Dict, List


class Scorer:
    """Compute precision, recall, and F1 metrics for structured extraction."""

    def __init__(self, config):
        """Initialize scorer with configuration."""
        self.config = config
        self.threshold = config.scoring.threshold

    def score(self, predicted: Dict, ground_truth: Dict) -> Dict[str, float]:
        """
        Score a single prediction against ground truth.

        Args:
            predicted: Extracted JSON data
            ground_truth: Expected/annotated JSON data

        Returns:
            Dict with precision, recall, f1 scores
        """
        if not isinstance(predicted, dict) or not isinstance(ground_truth, dict):
            return self._zero_scores()

        # Calculate metrics per field
        field_scores = {}
        for field in self.config.schema.fields:
            pred_value = predicted.get(field, "")
            gt_value = ground_truth.get(field, "")

            score = self._score_field(pred_value, gt_value, field)
            field_scores[field] = score

        # Aggregate scores across fields
        return self._aggregate_scores(field_scores)

    def _score_field(self, predicted_value, ground_truth_value, field_name: str) -> Dict[str, float]:
        """Score a single field."""
        # Handle different field types
        if field_name in ["skills", "experience", "education", "certifications"]:
            return self._score_list_field(predicted_value, ground_truth_value)
        else:
            return self._score_string_field(predicted_value, ground_truth_value)

    def _score_string_field(self, predicted: str, ground_truth: str) -> Dict[str, float]:
        """Score string fields using token overlap."""
        if not ground_truth:
            return {"precision": 1.0 if not predicted else 0.0, "recall": 1.0, "f1": 1.0}

        pred_tokens = set(str(predicted).lower().split())
        gt_tokens = set(str(ground_truth).lower().split())

        if not gt_tokens:
            return {"precision": 1.0 if not pred_tokens else 0.0, "recall": 1.0, "f1": 1.0}

        intersection = len(pred_tokens & gt_tokens)
        precision = intersection / len(pred_tokens) if pred_tokens else 0.0
        recall = intersection / len(gt_tokens) if gt_tokens else 0.0

        return {
            "precision": precision,
            "recall": recall,
            "f1": self._compute_f1(precision, recall)
        }

    def _score_list_field(self, predicted: List, ground_truth: List) -> Dict[str, float]:
        """Score list fields using item overlap."""
        if not isinstance(predicted, list):
            predicted = [] if not predicted else [predicted]
        if not isinstance(ground_truth, list):
            ground_truth = [] if not ground_truth else [ground_truth]

        if not ground_truth:
            return {"precision": 1.0 if not predicted else 0.0, "recall": 1.0, "f1": 1.0}

        # Simple overlap: count matching items
        pred_set = set(str(x).lower() for x in predicted)
        gt_set = set(str(x).lower() for x in ground_truth)

        intersection = len(pred_set & gt_set)
        precision = intersection / len(pred_set) if pred_set else 0.0
        recall = intersection / len(gt_set) if gt_set else 0.0

        return {
            "precision": precision,
            "recall": recall,
            "f1": self._compute_f1(precision, recall)
        }

    def _compute_f1(self, precision: float, recall: float) -> float:
        """Compute F1 score."""
        if precision + recall == 0:
            return 0.0
        return 2 * (precision * recall) / (precision + recall)

    def _aggregate_scores(self, field_scores: Dict[str, Dict]) -> Dict[str, float]:
        """Aggregate scores across all fields."""
        if not field_scores:
            return self._zero_scores()

        total_precision = sum(s["precision"] for s in field_scores.values())
        total_recall = sum(s["recall"] for s in field_scores.values())
        total_f1 = sum(s["f1"] for s in field_scores.values())

        num_fields = len(field_scores)

        return {
            "precision": total_precision / num_fields,
            "recall": total_recall / num_fields,
            "f1": total_f1 / num_fields,
            "field_scores": field_scores,
        }

    def _zero_scores(self) -> Dict[str, float]:
        """Return zero scores."""
        return {
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "field_scores": {},
        }

    def evaluate_batch(self, predictions: List[Dict], ground_truths: List[Dict]) -> Dict:
        """
        Evaluate a batch of predictions.

        Returns:
            Aggregated metrics across all samples
        """
        if len(predictions) != len(ground_truths):
            raise ValueError("Number of predictions must match ground truths")

        scores = [self.score(pred, gt) for pred, gt in zip(predictions, ground_truths)]

        if not scores:
            return self._zero_scores()

        # Average metrics
        avg_precision = sum(s["precision"] for s in scores) / len(scores)
        avg_recall = sum(s["recall"] for s in scores) / len(scores)
        avg_f1 = sum(s["f1"] for s in scores) / len(scores)

        return {
            "precision": avg_precision,
            "recall": avg_recall,
            "f1": avg_f1,
            "num_samples": len(scores),
            "individual_scores": scores,
        }

"""Data loader for resume dataset."""

import os
import json
from typing import List, Dict, Tuple
import random


class DataLoader:
    """Load and manage resume dataset with ground truth annotations."""

    def __init__(self, config):
        """Initialize data loader with configuration."""
        self.config = config
        self.resumes = []
        self.ground_truth = {}
        self.splits = {}
        self.total_pdfs = 0
        self.total_annotations = 0
        self.total_matches = 0
        self.unmatched_pdfs = []
        self.unmatched_annotations = []

    def load_data(self) -> Tuple[List[Dict], Dict]:
        """Load resume data and ground truth."""
        self._load_ground_truth()
        self._create_data_splits()
        self.resumes = list(self.ground_truth.values())
        return self.resumes, self.ground_truth

    def _load_ground_truth(self):
        """Load ground truth annotations from JSON files and match PDF paths."""
        gt_path = self.config.data.ground_truth_path
        dataset_path = self.config.data.dataset_path

        self.ground_truth = {}
        self.unmatched_pdfs = []
        self.unmatched_annotations = []
        self.total_pdfs = 0
        self.total_annotations = 0
        self.total_matches = 0

        if not os.path.exists(gt_path):
            raise FileNotFoundError(f"Ground truth path not found: {gt_path}")
        if not os.path.isdir(gt_path):
            raise ValueError(f"Ground truth path is not a directory: {gt_path}")
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Dataset path not found: {dataset_path}")
        if not os.path.isdir(dataset_path):
            raise ValueError(f"Dataset path is not a directory: {dataset_path}")

        pdf_index = {}
        for filename in os.listdir(dataset_path):
            if filename.lower().endswith(".pdf"):
                sample_key = os.path.splitext(filename)[0]
                pdf_index[sample_key] = os.path.join(dataset_path, filename)
        self.total_pdfs = len(pdf_index)

        annotation_index = {}
        for filename in os.listdir(gt_path):
            if filename.lower().endswith(".json"):
                sample_key = filename[: -len(".json")]
                if sample_key.endswith(".gold"):
                    sample_key = sample_key[: -len(".gold")]
                filepath = os.path.join(gt_path, filename)
                try:
                    with open(filepath, "r") as f:
                        annotation_index[sample_key] = json.load(f)
                except Exception as e:
                    print(f"Error loading ground truth {filename}: {e}")
                    continue
        self.total_annotations = len(annotation_index)

        for sample_key, annotation in annotation_index.items():
            pdf_path = pdf_index.get(sample_key)
            if not pdf_path:
                self.unmatched_annotations.append(sample_key)
                continue

            sample_record = {
                "pdf_path": pdf_path,
                **annotation,
            }
            self.ground_truth[sample_key] = sample_record
            self.total_matches += 1

        for sample_key in pdf_index:
            if sample_key not in self.ground_truth:
                self.unmatched_pdfs.append(sample_key)

        if self.total_matches == 0:
            raise RuntimeError(
                f"No matched PDF/annotation pairs found. "
                f"PDFs={self.total_pdfs}, annotations={self.total_annotations}, "
                f"dataset_path={dataset_path}, ground_truth_path={gt_path}"
            )

    def _create_data_splits(self):
        """Create train/val/test splits from ground truth."""
        keys = list(self.ground_truth.keys())
        random.shuffle(keys)

        total = len(keys)
        train_size = int(total * self.config.data.train_split)
        val_size = int(total * self.config.data.val_split)

        self.splits = {
            "train": keys[:train_size],
            "val": keys[train_size:train_size + val_size],
            "test": keys[train_size + val_size:],
        }

    def get_split(self, split: str = "train") -> Dict[str, Dict]:
        """Get data split (train/val/test)."""
        if split not in self.splits:
            raise ValueError(f"Unknown split: {split}")

        return {
            key: self.ground_truth[key] for key in self.splits[split]
        }

    def get_sample(self, key: str) -> Dict:
        """Get a specific ground truth sample."""
        return self.ground_truth.get(key, {})

    def add_sample(self, key: str, ground_truth_data: Dict):
        """Add a ground truth sample."""
        self.ground_truth[key] = ground_truth_data

    def get_stats(self) -> Dict:
        """Get dataset statistics."""
        return {
            "total_pdfs": self.total_pdfs,
            "total_annotations": self.total_annotations,
            "total_matches": self.total_matches,
            "total_samples": len(self.ground_truth),
            "train_samples": len(self.splits.get("train", [])),
            "val_samples": len(self.splits.get("val", [])),
            "test_samples": len(self.splits.get("test", [])),
            "schema_fields": self.config.schema.fields,
            "unmatched_pdfs": self.unmatched_pdfs,
            "unmatched_annotations": self.unmatched_annotations,
        }

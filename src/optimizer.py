"""Prompt optimization module using greedy strategy."""

import json
import random
from typing import Dict, List, Tuple


class PromptOptimizer:
    """Greedy prompt optimization for improving extraction quality."""

    def __init__(self, config, extractor, scorer, data_loader, logger):
        """
        Initialize optimizer.

        Args:
            config: Configuration object
            extractor: Extractor instance
            scorer: Scorer instance
            data_loader: DataLoader instance
            logger: Logger instance
        """
        self.config = config
        self.extractor = extractor
        self.scorer = scorer
        self.data_loader = data_loader
        self.logger = logger

        self.current_best_prompt = config.optimizer.baseline_prompt
        self.current_best_score = 0.0
        self.iteration_history = []
        self.no_improvement_count = 0

    def optimize(self, resume_state: Dict = None) -> Dict:
        """
        Run greedy prompt optimization loop.

        Args:
            resume_state: Previous state to resume from

        Returns:
            Optimization results
        """
        start_iteration = 0
        if resume_state:
            self.current_best_prompt = resume_state.get("best_prompt", self.current_best_prompt)
            self.current_best_score = resume_state.get("best_score", 0.0)
            self.iteration_history = resume_state.get("history", [])
            self.no_improvement_count = resume_state.get("no_improvement_count", 0)
            start_iteration = resume_state.get("iteration", 0)

        val_data = self.data_loader.get_split("val")
        val_resumes = list(val_data.values())[:self.config.optimizer.validation_set_size]

        self.logger.log_event("INFO", f"Starting optimization from iteration {start_iteration}")
        self.logger.log_event("INFO", f"Validation set size: {len(val_resumes)}")

        for iteration in range(start_iteration, self.config.optimizer.max_iterations):
            self.logger.log_event("INFO", f"=== Iteration {iteration} ===")

            # Generate prompt variations
            prompt_variations = self._generate_variations(self.current_best_prompt)

            # Evaluate each variation
            iteration_scores = {}
            for var_idx, prompt in enumerate(prompt_variations):
                self.logger.log_prompt(iteration, prompt, var_idx)

                # Extract and score
                score = self._evaluate_prompt(prompt, val_resumes)
                iteration_scores[var_idx] = {
                    "prompt": prompt,
                    "score": score,
                }

                self.logger.log_score(iteration, f"variation_{var_idx}", score["f1"])

            # Find best variation
            best_variation = max(
                iteration_scores.items(),
                key=lambda x: x[1]["score"]["f1"]
            )
            best_var_idx, best_var_data = best_variation
            best_var_score = best_var_data["score"]["f1"]

            # Update best if improved
            if best_var_score > self.current_best_score:
                self.current_best_prompt = best_var_data["prompt"]
                self.current_best_score = best_var_score
                self.no_improvement_count = 0
                improved = True
                self.logger.log_event(
                    "INFO",
                    f"Improved! New best F1: {best_var_score:.4f}"
                )
            else:
                improved = False
                self.no_improvement_count += 1
                self.logger.log_event(
                    "INFO",
                    f"No improvement. Best F1 remains: {self.current_best_score:.4f}"
                )

            # Log iteration results
            iteration_data = {
                "iteration": iteration,
                "best_score": self.current_best_score,
                "best_variation": best_var_idx,
                "improved": improved,
                "variation_scores": {
                    str(k): v["score"]["f1"]
                    for k, v in iteration_scores.items()
                },
                "no_improvement_count": self.no_improvement_count,
            }
            self.iteration_history.append(iteration_data)
            self.logger.log_iteration(iteration, iteration_data)

            # Early stopping
            if self.no_improvement_count >= self.config.optimizer.early_stopping_patience:
                self.logger.log_event(
                    "INFO",
                    f"Early stopping triggered after {iteration + 1} iterations"
                )
                break

        return self._get_results()

    def _generate_variations(self, base_prompt: str) -> List[str]:
        """Generate prompt variations using different strategies."""
        variations = [base_prompt]  # Keep baseline

        # Strategy 1: Add examples
        examples_prompt = self._add_examples_to_prompt(base_prompt)
        variations.append(examples_prompt)

        # Strategy 2: Increase clarity/detail
        detailed_prompt = self._make_prompt_more_detailed(base_prompt)
        variations.append(detailed_prompt)

        return variations[:self.config.optimizer.prompt_variations]

    def _add_examples_to_prompt(self, prompt: str) -> str:
        """Add examples to the prompt."""
        example_section = """

Example output format:
{
    "name": "Jane Smith",
    "email": "jane@example.com",
    "phone": "+1-555-1234",
    "skills": ["Python", "Machine Learning"],
    "experience": [{"title": "Data Scientist", "company": "TechCorp", "duration": "2020-2023"}],
    "education": [{"degree": "M.S. Computer Science", "institution": "University", "graduation": "2020"}],
    "certifications": ["AWS Certified"]
}
"""
        return prompt + example_section

    def _make_prompt_more_detailed(self, prompt: str) -> str:
        """Make the prompt more detailed with additional instructions."""
        additions = """

Additional instructions:
- Be thorough and extract all available information
- For lists, include all entries you can find
- If information is not found, use empty string or empty list
- Ensure JSON is valid and properly formatted
- Extract exact text from resume when possible
"""
        return prompt + additions

    def _evaluate_prompt(self, prompt: str, val_samples: List[Dict]) -> Dict:
        """
        Evaluate a prompt on validation samples.

        Returns average scores across samples.
        """
        predictions = []
        ground_truths = []

        for sample in val_samples:
            pdf_path = sample.get("pdf_path")
            if not pdf_path:
                self.logger.log_event("WARNING", "Skipping validation sample without PDF path", {"sample": sample})
                continue

            # Extract from actual PDF file
            try:
                result = self.extractor.extract_from_pdf(pdf_path, prompt)
            except Exception as exc:
                self.logger.log_event(
                    "WARNING",
                    "Skipping validation sample due to extraction error",
                    {"pdf_path": pdf_path, "error": str(exc)}
                )
                continue

            predictions.append(result["extracted"])
            ground_truths.append(self._get_ground_truth_annotation(sample))

        # Score
        return self.scorer.evaluate_batch(predictions, ground_truths)

    def _get_ground_truth_annotation(self, sample: Dict) -> Dict:
        """Return a ground truth annotation dictionary without PDF path metadata."""
        annotation = {}
        for field in self.config.schema.fields:
            if field in sample:
                annotation[field] = sample[field]
            else:
                annotation[field] = [] if field in ["skills", "experience", "education", "certifications"] else ""

        return annotation

    def _get_results(self) -> Dict:
        """Get optimization results."""
        return {
            "best_prompt": self.current_best_prompt,
            "best_score": self.current_best_score,
            "iterations": len(self.iteration_history),
            "history": self.iteration_history,
            "no_improvement_count": self.no_improvement_count,
        }

    def get_state(self) -> Dict:
        """Get current optimization state for resumption."""
        return {
            "best_prompt": self.current_best_prompt,
            "best_score": self.current_best_score,
            "history": self.iteration_history,
            "no_improvement_count": self.no_improvement_count,
            "iteration": len(self.iteration_history),
        }

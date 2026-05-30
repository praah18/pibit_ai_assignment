"""Report generation for optimization results."""

import json
import os
from typing import Dict
from datetime import datetime


class ReportGenerator:
    """Generate comprehensive reports from optimization results."""

    def __init__(self, config, logger):
        """Initialize report generator."""
        self.config = config
        self.logger = logger
        self.output_dir = "outputs"
        os.makedirs(self.output_dir, exist_ok=True)

    def generate_optimization_report(self, opt_results: Dict, filename: str = "REPORT.md") -> str:
        """
        Generate comprehensive optimization report.

        Args:
            opt_results: Results from optimizer
            filename: Output filename

        Returns:
            Path to generated report
        """
        filepath = os.path.join(self.output_dir, filename)

        with open(filepath, "w") as f:
            f.write(self._get_header())
            f.write(self._get_optimization_summary(opt_results))
            f.write(self._get_iteration_details(opt_results))
            f.write(self._get_logging_stats())
            f.write(self._get_api_stats())
            f.write(self._get_best_prompt(opt_results))
            f.write(self._get_recommendations())

        print(f"Report generated: {filepath}")
        return filepath

    def _get_header(self) -> str:
        """Generate report header."""
        return f"""# PIBIT AI Assignment - Automated Prompt Optimization Report

**Generated:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

---

## Executive Summary

This report documents the results of automated prompt optimization for structured extraction from resumes using LLMs with a greedy optimization strategy.

"""

    def _get_optimization_summary(self, opt_results: Dict) -> str:
        """Generate optimization summary section."""
        best_score = opt_results.get("best_score", 0.0)
        iterations = opt_results.get("iterations", 0)

        return f"""## Optimization Results

| Metric | Value |
|--------|-------|
| Best F1 Score | {best_score:.4f} |
| Total Iterations | {iterations} |
| Strategy | {self.config.optimizer.strategy} |
| Max Iterations Configured | {self.config.optimizer.max_iterations} |
| Early Stopping Patience | {self.config.optimizer.early_stopping_patience} |

### Performance Over Iterations

"""

    def _get_iteration_details(self, opt_results: Dict) -> str:
        """Generate detailed iteration-by-iteration results."""
        history = opt_results.get("history", [])

        text = "## Iteration Details\n\n"
        text += "| Iteration | Best Score | Improved | No Improvement Count |\n"
        text += "|-----------|-----------|----------|---------------------|\n"

        for item in history:
            iteration = item.get("iteration", 0)
            best_score = item.get("best_score", 0.0)
            improved = "✓" if item.get("improved") else "✗"
            no_imp_count = item.get("no_improvement_count", 0)

            text += f"| {iteration} | {best_score:.4f} | {improved} | {no_imp_count} |\n"

        text += "\n"
        return text

    def _get_logging_stats(self) -> str:
        """Generate logging statistics section."""
        log_summary = self.logger.get_summary()
        prompt_history = self.logger.get_prompt_history()

        text = "## Logging Statistics\n\n"
        text += f"- Total Prompts Logged: {log_summary['total_prompts']}\n"
        text += f"- Total Scores Logged: {log_summary['total_scores']}\n"
        text += f"- Total Iterations: {log_summary['iterations_logged']}\n"

        if prompt_history:
            text += f"\n### Prompt Analysis\n"
            avg_length = sum(p.get("prompt_length", 0) for p in prompt_history) / len(prompt_history)
            min_length = min(p.get("prompt_length", 0) for p in prompt_history)
            max_length = max(p.get("prompt_length", 0) for p in prompt_history)

            text += f"- Average Prompt Length: {avg_length:.0f} characters\n"
            text += f"- Min Prompt Length: {min_length} characters\n"
            text += f"- Max Prompt Length: {max_length} characters\n"

        text += "\n"
        return text

    def _get_api_stats(self) -> str:
        """Generate API statistics section."""
        api_stats = self.logger.get_api_call_stats()
        latency_stats = self.logger.get_latency_stats()

        text = "## API and Performance Statistics\n\n"

        text += "### API Calls\n"
        text += f"- Total API Calls: {api_stats.get('total_calls', 0)}\n"
        text += f"- Total Tokens: {api_stats.get('total_tokens', 0)}\n"
        text += f"- Total Cost: ${api_stats.get('total_cost', 0.0):.4f}\n"
        text += f"- Models Used: {', '.join(api_stats.get('models_used', []))}\n"

        text += "\n### Latency\n"
        text += f"- Total Time: {latency_stats.get('total_time', 0.0):.2f} seconds\n"
        text += f"- Average Latency: {latency_stats.get('avg_latency', 0.0):.4f} seconds\n"
        text += f"- Min Latency: {latency_stats.get('min_latency', 0.0):.4f} seconds\n"
        text += f"- Max Latency: {latency_stats.get('max_latency', 0.0):.4f} seconds\n"

        text += "\n"
        return text

    def _get_best_prompt(self, opt_results: Dict) -> str:
        """Generate best prompt section."""
        best_prompt = opt_results.get("best_prompt", "")

        text = "## Best Optimized Prompt\n\n"
        text += "```\n"
        text += best_prompt
        text += "\n```\n\n"
        return text

    def _get_recommendations(self) -> str:
        """Generate recommendations section."""
        best_score = self.logger.get_summary()
        latency_stats = self.logger.get_latency_stats()

        text = "## Recommendations\n\n"

        if best_score.get("total_scores", 0) < 10:
            text += "- **More Iterations:** Consider running more optimization iterations for better results.\n"

        if latency_stats.get("total_time", 0) > 300:
            text += "- **Optimize Latency:** High latency detected. Consider using a faster model or optimizing batch processing.\n"

        text += "- **Manual Review:** Review the best optimized prompt and consider fine-tuning it based on domain knowledge.\n"
        text += "- **Extended Testing:** Evaluate the best prompt on the test set before production deployment.\n"
        text += "- **Prompt Versioning:** Document and version the prompts for reproducibility.\n"

        text += "\n---\n\n"
        text += "*End of Report*\n"
        return text

    def generate_summary_json(self, opt_results: Dict, filename: str = "optimization_summary.json") -> str:
        """Generate JSON summary of results."""
        filepath = os.path.join(self.output_dir, filename)

        summary = {
            "timestamp": datetime.now().isoformat(),
            "optimization_results": opt_results,
            "config": self.config.to_dict(),
            "logging_stats": self.logger.get_summary(),
            "api_stats": self.logger.get_api_call_stats(),
            "latency_stats": self.logger.get_latency_stats(),
        }

        with open(filepath, "w") as f:
            json.dump(summary, f, indent=2)

        return filepath

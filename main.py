"""Main entry point for PIBIT AI assignment."""

import argparse
import sys
import json
from src.config import Config
from src.data_loader import DataLoader
from src.extractor import Extractor
from src.scorer import Scorer
from src.logger import Logger
from src.optimizer import PromptOptimizer
from src.resume_manager import ResumeManager
from src.report_generator import ReportGenerator


def main():
    """Main orchestration function."""
    parser = argparse.ArgumentParser(
        description="PIBIT AI - Automated Prompt Optimization for Structured Extraction"
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Path to configuration JSON file"
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        help="Maximum optimization iterations"
    )
    parser.add_argument(
        "--clear-checkpoints",
        action="store_true",
        help="Clear existing checkpoints and start fresh"
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Disable resumption from checkpoint"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging"
    )

    args = parser.parse_args()

    try:
        # Load configuration
        config = Config(args.config)

        # Override with command-line arguments
        if args.max_iterations:
            config.optimizer.max_iterations = args.max_iterations
        if args.debug:
            config.logging.log_level = "DEBUG"
        if args.no_resume:
            config.resume.auto_resume = False

        print("=" * 60)
        print("PIBIT AI - Automated Prompt Optimization")
        print("=" * 60)
        print()

        # Initialize components
        logger = Logger(config)
        logger.log_event("INFO", "Initializing components...")

        data_loader = DataLoader(config)
        data_loader.load_data()
        logger.log_event("INFO", f"Dataset stats: {data_loader.get_stats()}")

        extractor = Extractor(config)
        scorer = Scorer(config)

        # Resume or start fresh
        resume_manager = ResumeManager(config)

        if args.clear_checkpoints:
            logger.log_event("INFO", "Clearing existing checkpoints...")
            resume_manager.clear_checkpoints()

        resume_state = None
        if config.resume.auto_resume and resume_manager.should_resume():
            resume_state = resume_manager.get_resume_state()
            if resume_state:
                logger.log_event("INFO", "Resuming from checkpoint...")
            else:
                logger.log_event("WARNING", "Could not resume from checkpoint")

        # Run optimization
        logger.log_event("INFO", "Starting prompt optimization...")
        optimizer = PromptOptimizer(config, extractor, scorer, data_loader, logger)

        results = optimizer.optimize(resume_state)

        # Save checkpoint after optimization completes
        logger.log_event("INFO", "Saving final checkpoint...")
        resume_manager.save_checkpoint(results["iterations"], optimizer.get_state())

        # Evaluate on test set
        logger.log_event("INFO", "Evaluating on test set...")
        test_results = _evaluate_on_test_set(
            config, optimizer.current_best_prompt,
            extractor, scorer, data_loader, logger
        )

        logger.log_event("INFO", "Test Set Performance:")
        logger.log_event("INFO", f"  Precision: {test_results['precision']:.4f}")
        logger.log_event("INFO", f"  Recall: {test_results['recall']:.4f}")
        logger.log_event("INFO", f"  F1: {test_results['f1']:.4f}")

        # Generate reports
        logger.log_event("INFO", "Generating reports...")
        report_gen = ReportGenerator(config, logger)
        report_path = report_gen.generate_optimization_report(results)
        summary_path = report_gen.generate_summary_json(results)

        # Save logger outputs
        logger.save_logs()

        print()
        print("=" * 60)
        print("Optimization Complete!")
        print("=" * 60)
        print(f"Best F1 Score: {results['best_score']:.4f}")
        print(f"Total Iterations: {results['iterations']}")
        print(f"Test F1 Score: {test_results['f1']:.4f}")
        print()
        print(f"Report: {report_path}")
        print(f"Summary: {summary_path}")
        print()

        return 0

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


def _evaluate_on_test_set(config, best_prompt, extractor, scorer, data_loader, logger):
    """Evaluate best prompt on test set."""
    test_data = data_loader.get_split("test")
    test_samples = list(test_data.values())

    predictions = []
    ground_truths = []

    for sample in test_samples:
        pdf_path = sample.get("pdf_path")
        if not pdf_path:
            logger.log_event("WARNING", "Skipping test sample without PDF path", {"sample_key": sample.get("id", "unknown")})
            continue

        try:
            result = extractor.extract_from_pdf(pdf_path, best_prompt)
        except Exception as exc:
            logger.log_event(
                "WARNING",
                "Skipping test sample due to extraction error",
                {"pdf_path": pdf_path, "error": str(exc)}
            )
            continue

        predictions.append(result["extracted"])
        ground_truths.append(_get_ground_truth_annotation(sample, config))

    return scorer.evaluate_batch(predictions, ground_truths)


def _get_ground_truth_annotation(sample, config):
    """Return a ground truth annotation dictionary without PDF path metadata."""
    annotation = {}
    for field in config.schema.fields:
        if field in sample:
            annotation[field] = sample[field]
        else:
            annotation[field] = [] if field in ["skills", "experience", "education", "certifications"] else ""
    return annotation


if __name__ == "__main__":
    sys.exit(main())

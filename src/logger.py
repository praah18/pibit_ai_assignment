"""Logging system for tracking prompts, scores, latency, and API calls."""

import json
import os
from typing import Dict, Any, List
from datetime import datetime


class Logger:
    """Comprehensive logging for optimization process."""

    def __init__(self, config):
        """Initialize logger with configuration."""
        self.config = config
        self.log_dir = config.logging.log_dir
        os.makedirs(self.log_dir, exist_ok=True)

        self.logs = {
            "prompts": [],
            "scores": [],
            "api_calls": [],
            "latencies": [],
            "iterations": [],
        }

    def log_prompt(self, iteration: int, prompt: str, variation_num: int = 0):
        """Log a prompt variant."""
        if not self.config.logging.log_prompts:
            return

        entry = {
            "timestamp": datetime.now().isoformat(),
            "iteration": iteration,
            "variation": variation_num,
            "prompt": prompt,
            "prompt_length": len(prompt),
        }
        self.logs["prompts"].append(entry)

    def log_score(self, iteration: int, metric_name: str, value: float, details: Dict = None):
        """Log a score metric."""
        if not self.config.logging.log_scores:
            return

        entry = {
            "timestamp": datetime.now().isoformat(),
            "iteration": iteration,
            "metric": metric_name,
            "value": value,
        }
        if details:
            entry["details"] = details

        self.logs["scores"].append(entry)

    def log_api_call(self, model: str, tokens_used: int = 0, cost: float = 0.0):
        """Log an API call."""
        if not self.config.logging.log_api_calls:
            return

        entry = {
            "timestamp": datetime.now().isoformat(),
            "model": model,
            "tokens": tokens_used,
            "cost": cost,
        }
        self.logs["api_calls"].append(entry)

    def log_latency(self, operation: str, latency: float):
        """Log operation latency."""
        if not self.config.logging.log_latency:
            return

        entry = {
            "timestamp": datetime.now().isoformat(),
            "operation": operation,
            "latency_seconds": latency,
        }
        self.logs["latencies"].append(entry)

    def log_iteration(self, iteration: int, data: Dict[str, Any]):
        """Log entire iteration data."""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "iteration": iteration,
            **data,
        }
        self.logs["iterations"].append(entry)

    def log_event(self, level: str, message: str, extra_data: Dict = None):
        """Log a general event."""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "level": level,
            "message": message,
        }
        if extra_data:
            entry.update(extra_data)

        if self.config.logging.log_to_console:
            if extra_data:
                print(f"[{level}] {message} | details: {extra_data}")
            else:
                print(f"[{level}] {message}")

        if self.config.logging.log_to_file:
            self._append_to_log_file(entry)

    def _append_to_log_file(self, entry: Dict):
        """Append event to main log file."""
        log_file = os.path.join(self.log_dir, "events.jsonl")
        with open(log_file, "a") as f:
            f.write(json.dumps(entry) + "\n")

    def save_logs(self, filename: str = "optimization_logs.json"):
        """Save all logs to file."""
        filepath = os.path.join(self.log_dir, filename)
        with open(filepath, "w") as f:
            json.dump(self.logs, f, indent=2)
        self.log_event("INFO", f"Logs saved to {filepath}")

    def load_logs(self, filename: str = "optimization_logs.json") -> Dict:
        """Load logs from file."""
        filepath = os.path.join(self.log_dir, filename)
        if not os.path.exists(filepath):
            self.log_event("WARNING", f"Log file not found: {filepath}")
            return {}

        try:
            with open(filepath, "r") as f:
                self.logs = json.load(f)
            return self.logs
        except Exception as e:
            self.log_event("ERROR", f"Error loading logs: {e}")
            return {}

    def get_summary(self) -> Dict:
        """Get summary of all logs."""
        return {
            "total_prompts": len(self.logs["prompts"]),
            "total_scores": len(self.logs["scores"]),
            "total_api_calls": len(self.logs["api_calls"]),
            "total_latency": sum(l.get("latency_seconds", 0) for l in self.logs["latencies"]),
            "iterations_logged": len(self.logs["iterations"]),
        }

    def get_prompt_history(self) -> List[Dict]:
        """Get all logged prompts."""
        return self.logs["prompts"]

    def get_score_history(self) -> List[Dict]:
        """Get all logged scores."""
        return self.logs["scores"]

    def get_api_call_stats(self) -> Dict:
        """Get API call statistics."""
        calls = self.logs["api_calls"]
        if not calls:
            return {"total_calls": 0, "total_cost": 0.0, "models_used": []}

        return {
            "total_calls": len(calls),
            "total_cost": sum(c.get("cost", 0.0) for c in calls),
            "total_tokens": sum(c.get("tokens", 0) for c in calls),
            "models_used": list(set(c.get("model") for c in calls if c.get("model"))),
        }

    def get_latency_stats(self) -> Dict:
        """Get latency statistics."""
        latencies = self.logs["latencies"]
        if not latencies:
            return {"total_time": 0.0, "avg_latency": 0.0, "operations": []}

        lats = [l["latency_seconds"] for l in latencies]
        return {
            "total_time": sum(lats),
            "avg_latency": sum(lats) / len(lats),
            "min_latency": min(lats),
            "max_latency": max(lats),
            "operations_logged": len(latencies),
        }

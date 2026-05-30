"""Configuration management for PIBIT AI assignment."""

import os
from dataclasses import dataclass, asdict
from typing import Dict, List
import json


@dataclass
class ExtractBenchSchema:
    """Resume extraction schema based on ExtractBench."""
    fields: List[str] = None

    def __post_init__(self):
        if self.fields is None:
            self.fields = [
                "name",
                "email",
                "phone",
                "skills",
                "experience",
                "education",
                "certifications"
            ]


@dataclass
class ModelConfig:
    """LLM model configuration."""
    provider: str = "openai"
    name: str = "gemini-1.5-flash"
    api_key: str = ""
    temperature: float = 0.7
    max_tokens: int = 2000
    retry_attempts: int = 3
    retry_backoff_seconds: float = 1.0


@dataclass
class DataConfig:
    """Dataset configuration."""
    dataset_path: str = "/Users/prachirai/Desktop/extract-bench-main/dataset/hiring/resume/pdf+gold"
    ground_truth_path: str = "/Users/prachirai/Desktop/extract-bench-main/dataset/hiring/resume/pdf+gold"
    train_split: float = 0.7
    val_split: float = 0.2
    test_split: float = 0.1
    max_samples: int = -1  # -1 means all


@dataclass
class ScoringConfig:
    """Scoring and evaluation configuration."""
    metric: str = "f1"  # f1, precision, recall
    threshold: float = 0.5
    weights: Dict[str, float] = None

    def __post_init__(self):
        if self.weights is None:
            self.weights = {
                "precision": 0.5,
                "recall": 0.5,
                "f1": 1.0
            }


@dataclass
class OptimizerConfig:
    """Prompt optimization configuration."""
    strategy: str = "greedy"
    max_iterations: int = 5
    prompt_variations: int = 3
    baseline_prompt: str = ""
    validation_set_size: int = 10
    early_stopping_patience: int = 3

    def __post_init__(self):
        if not self.baseline_prompt:
            self.baseline_prompt = self._get_default_prompt()

    def _get_default_prompt(self) -> str:
        """Default baseline prompt for resume extraction."""
        return """Extract structured information from the following resume text.
Return a JSON object with the following fields:
- name: Full name of the person
- email: Email address
- phone: Phone number
- skills: List of technical and professional skills
- experience: Work experience entries (job title, company, dates)
- education: Educational background (degree, institution, graduation date)
- certifications: Professional certifications

Resume text:
{resume_text}

Return only valid JSON, no additional text."""


@dataclass
class LoggingConfig:
    """Logging configuration."""
    log_dir: str = "outputs/logs"
    log_level: str = "INFO"
    log_to_console: bool = True
    log_to_file: bool = True
    log_prompts: bool = True
    log_scores: bool = True
    log_latency: bool = True
    log_api_calls: bool = True


@dataclass
class ResumeConfig:
    """Resume/checkpoint configuration."""
    checkpoint_dir: str = "outputs/checkpoints"
    save_interval: int = 5  # Save every N iterations
    auto_resume: bool = True
    resume_from: str = ""  # Empty = start fresh, else load from checkpoint


class Config:
    """Main configuration class combining all sub-configs."""

    def __init__(self, config_file: str = None):
        """Initialize configuration from file or use defaults."""
        self.schema = ExtractBenchSchema()
        self.model = ModelConfig()
        self.data = DataConfig()
        self.scoring = ScoringConfig()
        self.optimizer = OptimizerConfig()
        self.logging = LoggingConfig()
        self.resume = ResumeConfig()

        # Load from file if provided
        if config_file and os.path.exists(config_file):
            self.load_from_file(config_file)

        # Override with environment variables
        self._load_env_overrides()
        
        # Ensure directories exist
        self._ensure_directories()

    def _load_env_overrides(self):
        """Load configuration overrides from environment variables."""
        if os.getenv("OPENAI_API_KEY"):
            self.model.api_key = os.getenv("OPENAI_API_KEY")
            self.model.provider = "openai"
        if os.getenv("GEMINI_API_KEY"):
            self.model.api_key = os.getenv("GEMINI_API_KEY")
            self.model.provider = "gemini"
        if os.getenv("AI_API_KEY") and not self.model.api_key:
            self.model.api_key = os.getenv("AI_API_KEY")
        if os.getenv("MODEL_NAME"):
            self.model.name = os.getenv("MODEL_NAME")
        if os.getenv("LLM_PROVIDER"):
            self.model.provider = os.getenv("LLM_PROVIDER")
        if os.getenv("DATASET_PATH"):
            self.data.dataset_path = os.getenv("DATASET_PATH")
        if os.getenv("GROUND_TRUTH_PATH"):
            self.data.ground_truth_path = os.getenv("GROUND_TRUTH_PATH")
        if os.getenv("LOG_LEVEL"):
            self.logging.log_level = os.getenv("LOG_LEVEL")

    def _ensure_directories(self):
        """Create required directories if they don't exist."""
        dirs = [
            self.data.dataset_path,
            self.data.ground_truth_path,
            self.logging.log_dir,
            self.resume.checkpoint_dir,
            "outputs"
        ]
        for d in dirs:
            os.makedirs(d, exist_ok=True)

    def load_from_file(self, config_file: str):
        """Load configuration from JSON file."""
        try:
            with open(config_file, "r") as f:
                config_dict = json.load(f)
            self._apply_dict_config(config_dict)
        except Exception as e:
            print(f"Warning: Could not load config from {config_file}: {e}")

    def _apply_dict_config(self, config_dict: Dict):
        """Apply a dictionary configuration."""
        if "model" in config_dict:
            for k, v in config_dict["model"].items():
                setattr(self.model, k, v)
        if "data" in config_dict:
            for k, v in config_dict["data"].items():
                setattr(self.data, k, v)
        if "scoring" in config_dict:
            for k, v in config_dict["scoring"].items():
                setattr(self.scoring, k, v)
        if "optimizer" in config_dict:
            for k, v in config_dict["optimizer"].items():
                setattr(self.optimizer, k, v)
        if "logging" in config_dict:
            for k, v in config_dict["logging"].items():
                setattr(self.logging, k, v)
        if "resume" in config_dict:
            for k, v in config_dict["resume"].items():
                setattr(self.resume, k, v)

    def to_dict(self) -> Dict:
        """Convert configuration to dictionary."""
        return {
            "schema": asdict(self.schema),
            "model": asdict(self.model),
            "data": asdict(self.data),
            "scoring": asdict(self.scoring),
            "optimizer": asdict(self.optimizer),
            "logging": asdict(self.logging),
            "resume": asdict(self.resume),
        }

    def save(self, filepath: str):
        """Save configuration to JSON file."""
        with open(filepath, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    def __repr__(self) -> str:
        """String representation of configuration."""
        return json.dumps(self.to_dict(), indent=2)

# PIBIT AI Assignment - Automated Prompt Optimization for Structured Extraction

A modular Python system for automatically optimizing LLM prompts to extract structured data (resumes) from PDFs with measurable quality metrics and resumable optimization loops.

## Features

✅ **Modular Architecture** - Clean separation of concerns with independent, testable modules  
✅ **PDF to JSON Extraction** - Extract resume data using LLM-powered prompts  
✅ **OCR Fallback Support** - Uses PyMuPDF, pdfplumber, then OCR for scanned PDFs  
✅ **ExtractBench Resume Schema** - Standardized extraction schema with 7 key fields  
✅ **Configurable Pipeline** - Customize models, datasets, and scoring strategies  
✅ **Quality Metrics** - Precision, Recall, and F1-score evaluation  
✅ **Greedy Optimization** - Automatic prompt improvement through iterative variation testing  
✅ **Comprehensive Logging** - Track prompts, scores, latency, and API calls  
✅ **Resume Support** - Save/load checkpoints to resume after interruption  
✅ **Reporting** - Generate Markdown and JSON reports with insights  
✅ **Unit Tests** - 21 comprehensive tests covering all modules

# demo Video

Watch the project overview video here:
https://drive.google.com/file/d/1MrYT41N84fi7NH-XSR_WWjHXtRFSiPD7/view?usp=sharing


## Project Structure

```
pibit_ai_assignment/
├── src/
│   ├── __init__.py
│   ├── config.py              # Configuration management
│   ├── data_loader.py         # Load and manage datasets
│   ├── extractor.py           # LLM-based extraction
│   ├── scorer.py              # Evaluation metrics
│   ├── optimizer.py           # Prompt optimization loop
│   ├── logger.py              # Comprehensive logging
│   ├── resume_manager.py      # Checkpoint management
│   └── report_generator.py    # Report generation
├── tests/
│   ├── __init__.py
│   ├── test_config.py         # Configuration tests
│   ├── test_scorer.py         # Scoring tests
│   ├── test_extractor.py      # Extraction tests
│   └── test_optimizer.py      # (Integration tests)
├── data/
│   ├── ground_truth/          # Sample resume annotations
│   └── resumes/               # Resume PDF/text files
├── outputs/
│   ├── logs/                  # Detailed logs
│   ├── checkpoints/           # Optimization checkpoints
│   ├── REPORT.md              # Markdown report
│   └── optimization_summary.json  # JSON summary
├── main.py                    # Entry point
├── requirements.txt           # Dependencies
└── README.md                  # This file
```

## Installation

### Prerequisites

- Python 3.8 or higher
- `tesseract` command-line tool installed and on PATH
- `poppler` utilities installed (`pdftoppm` / `pdftocairo`)

### Setup

```bash
# Clone or navigate to the project directory
cd pibit_ai_assignment

# Install Python dependencies
pip install -r requirements.txt

# On macOS, install system dependencies via Homebrew:
# brew install tesseract poppler

# On Ubuntu/Debian:
# sudo apt-get install tesseract-ocr poppler-utils
```
# Optional: For LLM integration
# pip install openai anthropic pypdf python-dotenv
```

## Usage

### Basic Usage

Run the optimization with default configuration:

```bash
python3 main.py
```

### Advanced Options

```bash
# Run with custom max iterations
python3 main.py --max-iterations 10

# Start fresh (clear previous checkpoints)
python3 main.py --clear-checkpoints

# Disable automatic resumption
python3 main.py --no-resume

# Enable debug logging
python3 main.py --debug

# Use custom configuration file
python3 main.py --config config.json
```

### Running Tests

```bash
# Run all unit tests
python3 -m unittest discover -s tests -p "test_*.py" -v

# Run specific test module
python3 -m unittest tests.test_scorer -v
```

## Configuration

### Default Configuration Structure

Configuration can be customized through:

1. **Config file** (JSON)
2. **Environment variables** (override file settings)
3. **Command-line arguments** (override everything)

#### Example Configuration File

```json
{
  "model": {
    "name": "gpt-3.5-turbo",
    "temperature": 0.7,
    "max_tokens": 2000
  },
  "data": {
    "dataset_path": "data/resumes",
    "ground_truth_path": "data/ground_truth",
    "train_split": 0.7,
    "val_split": 0.2,
    "test_split": 0.1
  },
  "optimizer": {
    "strategy": "greedy",
    "max_iterations": 5,
    "prompt_variations": 3,
    "validation_set_size": 10,
    "early_stopping_patience": 3
  },
  "logging": {
    "log_level": "INFO",
    "log_to_console": true,
    "log_to_file": true
  },
  "resume": {
    "auto_resume": true,
    "save_interval": 5
  }
}
```

### Schema Configuration

The ExtractBench Resume Schema extracts:

- **name** - Full name
- **email** - Email address
- **phone** - Phone number
- **skills** - List of technical/professional skills
- **experience** - Work experience entries
- **education** - Educational background
- **certifications** - Professional certifications

## Module Documentation

### Config (`config.py`)

Manages all configuration parameters with automatic directory creation and environment variable overrides.

```python
from src.config import Config

config = Config()
config.model.name = "gpt-4"
config.save("my_config.json")
```

### Data Loader (`data_loader.py`)

Loads ground truth annotations and creates train/val/test splits.

```python
from src.data_loader import DataLoader

loader = DataLoader(config)
loader.load_data()
train_data = loader.get_split("train")
```

### Extractor (`extractor.py`)

Extracts structured data from resume text using LLM with configurable prompts.

```python
from src.extractor import Extractor

extractor = Extractor(config)
result = extractor.extract(resume_text, custom_prompt)
# Returns: {extracted, latency, api_calls, prompt_used}
```

### Scorer (`scorer.py`)

Evaluates extraction quality with Precision, Recall, and F1 metrics.

```python
from src.scorer import Scorer

scorer = Scorer(config)
scores = scorer.score(predicted, ground_truth)
# Returns: {precision, recall, f1, field_scores}
```

### Logger (`logger.py`)

Comprehensive logging of prompts, scores, latency, and API calls.

```python
from src.logger import Logger

logger = Logger(config)
logger.log_score(iteration=0, metric_name="f1", value=0.85)
logger.save_logs()
```

### Optimizer (`optimizer.py`)

Greedy prompt optimization loop - tests variations and keeps best.

```python
from src.optimizer import PromptOptimizer

optimizer = PromptOptimizer(config, extractor, scorer, loader, logger)
results = optimizer.optimize()
# Returns: {best_prompt, best_score, iterations, history}
```

### Resume Manager (`resume_manager.py`)

Saves and loads optimization state for resumption.

```python
from src.resume_manager import ResumeManager

manager = ResumeManager(config)
manager.save_checkpoint(iteration=5, state=optimizer.get_state())
checkpoint = manager.load_checkpoint()
```

### Report Generator (`report_generator.py`)

Generates comprehensive Markdown and JSON reports.

```python
from src.report_generator import ReportGenerator

generator = ReportGenerator(config, logger)
generator.generate_optimization_report(results)
generator.generate_summary_json(results)
```

## Architecture & Design

### Greedy Optimization Strategy

1. **Initialize** with baseline prompt
2. **For each iteration**:
   - Generate prompt variations (add examples, increase clarity)
   - Evaluate each variation on validation set
   - Keep the best-performing prompt
   - Log metrics and state
3. **Early stopping** if no improvement for N iterations
4. **Evaluate** best prompt on test set

### Key Design Principles

- **Modularity**: Each component is independent and testable
- **Configurability**: All parameters are configurable via config files
- **Resumability**: Full state saved at checkpoints for continuation
- **Observability**: Comprehensive logging of decisions and metrics
- **Simplicity**: Clean, interview-friendly code

## Output Files

### Logs Directory (`outputs/logs/`)

- `events.jsonl` - Timestamped events log
- `optimization_logs.json` - Complete structured logs

### Checkpoints Directory (`outputs/checkpoints/`)

- `checkpoint_iter_N.json` - Snapshot of iteration N
- `latest.json` - Latest checkpoint for quick resumption

### Reports (`outputs/`)

- `REPORT.md` - Comprehensive Markdown report with analysis
- `optimization_summary.json` - Structured JSON summary

## Example Workflow

```bash
# 1. Run initial optimization
python3 main.py --max-iterations 5

# 2. Check results
cat outputs/REPORT.md

# 3. Resume optimization for more iterations
python3 main.py --max-iterations 10
# Automatically resumes from last checkpoint

# 4. Start fresh with different config
python3 main.py --config custom_config.json --clear-checkpoints
```

## Adding Custom Ground Truth Data

Place JSON files in `data/ground_truth/`:

```json
{
  "name": "John Doe",
  "email": "john@example.com",
  "phone": "+1-555-0123",
  "skills": ["Python", "Machine Learning"],
  "experience": ["Senior ML Engineer at TechCorp"],
  "education": ["M.S. Computer Science, Stanford"],
  "certifications": ["AWS Architect"]
}
```

## Testing

### Unit Tests (21 tests)

```bash
python3 -m unittest discover -s tests -v
```

Test coverage includes:

- Configuration loading/saving
- Schema validation
- Extraction functionality
- Scoring metrics
- Batch processing
- Invalid input handling

### Integration Testing

Run end-to-end optimization:

```bash
python3 main.py --max-iterations 3 --clear-checkpoints
```

Verify outputs:

```bash
ls -la outputs/logs/
ls -la outputs/checkpoints/
cat outputs/REPORT.md
```

## Performance Considerations

- **Latency**: Simulated extraction (0.05-0.2s each), real LLMs will be slower
- **API Calls**: Logged and counted; use cost tracking for budgeting
- **Checkpoints**: Auto-saved, old ones cleaned up (keeps last 5 by default)
- **Memory**: Efficient with streaming/batching support

## Extending the System

### Custom Scoring Metrics

```python
class CustomScorer(Scorer):
    def _score_field(self, predicted, ground_truth, field_name):
        # Custom logic
        return {"precision": p, "recall": r, "f1": f1}
```

### Custom Optimization Strategy

```python
class EvolutionaryOptimizer(PromptOptimizer):
    def optimize(self):
        # Genetic algorithm instead of greedy
        pass
```

### Real LLM Integration

Replace `Extractor._call_llm()`:

```python
import openai
def _call_llm(self, prompt):
    response = openai.ChatCompletion.create(
        model=self.config.model.name,
        messages=[{"role": "user", "content": prompt}],
        temperature=self.config.model.temperature
    )
    return json.loads(response.choices[0].message.content)
```

## Troubleshooting

### Issue: "No validation samples"

Add more ground truth files to `data/ground_truth/` - need 5+ samples for proper splits.

### Issue: Low F1 scores

- Increase max_iterations for more optimization
- Check prompt quality in REPORT.md
- Manually improve baseline prompt in config

### Issue: Checkpoint not loading

- Clear checkpoints: `python3 main.py --clear-checkpoints`
- Check `outputs/checkpoints/` directory exists
- Verify JSON files are valid

## Performance Metrics

Example run with 3 samples:

```
Best F1 Score: 0.0408
Total Iterations: 3
Test F1 Score: 0.2857
Total API Calls: 0 (simulated)
Total Time: ~2 seconds
```
## Results

Best Validation F1 Score: 0.8571

Test F1 Score: 0.7143

Model Used:
- Gemini 2.5 Flash

Features:
- OCR fallback
- Prompt optimization
- Checkpointing
- Automated evaluation
- Report generation
- 
## Requirements

- Python 3.8+
- No external dependencies for core functionality
- Optional: `openai`, `anthropic`, `pypdf` for enhanced features

## License

Educational assignment project - PIBIT AI

## Author

PIBIT AI Assignment
Built with modular architecture for extensibility and clarity

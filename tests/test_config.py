"""Unit tests for configuration module."""

import unittest
import os
import json
import tempfile
from src.config import Config, ModelConfig, DataConfig, ExtractBenchSchema


class TestExtractBenchSchema(unittest.TestCase):
    """Test ExtractBenchSchema."""

    def test_default_fields(self):
        """Test default schema fields."""
        schema = ExtractBenchSchema()
        self.assertEqual(len(schema.fields), 7)
        self.assertIn("name", schema.fields)
        self.assertIn("email", schema.fields)
        self.assertIn("skills", schema.fields)

    def test_custom_fields(self):
        """Test custom schema fields."""
        custom_fields = ["custom1", "custom2"]
        schema = ExtractBenchSchema(fields=custom_fields)
        self.assertEqual(schema.fields, custom_fields)


class TestConfig(unittest.TestCase):
    """Test Configuration class."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()

    def test_default_config(self):
        """Test default configuration."""
        config = Config()
        self.assertIsNotNone(config.model)
        self.assertIsNotNone(config.data)
        self.assertIsNotNone(config.scoring)
        self.assertIsNotNone(config.schema)

    def test_config_to_dict(self):
        """Test converting config to dictionary."""
        config = Config()
        config_dict = config.to_dict()

        self.assertIn("model", config_dict)
        self.assertIn("data", config_dict)
        self.assertIn("schema", config_dict)

    def test_config_save_and_load(self):
        """Test saving and loading configuration."""
        config_file = os.path.join(self.temp_dir, "test_config.json")

        # Save config
        config1 = Config()
        config1.model.name = "test-model"
        config1.save(config_file)

        self.assertTrue(os.path.exists(config_file))

        # Load config
        config2 = Config(config_file)
        self.assertEqual(config2.model.name, "test-model")

    def test_directories_created(self):
        """Test that required directories are created."""
        config = Config()
        self.assertTrue(os.path.exists(config.data.dataset_path))
        self.assertTrue(os.path.exists(config.logging.log_dir))
        self.assertTrue(os.path.exists(config.resume.checkpoint_dir))

    def test_baseline_prompt(self):
        """Test default baseline prompt."""
        config = Config()
        self.assertIn("Extract structured", config.optimizer.baseline_prompt)
        self.assertIn("name", config.optimizer.baseline_prompt)
        self.assertIn("email", config.optimizer.baseline_prompt)


if __name__ == "__main__":
    unittest.main()

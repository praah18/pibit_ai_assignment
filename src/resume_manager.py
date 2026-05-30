"""Resume manager for saving/loading optimization state after interruption."""

import json
import os
from typing import Dict, Optional
from datetime import datetime


class ResumeManager:
    """Save and restore optimization state for resumption after interruption."""

    def __init__(self, config):
        """Initialize resume manager."""
        self.config = config
        self.checkpoint_dir = config.resume.checkpoint_dir
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def save_checkpoint(self, iteration: int, state: Dict) -> str:
        """
        Save optimization checkpoint.

        Args:
            iteration: Current iteration number
            state: Complete optimization state dict

        Returns:
            Path to saved checkpoint
        """
        checkpoint_data = {
            "timestamp": datetime.now().isoformat(),
            "iteration": iteration,
            "state": state,
            "config": self.config.to_dict(),
        }

        filename = f"checkpoint_iter_{iteration}.json"
        filepath = os.path.join(self.checkpoint_dir, filename)

        with open(filepath, "w") as f:
            json.dump(checkpoint_data, f, indent=2)

        # Also update latest checkpoint
        latest_path = os.path.join(self.checkpoint_dir, "latest.json")
        with open(latest_path, "w") as f:
            json.dump(checkpoint_data, f, indent=2)

        return filepath

    def load_checkpoint(self, iteration: Optional[int] = None) -> Optional[Dict]:
        """
        Load optimization checkpoint.

        Args:
            iteration: Specific iteration to load. If None, loads latest.

        Returns:
            Checkpoint state or None if not found
        """
        if iteration is None:
            # Try to load latest checkpoint
            filepath = os.path.join(self.checkpoint_dir, "latest.json")
            if not os.path.exists(filepath):
                return None
        else:
            filepath = os.path.join(self.checkpoint_dir, f"checkpoint_iter_{iteration}.json")

        if not os.path.exists(filepath):
            return None

        try:
            with open(filepath, "r") as f:
                checkpoint = json.load(f)
            return checkpoint
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            return None

    def get_latest_iteration(self) -> Optional[int]:
        """Get the latest saved iteration number."""
        checkpoint = self.load_checkpoint()
        if checkpoint:
            return checkpoint.get("iteration")
        return None

    def list_checkpoints(self) -> list:
        """List all available checkpoints."""
        checkpoints = []
        for filename in os.listdir(self.checkpoint_dir):
            if filename.startswith("checkpoint_iter_") and filename.endswith(".json"):
                try:
                    iteration = int(filename.replace("checkpoint_iter_", "").replace(".json", ""))
                    filepath = os.path.join(self.checkpoint_dir, filename)
                    stat = os.stat(filepath)
                    checkpoints.append({
                        "iteration": iteration,
                        "filename": filename,
                        "size_bytes": stat.st_size,
                        "timestamp": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                    })
                except ValueError:
                    pass
        return sorted(checkpoints, key=lambda x: x["iteration"])

    def cleanup_old_checkpoints(self, keep_last_n: int = 5):
        """Remove old checkpoints, keeping only the last N."""
        checkpoints = self.list_checkpoints()
        if len(checkpoints) <= keep_last_n:
            return

        to_remove = checkpoints[:-keep_last_n]
        for checkpoint in to_remove:
            filepath = os.path.join(self.checkpoint_dir, checkpoint["filename"])
            try:
                os.remove(filepath)
                print(f"Removed old checkpoint: {checkpoint['filename']}")
            except Exception as e:
                print(f"Error removing checkpoint: {e}")

    def should_resume(self) -> bool:
        """Check if resumption is enabled and checkpoint exists."""
        if not self.config.resume.auto_resume:
            return False

        if self.config.resume.resume_from:
            return os.path.exists(os.path.join(
                self.checkpoint_dir,
                f"checkpoint_iter_{self.config.resume.resume_from}.json"
            ))

        return os.path.exists(os.path.join(self.checkpoint_dir, "latest.json"))

    def get_resume_state(self) -> Optional[Dict]:
        """Get state to resume from."""
        if not self.should_resume():
            return None

        if self.config.resume.resume_from:
            checkpoint = self.load_checkpoint(int(self.config.resume.resume_from))
        else:
            checkpoint = self.load_checkpoint()

        if checkpoint:
            return checkpoint.get("state")
        return None

    def clear_checkpoints(self):
        """Clear all saved checkpoints."""
        try:
            for filename in os.listdir(self.checkpoint_dir):
                if filename.endswith(".json"):
                    filepath = os.path.join(self.checkpoint_dir, filename)
                    os.remove(filepath)
            print("All checkpoints cleared")
        except Exception as e:
            print(f"Error clearing checkpoints: {e}")

"""Utility functions for FE-GAN training and evaluation.

Provides:
- seed_everything: Deterministic seeding for reproducibility
- Timer: Context manager for timing code blocks
- JSONLogger: Structured logging to JSON files
- hash_config: Deterministic hash of configuration dict
- git_sha: Get current git commit SHA
- get_device: Get available torch device
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import subprocess
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    """Set all random seeds for reproducibility.

    Args:
        seed: Integer seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # For full determinism (may impact performance)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def seed_everything_fast(seed: int) -> None:
    """Set random seeds with cudnn.benchmark=True for performance.

    Use this for production sweeps where exact reproducibility across
    different hardware is less critical than speed.

    Args:
        seed: Integer seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


class Timer:
    """Context manager for timing code blocks.

    Example:
        with Timer() as t:
            train_model()
        print(f"Training took {t.elapsed:.2f} seconds")
    """

    def __init__(self):
        self.start_time: float = 0.0
        self.end_time: float = 0.0
        self.elapsed: float = 0.0

    def __enter__(self) -> "Timer":
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, *args) -> None:
        self.end_time = time.perf_counter()
        self.elapsed = self.end_time - self.start_time


@contextmanager
def timer(name: str = ""):
    """Simple timing context manager with optional name.

    Example:
        with timer("training"):
            train_model()
        # Prints: training took 123.45 seconds
    """
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        prefix = f"{name} " if name else ""
        print(f"{prefix}took {elapsed:.2f} seconds")


class JSONLogger:
    """Simple JSON lines logger for structured logging.

    Each log entry is written as a single JSON line.
    """

    def __init__(self, path: str | Path):
        """Initialize logger.

        Args:
            path: Path to log file (will be created/appended).
        """
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, **kwargs) -> None:
        """Log a single entry with timestamp.

        Args:
            **kwargs: Key-value pairs to log.
        """
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **kwargs,
        }
        with open(self.path, "a") as f:
            f.write(json.dumps(entry) + "\n")

    def log_dict(self, data: dict[str, Any]) -> None:
        """Log a dictionary with timestamp.

        Args:
            data: Dictionary to log.
        """
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **data,
        }
        with open(self.path, "a") as f:
            f.write(json.dumps(entry) + "\n")


def hash_config(config: dict[str, Any]) -> str:
    """Compute deterministic SHA256 hash of config dict.

    Args:
        config: Configuration dictionary.

    Returns:
        First 12 characters of SHA256 hash.
    """
    # Sort keys for deterministic serialization
    config_str = json.dumps(config, sort_keys=True, default=str)
    return hashlib.sha256(config_str.encode()).hexdigest()[:12]


def git_sha() -> str:
    """Get current git commit SHA.

    Returns:
        Git commit SHA (short form) or 'unknown' if not in a git repo.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    return "unknown"


def get_device() -> torch.device:
    """Get the best available torch device.

    Returns:
        torch.device for CUDA if available, else CPU.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def get_system_info() -> dict[str, Any]:
    """Collect system information for reproducibility logging.

    Returns:
        Dictionary with torch version, CUDA info, hostname, etc.
    """
    info = {
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "cuda_available": torch.cuda.is_available(),
        "git_sha": git_sha(),
        "hostname": os.uname().nodename,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    if torch.cuda.is_available():
        info["cuda_version"] = torch.version.cuda
        info["cudnn_version"] = torch.backends.cudnn.version()
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["gpu_count"] = torch.cuda.device_count()

    return info


def save_json(data: dict[str, Any], path: str | Path) -> None:
    """Save dictionary to JSON file.

    Args:
        data: Dictionary to save.
        path: Output path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)


def load_json(path: str | Path) -> dict[str, Any]:
    """Load dictionary from JSON file.

    Args:
        path: Input path.

    Returns:
        Loaded dictionary.
    """
    with open(path) as f:
        return json.load(f)

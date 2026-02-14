"""Tests for utility functions."""

import json
import tempfile
from pathlib import Path

import numpy as np
import torch

from fe_gan.utils import (
    JSONLogger,
    Timer,
    get_device,
    get_system_info,
    git_sha,
    hash_config,
    load_json,
    save_json,
    seed_everything,
    seed_everything_fast,
)


def test_seed_everything_deterministic():
    """Test that seed_everything produces deterministic results."""
    seed_everything(42)
    a1 = torch.randn(10)
    np1 = np.random.randn(10)

    seed_everything(42)
    a2 = torch.randn(10)
    np2 = np.random.randn(10)

    assert torch.allclose(a1, a2)
    assert np.allclose(np1, np2)


def test_seed_everything_fast():
    """Test that seed_everything_fast sets seeds."""
    seed_everything_fast(42)
    a1 = torch.randn(10)

    seed_everything_fast(42)
    a2 = torch.randn(10)

    # Should be deterministic on same hardware
    assert torch.allclose(a1, a2)


def test_timer():
    """Test Timer context manager."""
    import time

    with Timer() as t:
        time.sleep(0.1)

    assert t.elapsed >= 0.1
    assert t.elapsed < 0.5  # Generous upper bound


def test_json_logger():
    """Test JSONLogger writes valid JSON lines."""
    with tempfile.TemporaryDirectory() as tmpdir:
        log_path = Path(tmpdir) / "test.jsonl"
        logger = JSONLogger(log_path)

        logger.log(epoch=1, loss=0.5)
        logger.log(epoch=2, loss=0.3)

        with open(log_path) as f:
            lines = f.readlines()

        assert len(lines) == 2
        entry1 = json.loads(lines[0])
        entry2 = json.loads(lines[1])

        assert entry1["epoch"] == 1
        assert entry1["loss"] == 0.5
        assert "timestamp" in entry1

        assert entry2["epoch"] == 2


def test_hash_config_deterministic():
    """Test that hash_config is deterministic."""
    config = {"lr": 0.001, "batch_size": 32, "model": "fegan"}

    h1 = hash_config(config)
    h2 = hash_config(config)

    assert h1 == h2
    assert len(h1) == 12


def test_hash_config_different_for_different_configs():
    """Test that different configs produce different hashes."""
    config1 = {"lr": 0.001}
    config2 = {"lr": 0.002}

    assert hash_config(config1) != hash_config(config2)


def test_git_sha():
    """Test git_sha returns a string."""
    sha = git_sha()
    assert isinstance(sha, str)
    assert len(sha) > 0  # Either a SHA or 'unknown'


def test_get_device():
    """Test get_device returns a valid device."""
    device = get_device()
    assert isinstance(device, torch.device)
    # Should be 'cuda' or 'cpu'
    assert device.type in ("cuda", "cpu")


def test_get_system_info():
    """Test get_system_info returns expected keys."""
    info = get_system_info()

    assert "torch_version" in info
    assert "numpy_version" in info
    assert "cuda_available" in info
    assert "git_sha" in info
    assert "hostname" in info
    assert "timestamp" in info


def test_save_load_json():
    """Test save_json and load_json round-trip."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "subdir" / "test.json"
        data = {"a": 1, "b": [1, 2, 3], "c": {"nested": True}}

        save_json(data, path)
        loaded = load_json(path)

        assert loaded == data

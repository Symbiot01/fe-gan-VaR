"""Tests for sweep script."""

import tempfile
from pathlib import Path

import pytest
import torch

from scripts.sweep import SweepTask, parse_seeds, run_sweep_sequential


class TestParseSeeds:
    """Tests for parse_seeds function."""

    def test_comma_separated(self):
        """Test comma-separated seeds."""
        seeds = parse_seeds("0,1,2,3")
        assert seeds == [0, 1, 2, 3]

    def test_range(self):
        """Test range syntax."""
        seeds = parse_seeds("0-4")
        assert seeds == [0, 1, 2, 3, 4]

    def test_mixed(self):
        """Test mixed syntax."""
        seeds = parse_seeds("0-2,5,8-10")
        assert seeds == [0, 1, 2, 5, 8, 9, 10]

    def test_single(self):
        """Test single seed."""
        seeds = parse_seeds("42")
        assert seeds == [42]


class TestSweepTask:
    """Tests for SweepTask dataclass."""

    def test_task_creation(self):
        """Test task creation."""
        task = SweepTask(
            config_name="test",
            config_path=Path("configs/baseline_wgan.yaml"),
            seed=0,
            epochs=10,
            out_dir=Path("results/test"),
        )

        assert task.config_name == "test"
        assert task.seed == 0
        assert task.epochs == 10


class TestSweepSequential:
    """Tests for sequential sweep execution."""

    @pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="CUDA not available",
    )
    def test_sweep_two_seeds(self):
        """Test sweep with 2 seeds completes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Create tasks
            tasks = []
            for seed in [0, 1]:
                task = SweepTask(
                    config_name="baseline_wgan",
                    config_path=Path("configs/baseline_wgan.yaml"),
                    seed=seed,
                    epochs=2,  # Very short
                    out_dir=tmpdir / f"run_s{seed}",
                )
                tasks.append(task)

            # Run sweep
            device = torch.device("cuda")
            results = run_sweep_sequential(tasks, device)

            # Check results
            assert len(results) == 2

            for r in results:
                assert "var_err_median" in r
                assert "es_err_median" in r
                assert "total_wall_seconds" in r

    def test_sweep_cpu_fallback(self):
        """Test sweep works on CPU."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            task = SweepTask(
                config_name="baseline_wgan",
                config_path=Path("configs/baseline_wgan.yaml"),
                seed=0,
                epochs=1,
                out_dir=tmpdir / "run_s0",
            )

            device = torch.device("cpu")
            results = run_sweep_sequential([task], device)

            assert len(results) == 1
            assert "var_err_median" in results[0]


class TestSweepOutputs:
    """Tests for sweep output files."""

    def test_sweep_creates_files(self):
        """Test sweep creates expected output files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            task = SweepTask(
                config_name="test_config",
                config_path=Path("configs/baseline_wgan.yaml"),
                seed=0,
                epochs=1,
                out_dir=tmpdir / "run_s0",
            )

            device = torch.device("cpu")
            results = run_sweep_sequential([task], device)

            # Check files were created in task output dir
            out_dir = tmpdir / "run_s0"
            assert (out_dir / "eval.json").exists()
            assert (out_dir / "config.json").exists()
            assert (out_dir / "metrics.csv").exists()

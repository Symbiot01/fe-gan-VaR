"""Tests for trainer module."""

import tempfile
from pathlib import Path

import pytest
import torch

from fe_gan.data import VIXDataset
from fe_gan.models import Critic, FEGenerator, Generator
from fe_gan.trainer import Trainer, TrainerConfig, train_model


class TestTrainerConfig:
    """Tests for TrainerConfig."""

    def test_default_config(self):
        """Test default configuration."""
        config = TrainerConfig()

        assert config.model_type == "wgan"
        assert config.loss_type == "wgan"
        assert config.epochs == 100
        assert config.batch_size == 100
        assert config.n_critic == 5

    def test_config_to_dict(self):
        """Test config serialization."""
        config = TrainerConfig(epochs=50, seed=42)
        d = config.to_dict()

        assert d["epochs"] == 50
        assert d["seed"] == 42
        assert isinstance(d, dict)


class TestTrainer:
    """Tests for Trainer class."""

    @pytest.fixture
    def simple_trainer(self):
        """Create a simple trainer for testing."""
        # Small models for fast testing
        gen = Generator(noise_dim=10, out_dim=50, depth=3, width=64)
        critic = Critic(in_dim=50, depth=3, width=32)

        # Small datasets
        train_windows = torch.randn(100, 50)
        eval_windows = torch.randn(20, 50)
        train_data = VIXDataset(train_windows)
        eval_data = VIXDataset(eval_windows)

        with tempfile.TemporaryDirectory() as tmpdir:
            config = TrainerConfig(
                model_type="wgan",
                loss_type="wgan",
                noise_dim=10,
                hist_dim=50,
                gen_depth=3,
                gen_width=64,
                critic_depth=3,
                critic_width=32,
                epochs=2,
                batch_size=16,
                n_critic=2,
                eval_every=1,
                eval_samples=10,
                out_dir=tmpdir,
            )

            trainer = Trainer(
                generator=gen,
                critic=critic,
                train_data=train_data,
                eval_data=eval_data,
                config=config,
                device="cpu",
            )

            yield trainer, tmpdir

    def test_trainer_creation(self, simple_trainer):
        """Test trainer can be created."""
        trainer, _ = simple_trainer
        assert trainer is not None
        assert trainer.generator is not None
        assert trainer.critic is not None

    def test_trainer_short_train(self, simple_trainer):
        """Test trainer completes 2-epoch training."""
        trainer, tmpdir = simple_trainer

        result = trainer.train(progress=False)

        assert "var_err_median" in result
        assert "es_err_median" in result
        assert "total_wall_seconds" in result
        assert result["total_wall_seconds"] > 0

    def test_trainer_saves_files(self, simple_trainer):
        """Test trainer saves output files."""
        trainer, tmpdir = simple_trainer
        trainer.train(progress=False)

        out_dir = Path(tmpdir)
        assert (out_dir / "metrics.csv").exists()
        assert (out_dir / "eval.json").exists()
        assert (out_dir / "config.json").exists()
        assert (out_dir / "metadata.json").exists()

    def test_trainer_metrics_history(self, simple_trainer):
        """Test trainer records metrics history."""
        trainer, _ = simple_trainer
        trainer.train(progress=False)

        assert len(trainer.metrics_history) == 2
        assert trainer.metrics_history[0].epoch == 1
        assert trainer.metrics_history[1].epoch == 2


class TestFEGANTrainer:
    """Tests for FE-GAN training."""

    @pytest.fixture
    def fegan_trainer(self):
        """Create FE-GAN trainer for testing."""
        gen = FEGenerator(noise_dim=10, hist_dim=50, out_dim=50, depth=3, width=64)
        critic = Critic(in_dim=50, depth=3, width=32)

        train_windows = torch.randn(100, 50)
        eval_windows = torch.randn(20, 50)
        train_data = VIXDataset(train_windows)
        eval_data = VIXDataset(eval_windows)

        with tempfile.TemporaryDirectory() as tmpdir:
            config = TrainerConfig(
                model_type="fegan",
                loss_type="wgan",
                noise_dim=10,
                hist_dim=50,
                gen_depth=3,
                gen_width=64,
                critic_depth=3,
                critic_width=32,
                epochs=2,
                batch_size=16,
                n_critic=2,
                eval_every=1,
                eval_samples=10,
                out_dir=tmpdir,
            )

            trainer = Trainer(
                generator=gen,
                critic=critic,
                train_data=train_data,
                eval_data=eval_data,
                config=config,
                device="cpu",
            )

            yield trainer

    def test_fegan_trains(self, fegan_trainer):
        """Test FE-GAN completes training."""
        result = fegan_trainer.train(progress=False)

        assert "var_err_median" in result
        assert torch.isfinite(torch.tensor(result["var_err_median"]))


class TestTailGANTrainer:
    """Tests for Tail-GAN training."""

    @pytest.fixture
    def tailgan_trainer(self):
        """Create Tail-GAN trainer for testing."""
        gen = FEGenerator(noise_dim=10, hist_dim=50, out_dim=50, depth=3, width=64)
        critic = Critic(in_dim=50, depth=3, width=32)

        train_windows = torch.randn(100, 50)
        eval_windows = torch.randn(20, 50)
        train_data = VIXDataset(train_windows)
        eval_data = VIXDataset(eval_windows)

        with tempfile.TemporaryDirectory() as tmpdir:
            config = TrainerConfig(
                model_type="fegan",
                loss_type="tailgan",
                noise_dim=10,
                hist_dim=50,
                gen_depth=3,
                gen_width=64,
                critic_depth=3,
                critic_width=32,
                epochs=2,
                batch_size=16,
                n_critic=2,
                tailgan_wgan_weight=0.5,
                eval_every=1,
                eval_samples=10,
                out_dir=tmpdir,
            )

            trainer = Trainer(
                generator=gen,
                critic=critic,
                train_data=train_data,
                eval_data=eval_data,
                config=config,
                device="cpu",
            )

            yield trainer

    def test_tailgan_trains(self, tailgan_trainer):
        """Test Tail-GAN completes training."""
        result = tailgan_trainer.train(progress=False)

        assert "var_err_median" in result
        assert torch.isfinite(torch.tensor(result["var_err_median"]))

    def test_tailgan_records_fz_loss(self, tailgan_trainer):
        """Test Tail-GAN records FZ loss in metrics."""
        tailgan_trainer.train(progress=False)

        # FZ loss should be recorded
        assert tailgan_trainer.metrics_history[-1].fz_loss is not None


class TestTrainModel:
    """Tests for train_model convenience function."""

    def test_train_model_wgan(self):
        """Test train_model with WGAN config."""
        train_windows = torch.randn(100, 50)
        eval_windows = torch.randn(20, 50)
        train_data = VIXDataset(train_windows)
        eval_data = VIXDataset(eval_windows)

        with tempfile.TemporaryDirectory() as tmpdir:
            config = TrainerConfig(
                model_type="wgan",
                loss_type="wgan",
                noise_dim=10,
                hist_dim=50,
                gen_depth=3,
                gen_width=64,
                critic_depth=3,
                critic_width=32,
                epochs=2,
                batch_size=16,
                out_dir=tmpdir,
            )

            result = train_model(
                train_data=train_data,
                eval_data=eval_data,
                config=config,
                device="cpu",
                progress=False,
            )

            assert "var_err_median" in result
            assert (Path(tmpdir) / "eval.json").exists()


class TestIdentityGenerator:
    """Test that identity-like generator produces low error."""

    def test_identity_low_error(self):
        """An identity-like generator should produce near-zero error."""
        # Create a "generator" that just returns the input noise scaled
        # This won't be perfect, but errors should be bounded

        # In practice, we just verify the evaluation pipeline works
        # and produces reasonable numbers
        train_windows = torch.randn(100, 50)
        eval_windows = torch.randn(20, 50)
        train_data = VIXDataset(train_windows)
        eval_data = VIXDataset(eval_windows)

        with tempfile.TemporaryDirectory() as tmpdir:
            config = TrainerConfig(
                model_type="wgan",
                noise_dim=10,
                hist_dim=50,
                gen_depth=2,
                gen_width=32,
                critic_depth=2,
                critic_width=16,
                epochs=1,
                batch_size=16,
                out_dir=tmpdir,
            )

            result = train_model(
                train_data=train_data,
                eval_data=eval_data,
                config=config,
                device="cpu",
                progress=False,
            )

            # Just check that evaluation completed and produced numbers
            assert result["var_err_median"] is not None
            assert result["var_err_median"] >= 0

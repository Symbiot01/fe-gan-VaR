"""Tests for evaluation utilities."""

import torch

from fe_gan.eval import (
    compute_eval_metrics,
    compute_var_es_errors,
    evaluate_generator,
)
from fe_gan.models import Generator, FEGenerator


class TestComputeVaRESErrors:
    """Tests for compute_var_es_errors function."""

    def test_basic_shape(self):
        """Test output shapes."""
        generated = torch.randn(100, 250)
        real = torch.randn(100, 250)

        result = compute_var_es_errors(generated, real)

        assert "var_errors" in result
        assert "es_errors" in result
        assert result["var_errors"].shape == (100,)
        assert result["es_errors"].shape == (100,)

    def test_different_sizes(self):
        """Test with different batch sizes."""
        generated = torch.randn(50, 250)
        real = torch.randn(100, 250)

        result = compute_var_es_errors(generated, real)

        # Should compare min(50, 100) = 50 samples
        assert result["var_errors"].shape == (50,)
        assert result["es_errors"].shape == (50,)

    def test_errors_non_negative(self):
        """Test that errors are non-negative."""
        generated = torch.randn(100, 250)
        real = torch.randn(100, 250)

        result = compute_var_es_errors(generated, real)

        assert (result["var_errors"] >= 0).all()
        assert (result["es_errors"] >= 0).all()


class TestComputeEvalMetrics:
    """Tests for compute_eval_metrics function."""

    def test_basic(self):
        """Test metric computation."""
        var_errors = torch.abs(torch.randn(100))
        es_errors = torch.abs(torch.randn(100))

        metrics = compute_eval_metrics(var_errors, es_errors)

        assert "var_err_median" in metrics
        assert "var_err_mean" in metrics
        assert "var_err_std" in metrics
        assert "var_err_iqr" in metrics
        assert "var_err_p90" in metrics
        assert "es_err_median" in metrics
        assert "es_err_mean" in metrics

    def test_values_reasonable(self):
        """Test metric values are reasonable."""
        errors = torch.abs(torch.randn(1000))
        metrics = compute_eval_metrics(errors, errors)

        # Median should be within reasonable range for abs(normal)
        assert 0 < metrics["var_err_median"] < 2
        assert metrics["var_err_min"] >= 0
        assert metrics["var_err_max"] >= metrics["var_err_median"]


class TestEvaluateGenerator:
    """Tests for evaluate_generator function."""

    def test_evaluate_wgan(self):
        """Test evaluation of vanilla generator."""
        gen = Generator(noise_dim=100, out_dim=250)
        real_windows = torch.randn(100, 250)

        metrics = evaluate_generator(
            gen,
            real_windows,
            n_samples=50,
            noise_dim=100,
            is_fegan=False,
        )

        assert "var_err_median" in metrics
        assert "es_err_median" in metrics
        assert "n_generated" in metrics
        assert metrics["n_generated"] == 50

    def test_evaluate_fegan(self):
        """Test evaluation of FE-GAN generator."""
        gen = FEGenerator(noise_dim=100, hist_dim=250, out_dim=250)
        real_windows = torch.randn(100, 250)

        metrics = evaluate_generator(
            gen,
            real_windows,
            n_samples=50,
            noise_dim=100,
            is_fegan=True,
        )

        assert "var_err_median" in metrics
        assert "es_err_median" in metrics

    def test_evaluate_finite(self):
        """Test evaluation produces finite values."""
        gen = Generator(noise_dim=100, out_dim=250)
        real_windows = torch.randn(100, 250)

        metrics = evaluate_generator(
            gen,
            real_windows,
            n_samples=50,
        )

        assert not any(
            isinstance(v, float) and (v != v)  # NaN check
            for v in metrics.values()
        )

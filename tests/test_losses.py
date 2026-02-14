"""Tests for loss functions."""

import pytest
import torch
import torch.nn as nn
import numpy as np
from scipy import stats

from fe_gan.losses import (
    clip_weights,
    empirical_quantile,
    empirical_var_es,
    fissler_ziegel_loss,
    fissler_ziegel_score,
    soft_empirical_var_es,
    tail_gan_gen_loss,
    var_es_error,
    wgan_critic_loss,
    wgan_gen_loss,
)


class TestWGANLosses:
    """Tests for WGAN loss functions."""

    def test_wgan_critic_loss_basic(self):
        """Test critic loss computation."""
        real_score = torch.tensor([[1.0], [2.0], [3.0]])
        fake_score = torch.tensor([[0.5], [0.5], [0.5]])

        loss = wgan_critic_loss(real_score, fake_score)

        # Loss = E[fake] - E[real] = 0.5 - 2.0 = -1.5
        assert torch.isclose(loss, torch.tensor(-1.5))

    def test_wgan_critic_loss_gradient(self):
        """Test critic loss gradient flow."""
        real_score = torch.randn(8, 1, requires_grad=True)
        fake_score = torch.randn(8, 1, requires_grad=True)

        loss = wgan_critic_loss(real_score, fake_score)
        loss.backward()

        assert real_score.grad is not None
        assert fake_score.grad is not None

    def test_wgan_gen_loss_basic(self):
        """Test generator loss computation."""
        fake_score = torch.tensor([[1.0], [2.0], [3.0]])

        loss = wgan_gen_loss(fake_score)

        # Loss = -E[fake] = -2.0
        assert torch.isclose(loss, torch.tensor(-2.0))

    def test_wgan_gen_loss_gradient(self):
        """Test generator loss gradient flow."""
        fake_score = torch.randn(8, 1, requires_grad=True)

        loss = wgan_gen_loss(fake_score)
        loss.backward()

        assert fake_score.grad is not None


class TestWeightClipping:
    """Tests for weight clipping."""

    def test_clip_weights_basic(self):
        """Test weight clipping clamps values."""
        model = nn.Linear(10, 10)

        # Initialize with large weights
        nn.init.uniform_(model.weight, -1.0, 1.0)

        clip_weights(model, clip_value=0.01)

        # All weights should be in [-0.01, 0.01]
        assert model.weight.abs().max() <= 0.01

    def test_clip_weights_multilayer(self):
        """Test clipping on multi-layer model."""
        model = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 10),
        )

        # Initialize with large weights
        for m in model.modules():
            if isinstance(m, nn.Linear):
                nn.init.uniform_(m.weight, -1.0, 1.0)

        clip_weights(model, clip_value=0.01)

        # All weights should be clipped
        for m in model.modules():
            if isinstance(m, nn.Linear):
                assert m.weight.abs().max() <= 0.01

    def test_clip_weights_preserves_small(self):
        """Test clipping preserves small weights."""
        model = nn.Linear(10, 10)

        # Initialize with small weights
        nn.init.uniform_(model.weight, -0.005, 0.005)
        original = model.weight.clone()

        clip_weights(model, clip_value=0.01)

        # Weights should be unchanged
        assert torch.allclose(model.weight, original)


class TestEmpiricalQuantile:
    """Tests for empirical quantile estimation."""

    def test_quantile_median(self):
        """Test median estimation."""
        x = torch.arange(1, 101, dtype=torch.float32)
        q = empirical_quantile(x, 0.5)

        assert torch.isclose(q, torch.tensor(50.5), atol=0.5)

    def test_quantile_5th(self):
        """Test 5th percentile."""
        x = torch.arange(1, 101, dtype=torch.float32)
        q = empirical_quantile(x, 0.05)

        # 5th percentile of 1-100 is ~5
        assert torch.isclose(q, torch.tensor(5.0), atol=1.0)

    def test_quantile_batched(self):
        """Test batched quantile computation."""
        x = torch.randn(4, 100)
        q = empirical_quantile(x, 0.5)

        assert q.shape == (4,)


class TestEmpiricalVaRES:
    """Tests for VaR and ES estimation."""

    def test_var_es_shape_1d(self):
        """Test VaR/ES shapes for 1D input."""
        returns = torch.randn(1000)
        var, es = empirical_var_es(returns, alpha=0.05)

        assert var.shape == ()
        assert es.shape == ()

    def test_var_es_shape_2d(self):
        """Test VaR/ES shapes for 2D input."""
        returns = torch.randn(8, 1000)
        var, es = empirical_var_es(returns, alpha=0.05)

        assert var.shape == (8,)
        assert es.shape == (8,)

    def test_es_more_extreme_than_var(self):
        """Test ES is more extreme (lower) than VaR."""
        returns = torch.randn(10000)
        var, es = empirical_var_es(returns, alpha=0.05)

        # For left tail, ES should be lower (more negative) than VaR
        assert es < var

    def test_var_es_normal_distribution(self):
        """Test VaR/ES on standard normal matches theoretical."""
        torch.manual_seed(42)
        n_samples = 50000
        returns = torch.randn(n_samples)

        var, es = empirical_var_es(returns, alpha=0.05)

        # Theoretical 5% VaR for N(0,1) is approximately -1.645
        theoretical_var = stats.norm.ppf(0.05)

        # Theoretical 5% ES for N(0,1)
        # ES = E[X | X < VaR] = -phi(VaR) / (alpha) for standard normal
        # ES = -phi(-1.645) / 0.05 ≈ -2.063
        theoretical_es = -stats.norm.pdf(stats.norm.ppf(0.05)) / 0.05

        assert abs(var.item() - theoretical_var) < 0.1, (
            f"VaR {var.item():.3f} vs theoretical {theoretical_var:.3f}"
        )
        assert abs(es.item() - theoretical_es) < 0.1, (
            f"ES {es.item():.3f} vs theoretical {theoretical_es:.3f}"
        )


class TestSoftEmpiricalVaRES:
    """Tests for differentiable VaR/ES."""

    def test_soft_var_es_gradient(self):
        """Test gradients flow through soft VaR/ES."""
        returns = torch.randn(8, 100, requires_grad=True)
        var, es = soft_empirical_var_es(returns, alpha=0.05)

        loss = var.sum() + es.sum()
        loss.backward()

        assert returns.grad is not None
        assert not torch.isnan(returns.grad).any()

    def test_soft_var_es_similar_to_hard(self):
        """Test soft VaR/ES approximates hard version."""
        torch.manual_seed(42)
        returns = torch.randn(1000)

        hard_var, hard_es = empirical_var_es(returns, alpha=0.05)
        soft_var, soft_es = soft_empirical_var_es(returns, alpha=0.05, temperature=0.01)

        # Should be close (lower temp = closer to hard)
        assert torch.isclose(soft_var, hard_var, atol=0.5)


class TestFisslerZiegelScore:
    """Tests for Fissler-Ziegel scoring function."""

    def test_fz_score_finite(self):
        """Test FZ score produces finite values."""
        returns = torch.randn(100)
        # VaR and ES should be negative for left tail
        var = torch.tensor(-1.5)
        es = torch.tensor(-2.0)

        score = fissler_ziegel_score(returns, var, es, alpha=0.05)

        assert torch.isfinite(score)

    def test_fz_score_gradient(self):
        """Test FZ score has valid gradients."""
        returns = torch.randn(100, requires_grad=True)
        var = torch.tensor(-1.5, requires_grad=True)
        es = torch.tensor(-2.0, requires_grad=True)

        score = fissler_ziegel_score(returns, var, es, alpha=0.05)
        score.backward()

        assert returns.grad is not None
        assert var.grad is not None
        assert es.grad is not None
        assert torch.isfinite(returns.grad).all()
        assert torch.isfinite(var.grad).all()
        assert torch.isfinite(es.grad).all()

    def test_fz_score_changes_with_var_es(self):
        """Test FZ score changes when VaR/ES are perturbed.

        The FZ score should be sensitive to VaR and ES estimates.
        """
        torch.manual_seed(42)

        # Generate samples from known distribution
        n_samples = 50000
        returns = torch.randn(n_samples)

        # Compute true VaR and ES
        true_var, true_es = empirical_var_es(returns, alpha=0.05)

        # Score at true values
        score_true = fissler_ziegel_score(
            returns, true_var, true_es, alpha=0.05
        )

        # Score at perturbed values
        perturbed_var = true_var + 0.5
        perturbed_es = true_es + 0.5
        score_perturbed = fissler_ziegel_score(
            returns, perturbed_var, perturbed_es, alpha=0.05
        )

        # The scores should be different (FZ is sensitive to VaR/ES)
        assert not torch.isclose(score_true, score_perturbed, atol=0.01), (
            f"Scores should differ: true={score_true:.4f}, perturbed={score_perturbed:.4f}"
        )

        # Both scores should be finite
        assert torch.isfinite(score_true)
        assert torch.isfinite(score_perturbed)


class TestFisslerZiegelLoss:
    """Tests for FZ loss function."""

    def test_fz_loss_finite(self):
        """Test FZ loss produces finite values."""
        returns = torch.randn(8, 250)

        loss = fissler_ziegel_loss(returns, alpha=0.05)

        assert torch.isfinite(loss)

    def test_fz_loss_gradient(self):
        """Test FZ loss has valid gradients."""
        returns = torch.randn(8, 250, requires_grad=True)

        loss = fissler_ziegel_loss(returns, alpha=0.05)
        loss.backward()

        assert returns.grad is not None
        assert torch.isfinite(returns.grad).all()


class TestTailGANLoss:
    """Tests for Tail-GAN generator loss."""

    def test_tail_gan_loss_pure_fz(self):
        """Test Tail-GAN loss with pure FZ (wgan_weight=0)."""
        fake_returns = torch.randn(8, 250)

        loss = tail_gan_gen_loss(fake_returns, wgan_weight=0.0)

        assert torch.isfinite(loss)

    def test_tail_gan_loss_combined(self):
        """Test Tail-GAN loss with combined WGAN and FZ."""
        fake_returns = torch.randn(8, 250)
        fake_score = torch.randn(8, 1)

        loss = tail_gan_gen_loss(
            fake_returns, wgan_weight=0.5, fake_score=fake_score
        )

        assert torch.isfinite(loss)

    def test_tail_gan_loss_gradient(self):
        """Test gradients flow through Tail-GAN loss."""
        fake_returns = torch.randn(8, 250, requires_grad=True)

        loss = tail_gan_gen_loss(fake_returns, wgan_weight=0.0)
        loss.backward()

        assert fake_returns.grad is not None
        assert not torch.isnan(fake_returns.grad).any()
        assert torch.isfinite(fake_returns.grad).all()


class TestVaRESError:
    """Tests for VaR/ES error computation."""

    def test_var_es_error_same_distribution(self):
        """Test error is small for same distribution."""
        torch.manual_seed(42)

        # Two samples from same distribution
        generated = torch.randn(10000)
        real = torch.randn(10000)

        var_err, es_err = var_es_error(generated, real, alpha=0.05)

        # Errors should be small (sampling variance)
        assert var_err < 0.2, f"VaR error {var_err:.3f} too large"
        assert es_err < 0.3, f"ES error {es_err:.3f} too large"

    def test_var_es_error_different_distribution(self):
        """Test error is large for different distributions."""
        torch.manual_seed(42)

        generated = torch.randn(10000)
        real = torch.randn(10000) * 2 + 1  # Different mean and std

        var_err, es_err = var_es_error(generated, real, alpha=0.05)

        # Errors should be larger
        assert var_err > 0.5
        assert es_err > 0.5

    def test_var_es_error_batched(self):
        """Test batched error computation."""
        generated = torch.randn(4, 1000)
        real = torch.randn(4, 1000)

        var_err, es_err = var_es_error(generated, real, alpha=0.05)

        assert var_err.shape == (4,)
        assert es_err.shape == (4,)


class TestNumericalStability:
    """Tests for numerical stability."""

    def test_var_es_extreme_values(self):
        """Test VaR/ES handles extreme values."""
        returns = torch.randn(1000)
        returns[0] = -10.0  # Extreme negative
        returns[1] = 10.0  # Extreme positive

        var, es = empirical_var_es(returns, alpha=0.05)

        assert torch.isfinite(var)
        assert torch.isfinite(es)

    def test_fz_loss_small_batch(self):
        """Test FZ loss with small batch size."""
        returns = torch.randn(2, 250)

        loss = fissler_ziegel_loss(returns, alpha=0.05)

        assert torch.isfinite(loss)

    def test_gradients_small_batch(self):
        """Test gradients are stable with small batch."""
        returns = torch.randn(2, 250, requires_grad=True)

        loss = fissler_ziegel_loss(returns, alpha=0.05)
        loss.backward()

        assert torch.isfinite(returns.grad).all()

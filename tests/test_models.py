"""Tests for neural network models."""

import pytest
import torch

from fe_gan.models import (
    Critic,
    FEGenerator,
    Generator,
    HistoryPreproc,
    count_parameters,
    create_critic,
    create_generator,
    get_model_info,
)


class TestGenerator:
    """Tests for vanilla Generator."""

    def test_generator_forward_shape(self):
        """Test generator output shape."""
        gen = Generator(noise_dim=100, out_dim=250, depth=10, width=1000)

        z = torch.randn(8, 100)
        out = gen(z)

        assert out.shape == (8, 250)

    def test_generator_batch_sizes(self):
        """Test generator with various batch sizes."""
        gen = Generator(noise_dim=100, out_dim=250)
        gen.eval()  # BatchNorm requires batch_size > 1 in training mode

        for batch_size in [1, 8, 32, 100]:
            z = torch.randn(batch_size, 100)
            with torch.no_grad():
                out = gen(z)
            assert out.shape == (batch_size, 250)

    def test_generator_param_count(self):
        """Test generator has reasonable param count."""
        gen = Generator(noise_dim=100, out_dim=250, depth=10, width=1000)

        n_params = count_parameters(gen)

        # Per plan: ~10-12M params
        assert 8_000_000 < n_params < 15_000_000, f"Params {n_params} outside expected range"

    def test_generator_no_nan(self):
        """Test generator produces no NaN outputs."""
        gen = Generator()

        z = torch.randn(32, 100)
        out = gen(z)

        assert not torch.isnan(out).any(), "Generator produced NaN"
        assert not torch.isinf(out).any(), "Generator produced Inf"

    def test_generator_gradients(self):
        """Test gradients flow through generator."""
        gen = Generator()

        z = torch.randn(8, 100, requires_grad=True)
        out = gen(z)
        loss = out.sum()
        loss.backward()

        assert z.grad is not None
        assert not torch.isnan(z.grad).any()

    def test_generator_different_configs(self):
        """Test generator with different configurations."""
        configs = [
            {"noise_dim": 50, "out_dim": 100, "depth": 5, "width": 500},
            {"noise_dim": 200, "out_dim": 250, "depth": 8, "width": 800},
        ]

        for config in configs:
            gen = Generator(**config)
            z = torch.randn(4, config["noise_dim"])
            out = gen(z)
            assert out.shape == (4, config["out_dim"])


class TestCritic:
    """Tests for Critic (Discriminator)."""

    def test_critic_forward_shape(self):
        """Test critic output shape."""
        critic = Critic(in_dim=250, depth=5, width=100)

        x = torch.randn(8, 250)
        score = critic(x)

        assert score.shape == (8, 1)

    def test_critic_batch_sizes(self):
        """Test critic with various batch sizes."""
        critic = Critic()

        for batch_size in [1, 8, 32, 100]:
            x = torch.randn(batch_size, 250)
            score = critic(x)
            assert score.shape == (batch_size, 1)

    def test_critic_no_nan(self):
        """Test critic produces no NaN outputs."""
        critic = Critic()

        x = torch.randn(32, 250)
        score = critic(x)

        assert not torch.isnan(score).any()
        assert not torch.isinf(score).any()

    def test_critic_gradients(self):
        """Test gradients flow through critic."""
        critic = Critic()

        x = torch.randn(8, 250, requires_grad=True)
        score = critic(x)
        loss = score.sum()
        loss.backward()

        assert x.grad is not None
        assert not torch.isnan(x.grad).any()

    def test_critic_no_batchnorm(self):
        """Test critic has no BatchNorm layers (per WGAN paper)."""
        critic = Critic()

        for module in critic.modules():
            assert not isinstance(module, torch.nn.BatchNorm1d), (
                "Critic should not have BatchNorm"
            )


class TestHistoryPreproc:
    """Tests for HistoryPreproc module."""

    def test_preproc_forward_shape(self):
        """Test preprocessor output shape."""
        preproc = HistoryPreproc(hist_dim=250, feat_dim=250, hidden=512)

        hist = torch.randn(8, 250)
        feat = preproc(hist)

        assert feat.shape == (8, 250)

    def test_preproc_different_dims(self):
        """Test preprocessor with different dimensions."""
        preproc = HistoryPreproc(hist_dim=100, feat_dim=50, hidden=256)

        hist = torch.randn(8, 100)
        feat = preproc(hist)

        assert feat.shape == (8, 50)

    def test_preproc_no_nan(self):
        """Test preprocessor produces no NaN."""
        preproc = HistoryPreproc()

        hist = torch.randn(32, 250)
        feat = preproc(hist)

        assert not torch.isnan(feat).any()


class TestFEGenerator:
    """Tests for Feature-Enriched Generator."""

    def test_fegen_forward_shape(self):
        """Test FE-GAN output shape."""
        fegen = FEGenerator(noise_dim=100, hist_dim=250, out_dim=250)

        z = torch.randn(8, 100)
        hist = torch.randn(8, 250)
        out = fegen(z, hist)

        assert out.shape == (8, 250)

    def test_fegen_batch_sizes(self):
        """Test FE-GAN with various batch sizes."""
        fegen = FEGenerator()
        fegen.eval()  # BatchNorm requires batch_size > 1 in training mode

        for batch_size in [1, 8, 32, 100]:
            z = torch.randn(batch_size, 100)
            hist = torch.randn(batch_size, 250)
            with torch.no_grad():
                out = fegen(z, hist)
            assert out.shape == (batch_size, 250)

    def test_fegen_param_count(self):
        """Test FE-GAN has reasonable param count."""
        fegen = FEGenerator()

        n_params = count_parameters(fegen)

        # Should have more params than vanilla generator due to preproc
        gen = Generator()
        gen_params = count_parameters(gen)

        assert n_params > gen_params, "FE-GAN should have more params than vanilla"

    def test_fegen_no_nan(self):
        """Test FE-GAN produces no NaN."""
        fegen = FEGenerator()

        z = torch.randn(32, 100)
        hist = torch.randn(32, 250)
        out = fegen(z, hist)

        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()

    def test_fegen_gradients(self):
        """Test gradients flow through FE-GAN."""
        fegen = FEGenerator()

        z = torch.randn(8, 100, requires_grad=True)
        hist = torch.randn(8, 250, requires_grad=True)
        out = fegen(z, hist)
        loss = out.sum()
        loss.backward()

        assert z.grad is not None
        assert hist.grad is not None
        assert not torch.isnan(z.grad).any()
        assert not torch.isnan(hist.grad).any()

    def test_fegen_history_affects_output(self):
        """Test that different history produces different output."""
        fegen = FEGenerator()
        fegen.eval()  # Disable BatchNorm noise

        z = torch.randn(8, 100)
        hist1 = torch.randn(8, 250)
        hist2 = torch.randn(8, 250)

        with torch.no_grad():
            out1 = fegen(z, hist1)
            out2 = fegen(z, hist2)

        # Different history should produce different output
        assert not torch.allclose(out1, out2), "FE-GAN should use history"


class TestFactoryFunctions:
    """Tests for factory functions."""

    def test_create_generator_wgan(self):
        """Test creating vanilla generator."""
        gen = create_generator(model_type="wgan")

        assert isinstance(gen, Generator)

        z = torch.randn(4, 100)
        out = gen(z)
        assert out.shape == (4, 250)

    def test_create_generator_fegan(self):
        """Test creating FE-GAN generator."""
        gen = create_generator(model_type="fegan")

        assert isinstance(gen, FEGenerator)

        z = torch.randn(4, 100)
        hist = torch.randn(4, 250)
        out = gen(z, hist)
        assert out.shape == (4, 250)

    def test_create_generator_invalid(self):
        """Test invalid model type raises error."""
        with pytest.raises(ValueError, match="Unknown model_type"):
            create_generator(model_type="invalid")

    def test_create_critic(self):
        """Test creating critic."""
        critic = create_critic()

        assert isinstance(critic, Critic)

        x = torch.randn(4, 250)
        score = critic(x)
        assert score.shape == (4, 1)


class TestModelInfo:
    """Tests for model info utilities."""

    def test_count_parameters(self):
        """Test parameter counting."""
        gen = Generator()
        n_params = count_parameters(gen)

        assert n_params > 0
        assert isinstance(n_params, int)

    def test_get_model_info(self):
        """Test model info extraction."""
        gen = Generator()
        info = get_model_info(gen)

        assert "class" in info
        assert info["class"] == "Generator"
        assert "trainable_params" in info
        assert "total_params" in info


class TestGPU:
    """Tests for GPU compatibility (skipped if no GPU)."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_generator_gpu(self):
        """Test generator on GPU."""
        gen = Generator().cuda()

        z = torch.randn(8, 100, device="cuda")
        out = gen(z)

        assert out.device.type == "cuda"
        assert out.shape == (8, 250)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_fegen_gpu(self):
        """Test FE-GAN on GPU."""
        fegen = FEGenerator().cuda()

        z = torch.randn(8, 100, device="cuda")
        hist = torch.randn(8, 250, device="cuda")
        out = fegen(z, hist)

        assert out.device.type == "cuda"
        assert out.shape == (8, 250)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_critic_gpu(self):
        """Test critic on GPU."""
        critic = Critic().cuda()

        x = torch.randn(8, 250, device="cuda")
        score = critic(x)

        assert score.device.type == "cuda"
        assert score.shape == (8, 1)

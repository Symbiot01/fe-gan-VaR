"""Trainer for WGAN and FE-GAN models.

Implements the training loop with:
- WGAN training with weight clipping
- Optional Tail-GAN loss for generator
- Periodic evaluation during training
- Inline final evaluation at the end
"""

from __future__ import annotations

import csv
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import torch
import torch.nn as nn
import torch.optim as optim
from torch import Tensor
from tqdm import tqdm

from fe_gan.data import VIXDataset
from fe_gan.eval import evaluate_generator
from fe_gan.losses import (
    clip_weights,
    fissler_ziegel_loss,
    tail_gan_gen_loss,
    wgan_critic_loss,
    wgan_gen_loss,
)
from fe_gan.utils import (
    get_system_info,
    hash_config,
    save_json,
    seed_everything,
)


@dataclass
class TrainerConfig:
    """Configuration for training."""

    # Model type
    model_type: Literal["wgan", "fegan"] = "wgan"
    loss_type: Literal["wgan", "tailgan"] = "wgan"

    # Architecture
    noise_dim: int = 100
    gen_depth: int = 10
    gen_width: int = 1000
    critic_depth: int = 5
    critic_width: int = 100

    # FE-GAN specific
    hist_dim: int = 250
    preproc_hidden: int = 512
    preproc_feat_dim: int = 250

    # Training
    epochs: int = 100
    batch_size: int = 100
    n_critic: int = 5
    clip_value: float = 0.01
    lr: float = 5e-5

    # Tail-GAN specific
    alpha: float = 0.05
    tailgan_wgan_weight: float = 0.5
    tailgan_fz_loss_clip_value: float | None = None
    generator_grad_clip_norm: float | None = None

    # Evaluation
    eval_every: int = 10
    eval_samples: int = 1000

    # Seeding
    seed: int = 0

    # Output
    out_dir: str = "results/run"
    save_checkpoint: bool = False

    def __post_init__(self) -> None:
        """Validate values that affect training stability."""
        if not 0.0 < self.alpha < 1.0:
            raise ValueError("alpha must be between 0 and 1")
        if not 0.0 <= self.tailgan_wgan_weight <= 1.0:
            raise ValueError("tailgan_wgan_weight must be between 0 and 1")
        if self.tailgan_fz_loss_clip_value is not None and self.tailgan_fz_loss_clip_value <= 0:
            raise ValueError("tailgan_fz_loss_clip_value must be positive")
        if self.generator_grad_clip_norm is not None and self.generator_grad_clip_norm <= 0:
            raise ValueError("generator_grad_clip_norm must be positive")

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "model_type": self.model_type,
            "loss_type": self.loss_type,
            "noise_dim": self.noise_dim,
            "gen_depth": self.gen_depth,
            "gen_width": self.gen_width,
            "critic_depth": self.critic_depth,
            "critic_width": self.critic_width,
            "hist_dim": self.hist_dim,
            "preproc_hidden": self.preproc_hidden,
            "preproc_feat_dim": self.preproc_feat_dim,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "n_critic": self.n_critic,
            "clip_value": self.clip_value,
            "lr": self.lr,
            "alpha": self.alpha,
            "tailgan_wgan_weight": self.tailgan_wgan_weight,
            "tailgan_fz_loss_clip_value": self.tailgan_fz_loss_clip_value,
            "generator_grad_clip_norm": self.generator_grad_clip_norm,
            "eval_every": self.eval_every,
            "eval_samples": self.eval_samples,
            "seed": self.seed,
            "out_dir": self.out_dir,
            "save_checkpoint": self.save_checkpoint,
        }


@dataclass
class TrainingMetrics:
    """Metrics collected during training."""

    epoch: int = 0
    d_loss: float = 0.0
    g_loss: float = 0.0
    fz_loss: float | None = None
    var_err_median: float | None = None
    es_err_median: float | None = None
    wall_seconds: float = 0.0


class Trainer:
    """WGAN / FE-GAN trainer with inline evaluation."""

    def __init__(
        self,
        generator: nn.Module,
        critic: nn.Module,
        train_data: VIXDataset,
        eval_data: VIXDataset,
        config: TrainerConfig,
        device: torch.device | str = "cuda",
    ):
        """Initialize trainer.

        Args:
            generator: Generator model.
            critic: Critic model.
            train_data: Training dataset.
            eval_data: Evaluation dataset.
            config: Training configuration.
            device: Device to train on.
        """
        self.generator = generator
        self.critic = critic
        self.train_data = train_data
        self.eval_data = eval_data
        self.config = config
        self.device = torch.device(device)

        # Move models to device
        self.generator = self.generator.to(self.device)
        self.critic = self.critic.to(self.device)

        # Move data to device for faster sampling
        self.train_data = self.train_data.to(self.device)
        self.eval_data = self.eval_data.to(self.device)

        # Optimizers (RMSprop as per paper)
        self.opt_g = optim.RMSprop(self.generator.parameters(), lr=config.lr)
        self.opt_c = optim.RMSprop(self.critic.parameters(), lr=config.lr)

        # Metrics history
        self.metrics_history: list[TrainingMetrics] = []

        # Determine if FE-GAN
        self.is_fegan = config.model_type == "fegan"

        # Create output directory
        self.out_dir = Path(config.out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def _sample_batch(self) -> Tensor:
        """Sample a batch of real windows."""
        return self.train_data.sample_batch(self.config.batch_size)

    def _sample_noise(self, batch_size: int | None = None) -> Tensor:
        """Sample noise for generator."""
        bs = batch_size or self.config.batch_size
        return torch.randn(bs, self.config.noise_dim, device=self.device)

    def _generate(self, z: Tensor, hist: Tensor | None = None) -> Tensor:
        """Generate fake samples."""
        if self.is_fegan:
            if hist is None:
                hist = self._sample_batch()
            return self.generator(z, hist)
        else:
            return self.generator(z)

    def _train_critic_step(self, real: Tensor) -> float:
        """One critic training step.

        Args:
            real: Real samples batch.

        Returns:
            Critic loss value.
        """
        self.opt_c.zero_grad()

        # Generate fake samples
        with torch.no_grad():
            z = self._sample_noise()
            fake = self._generate(z, real if self.is_fegan else None)

        # Critic scores
        real_score = self.critic(real)
        fake_score = self.critic(fake)

        # WGAN critic loss
        c_loss = wgan_critic_loss(real_score, fake_score)

        c_loss.backward()
        self.opt_c.step()

        # Weight clipping
        clip_weights(self.critic, self.config.clip_value)

        return float(c_loss.detach())

    def _train_generator_step(self, real: Tensor) -> tuple[float, float | None]:
        """One generator training step.

        Args:
            real: Real samples (used as history for FE-GAN).

        Returns:
            Tuple of (total_loss, fz_loss if Tail-GAN else None).
        """
        self.opt_g.zero_grad()

        z = self._sample_noise()
        fake = self._generate(z, real if self.is_fegan else None)

        # Critic score for fake
        fake_score = self.critic(fake)

        if self.config.loss_type == "tailgan":
            # Combined WGAN + Tail-GAN loss
            g_loss = tail_gan_gen_loss(
                fake,
                wgan_weight=self.config.tailgan_wgan_weight,
                fake_score=fake_score,
                alpha=self.config.alpha,
                fz_loss_clip_value=self.config.tailgan_fz_loss_clip_value,
            )
            # Also compute FZ loss for logging
            fz_loss = float(fissler_ziegel_loss(fake.detach(), self.config.alpha))
        else:
            # Pure WGAN loss
            g_loss = wgan_gen_loss(fake_score)
            fz_loss = None

        if not torch.isfinite(g_loss):
            raise FloatingPointError(f"Non-finite generator loss for seed {self.config.seed}")

        g_loss.backward()
        if self.config.generator_grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                self.generator.parameters(),
                max_norm=self.config.generator_grad_clip_norm,
                error_if_nonfinite=True,
            )
        self.opt_g.step()

        return float(g_loss.detach()), fz_loss

    def _evaluate(self) -> dict[str, float]:
        """Run evaluation on current generator."""
        metrics = evaluate_generator(
            self.generator,
            self.eval_data.get_all(),
            n_samples=self.config.eval_samples,
            noise_dim=self.config.noise_dim,
            alpha=self.config.alpha,
            device=self.device,
            is_fegan=self.is_fegan,
        )
        return metrics

    def train(self, progress: bool = True) -> dict[str, Any]:
        """Run full training loop.

        Args:
            progress: Whether to show progress bar.

        Returns:
            Dictionary with final evaluation metrics.
        """
        # Seed for reproducibility
        seed_everything(self.config.seed)

        start_time = time.perf_counter()
        epochs_iter = range(1, self.config.epochs + 1)
        if progress:
            epochs_iter = tqdm(epochs_iter, desc="Training", leave=False)

        for epoch in epochs_iter:
            epoch_start = time.perf_counter()

            # Train critic for n_critic steps
            d_losses = []
            for _ in range(self.config.n_critic):
                real = self._sample_batch()
                d_loss = self._train_critic_step(real)
                d_losses.append(d_loss)

            # Train generator for 1 step
            real = self._sample_batch()
            g_loss, fz_loss = self._train_generator_step(real)

            # Collect metrics
            epoch_time = time.perf_counter() - epoch_start
            metrics = TrainingMetrics(
                epoch=epoch,
                d_loss=sum(d_losses) / len(d_losses),
                g_loss=g_loss,
                fz_loss=fz_loss,
                wall_seconds=epoch_time,
            )

            # Periodic evaluation
            if epoch % self.config.eval_every == 0 or epoch == self.config.epochs:
                eval_metrics = self._evaluate()
                metrics.var_err_median = eval_metrics.get("var_err_median")
                metrics.es_err_median = eval_metrics.get("es_err_median")

            self.metrics_history.append(metrics)

            # Update progress bar
            if progress and hasattr(epochs_iter, "set_postfix"):
                epochs_iter.set_postfix(
                    D=f"{metrics.d_loss:.3f}",
                    G=f"{metrics.g_loss:.3f}",
                )

        total_time = time.perf_counter() - start_time

        # Final evaluation
        final_eval = self._evaluate()
        final_eval["total_wall_seconds"] = total_time
        final_eval["epochs"] = self.config.epochs

        # Save results
        self._save_results(final_eval)

        return final_eval

    def _save_results(self, final_eval: dict[str, Any]) -> None:
        """Save training results to disk."""
        # Save metrics CSV
        metrics_path = self.out_dir / "metrics.csv"
        with open(metrics_path, "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "epoch",
                    "d_loss",
                    "g_loss",
                    "fz_loss",
                    "var_err_median",
                    "es_err_median",
                    "wall_seconds",
                ],
            )
            writer.writeheader()
            for m in self.metrics_history:
                writer.writerow(
                    {
                        "epoch": m.epoch,
                        "d_loss": m.d_loss,
                        "g_loss": m.g_loss,
                        "fz_loss": m.fz_loss,
                        "var_err_median": m.var_err_median,
                        "es_err_median": m.es_err_median,
                        "wall_seconds": m.wall_seconds,
                    }
                )

        # Save eval JSON
        eval_path = self.out_dir / "eval.json"
        save_json(final_eval, eval_path)

        # Save config
        config_path = self.out_dir / "config.json"
        save_json(self.config.to_dict(), config_path)

        # Save metadata
        metadata = get_system_info()
        metadata["config_hash"] = hash_config(self.config.to_dict())
        metadata["seed"] = self.config.seed
        metadata_path = self.out_dir / "metadata.json"
        save_json(metadata, metadata_path)

        # Optionally save checkpoint
        if self.config.save_checkpoint:
            ckpt_path = self.out_dir / "final_gen.pt"
            torch.save(self.generator.state_dict(), ckpt_path)


def train_model(
    train_data: VIXDataset,
    eval_data: VIXDataset,
    config: TrainerConfig,
    device: torch.device | str = "cuda",
    progress: bool = True,
) -> dict[str, Any]:
    """Train a model with given config.

    Convenience function that creates models and trainer.

    Args:
        train_data: Training dataset.
        eval_data: Evaluation dataset.
        config: Training configuration.
        device: Device to use.
        progress: Whether to show progress.

    Returns:
        Final evaluation metrics.
    """
    from fe_gan.models import create_critic, create_generator

    # Create models
    generator = create_generator(
        model_type=config.model_type,
        noise_dim=config.noise_dim,
        out_dim=config.hist_dim,  # output same size as window
        depth=config.gen_depth,
        width=config.gen_width,
    )

    critic = create_critic(
        in_dim=config.hist_dim,
        depth=config.critic_depth,
        width=config.critic_width,
    )

    # Create trainer
    trainer = Trainer(
        generator=generator,
        critic=critic,
        train_data=train_data,
        eval_data=eval_data,
        config=config,
        device=device,
    )

    # Train
    return trainer.train(progress=progress)

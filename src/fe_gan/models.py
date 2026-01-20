"""Neural network models for FE-GAN.

Implements the architecture from Chen (2024):
- Generator: 10-layer MLP with BatchNorm, generates 250-day return windows
- Critic: 5-layer MLP (no BN), scores real vs generated windows
- HistoryPreproc: Preprocesses historical context for FE-GAN
- FEGenerator: Feature-enriched generator using historical context

Architecture details (from paper Section 2.1):
- Generator: Linear(100, 1000) -> [Linear(1000, 1000) + BN + ReLU] x 8 -> Linear(1000, 250)
- Critic: [Linear + ReLU] x 4 -> Linear -> 1 (no BN)
- HistoryPreproc: Linear(250, 512) -> ReLU -> Linear(512, 250)
"""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn
from torch import Tensor


class Generator(nn.Module):
    """WGAN Generator for financial time series.

    Generates synthetic 250-day log return windows from noise.

    Architecture:
        - Input: noise vector [B, noise_dim]
        - 10 linear layers with width 1000
        - BatchNorm + ReLU after each hidden layer
        - Output: [B, out_dim]

    Args:
        noise_dim: Dimension of input noise vector (default 100).
        out_dim: Dimension of output (default 250 for 250 trading days).
        depth: Number of linear layers (default 10).
        width: Width of hidden layers (default 1000).
    """

    def __init__(
        self,
        noise_dim: int = 100,
        out_dim: int = 250,
        depth: int = 10,
        width: int = 1000,
    ):
        super().__init__()

        self.noise_dim = noise_dim
        self.out_dim = out_dim
        self.depth = depth
        self.width = width

        layers: list[nn.Module] = []

        # First layer: noise_dim -> width
        layers.append(nn.Linear(noise_dim, width))
        layers.append(nn.BatchNorm1d(width))
        layers.append(nn.ReLU())

        # Hidden layers: width -> width
        for _ in range(depth - 2):
            layers.append(nn.Linear(width, width))
            layers.append(nn.BatchNorm1d(width))
            layers.append(nn.ReLU())

        # Output layer: width -> out_dim
        layers.append(nn.Linear(width, out_dim))

        self.layers = nn.Sequential(*layers)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights using Xavier uniform."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, z: Tensor) -> Tensor:
        """Generate synthetic returns from noise.

        Args:
            z: Noise tensor [B, noise_dim].

        Returns:
            Synthetic returns [B, out_dim].
        """
        return self.layers(z)


class Critic(nn.Module):
    """WGAN Critic (Discriminator) for financial time series.

    Scores real vs generated windows. Higher score = more "real-like".

    Architecture:
        - Input: return window [B, in_dim]
        - 5 linear layers with width 100
        - ReLU activation (no BatchNorm per WGAN paper)
        - Output: scalar score [B, 1]

    Args:
        in_dim: Dimension of input (default 250).
        depth: Number of linear layers (default 5).
        width: Width of hidden layers (default 100).
    """

    def __init__(
        self,
        in_dim: int = 250,
        depth: int = 5,
        width: int = 100,
    ):
        super().__init__()

        self.in_dim = in_dim
        self.depth = depth
        self.width = width

        layers: list[nn.Module] = []

        # First layer: in_dim -> width
        layers.append(nn.Linear(in_dim, width))
        layers.append(nn.ReLU())

        # Hidden layers: width -> width
        for _ in range(depth - 2):
            layers.append(nn.Linear(width, width))
            layers.append(nn.ReLU())

        # Output layer: width -> 1 (score)
        layers.append(nn.Linear(width, 1))

        self.layers = nn.Sequential(*layers)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights using Xavier uniform."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: Tensor) -> Tensor:
        """Score a return window.

        Args:
            x: Return window [B, in_dim].

        Returns:
            Score [B, 1]. Higher = more real-like.
        """
        return self.layers(x)


class HistoryPreproc(nn.Module):
    """Preprocessor for historical context in FE-GAN.

    Transforms the historical return sequence into a feature representation
    that is concatenated with noise before feeding to the generator.

    Architecture:
        - Input: history window [B, hist_dim]
        - Linear(hist_dim, hidden) -> ReLU -> Linear(hidden, feat_dim)
        - Output: features [B, feat_dim]

    Args:
        hist_dim: Dimension of history input (default 250).
        feat_dim: Dimension of output features (default 250).
        hidden: Hidden layer width (default 512).
    """

    def __init__(
        self,
        hist_dim: int = 250,
        feat_dim: int = 250,
        hidden: int = 512,
    ):
        super().__init__()

        self.hist_dim = hist_dim
        self.feat_dim = feat_dim
        self.hidden = hidden

        self.layers = nn.Sequential(
            nn.Linear(hist_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, feat_dim),
            nn.ReLU(),
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights using Xavier uniform."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, hist: Tensor) -> Tensor:
        """Transform historical context.

        Args:
            hist: History window [B, hist_dim].

        Returns:
            Features [B, feat_dim].
        """
        return self.layers(hist)


class FEGenerator(nn.Module):
    """Feature-Enriched Generator (FE-GAN).

    Incorporates historical context into the generator by:
    1. Preprocessing history through HistoryPreproc
    2. Concatenating preprocessed features with noise
    3. Feeding to a modified Generator body

    Architecture:
        - Input: (noise [B, noise_dim], history [B, hist_dim])
        - HistoryPreproc(history) -> [B, feat_dim]
        - Concat with noise -> [B, noise_dim + feat_dim]
        - Generator body (first layer adjusted) -> [B, out_dim]

    Args:
        noise_dim: Dimension of noise (default 100).
        hist_dim: Dimension of history input (default 250).
        out_dim: Dimension of output (default 250).
        depth: Depth of generator body (default 10).
        width: Width of generator hidden layers (default 1000).
        preproc_hidden: Hidden dim of preprocessor (default 512).
        preproc_feat_dim: Output dim of preprocessor (default 250).
    """

    def __init__(
        self,
        noise_dim: int = 100,
        hist_dim: int = 250,
        out_dim: int = 250,
        depth: int = 10,
        width: int = 1000,
        preproc_hidden: int = 512,
        preproc_feat_dim: int = 250,
    ):
        super().__init__()

        self.noise_dim = noise_dim
        self.hist_dim = hist_dim
        self.out_dim = out_dim
        self.feat_dim = preproc_feat_dim

        # History preprocessor
        self.preproc = HistoryPreproc(
            hist_dim=hist_dim,
            feat_dim=preproc_feat_dim,
            hidden=preproc_hidden,
        )

        # Generator body with adjusted input size
        input_dim = noise_dim + preproc_feat_dim

        layers: list[nn.Module] = []

        # First layer: combined input -> width
        layers.append(nn.Linear(input_dim, width))
        layers.append(nn.BatchNorm1d(width))
        layers.append(nn.ReLU())

        # Hidden layers
        for _ in range(depth - 2):
            layers.append(nn.Linear(width, width))
            layers.append(nn.BatchNorm1d(width))
            layers.append(nn.ReLU())

        # Output layer
        layers.append(nn.Linear(width, out_dim))

        self.body = nn.Sequential(*layers)

        # Initialize body weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights using Xavier uniform."""
        for m in self.body.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, z: Tensor, hist: Tensor) -> Tensor:
        """Generate returns conditioned on history.

        Args:
            z: Noise tensor [B, noise_dim].
            hist: History window [B, hist_dim].

        Returns:
            Synthetic returns [B, out_dim].
        """
        # Preprocess history
        feat = self.preproc(hist)

        # Concatenate noise and features
        combined = torch.cat([z, feat], dim=1)

        # Generate through body
        return self.body(combined)


def create_generator(
    model_type: Literal["wgan", "fegan"] = "wgan",
    noise_dim: int = 100,
    out_dim: int = 250,
    hist_dim: int = 250,
    depth: int = 10,
    width: int = 1000,
) -> Generator | FEGenerator:
    """Factory function to create appropriate generator.

    Args:
        model_type: 'wgan' for vanilla generator, 'fegan' for FE-GAN.
        noise_dim: Dimension of noise input.
        out_dim: Dimension of output.
        hist_dim: Dimension of history (for fegan only).
        depth: Number of layers.
        width: Hidden layer width.

    Returns:
        Generator or FEGenerator instance.
    """
    if model_type == "wgan":
        return Generator(
            noise_dim=noise_dim,
            out_dim=out_dim,
            depth=depth,
            width=width,
        )
    elif model_type == "fegan":
        return FEGenerator(
            noise_dim=noise_dim,
            hist_dim=hist_dim,
            out_dim=out_dim,
            depth=depth,
            width=width,
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}")


def create_critic(
    in_dim: int = 250,
    depth: int = 5,
    width: int = 100,
) -> Critic:
    """Factory function to create critic.

    Args:
        in_dim: Dimension of input.
        depth: Number of layers.
        width: Hidden layer width.

    Returns:
        Critic instance.
    """
    return Critic(in_dim=in_dim, depth=depth, width=width)


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters in a model.

    Args:
        model: PyTorch module.

    Returns:
        Number of trainable parameters.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model_info(model: nn.Module) -> dict:
    """Get model summary information.

    Args:
        model: PyTorch module.

    Returns:
        Dictionary with model info.
    """
    return {
        "class": model.__class__.__name__,
        "trainable_params": count_parameters(model),
        "total_params": sum(p.numel() for p in model.parameters()),
    }

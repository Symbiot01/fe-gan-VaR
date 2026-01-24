"""Loss functions for WGAN and Tail-GAN training.

Implements:
- WGAN critic and generator losses with weight clipping
- Empirical VaR and ES estimation (differentiable)
- Fissler-Ziegel scoring function (FZ0 variant from Cont et al. 2022)
- Tail-GAN generator loss

Key references:
- Arjovsky et al. 2017: Wasserstein GAN
- Cont et al. 2022: Tail-GAN (eq. 15 for FZ scoring)
- Fissler & Ziegel 2016: Higher-order elicitability
"""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn
from torch import Tensor


# =============================================================================
# WGAN Losses
# =============================================================================


def wgan_critic_loss(real_score: Tensor, fake_score: Tensor) -> Tensor:
    """WGAN critic loss (Wasserstein distance estimator).

    The critic tries to maximize E[D(real)] - E[D(fake)],
    so we minimize -(E[D(real)] - E[D(fake)]) = E[D(fake)] - E[D(real)].

    Args:
        real_score: Critic scores for real samples [B, 1].
        fake_score: Critic scores for fake samples [B, 1].

    Returns:
        Scalar loss tensor.
    """
    return fake_score.mean() - real_score.mean()


def wgan_gen_loss(fake_score: Tensor) -> Tensor:
    """WGAN generator loss.

    The generator tries to maximize E[D(fake)], so we minimize -E[D(fake)].

    Args:
        fake_score: Critic scores for generated samples [B, 1].

    Returns:
        Scalar loss tensor.
    """
    return -fake_score.mean()


def clip_weights(model: nn.Module, clip_value: float = 0.01) -> None:
    """Clip model weights to [-clip_value, clip_value].

    Standard WGAN weight clipping to enforce Lipschitz constraint.

    Args:
        model: Neural network module (typically the critic).
        clip_value: Maximum absolute weight value.
    """
    for p in model.parameters():
        p.data.clamp_(-clip_value, clip_value)


# =============================================================================
# VaR and ES Estimation
# =============================================================================


def empirical_quantile(x: Tensor, alpha: float) -> Tensor:
    """Compute empirical quantile of samples.

    Uses linear interpolation for quantile estimation.

    Args:
        x: Samples tensor [N] or [B, N].
        alpha: Quantile level (e.g., 0.05 for 5th percentile).

    Returns:
        Quantile value(s). Shape [] for 1D input, [B] for 2D input.
    """
    if x.dim() == 1:
        x = x.unsqueeze(0)
        squeeze = True
    else:
        squeeze = False

    # Sort along the last dimension
    sorted_x, _ = torch.sort(x, dim=-1)
    n = sorted_x.shape[-1]

    # Compute index (linear interpolation)
    idx = alpha * (n - 1)
    idx_low = int(idx)
    idx_high = min(idx_low + 1, n - 1)
    weight = idx - idx_low

    # Interpolate
    q_low = sorted_x[:, idx_low]
    q_high = sorted_x[:, idx_high]
    quantile = q_low + weight * (q_high - q_low)

    if squeeze:
        quantile = quantile.squeeze(0)

    return quantile


def empirical_var_es(
    returns: Tensor,
    alpha: float = 0.05,
) -> tuple[Tensor, Tensor]:
    """Compute empirical Value-at-Risk and Expected Shortfall.

    VaR_alpha is the alpha-quantile of returns (left tail).
    ES_alpha is the expected value of returns below VaR_alpha.

    Note: For financial risk, VaR and ES are typically positive numbers
    representing potential loss. Here we use the standard definition where
    VaR is the negative of the alpha-quantile of log returns.

    Args:
        returns: Log returns tensor [B, T] or [T].
        alpha: Risk level (default 0.05 for 5% tail).

    Returns:
        Tuple of (VaR, ES) tensors.
        - For [T] input: shape []
        - For [B, T] input: shape [B]

    VaR is the alpha-quantile (left tail cutoff).
    ES is the average of returns below VaR.
    """
    if returns.dim() == 1:
        returns = returns.unsqueeze(0)
        squeeze = True
    else:
        squeeze = False

    # Sort returns
    sorted_returns, _ = torch.sort(returns, dim=-1)
    n = sorted_returns.shape[-1]

    # VaR: alpha-quantile
    var = empirical_quantile(returns, alpha)

    # ES: mean of returns below VaR
    # Use soft indicator for differentiability
    # Hard version: mask = returns <= var.unsqueeze(-1)
    # Soft version: use straight-through for gradients

    # Number of samples in tail
    k = max(1, int(alpha * n))

    # ES as mean of k lowest values
    es = sorted_returns[:, :k].mean(dim=-1)

    if squeeze:
        var = var.squeeze(0)
        es = es.squeeze(0)

    return var, es


def soft_empirical_var_es(
    returns: Tensor,
    alpha: float = 0.05,
    temperature: float = 0.1,
) -> tuple[Tensor, Tensor]:
    """Differentiable VaR/ES using soft sorting.

    Uses soft indicator functions for smooth gradients.
    More numerically stable for gradient-based optimization.

    Args:
        returns: Log returns tensor [B, T].
        alpha: Risk level.
        temperature: Softness of the indicator (lower = sharper).

    Returns:
        Tuple of (VaR, ES) tensors of shape [B].
    """
    if returns.dim() == 1:
        returns = returns.unsqueeze(0)
        squeeze = True
    else:
        squeeze = False

    # Compute VaR via quantile
    var = empirical_quantile(returns, alpha)  # [B]

    # Compute soft indicator: P(return <= VaR)
    # Using sigmoid for smooth approximation
    diff = (var.unsqueeze(-1) - returns) / temperature  # [B, T]
    weights = torch.sigmoid(diff)  # Soft indicator

    # Normalize weights
    weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-8)

    # ES as weighted average
    es = (returns * weights).sum(dim=-1)  # [B]

    if squeeze:
        var = var.squeeze(0)
        es = es.squeeze(0)

    return var, es


# =============================================================================
# Fissler-Ziegel Score (Tail-GAN Loss)
# =============================================================================


def fissler_ziegel_score(
    returns: Tensor,
    var: Tensor,
    es: Tensor,
    alpha: float = 0.05,
    variant: Literal["FZ0", "FZ1"] = "FZ0",
) -> Tensor:
    """Compute Fissler-Ziegel scoring function.

    This is a strictly consistent scoring function for jointly evaluating
    VaR and ES, from Fissler & Ziegel (2016) and Cont et al. (2022).

    FZ0 variant (eq. 15 from Cont et al. 2022):
        S(v, e, y) = (1/e) * (e - v + (v - y)^+ / alpha) + log(-e) - 1

    where:
        - v = VaR (typically negative for left tail)
        - e = ES (should be more negative than VaR, e < v < 0)
        - y = realized return
        - (x)^+ = max(0, x)

    The score is minimized when (v, e) equals the true (VaR, ES).

    Args:
        returns: Realized returns [N] (1D array of samples).
        var: Value-at-Risk estimate (scalar tensor).
        es: Expected Shortfall estimate (scalar tensor, must be < var).
        alpha: Risk level.
        variant: Scoring variant ('FZ0' or 'FZ1').

    Returns:
        Scalar score tensor (lower is better).
    """
    # Flatten to 1D for simplicity
    returns = returns.flatten()

    # Ensure var and es are scalars
    var = var.reshape([]) if var.dim() > 0 else var
    es = es.reshape([]) if es.dim() > 0 else es

    eps = 1e-8

    if variant == "FZ0":
        # Ensure ES is negative for the log term
        safe_es = es.clamp(max=-eps)

        # Exceedance term: max(0, VaR - return) for returns below VaR
        exceedance = torch.relu(var - returns)  # [N]

        # FZ0: S(v, e, y) = (1/e)(e - v + exceedance/alpha) + log(-e) - 1
        # Per-sample score
        per_sample = (safe_es - var + exceedance / alpha) / safe_es + torch.log(-safe_es) - 1

        # Average score
        score = per_sample.mean()

    elif variant == "FZ1":
        raise NotImplementedError("FZ1 variant not implemented")
    else:
        raise ValueError(f"Unknown variant: {variant}")

    return score


def fissler_ziegel_loss(
    returns: Tensor,
    alpha: float = 0.05,
    variant: Literal["FZ0", "FZ1"] = "FZ0",
) -> Tensor:
    """Compute FZ loss using empirical VaR/ES from the returns.

    This is a self-consistent loss: we estimate VaR and ES from the same
    returns we're scoring. Minimizing this encourages the generator to
    produce returns with accurate tail statistics.

    For batched inputs [B, T], computes loss per batch element and averages.

    Args:
        returns: Generated returns [T] or [B, T].
        alpha: Risk level.
        variant: FZ variant.

    Returns:
        Scalar loss tensor.
    """
    if returns.dim() == 1:
        # Single sample
        var, es = empirical_var_es(returns, alpha)
        return fissler_ziegel_score(returns, var, es, alpha, variant)

    # Batched: compute loss per batch element and average
    batch_size = returns.shape[0]
    scores = []

    for i in range(batch_size):
        batch_returns = returns[i]
        var, es = empirical_var_es(batch_returns, alpha)
        score = fissler_ziegel_score(batch_returns, var, es, alpha, variant)
        scores.append(score)

    return torch.stack(scores).mean()


def tail_gan_gen_loss(
    fake_returns: Tensor,
    real_returns: Tensor | None = None,
    alpha: float = 0.05,
    wgan_weight: float = 0.0,
    fake_score: Tensor | None = None,
) -> Tensor:
    """Tail-GAN generator loss combining WGAN and FZ losses.

    The generator is trained to:
    1. Fool the critic (WGAN loss)
    2. Match tail risk statistics (FZ loss)

    When wgan_weight > 0 and fake_score is provided, combines both losses:
        loss = wgan_weight * wgan_loss + (1 - wgan_weight) * fz_loss

    Args:
        fake_returns: Generated returns [B, T].
        real_returns: Real returns for reference (optional, not used in current impl).
        alpha: Risk level for VaR/ES.
        wgan_weight: Weight for WGAN loss (0 = pure FZ, 1 = pure WGAN).
        fake_score: Critic scores for generated samples (needed if wgan_weight > 0).

    Returns:
        Scalar loss tensor.
    """
    # FZ loss component
    fz_loss = fissler_ziegel_loss(fake_returns, alpha)

    if wgan_weight > 0 and fake_score is not None:
        wgan_loss = wgan_gen_loss(fake_score)
        return wgan_weight * wgan_loss + (1 - wgan_weight) * fz_loss
    else:
        return fz_loss


# =============================================================================
# Evaluation Metrics
# =============================================================================


def var_es_error(
    generated: Tensor,
    real: Tensor,
    alpha: float = 0.05,
) -> tuple[Tensor, Tensor]:
    """Compute absolute errors in VaR and ES.

    Args:
        generated: Generated returns [B, T] or [T].
        real: Real returns [B, T] or [T].
        alpha: Risk level.

    Returns:
        Tuple of (|VaR_err|, |ES_err|) tensors.
    """
    gen_var, gen_es = empirical_var_es(generated, alpha)
    real_var, real_es = empirical_var_es(real, alpha)

    var_err = torch.abs(gen_var - real_var)
    es_err = torch.abs(gen_es - real_es)

    return var_err, es_err

"""Evaluation utilities for FE-GAN.

Computes VaR and ES errors between generated and real samples.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from fe_gan.losses import empirical_var_es


def compute_var_es_errors(
    generated_windows: Tensor,
    real_windows: Tensor,
    alpha: float = 0.05,
) -> dict[str, Tensor]:
    """Compute VaR and ES errors between generated and real windows.

    For each pair of (generated, real) windows, computes the absolute
    difference in empirical VaR and ES.

    Args:
        generated_windows: Generated windows [N, T].
        real_windows: Real windows [M, T].
        alpha: Risk level for VaR/ES.

    Returns:
        Dictionary with:
        - var_errors: [min(N, M)] tensor of |VaR_gen - VaR_real|
        - es_errors: [min(N, M)] tensor of |ES_gen - ES_real|
        - gen_var: [N] tensor of VaR from generated
        - gen_es: [N] tensor of ES from generated
        - real_var: [M] tensor of VaR from real
        - real_es: [M] tensor of ES from real
    """
    # Compute VaR and ES for each window
    gen_var, gen_es = empirical_var_es(generated_windows, alpha)
    real_var, real_es = empirical_var_es(real_windows, alpha)

    # Compute per-sample errors (compare pairwise)
    n_compare = min(len(generated_windows), len(real_windows))
    var_errors = torch.abs(gen_var[:n_compare] - real_var[:n_compare])
    es_errors = torch.abs(gen_es[:n_compare] - real_es[:n_compare])

    return {
        "var_errors": var_errors,
        "es_errors": es_errors,
        "gen_var": gen_var,
        "gen_es": gen_es,
        "real_var": real_var,
        "real_es": real_es,
    }


def compute_eval_metrics(
    var_errors: Tensor,
    es_errors: Tensor,
) -> dict[str, float]:
    """Compute summary statistics of VaR and ES errors.

    Args:
        var_errors: Tensor of absolute VaR errors.
        es_errors: Tensor of absolute ES errors.

    Returns:
        Dictionary with median, IQR, 90th percentile for each error type.
    """

    def _quantile(x: Tensor, q: float) -> float:
        sorted_x, _ = torch.sort(x)
        idx = int(q * (len(x) - 1))
        return float(sorted_x[idx])

    def _stats(x: Tensor, prefix: str) -> dict[str, float]:
        return {
            f"{prefix}_median": float(x.median()),
            f"{prefix}_mean": float(x.mean()),
            f"{prefix}_std": float(x.std()),
            f"{prefix}_iqr": _quantile(x, 0.75) - _quantile(x, 0.25),
            f"{prefix}_p90": _quantile(x, 0.90),
            f"{prefix}_min": float(x.min()),
            f"{prefix}_max": float(x.max()),
        }

    metrics = {}
    metrics.update(_stats(var_errors, "var_err"))
    metrics.update(_stats(es_errors, "es_err"))

    return metrics


def evaluate_generator(
    generator: torch.nn.Module,
    real_windows: Tensor,
    n_samples: int = 1000,
    noise_dim: int = 100,
    alpha: float = 0.05,
    device: torch.device | str = "cpu",
    is_fegan: bool = False,
) -> dict[str, Any]:
    """Evaluate a trained generator against real data.

    Generates synthetic windows and compares their VaR/ES to real windows.

    Args:
        generator: Trained generator model.
        real_windows: Real evaluation windows [M, T].
        n_samples: Number of synthetic windows to generate.
        noise_dim: Dimension of noise input.
        alpha: Risk level for VaR/ES.
        device: Device to use.
        is_fegan: Whether this is an FE-GAN (needs history input).

    Returns:
        Dictionary with evaluation metrics and raw errors.
    """
    generator.eval()
    device = torch.device(device)

    # Move real windows to device
    real_windows = real_windows.to(device)

    with torch.no_grad():
        # Generate synthetic windows
        z = torch.randn(n_samples, noise_dim, device=device)

        if is_fegan:
            # For FE-GAN, use real windows as history context
            # Sample random real windows to condition on
            hist_idx = torch.randint(len(real_windows), (n_samples,))
            history = real_windows[hist_idx]
            generated = generator(z, history)
        else:
            generated = generator(z)

    # Compute errors
    error_data = compute_var_es_errors(generated, real_windows, alpha)

    # Compute summary metrics
    metrics = compute_eval_metrics(
        error_data["var_errors"],
        error_data["es_errors"],
    )

    # Add sample counts
    metrics["n_generated"] = n_samples
    metrics["n_real"] = len(real_windows)
    metrics["alpha"] = alpha

    return metrics

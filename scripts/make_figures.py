"""Generate paper-style figures from sweep results.

Usage:
    python -m scripts.make_figures --results results/sweep_20260907_120000

Produces:
- fig3_var_es_boxplot.png: VaR and ES error boxplots (WGAN vs FE-GAN)
- fig10_tailgan_boxplot.png: FE-GAN-WGAN vs FE-GAN-TailGAN comparison
- convergence_var.png: Median VaR error vs epoch with IQR band
- wallclock_bar.png: Wall clock time to target error
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def load_aggregate(results_dir: Path) -> pd.DataFrame:
    """Load aggregate.csv into a DataFrame."""
    aggregate_path = results_dir / "aggregate.csv"
    if not aggregate_path.exists():
        raise FileNotFoundError(f"aggregate.csv not found in {results_dir}")

    return pd.read_csv(aggregate_path)


def load_metrics_files(results_dir: Path) -> dict[str, list[pd.DataFrame]]:
    """Load all metrics.csv files grouped by config.

    Returns:
        Dictionary mapping config_name to list of DataFrames.
    """
    per_run_dir = results_dir / "per_run"
    if not per_run_dir.exists():
        return {}

    metrics_by_config: dict[str, list[pd.DataFrame]] = {}

    for run_dir in per_run_dir.iterdir():
        if not run_dir.is_dir():
            continue

        metrics_path = run_dir / "metrics.csv"
        if not metrics_path.exists():
            continue

        # Parse config name
        parts = run_dir.name.rsplit("_s", 1)
        config_name = parts[0] if len(parts) == 2 else run_dir.name

        df = pd.read_csv(metrics_path)
        if config_name not in metrics_by_config:
            metrics_by_config[config_name] = []
        metrics_by_config[config_name].append(df)

    return metrics_by_config


def fig3_var_es_boxplot(df: pd.DataFrame, out_path: Path):
    """Create Figure 3 style boxplot: VaR and ES errors, WGAN vs FE-GAN.

    Args:
        df: Aggregate DataFrame.
        out_path: Output path for PNG.
    """
    # Filter to relevant configs
    wgan_df = df[df["config"] == "baseline_wgan"].copy()
    fegan_df = df[df["config"] == "fegan_hist_wgan"].copy()

    if wgan_df.empty or fegan_df.empty:
        print("  Warning: Missing configs for Figure 3")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # VaR error boxplot
    ax = axes[0]
    data = [
        wgan_df["var_err_median"].dropna().values,
        fegan_df["var_err_median"].dropna().values,
    ]
    bp = ax.boxplot(data, labels=["WGAN", "FE-GAN"], patch_artist=True)
    bp["boxes"][0].set_facecolor("lightcoral")
    bp["boxes"][1].set_facecolor("lightgreen")
    ax.set_ylabel("|VaR Error|")
    ax.set_title("VaR Error Distribution (100 seeds)")
    ax.grid(True, alpha=0.3)

    # ES error boxplot
    ax = axes[1]
    data = [
        wgan_df["es_err_median"].dropna().values,
        fegan_df["es_err_median"].dropna().values,
    ]
    bp = ax.boxplot(data, labels=["WGAN", "FE-GAN"], patch_artist=True)
    bp["boxes"][0].set_facecolor("lightcoral")
    bp["boxes"][1].set_facecolor("lightgreen")
    ax.set_ylabel("|ES Error|")
    ax.set_title("ES Error Distribution (100 seeds)")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


def fig10_tailgan_boxplot(df: pd.DataFrame, out_path: Path):
    """Create Figure 10 style boxplot: FE-GAN-WGAN vs FE-GAN-TailGAN.

    Args:
        df: Aggregate DataFrame.
        out_path: Output path for PNG.
    """
    fegan_wgan = df[df["config"] == "fegan_hist_wgan"].copy()
    fegan_tailgan = df[df["config"] == "fegan_hist_tailgan"].copy()

    if fegan_wgan.empty or fegan_tailgan.empty:
        print("  Warning: Missing configs for Figure 10")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # VaR error
    ax = axes[0]
    data = [
        fegan_wgan["var_err_median"].dropna().values,
        fegan_tailgan["var_err_median"].dropna().values,
    ]
    bp = ax.boxplot(data, labels=["FE-GAN + WGAN", "FE-GAN + Tail-GAN"], patch_artist=True)
    bp["boxes"][0].set_facecolor("lightblue")
    bp["boxes"][1].set_facecolor("lightyellow")
    ax.set_ylabel("|VaR Error|")
    ax.set_title("VaR Error: WGAN vs Tail-GAN Loss")
    ax.grid(True, alpha=0.3)

    # ES error
    ax = axes[1]
    data = [
        fegan_wgan["es_err_median"].dropna().values,
        fegan_tailgan["es_err_median"].dropna().values,
    ]
    bp = ax.boxplot(data, labels=["FE-GAN + WGAN", "FE-GAN + Tail-GAN"], patch_artist=True)
    bp["boxes"][0].set_facecolor("lightblue")
    bp["boxes"][1].set_facecolor("lightyellow")
    ax.set_ylabel("|ES Error|")
    ax.set_title("ES Error: WGAN vs Tail-GAN Loss")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


def convergence_plot(
    metrics_by_config: dict[str, list[pd.DataFrame]],
    out_path: Path,
):
    """Create convergence plot: median VaR error vs epoch with IQR band.

    Args:
        metrics_by_config: Dictionary of config -> list of metrics DataFrames.
        out_path: Output path for PNG.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = {
        "baseline_wgan": "red",
        "fegan_hist_wgan": "green",
        "fegan_hist_tailgan": "blue",
    }
    labels = {
        "baseline_wgan": "Baseline WGAN",
        "fegan_hist_wgan": "FE-GAN (WGAN)",
        "fegan_hist_tailgan": "FE-GAN (Tail-GAN)",
    }

    for config_name, dfs in metrics_by_config.items():
        if not dfs:
            continue

        # Stack all metrics
        # Find common epochs (with var_err_median values)
        valid_dfs = [df[df["var_err_median"].notna()] for df in dfs]
        if not valid_dfs:
            continue

        # Combine by epoch
        combined = pd.concat(valid_dfs, ignore_index=True)
        grouped = combined.groupby("epoch")["var_err_median"]

        epochs = sorted(grouped.groups.keys())
        medians = [grouped.get_group(e).median() for e in epochs]
        q25 = [grouped.get_group(e).quantile(0.25) for e in epochs]
        q75 = [grouped.get_group(e).quantile(0.75) for e in epochs]

        color = colors.get(config_name, "gray")
        label = labels.get(config_name, config_name)

        ax.plot(epochs, medians, color=color, label=label, linewidth=2)
        ax.fill_between(epochs, q25, q75, color=color, alpha=0.2)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("|VaR Error| (median)")
    ax.set_title("Convergence: VaR Error vs Training Epoch")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


def wallclock_bar(df: pd.DataFrame, out_path: Path):
    """Create wall clock comparison bar chart.

    Args:
        df: Aggregate DataFrame.
        out_path: Output path for PNG.
    """
    # Group by config and compute mean wall time
    summary = df.groupby("config")["wall_seconds"].agg(["mean", "std"]).reset_index()

    fig, ax = plt.subplots(figsize=(8, 5))

    x = range(len(summary))
    bars = ax.bar(
        x,
        summary["mean"],
        yerr=summary["std"],
        capsize=5,
        color=["lightcoral", "lightgreen", "lightblue"][: len(summary)],
        edgecolor="black",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(summary["config"], rotation=15, ha="right")
    ax.set_ylabel("Wall Clock Time (seconds)")
    ax.set_title("Training Time per Configuration")
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate figures from sweep results")
    parser.add_argument(
        "--results",
        type=Path,
        required=True,
        help="Path to sweep results directory",
    )
    args = parser.parse_args()

    results_dir = args.results
    figures_dir = results_dir / "figures"
    figures_dir.mkdir(exist_ok=True)

    print(f"Generating figures from: {results_dir}")
    print(f"Output: {figures_dir}")
    print()

    # Load data
    try:
        df = load_aggregate(results_dir)
        print(f"Loaded {len(df)} rows from aggregate.csv")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    metrics_by_config = load_metrics_files(results_dir)
    print(f"Loaded metrics for {len(metrics_by_config)} configurations")
    print()

    # Generate figures
    print("Generating figures:")

    fig3_var_es_boxplot(df, figures_dir / "fig3_var_es_boxplot.png")
    fig10_tailgan_boxplot(df, figures_dir / "fig10_tailgan_boxplot.png")

    if metrics_by_config:
        convergence_plot(metrics_by_config, figures_dir / "convergence_var.png")

    wallclock_bar(df, figures_dir / "wallclock_bar.png")

    print()
    print("Done!")


if __name__ == "__main__":
    main()

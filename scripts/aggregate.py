"""Aggregate per-run eval.json files into a single aggregate.csv.

Usage:
    python -m scripts.aggregate --results results/sweep_20260907_120000

This script walks through all per_run/*/eval.json files and combines them
into a single aggregate.csv with summary statistics per configuration.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Aggregate sweep results")
    parser.add_argument(
        "--results",
        type=Path,
        required=True,
        help="Path to sweep results directory",
    )
    args = parser.parse_args()

    results_dir = args.results
    per_run_dir = results_dir / "per_run"

    if not per_run_dir.exists():
        print(f"Error: per_run directory not found: {per_run_dir}")
        return

    # Collect all eval.json files
    rows = []

    for run_dir in sorted(per_run_dir.iterdir()):
        if not run_dir.is_dir():
            continue

        eval_path = run_dir / "eval.json"
        config_path = run_dir / "config.json"

        if not eval_path.exists():
            print(f"Warning: No eval.json in {run_dir}")
            continue

        with open(eval_path) as f:
            eval_data = json.load(f)

        # Parse config name and seed from directory name (e.g., "baseline_wgan_s0")
        parts = run_dir.name.rsplit("_s", 1)
        if len(parts) == 2:
            config_name = parts[0]
            seed = int(parts[1])
        else:
            config_name = run_dir.name
            seed = 0

        # Get config details if available
        epochs = eval_data.get("epochs", 0)

        row = {
            "config": config_name,
            "seed": seed,
            "epochs": epochs,
            "var_err_median": eval_data.get("var_err_median"),
            "var_err_mean": eval_data.get("var_err_mean"),
            "var_err_std": eval_data.get("var_err_std"),
            "var_err_iqr": eval_data.get("var_err_iqr"),
            "var_err_p90": eval_data.get("var_err_p90"),
            "es_err_median": eval_data.get("es_err_median"),
            "es_err_mean": eval_data.get("es_err_mean"),
            "es_err_std": eval_data.get("es_err_std"),
            "es_err_iqr": eval_data.get("es_err_iqr"),
            "es_err_p90": eval_data.get("es_err_p90"),
            "wall_seconds": eval_data.get("total_wall_seconds"),
        }
        rows.append(row)

    if not rows:
        print("No results found!")
        return

    # Sort by config, then seed
    rows.sort(key=lambda x: (x["config"], x["seed"]))

    # Write aggregate CSV
    aggregate_path = results_dir / "aggregate.csv"
    fieldnames = [
        "config",
        "seed",
        "epochs",
        "var_err_median",
        "var_err_mean",
        "var_err_std",
        "var_err_iqr",
        "var_err_p90",
        "es_err_median",
        "es_err_mean",
        "es_err_std",
        "es_err_iqr",
        "es_err_p90",
        "wall_seconds",
    ]

    with open(aggregate_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Aggregated {len(rows)} runs to: {aggregate_path}")

    # Print summary statistics per config
    print("\nSummary by configuration:")
    print("-" * 60)

    configs = sorted(set(r["config"] for r in rows))
    for config in configs:
        config_rows = [r for r in rows if r["config"] == config]
        n = len(config_rows)

        var_medians = [r["var_err_median"] for r in config_rows if r["var_err_median"]]
        es_medians = [r["es_err_median"] for r in config_rows if r["es_err_median"]]

        if var_medians:
            var_median_of_medians = sorted(var_medians)[len(var_medians) // 2]
            var_mean_of_medians = sum(var_medians) / len(var_medians)
        else:
            var_median_of_medians = None
            var_mean_of_medians = None

        if es_medians:
            es_median_of_medians = sorted(es_medians)[len(es_medians) // 2]
            es_mean_of_medians = sum(es_medians) / len(es_medians)
        else:
            es_median_of_medians = None
            es_mean_of_medians = None

        print(f"{config}:")
        print(f"  Runs: {n}")
        if var_median_of_medians:
            print(f"  VaR error (median of medians): {var_median_of_medians:.4f}")
        if es_median_of_medians:
            print(f"  ES error (median of medians): {es_median_of_medians:.4f}")
        print()


if __name__ == "__main__":
    main()

"""Train a single FE-GAN model.

Usage:
    python -m scripts.train --config configs/fegan_hist_wgan.yaml --seed 0
    python -m scripts.train --config configs/baseline_wgan.yaml --seed 0 --epochs 20

See configs/ for available configurations.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf

from fe_gan.data import load_vix_dataset
from fe_gan.trainer import TrainerConfig, train_model
from fe_gan.utils import get_device


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train FE-GAN model")

    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to config YAML file",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override epochs from config",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Override batch size from config",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output directory (default: results/<config_name>_s<seed>)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (default: auto-detect)",
    )
    parser.add_argument(
        "--save_ckpt",
        action="store_true",
        help="Save final generator checkpoint",
    )
    parser.add_argument(
        "--no_progress",
        action="store_true",
        help="Disable progress bar",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Load config
    if not args.config.exists():
        print(f"Error: Config file not found: {args.config}")
        sys.exit(1)

    cfg = OmegaConf.load(args.config)

    # Create TrainerConfig from YAML
    config = TrainerConfig(
        model_type=cfg.get("model_type", "wgan"),
        loss_type=cfg.get("loss_type", "wgan"),
        noise_dim=cfg.get("noise_dim", 100),
        gen_depth=cfg.get("gen_depth", 10),
        gen_width=cfg.get("gen_width", 1000),
        critic_depth=cfg.get("critic_depth", 5),
        critic_width=cfg.get("critic_width", 100),
        hist_dim=cfg.get("hist_dim", 250),
        preproc_hidden=cfg.get("preproc_hidden", 512),
        preproc_feat_dim=cfg.get("preproc_feat_dim", 250),
        epochs=args.epochs or cfg.get("epochs", 100),
        batch_size=args.batch_size or cfg.get("batch_size", 100),
        n_critic=cfg.get("n_critic", 5),
        clip_value=cfg.get("clip_value", 0.01),
        lr=cfg.get("lr", 5e-5),
        alpha=cfg.get("alpha", 0.05),
        tailgan_wgan_weight=cfg.get("tailgan_wgan_weight", 0.5),
        eval_every=cfg.get("eval_every", 10),
        eval_samples=cfg.get("eval_samples", 1000),
        seed=args.seed,
        save_checkpoint=args.save_ckpt,
    )

    # Set output directory
    if args.out:
        config.out_dir = str(args.out)
    else:
        config_name = args.config.stem
        config.out_dir = f"results/{config_name}_s{args.seed}"

    # Determine device
    if args.device:
        device = torch.device(args.device)
    else:
        device = get_device()

    print(f"Configuration: {args.config.name}")
    print(f"  Model type: {config.model_type}")
    print(f"  Loss type: {config.loss_type}")
    print(f"  Epochs: {config.epochs}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Seed: {config.seed}")
    print(f"  Device: {device}")
    print(f"  Output: {config.out_dir}")

    # Load data
    print("\nLoading data...")
    train_data, eval_data = load_vix_dataset(
        window_size=config.hist_dim,
        n_eval=100,
        device=device,
    )
    print(f"  Train windows: {len(train_data)}")
    print(f"  Eval windows: {len(eval_data)}")

    # Train
    print("\nTraining...")
    final_eval = train_model(
        train_data=train_data,
        eval_data=eval_data,
        config=config,
        device=device,
        progress=not args.no_progress,
    )

    # Print results
    print("\nFinal evaluation:")
    print(f"  VaR error median: {final_eval['var_err_median']:.4f}")
    print(f"  ES error median: {final_eval['es_err_median']:.4f}")
    print(f"  Total wall time: {final_eval['total_wall_seconds']:.1f}s")
    print(f"\nResults saved to: {config.out_dir}")


if __name__ == "__main__":
    main()

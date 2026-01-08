# FE-GAN for VaR Estimation

This repository implements and verifies the claims from Chen (2024) "Risk Management with Feature-Enriched Generative Adversarial Networks (FE-GAN)".

## Overview

FE-GAN extends Wasserstein GANs (WGANs) with historical context to improve Value-at-Risk (VaR) and Expected Shortfall (ES) estimation for financial time series. This implementation verifies three main claims:

1. **FE-GAN reduces VaR/ES error** compared to vanilla WGAN
2. **FE-GAN converges faster** (fewer epochs to target accuracy)
3. **Tail-GAN loss improves ES** when combined with FE-GAN

## Quick Start

```bash
# Setup
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

# Run tests
make test

# Run smoke tests (20 epochs each, ~5 min on GPU)
make smoke

# Run a single training
python -m scripts.train --config configs/fegan_hist_wgan.yaml --seed 0

# Run full sweep on A100 (300 runs, ~40-70 min)
bash scripts/a100_pipeline.sh
```

## Project Structure

```
fe-gan-VaR/
├── configs/                # Training configurations
│   ├── baseline_wgan.yaml
│   ├── fegan_hist_wgan.yaml
│   ├── fegan_hist_tailgan.yaml
│   └── sweep_plan_a.yaml
├── data/
│   └── vix_2014_2019.csv   # VIX data (committed for reproducibility)
├── notebooks/
│   └── legacy_GAN.ipynb    # Original notebook (reference only)
├── papers/                 # Reference papers
├── scripts/
│   ├── train.py            # Single training run
│   ├── sweep.py            # Multi-seed sweep
│   ├── aggregate.py        # Aggregate results
│   ├── make_figures.py     # Generate paper figures
│   └── a100_pipeline.sh    # Full A100 pipeline
├── src/fe_gan/
│   ├── data.py             # Data loading and windowing
│   ├── models.py           # Generator, Critic, FEGenerator
│   ├── losses.py           # WGAN and Tail-GAN losses
│   ├── trainer.py          # Training loop
│   ├── eval.py             # VaR/ES evaluation
│   └── utils.py            # Utilities
├── tests/                  # Unit tests
├── results/                # Training outputs (gitignored)
├── Dockerfile.a100         # A100 deployment image
├── Makefile                # Common commands
├── pyproject.toml          # Dependencies
└── REPORT.md               # Verification report
```

## Architecture

Following Chen (2024) Section 2.1:

- **Generator**: 10 linear layers, width 1000, BatchNorm + ReLU
- **Critic**: 5 linear layers, width 100, ReLU (no BatchNorm)
- **FE-GAN Preprocessor**: 2 layers (250 → 512 → 250)
- **Training**: RMSprop lr=5e-5, batch=100, n_critic=5, clip=0.01

## Data

VIX index data from CBOE (2014-2019):
- 1509 trading days
- Log returns computed as `ln(Close_t / Close_{t-1})`
- Rolling windows of 250 days
- Train/eval split: last 100 windows for evaluation

## Configurations

| Config | Model | Loss | Epochs | Description |
|--------|-------|------|--------|-------------|
| `baseline_wgan` | WGAN | WGAN | 300 | Vanilla WGAN baseline |
| `fegan_hist_wgan` | FE-GAN | WGAN | 100 | FE-GAN with historical context |
| `fegan_hist_tailgan` | FE-GAN | Tail-GAN | 100 | FE-GAN with Fissler-Ziegel loss |

## A100 Pipeline

The full verification runs 300 experiments (3 configs × 100 seeds):

```bash
# On A100 VM
bash scripts/a100_pipeline.sh --workers 10

# Expected runtime: 40-70 minutes
# Output: <5 MB (CSV, JSON, PNG only, no checkpoints)
```

## Results

See [REPORT.md](REPORT.md) for the full verification report with figures.

## References

1. Chen (2024). Risk Management with Feature-Enriched Generative Adversarial Networks (FE-GAN).
2. Cont et al. (2022). Tail-GAN: Learning to Simulate Tail Risk Scenarios.
3. Arjovsky et al. (2017). Wasserstein GAN.

## License

MIT

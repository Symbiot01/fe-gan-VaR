# FE-GAN Verification Report

This report documents the verification of claims from Chen (2024) "Risk Management with Feature-Enriched Generative Adversarial Networks (FE-GAN)".

## Setup

### Hardware
- **A100 GPU**: NVIDIA A100 40GB (for production sweep)
- **RTX 3060**: Local development and smoke tests

### Data
- **Source**: CBOE VIX History (2014-01-02 to 2019-12-31)
- **Rows**: 1509 trading days
- **Windows**: 1160 training windows, 100 evaluation windows (size 250)

### Configurations

| Config | Model | Loss | Epochs | Seeds |
|--------|-------|------|--------|-------|
| baseline_wgan | WGAN | WGAN | 300 | 100 |
| fegan_hist_wgan | FE-GAN | WGAN | 100 | 100 |
| fegan_hist_tailgan | FE-GAN | Tail-GAN | 100 | 100 |

### Architecture (per paper)
- **Generator**: 10 layers, width 1000, BatchNorm + ReLU
- **Critic**: 5 layers, width 100, ReLU (no BatchNorm)
- **FE-GAN Preprocessor**: 2 layers (250 → 512 → 250)
- **Training**: RMSprop lr=5e-5, batch=100, n_critic=5, clip=0.01

## Results

### Claim 1: FE-GAN reduces VaR and ES error vs baseline WGAN

**Paper claim**: Historical FE-GAN significantly reduces |VaR_5%| and |ES_5%| error compared to vanilla WGAN.

| Metric | WGAN (median ± IQR) | FE-GAN (median ± IQR) | Improvement |
|--------|---------------------|------------------------|-------------|
| \|VaR error\| | TODO ± TODO | TODO ± TODO | TODO% |
| \|ES error\| | TODO ± TODO | TODO ± TODO | TODO% |

![VaR and ES Error Boxplot](results/sweep_TIMESTAMP/figures/fig3_var_es_boxplot.png)

**Verdict**: TODO

### Claim 2: FE-GAN converges faster than WGAN

**Paper claim**: FE-GAN reaches target error level in fewer epochs and less wall-clock time.

| Metric | WGAN | FE-GAN | Speedup |
|--------|------|--------|---------|
| Epochs to reach VaR < X | TODO | TODO | TODOx |
| Wall-clock time (s) | TODO | TODO | TODOx |

![Convergence Plot](results/sweep_TIMESTAMP/figures/convergence_var.png)

**Verdict**: TODO

### Claim 3: Tail-GAN loss improves ES error

**Paper claim**: Under FE-GAN, Tail-GAN loss (Fissler-Ziegel score) further improves ES error compared to WGAN loss.

| Metric | FE-GAN + WGAN | FE-GAN + Tail-GAN | Improvement |
|--------|---------------|-------------------|-------------|
| \|VaR error\| | TODO | TODO | TODO% |
| \|ES error\| | TODO | TODO | TODO% |

![Tail-GAN Comparison](results/sweep_TIMESTAMP/figures/fig10_tailgan_boxplot.png)

**Verdict**: TODO

## Summary

| Claim | Verified? | Notes |
|-------|-----------|-------|
| 1. FE-GAN reduces VaR/ES error | TODO | |
| 2. FE-GAN converges faster | TODO | |
| 3. Tail-GAN improves ES | TODO | |

## Reproducibility

### Commands
```bash
# Run full sweep on A100
bash scripts/a100_pipeline.sh --workers 10

# Or step by step:
python -m scripts.sweep --sweep configs/sweep_plan_a.yaml --workers 10
python -m scripts.aggregate --results results/sweep_TIMESTAMP
python -m scripts.make_figures --results results/sweep_TIMESTAMP
```

### Environment
- Python: 3.10+
- PyTorch: 2.2+
- See `pyproject.toml` for full dependencies

### Data Integrity
- VIX data SHA256: `2039576bcfcb64fbd360bee9a87b5b08a8df35c73d0ef1cc34faef602f68fa4a`
- Committed to `data/vix_2014_2019.csv` for exact reproducibility

## References

1. Chen (2024). Risk Management with Feature-Enriched Generative Adversarial Networks (FE-GAN).
2. Cont et al. (2022). Tail-GAN: Learning to Simulate Tail Risk Scenarios.
3. Arjovsky et al. (2017). Wasserstein GAN.
4. Fissler & Ziegel (2016). Higher-order elicitability and Osband's principle.

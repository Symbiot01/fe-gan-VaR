# FE-GAN Verification Report

Verification of claims from Chen (2024), *Risk Management with Feature-Enriched Generative Adversarial Networks (FE-GAN)*, on VIX log returns (historical context only).

**Sweep ID:** `results/sweep_20260908_112900`  
**Git SHA:** `5c7dcc8`  
**Completed:** 2026-09-08 (300/300 runs)

## Setup

### Hardware
- **Production sweep:** GCP `g2-standard-48`, **4× NVIDIA L4** (24 GB each), 40 workers (10 per GPU), multi-GPU sharding
- **Earlier reference:** 1× NVIDIA A100 40GB
- **Local:** RTX 3060 for development / smoke tests

### Data
- **Source:** CBOE VIX History (`data/vix_2014_2019.csv`)
- **Slice:** 2014-01-02 to 2019-12-31
- **Windows:** 1160 train / 100 eval, length 250
- **SHA256:** `2039576bcfcb64fbd360bee9a87b5b08a8df35c73d0ef1cc34faef602f68fa4a`

### Configurations

| Config | Model | Loss | Epochs | Seeds |
|--------|-------|------|--------|-------|
| baseline_wgan | WGAN | WGAN | 300 | 100 |
| fegan_hist_wgan | FE-GAN (historical) | WGAN | 300 | 100 |
| fegan_hist_tailgan | FE-GAN (historical) | Tail-GAN (FZ0) | 300 | 100 |

### Architecture (Chen 2024 §2.1)
- **Generator:** 10 layers, width 1000, BatchNorm + ReLU
- **Critic:** 5 layers, width 100, ReLU (no BatchNorm)
- **FE-GAN preprocessor:** 2 layers (250 → 512 → 250)
- **Optim:** RMSprop `lr=5e-5`, batch 100, `n_critic=5`, clip `0.01`, bf16 AMP

### Wall-clock
- Parallel sweep wall time (40 workers): **~464 s (~7.7 min)**
- Median seconds / run: baseline **24.2 s**, FE-GAN+WGAN **32.5 s**, FE-GAN+Tail-GAN **113.5 s**

## Results

Metrics below are the **median across 100 seeds** of each run’s **median |error|** over eval windows (same protocol as `scripts/aggregate.py`).

### Claim 1: FE-GAN reduces VaR/ES error vs baseline WGAN

**Paper claim:** Historical FE-GAN significantly reduces 5% VaR and ES error vs vanilla WGAN.

| Metric | WGAN median (IQR) | FE-GAN+WGAN median (IQR) | Δ vs baseline |
|--------|-------------------|---------------------------|---------------|
| \|VaR error\| | **0.0208** (0.0052) | **0.0218** (0.0053) | **−4.6%** (worse) |
| \|ES error\| | **0.0227** (0.0043) | **0.0234** (0.0028) | **−2.9%** (worse) |

![VaR and ES Error Boxplot](results/sweep_20260908_112900/figures/fig3_var_es_boxplot.png)

**Verdict: Not verified.** Under this implementation and protocol, FE-GAN+WGAN is statistically tied with / slightly worse than baseline WGAN. It does **not** reproduce the paper’s large error reduction on historical VIX.

### Claim 2: FE-GAN converges faster than WGAN

**Paper claim:** FE-GAN reaches target accuracy in fewer epochs / less wall-clock time.

| Metric | WGAN | FE-GAN+WGAN | Result |
|--------|------|-------------|--------|
| Median epochs to \|VaR\| < 0.05 | 190 | 190 | same |
| Median epochs to \|VaR\| < 0.03 | 200 | 200 | same |
| Median epochs to \|VaR\| < 0.025 | 210 | 210 | same |
| Median wall-clock / run (300 ep) | 24.2 s | 32.5 s | FE-GAN **slower** |

Median \|VaR error\| by epoch (across seeds):

| Epoch | WGAN | FE-GAN+WGAN |
|-------|------|-------------|
| 50 | 1.64 | 1.64 |
| 100 | 1.23 | 1.26 |
| 150 | 0.68 | 0.71 |
| 200 | 0.036 | 0.040 |
| 300 | 0.020 | 0.022 |

![Convergence Plot](results/sweep_20260908_112900/figures/convergence_var.png)

**Verdict: Not verified.** Convergence trajectories match closely; FE-GAN is not faster in epochs and is slower per run in wall-clock (extra history path).

### Claim 3: Tail-GAN loss improves ES under FE-GAN

**Paper claim:** With FE-GAN, Tail-GAN (Fissler–Ziegel) improves ES vs WGAN loss.

| Metric | FE-GAN + WGAN | FE-GAN + Tail-GAN | Δ |
|--------|---------------|-------------------|---|
| \|VaR error\| | **0.0218** | **1.0127** | much worse |
| \|ES error\| | **0.0234** | **2.1793** | much worse |

![Tail-GAN Comparison](results/sweep_20260908_112900/figures/fig10_tailgan_boxplot.png)

**Verdict: Not verified.** Tail-GAN under this FZ0 + WGAN hybrid training is unstable / poorly calibrated on this sweep. Anchoring FZ to real VaR/ES in a prior experiment also failed (VaR/ES errors ~10). Treat Tail-GAN here as an incomplete reproduction of Cont et al., not a confirmation of Chen §3.

## Summary

| Claim | Verified? | Notes |
|-------|-----------|-------|
| 1. FE-GAN reduces VaR/ES error | **No** | ~tied with baseline; tiny regression |
| 2. FE-GAN converges faster | **No** | Same epoch curve; higher wall time / run |
| 3. Tail-GAN improves ES | **No** | Large degradation vs FE-GAN+WGAN |

**What did work well:** the baseline WGAN and FE-GAN+WGAN stacks both reach low absolute errors (~0.02 VaR/ES) and the multi-GPU pipeline is solid (300/300, ~8 min on 4× L4).

**Likely gaps vs the paper (for follow-up):** evaluation matching (paper’s “100 models” distributions), Tail-GAN scoring-function details / gradient path, history preprocessing, and possibly data cleaning beyond the committed CBOE slice.

## Reproducibility

```bash
# On multi-GPU VM (auto workers = 3×GPU count; override for max throughput)
git pull
pip install -e ".[dev]"
bash scripts/a100_pipeline.sh --workers 40   # used for this report on 4× L4

# Aggregate / figures (already done by pipeline)
python -m scripts.aggregate --results results/sweep_20260908_112900
python -m scripts.make_figures --results results/sweep_20260908_112900
```

### Environment (sweep host)
- Hostname: `instance-20260908-110107`
- PyTorch: `2.6.0+cu124`
- CUDA: 12.4
- GPU: 4× NVIDIA L4

### Artifacts
- `results/sweep_20260908_112900/aggregate.csv`
- `results/sweep_20260908_112900/index.csv`
- `results/sweep_20260908_112900/sweep_meta.json`
- `results/sweep_20260908_112900/figures/*.png`
- `results/sweep_20260908_112900/per_run/*/eval.json` (300 runs)

## References

1. Chen (2024). Risk Management with Feature-Enriched Generative Adversarial Networks (FE-GAN). arXiv:2411.15519.
2. Cont et al. (2022). Tail-GAN: Learning to Simulate Tail Risk Scenarios. arXiv:2203.01664.
3. Arjovsky et al. (2017). Wasserstein GAN. arXiv:1701.07875.
4. Fissler & Ziegel (2016). Higher-order elicitability and Osband’s principle.

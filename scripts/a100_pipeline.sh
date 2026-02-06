#!/bin/bash
# A100 Pipeline: sweep + aggregate + figures in one command
#
# Usage:
#   bash scripts/a100_pipeline.sh              # Full sweep (3 configs x 100 seeds)
#   bash scripts/a100_pipeline.sh --dry-run    # Smoke test (2 seeds x 1 config x 5 epochs)
#   bash scripts/a100_pipeline.sh --workers 20 # Use 20 parallel workers
#
# This script is designed to run on an A100 VM. It:
# 1. Verifies data integrity (SHA256 check)
# 2. Runs unit tests
# 3. Runs smoke tests (unless --skip-smoke)
# 4. Runs the full sweep
# 5. Aggregates results
# 6. Generates figures
#
# Only small artifacts (CSV/JSON/PNG, <5 MB total) need to be rsync'd back.

set -e  # Exit on error

# Defaults
WORKERS=${WORKERS:-10}
DRY_RUN=false
SKIP_SMOKE=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --skip-smoke)
            SKIP_SMOKE=true
            shift
            ;;
        --workers)
            WORKERS="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "=========================================="
echo "FE-GAN A100 Pipeline"
echo "=========================================="
echo "Workers: $WORKERS"
echo "Dry run: $DRY_RUN"
echo "Skip smoke: $SKIP_SMOKE"
echo ""

# Step 1: Verify data integrity
echo "[1/6] Verifying data integrity..."
python -c "
from fe_gan.data import verify_data_integrity, DEFAULT_DATA_PATH
if not verify_data_integrity(DEFAULT_DATA_PATH):
    raise RuntimeError('Data integrity check failed! SHA256 mismatch.')
print('  Data integrity verified.')
"

# Step 2: Run unit tests
echo ""
echo "[2/6] Running unit tests..."
python -m pytest tests/ -v --tb=short -q
echo "  Unit tests passed."

# Step 3: Run smoke tests (unless skipped)
if [ "$SKIP_SMOKE" = false ]; then
    echo ""
    echo "[3/6] Running smoke tests..."
    mkdir -p results/smoke
    
    python -m scripts.train \
        --config configs/baseline_wgan.yaml \
        --seed 0 --epochs 20 --batch_size 32 \
        --out results/smoke/baseline_wgan_s0 \
        --no_progress

    python -m scripts.train \
        --config configs/fegan_hist_wgan.yaml \
        --seed 0 --epochs 20 --batch_size 32 \
        --out results/smoke/fegan_hist_wgan_s0 \
        --no_progress

    python -m scripts.train \
        --config configs/fegan_hist_tailgan.yaml \
        --seed 0 --epochs 20 --batch_size 32 \
        --out results/smoke/fegan_hist_tailgan_s0 \
        --no_progress

    echo "  Smoke tests passed."
else
    echo ""
    echo "[3/6] Skipping smoke tests..."
fi

# Step 4: Run sweep
echo ""
echo "[4/6] Running sweep..."

if [ "$DRY_RUN" = true ]; then
    # Dry run: 2 seeds x 1 config x 5 epochs
    python -m scripts.sweep \
        --config configs/fegan_hist_wgan.yaml \
        --seeds 0,1 \
        --epochs 5 \
        --workers 2 \
        --cudnn_benchmark
else
    # Full sweep: 3 configs x 100 seeds
    python -m scripts.sweep \
        --sweep configs/sweep_plan_a.yaml \
        --workers "$WORKERS" \
        --cudnn_benchmark
fi

# Get the latest sweep directory
SWEEP_DIR=$(ls -td results/sweep_* 2>/dev/null | head -1)

if [ -z "$SWEEP_DIR" ]; then
    echo "Error: No sweep directory found"
    exit 1
fi

echo "  Sweep completed: $SWEEP_DIR"

# Step 5: Aggregate results
echo ""
echo "[5/6] Aggregating results..."
python -m scripts.aggregate --results "$SWEEP_DIR"
echo "  Aggregation completed."

# Step 6: Generate figures
echo ""
echo "[6/6] Generating figures..."
python -m scripts.make_figures --results "$SWEEP_DIR"
echo "  Figures generated."

# Summary
echo ""
echo "=========================================="
echo "Pipeline completed!"
echo "=========================================="
echo "Results directory: $SWEEP_DIR"
echo ""
echo "Output files:"
ls -lh "$SWEEP_DIR"/*.csv "$SWEEP_DIR"/*.json 2>/dev/null || true
ls -lh "$SWEEP_DIR"/figures/*.png 2>/dev/null || true
echo ""
echo "Total size:"
du -sh "$SWEEP_DIR"

echo ""
echo "To rsync results back:"
echo "  rsync -avz $SWEEP_DIR/{*.csv,*.json,figures/} user@local:/path/to/results/"

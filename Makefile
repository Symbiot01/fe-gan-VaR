.PHONY: setup test smoke sweep-local a100 figures report clean lint help

# Default Python
PYTHON ?= python3

# Default workers for sweep
WORKERS ?= 10

# Results directory timestamp
TS := $(shell date +%Y%m%d_%H%M%S)

help:
	@echo "FE-GAN VaR Verification Pipeline"
	@echo ""
	@echo "Local development:"
	@echo "  make setup       - Install package in dev mode"
	@echo "  make test        - Run unit tests"
	@echo "  make lint        - Run linter"
	@echo "  make smoke       - Run smoke tests (3 configs x 20 epochs)"
	@echo "  make sweep-local - Run small local sweep (2 seeds x 1 config)"
	@echo ""
	@echo "A100 pipeline:"
	@echo "  make a100        - Run full A100 pipeline (sweep + aggregate + figures)"
	@echo "  make sweep-test  - Dry run sweep (2 seeds x 1 config x 5 epochs)"
	@echo ""
	@echo "Post-processing:"
	@echo "  make figures     - Generate figures from latest results"
	@echo "  make report      - Open REPORT.md for editing"
	@echo ""
	@echo "Utilities:"
	@echo "  make fetch-data  - Regenerate data/vix_2014_2019.csv from CBOE"
	@echo "  make clean       - Remove results and caches"

setup:
	$(PYTHON) -m pip install -e ".[dev]"

test:
	$(PYTHON) -m pytest tests/ -v --tb=short

lint:
	$(PYTHON) -m ruff check src/ scripts/ tests/
	$(PYTHON) -m ruff format --check src/ scripts/ tests/

smoke:
	@echo "Running smoke tests (3 configs x 20 epochs)..."
	$(PYTHON) -m scripts.train --config configs/baseline_wgan.yaml --seed 0 --epochs 20 --batch_size 32 --out results/smoke/baseline_wgan_s0
	$(PYTHON) -m scripts.train --config configs/fegan_hist_wgan.yaml --seed 0 --epochs 20 --batch_size 32 --out results/smoke/fegan_hist_wgan_s0
	$(PYTHON) -m scripts.train --config configs/fegan_hist_tailgan.yaml --seed 0 --epochs 20 --batch_size 32 --out results/smoke/fegan_hist_tailgan_s0
	@echo "Smoke tests completed. Check results/smoke/"

sweep-local:
	@echo "Running local sweep (2 seeds x 1 config x 20 epochs)..."
	$(PYTHON) -m scripts.sweep --config configs/fegan_hist_wgan.yaml --seeds 0,1 --epochs 20 --workers 2 --out results/sweep_local_$(TS)

sweep-test:
	@echo "Running sweep dry-run (2 seeds x 1 config x 5 epochs)..."
	$(PYTHON) -m scripts.sweep --config configs/fegan_hist_wgan.yaml --seeds 0,1 --epochs 5 --workers 2 --out results/sweep_test_$(TS)

a100:
	@echo "Running A100 pipeline..."
	bash scripts/a100_pipeline.sh --workers $(WORKERS)

figures:
	@echo "Generating figures from latest results..."
	$(PYTHON) -m scripts.make_figures --results $$(ls -td results/sweep_* | head -1)

report:
	@echo "Opening REPORT.md..."
	@if [ ! -f REPORT.md ]; then echo "# FE-GAN Verification Report\n\nTODO: Fill in after sweep completes." > REPORT.md; fi
	@echo "REPORT.md ready for editing"

fetch-data:
	$(PYTHON) -m scripts.fetch_data --force

clean:
	rm -rf results/
	rm -rf __pycache__/
	rm -rf src/fe_gan/__pycache__/
	rm -rf tests/__pycache__/
	rm -rf scripts/__pycache__/
	rm -rf .pytest_cache/
	rm -rf *.egg-info/
	rm -rf build/
	rm -rf dist/

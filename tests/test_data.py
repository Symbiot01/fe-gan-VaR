"""Tests for data loading and preprocessing."""

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from fe_gan.data import (
    DEFAULT_DATA_PATH,
    VIXDataset,
    compute_data_sha256,
    extract_sha256_from_header,
    get_data_stats,
    load_vix,
    load_vix_dataset,
    make_windows,
    train_eval_split,
    verify_data_integrity,
)


class TestDataFile:
    """Tests for the committed data file."""

    def test_data_file_exists(self):
        """Test that the committed data file exists."""
        assert DEFAULT_DATA_PATH.exists(), f"Data file not found: {DEFAULT_DATA_PATH}"

    def test_sha256_matches_header(self):
        """Test that SHA256 in header matches computed hash."""
        assert verify_data_integrity(DEFAULT_DATA_PATH), "SHA256 mismatch in data file"

    def test_extract_sha256_from_header(self):
        """Test SHA256 extraction from header."""
        sha = extract_sha256_from_header(DEFAULT_DATA_PATH)
        assert sha is not None
        assert len(sha) == 64  # SHA256 is 64 hex chars

    def test_compute_data_sha256(self):
        """Test SHA256 computation."""
        sha = compute_data_sha256(DEFAULT_DATA_PATH)
        assert len(sha) == 64


class TestLoadVIX:
    """Tests for load_vix function."""

    def test_load_vix_basic(self):
        """Test basic loading of VIX data."""
        df = load_vix()

        assert "Date" in df.columns
        assert "Close" in df.columns
        assert "LogReturn" in df.columns

    def test_load_vix_shape(self):
        """Test VIX data has expected shape."""
        df = load_vix()

        # Should have ~1509 rows (per plan)
        assert 1450 <= len(df) <= 1520, f"Expected ~1509 rows, got {len(df)}"

    def test_load_vix_no_nan(self):
        """Test no NaN values in data."""
        df = load_vix()

        assert df["Close"].isna().sum() == 0
        assert df["LogReturn"].isna().sum() == 0

    def test_load_vix_log_return_stats(self):
        """Test log returns have reasonable statistics."""
        df = load_vix()

        lr = df["LogReturn"]

        # Mean should be near zero
        assert abs(lr.mean()) < 0.01, f"Mean log return {lr.mean()} too far from 0"

        # Std should be in reasonable range (VIX is volatile)
        assert 0.01 < lr.std() < 0.2, f"Std {lr.std()} outside expected range"

    def test_load_vix_date_range(self):
        """Test data covers expected date range."""
        df = load_vix()

        start = df["Date"].min()
        end = df["Date"].max()

        # Should start in early 2014
        assert start.year == 2014
        assert start.month == 1

        # Should end in late 2019
        assert end.year == 2019
        assert end.month == 12


class TestMakeWindows:
    """Tests for make_windows function."""

    def test_make_windows_shape(self):
        """Test windows have correct shape."""
        returns = np.random.randn(300).astype(np.float32)
        windows = make_windows(returns, window_size=100)

        assert windows.shape == (201, 100)

    def test_make_windows_values(self):
        """Test window values match source."""
        returns = np.arange(10).astype(np.float32)
        windows = make_windows(returns, window_size=5)

        assert windows.shape == (6, 5)
        assert torch.allclose(windows[0], torch.tensor([0, 1, 2, 3, 4], dtype=torch.float32))
        assert torch.allclose(windows[1], torch.tensor([1, 2, 3, 4, 5], dtype=torch.float32))
        assert torch.allclose(windows[-1], torch.tensor([5, 6, 7, 8, 9], dtype=torch.float32))

    def test_make_windows_from_tensor(self):
        """Test make_windows works with tensor input."""
        returns = torch.randn(300)
        windows = make_windows(returns, window_size=100)

        assert windows.shape == (201, 100)

    def test_make_windows_too_short(self):
        """Test error for returns shorter than window."""
        returns = np.random.randn(50).astype(np.float32)

        with pytest.raises(ValueError, match="shorter than window"):
            make_windows(returns, window_size=100)

    def test_make_windows_real_data(self):
        """Test make_windows on real VIX data."""
        df = load_vix()
        returns = df["LogReturn"].values
        windows = make_windows(returns, window_size=250)

        # Per plan: ~1509 rows -> ~1260 windows
        assert 1200 <= len(windows) <= 1300, f"Expected ~1260 windows, got {len(windows)}"
        assert windows.shape[1] == 250


class TestTrainEvalSplit:
    """Tests for train_eval_split function."""

    def test_split_sizes(self):
        """Test split produces correct sizes."""
        windows = torch.randn(1000, 250)
        train, eval_ = train_eval_split(windows, n_eval=100)

        assert len(train) == 900
        assert len(eval_) == 100

    def test_split_no_overlap(self):
        """Test eval windows are from end of sequence."""
        windows = torch.arange(500).reshape(100, 5).float()
        train, eval_ = train_eval_split(windows, n_eval=20)

        # Last 20 windows should be in eval
        assert torch.allclose(eval_, windows[-20:])
        assert torch.allclose(train, windows[:-20])

    def test_split_shuffle(self):
        """Test shuffle doesn't affect eval."""
        windows = torch.arange(500).reshape(100, 5).float()

        train1, eval1 = train_eval_split(windows, n_eval=20, shuffle_train=False)
        train2, eval2 = train_eval_split(windows, n_eval=20, shuffle_train=True, seed=42)

        # Eval should be identical
        assert torch.allclose(eval1, eval2)

        # Train should be different (shuffled)
        assert not torch.allclose(train1, train2)

    def test_split_deterministic_shuffle(self):
        """Test shuffle is deterministic with seed."""
        windows = torch.randn(100, 5)

        train1, _ = train_eval_split(windows, n_eval=20, shuffle_train=True, seed=42)
        train2, _ = train_eval_split(windows, n_eval=20, shuffle_train=True, seed=42)

        assert torch.allclose(train1, train2)

    def test_split_too_few_windows(self):
        """Test error when not enough windows."""
        windows = torch.randn(50, 250)

        with pytest.raises(ValueError, match="Not enough windows"):
            train_eval_split(windows, n_eval=100)


class TestVIXDataset:
    """Tests for VIXDataset class."""

    def test_dataset_len(self):
        """Test dataset length."""
        windows = torch.randn(100, 250)
        dataset = VIXDataset(windows)

        assert len(dataset) == 100

    def test_dataset_getitem(self):
        """Test dataset indexing."""
        windows = torch.randn(100, 250)
        dataset = VIXDataset(windows)

        item = dataset[0]
        assert item.shape == (250,)
        assert torch.allclose(item, windows[0])

    def test_dataset_sample_batch(self):
        """Test batch sampling."""
        windows = torch.randn(100, 250)
        dataset = VIXDataset(windows)

        batch = dataset.sample_batch(32)
        assert batch.shape == (32, 250)

    def test_dataset_get_all(self):
        """Test get_all returns all windows."""
        windows = torch.randn(100, 250)
        dataset = VIXDataset(windows)

        all_windows = dataset.get_all()
        assert torch.allclose(all_windows, windows)

    def test_dataset_device(self):
        """Test dataset device handling."""
        windows = torch.randn(100, 250)
        dataset = VIXDataset(windows, device="cpu")

        assert dataset.windows.device == torch.device("cpu")


class TestLoadVIXDataset:
    """Tests for load_vix_dataset convenience function."""

    def test_load_vix_dataset(self):
        """Test full data loading pipeline."""
        train, eval_ = load_vix_dataset(window_size=250, n_eval=100)

        # Check types
        assert isinstance(train, VIXDataset)
        assert isinstance(eval_, VIXDataset)

        # Check sizes
        assert len(eval_) == 100
        assert len(train) > 1000  # Should have > 1000 training windows


class TestGetDataStats:
    """Tests for get_data_stats function."""

    def test_get_data_stats(self):
        """Test data stats extraction."""
        stats = get_data_stats()

        assert "n_rows" in stats
        assert "date_start" in stats
        assert "date_end" in stats
        assert "log_return_mean" in stats
        assert "log_return_std" in stats

        # Sanity checks
        assert 1450 <= stats["n_rows"] <= 1520
        assert stats["date_start"].startswith("2014")
        assert stats["date_end"].startswith("2019")

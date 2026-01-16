"""Data loading and preprocessing for FE-GAN.

Provides:
- load_vix: Load VIX data from CSV
- make_windows: Create rolling windows from log returns
- train_eval_split: Split windows into train and eval sets
- VIXDataset: PyTorch dataset for VIX windows
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import Tensor

# Default data path (relative to project root)
DEFAULT_DATA_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "vix_2014_2019.csv"


def load_vix(path: str | Path = DEFAULT_DATA_PATH) -> pd.DataFrame:
    """Load VIX data from CSV file.

    The CSV file should have a header comment section (lines starting with #)
    followed by columns: Date, Close, LogReturn.

    Args:
        path: Path to VIX CSV file.

    Returns:
        DataFrame with Date, Close, LogReturn columns.

    Raises:
        FileNotFoundError: If file doesn't exist.
        ValueError: If data validation fails.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"VIX data file not found: {path}")

    # Read file, skipping comment lines
    df = pd.read_csv(path, comment="#")

    # Parse date
    df["Date"] = pd.to_datetime(df["Date"])

    # Validate required columns
    required_cols = ["Date", "Close", "LogReturn"]
    missing = set(required_cols) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Validate no NaN
    for col in required_cols:
        if col == "Date":
            continue
        nan_count = df[col].isna().sum()
        if nan_count > 0:
            raise ValueError(f"Found {nan_count} NaN values in {col}")

    return df.sort_values("Date").reset_index(drop=True)


def extract_sha256_from_header(path: str | Path) -> str | None:
    """Extract SHA256 from CSV header comment.

    Args:
        path: Path to VIX CSV file.

    Returns:
        SHA256 string or None if not found.
    """
    path = Path(path)
    with open(path) as f:
        for line in f:
            if not line.startswith("#"):
                break
            match = re.search(r"SHA256.*?:\s*([a-f0-9]{64})", line, re.IGNORECASE)
            if match:
                return match.group(1)
    return None


def compute_data_sha256(path: str | Path) -> str:
    """Compute SHA256 of data section (excluding header comments).

    Args:
        path: Path to VIX CSV file.

    Returns:
        SHA256 hex string.
    """
    path = Path(path)
    lines = []
    with open(path) as f:
        for line in f:
            if not line.startswith("#"):
                lines.append(line)

    data_content = "".join(lines)
    return hashlib.sha256(data_content.encode()).hexdigest()


def verify_data_integrity(path: str | Path) -> bool:
    """Verify data SHA256 matches header.

    Args:
        path: Path to VIX CSV file.

    Returns:
        True if SHA256 matches, False otherwise.
    """
    expected = extract_sha256_from_header(path)
    if expected is None:
        return False

    actual = compute_data_sha256(path)
    return expected == actual


def make_windows(returns: np.ndarray | Tensor, window_size: int = 250) -> Tensor:
    """Create rolling windows from log returns.

    Uses a sliding window with stride 1 to create overlapping sequences.

    Args:
        returns: 1D array of log returns.
        window_size: Size of each window (default 250 for ~1 year of trading days).

    Returns:
        Tensor of shape [N, window_size] where N = len(returns) - window_size + 1.

    Raises:
        ValueError: If returns is too short for the window size.
    """
    if isinstance(returns, Tensor):
        returns = returns.numpy()
    returns = np.asarray(returns, dtype=np.float32)

    n = len(returns)
    if n < window_size:
        raise ValueError(f"Returns length {n} is shorter than window size {window_size}")

    # Create rolling windows using stride tricks
    n_windows = n - window_size + 1

    # Use numpy stride tricks for efficient window creation
    strides = (returns.strides[0], returns.strides[0])
    windows = np.lib.stride_tricks.as_strided(
        returns,
        shape=(n_windows, window_size),
        strides=strides,
    )

    # Make a copy to avoid memory issues with stride tricks
    windows = windows.copy()

    return torch.from_numpy(windows)


def train_eval_split(
    windows: Tensor,
    n_eval: int = 100,
    shuffle_train: bool = False,
    seed: int = 42,
) -> tuple[Tensor, Tensor]:
    """Split windows into train and eval sets.

    The last n_eval windows are used for evaluation (approximately non-overlapping
    with training due to time ordering).

    Args:
        windows: Tensor of shape [N, window_size].
        n_eval: Number of windows to reserve for evaluation.
        shuffle_train: Whether to shuffle training windows.
        seed: Random seed for shuffling.

    Returns:
        Tuple of (train_windows, eval_windows).

    Raises:
        ValueError: If not enough windows for eval split.
    """
    n_total = len(windows)
    if n_total <= n_eval:
        raise ValueError(f"Not enough windows ({n_total}) for eval split ({n_eval})")

    # Time-based split: last n_eval for evaluation
    train_windows = windows[:-n_eval]
    eval_windows = windows[-n_eval:]

    if shuffle_train:
        # Shuffle training windows
        rng = np.random.default_rng(seed)
        perm = rng.permutation(len(train_windows))
        train_windows = train_windows[perm]

    return train_windows, eval_windows


class VIXDataset:
    """Simple dataset wrapper for VIX windows.

    Supports indexing and batching for training loops.
    """

    def __init__(
        self,
        windows: Tensor,
        device: torch.device | str = "cpu",
    ):
        """Initialize dataset.

        Args:
            windows: Tensor of shape [N, window_size].
            device: Device to store data on.
        """
        self.windows = windows.to(device)
        self.device = device

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> Tensor:
        return self.windows[idx]

    def sample_batch(self, batch_size: int) -> Tensor:
        """Sample a random batch of windows.

        Args:
            batch_size: Number of windows to sample.

        Returns:
            Tensor of shape [batch_size, window_size].
        """
        indices = torch.randint(len(self), (batch_size,), device=self.device)
        return self.windows[indices]

    def get_all(self) -> Tensor:
        """Get all windows as a single tensor."""
        return self.windows

    def to(self, device: torch.device | str) -> "VIXDataset":
        """Move dataset to device."""
        return VIXDataset(self.windows, device=device)


def load_vix_dataset(
    path: str | Path = DEFAULT_DATA_PATH,
    window_size: int = 250,
    n_eval: int = 100,
    device: torch.device | str = "cpu",
    verify_integrity: bool = True,
) -> tuple[VIXDataset, VIXDataset]:
    """Load VIX data and create train/eval datasets.

    Convenience function that combines load_vix, make_windows, and train_eval_split.

    Args:
        path: Path to VIX CSV file.
        window_size: Size of each window.
        n_eval: Number of windows for evaluation.
        device: Device to store data on.
        verify_integrity: Whether to verify SHA256 checksum.

    Returns:
        Tuple of (train_dataset, eval_dataset).
    """
    if verify_integrity and not verify_data_integrity(path):
        import warnings

        warnings.warn(
            f"Data integrity check failed for {path}. "
            "SHA256 mismatch or missing checksum in header."
        )

    df = load_vix(path)
    returns = df["LogReturn"].values
    windows = make_windows(returns, window_size)

    train_windows, eval_windows = train_eval_split(windows, n_eval)

    train_dataset = VIXDataset(train_windows, device=device)
    eval_dataset = VIXDataset(eval_windows, device=device)

    return train_dataset, eval_dataset


def get_data_stats(path: str | Path = DEFAULT_DATA_PATH) -> dict[str, Any]:
    """Get summary statistics of the VIX data.

    Args:
        path: Path to VIX CSV file.

    Returns:
        Dictionary with summary statistics.
    """
    df = load_vix(path)

    return {
        "n_rows": len(df),
        "date_start": str(df["Date"].min().date()),
        "date_end": str(df["Date"].max().date()),
        "close_mean": float(df["Close"].mean()),
        "close_std": float(df["Close"].std()),
        "close_min": float(df["Close"].min()),
        "close_max": float(df["Close"].max()),
        "log_return_mean": float(df["LogReturn"].mean()),
        "log_return_std": float(df["LogReturn"].std()),
        "log_return_min": float(df["LogReturn"].min()),
        "log_return_max": float(df["LogReturn"].max()),
    }

"""Fetch VIX data from CBOE (primary) or yfinance (fallback).

This script downloads VIX historical data, computes log returns,
and saves to data/vix_2014_2019.csv with a header comment containing
source URL, fetch timestamp, and SHA256 checksum.

Usage:
    python -m scripts.fetch_data           # refuses if file exists
    python -m scripts.fetch_data --force   # overwrites existing file
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import requests

# Project root (parent of scripts/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_FILE = DATA_DIR / "vix_2014_2019.csv"

# Date range per paper
START_DATE = "2014-01-02"
END_DATE = "2019-12-31"

# CBOE official VIX history URL
CBOE_URL = "https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX_History.csv"


def fetch_from_cboe() -> pd.DataFrame:
    """Fetch VIX data from CBOE official source.

    Returns:
        DataFrame with Date, Close columns.

    Raises:
        requests.RequestException: If download fails.
        ValueError: If data validation fails.
    """
    print(f"Fetching VIX data from CBOE: {CBOE_URL}")

    response = requests.get(CBOE_URL, timeout=60)
    response.raise_for_status()

    # CBOE CSV has columns: DATE, OPEN, HIGH, LOW, CLOSE
    # Use DATE as index, parse it
    from io import StringIO

    df = pd.read_csv(StringIO(response.text))

    # Normalize column names
    df.columns = df.columns.str.strip().str.upper()

    # Parse date column
    date_col = None
    for col in ["DATE", "TRADE DATE"]:
        if col in df.columns:
            date_col = col
            break

    if date_col is None:
        raise ValueError(f"No date column found. Columns: {list(df.columns)}")

    df["Date"] = pd.to_datetime(df[date_col], format="mixed")
    df = df.sort_values("Date").reset_index(drop=True)

    # Get close column
    close_col = None
    for col in ["CLOSE", "VIX CLOSE"]:
        if col in df.columns:
            close_col = col
            break

    if close_col is None:
        raise ValueError(f"No close column found. Columns: {list(df.columns)}")

    df["Close"] = pd.to_numeric(df[close_col], errors="coerce")

    # Filter to date range
    mask = (df["Date"] >= START_DATE) & (df["Date"] <= END_DATE)
    df = df.loc[mask, ["Date", "Close"]].copy()

    return df


def fetch_from_yfinance() -> pd.DataFrame:
    """Fallback: Fetch VIX data from yfinance.

    Returns:
        DataFrame with Date, Close columns.
    """
    print("CBOE fetch failed, trying yfinance fallback...")

    import yfinance as yf

    ticker = yf.Ticker("^VIX")
    df = ticker.history(start=START_DATE, end="2020-01-01", auto_adjust=True)

    df = df.reset_index()
    df = df.rename(columns={"index": "Date"})
    if "Date" not in df.columns and "Datetime" in df.columns:
        df = df.rename(columns={"Datetime": "Date"})

    # Ensure Date is datetime without timezone
    df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None)

    # Filter to date range
    mask = (df["Date"] >= START_DATE) & (df["Date"] <= END_DATE)
    df = df.loc[mask, ["Date", "Close"]].copy()

    return df


def validate_data(df: pd.DataFrame) -> None:
    """Validate the fetched data.

    Args:
        df: DataFrame with Date, Close columns.

    Raises:
        ValueError: If validation fails.
    """
    # Check row count (expect ~1509 trading days)
    n_rows = len(df)
    if not (1450 <= n_rows <= 1520):
        raise ValueError(f"Expected 1450-1520 rows, got {n_rows}")

    # Check for NaN in Close
    nan_count = df["Close"].isna().sum()
    if nan_count > 0:
        raise ValueError(f"Found {nan_count} NaN values in Close column")

    # Check for gaps > 5 trading days
    df = df.sort_values("Date").reset_index(drop=True)
    date_diffs = df["Date"].diff().dt.days
    max_gap = date_diffs.max()
    if max_gap > 5:
        gap_idx = date_diffs.idxmax()
        raise ValueError(
            f"Found gap of {max_gap} days between {df.loc[gap_idx - 1, 'Date']} "
            f"and {df.loc[gap_idx, 'Date']}"
        )

    # Check date range
    actual_start = df["Date"].min()
    actual_end = df["Date"].max()
    if actual_start > pd.Timestamp(START_DATE) + pd.Timedelta(days=7):
        raise ValueError(f"Data starts too late: {actual_start}")
    if actual_end < pd.Timestamp(END_DATE) - pd.Timedelta(days=7):
        raise ValueError(f"Data ends too early: {actual_end}")

    print(f"Validation passed: {n_rows} rows, {actual_start.date()} to {actual_end.date()}")


def compute_log_returns(df: pd.DataFrame) -> pd.DataFrame:
    """Compute log returns from Close prices.

    Args:
        df: DataFrame with Date, Close columns.

    Returns:
        DataFrame with Date, Close, LogReturn columns (first row dropped).
    """
    df = df.sort_values("Date").reset_index(drop=True)
    df["LogReturn"] = np.log(df["Close"] / df["Close"].shift(1))

    # Drop first row (NaN log return)
    df = df.iloc[1:].reset_index(drop=True)

    return df


def compute_sha256(content: str) -> str:
    """Compute SHA256 hash of content."""
    return hashlib.sha256(content.encode()).hexdigest()


def save_with_header(df: pd.DataFrame, path: Path, source_url: str) -> str:
    """Save DataFrame with header comment containing metadata.

    Args:
        df: DataFrame to save.
        path: Output path.
        source_url: URL where data was fetched from.

    Returns:
        SHA256 of the data (excluding header).
    """
    # Convert DataFrame to CSV string (without header comment)
    data_csv = df.to_csv(index=False)
    data_sha256 = compute_sha256(data_csv)

    # Create header comment
    timestamp = datetime.now(UTC).isoformat()
    header = f"""# VIX Historical Data for FE-GAN Verification
# Source: {source_url}
# Date range: {START_DATE} to {END_DATE}
# Fetched: {timestamp}
# SHA256 (data only): {data_sha256}
# Rows: {len(df)}
#
"""

    # Write file
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write(header)
        f.write(data_csv)

    return data_sha256


def main():
    parser = argparse.ArgumentParser(description="Fetch VIX data for FE-GAN")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing data file",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_FILE,
        help="Output file path",
    )
    args = parser.parse_args()

    output_path = args.output

    # Check if file exists
    if output_path.exists() and not args.force:
        print(f"Error: {output_path} already exists. Use --force to overwrite.")
        sys.exit(1)

    # Try CBOE first, fall back to yfinance
    source_url = CBOE_URL
    try:
        df = fetch_from_cboe()
    except Exception as e:
        print(f"CBOE fetch failed: {e}")
        try:
            df = fetch_from_yfinance()
            source_url = "yfinance ^VIX (fallback)"
        except Exception as e2:
            print(f"yfinance fetch also failed: {e2}")
            sys.exit(1)

    # Validate
    validate_data(df)

    # Compute log returns
    df = compute_log_returns(df)
    print(f"Computed log returns: {len(df)} rows")

    # Save
    sha256 = save_with_header(df, output_path, source_url)
    print(f"Saved to {output_path}")
    print(f"SHA256: {sha256}")


if __name__ == "__main__":
    main()

"""
============================================================
src/data_loader.py
NIFTY50-VQC Project

PURPOSE:
    Loads and combines all NIFTY 50 yearly CSV files into a
    single clean, chronologically sorted DataFrame.

WHAT THIS MODULE DOES:
    1. Locates all 11 yearly CSV files (2015–2025)
    2. Standardises column names
    3. Parses the "DD Mon YYYY" date format into datetime
    4. Converts OHLC columns to float (removes quote strings)
    5. Drops the constant "Index Name" column
    6. Sorts each file chronologically (ascending by date)
    7. Detects missing values in each file
    8. Detects duplicate dates in each file
    9. Combines all 11 years into one DataFrame
    10. Sorts the combined DataFrame chronologically
    11. Reports all data quality findings
    12. (Optionally) saves to data/processed/nifty50_combined.csv

RAW DATA FORMAT (as found in the CSV files):
    "Index Name","Date","Open","High","Low","Close"
    "NIFTY 50","31 Dec 2015","7897.80","7955.55","7891.15","7946.35"

    Note: Files are reverse-chronological (Dec→Jan).
    Note: All values are quoted strings, not numbers yet.
    Note: 2016 filename contains "(1)" — handled gracefully.

IMPORTANT:
    Raw files are NEVER modified.
    This module is READ-ONLY with respect to raw data.

USAGE:
    from src.data_loader import load_nifty50_data, DATA_DIR

    df = load_nifty50_data(verbose=True)
    # Returns a clean pandas DataFrame with columns:
    # date, open, high, low, close

Author: NIFTY50-VQC Project
Date:   2026-10-02
============================================================
"""

import os
import glob
import pandas as pd
import numpy as np

# ============================================================
# CONFIGURATION — Paths
# ============================================================

# Root directory of the project (parent of src/)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Location of raw CSV files (the extracted preview folder)
DATA_DIR = os.path.join(PROJECT_ROOT, "NIFTY50_DATASET_preview", "NIFTY50_DATASET")

# Location where processed output will be saved
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")

# Expected output file
COMBINED_CSV = os.path.join(PROCESSED_DIR, "nifty50_combined.csv")

# Expected year range
EXPECTED_YEARS = list(range(2015, 2026))  # 2015 to 2025 inclusive


# ============================================================
# HELPER: Parse "31 Dec 2015" → datetime
# ============================================================

def _parse_nse_date(date_str):
    """
    Parse NSE-format date strings like "31 Dec 2015" into pandas datetime.

    The NSE (National Stock Exchange of India) uses this text format
    in historical data exports. pandas read_csv cannot auto-detect it,
    so we parse it explicitly.

    Args:
        date_str (str): Date string like "31 Dec 2015"

    Returns:
        pd.Timestamp: Parsed date, or pd.NaT if parsing fails
    """
    try:
        return pd.to_datetime(date_str.strip(), format="%d %b %Y")
    except Exception:
        return pd.NaT


# ============================================================
# HELPER: Load a single yearly CSV file
# ============================================================

def _load_single_file(filepath, verbose=True):
    """
    Load one NIFTY 50 yearly CSV file and return a clean DataFrame.

    Steps performed:
    1. Read raw CSV (all columns as strings initially)
    2. Standardise column names to lowercase
    3. Drop the 'index name' column (constant 'NIFTY 50' — not useful)
    4. Parse the 'date' column from "DD Mon YYYY" format
    5. Convert open, high, low, close to float
    6. Sort rows chronologically (ascending by date)
    7. Report missing values and duplicate dates

    Args:
        filepath (str): Full path to the CSV file
        verbose (bool): If True, print detailed per-file report

    Returns:
        pd.DataFrame or None: Clean DataFrame, or None if loading failed
    """
    filename = os.path.basename(filepath)

    if verbose:
        print(f"\n  Loading: {filename}")
        print(f"  {'─' * 55}")

    # ── Step 1: Read raw CSV ──────────────────────────────────
    try:
        df_raw = pd.read_csv(
            filepath,
            dtype=str,          # Read everything as string first
            encoding='utf-8',   # Try UTF-8 first
            skipinitialspace=True
        )
    except UnicodeDecodeError:
        # Some NSE files may use cp1252 encoding
        df_raw = pd.read_csv(
            filepath,
            dtype=str,
            encoding='cp1252',
            skipinitialspace=True
        )

    if verbose:
        print(f"  Raw shape       : {df_raw.shape[0]} rows × {df_raw.shape[1]} columns")
        print(f"  Raw columns     : {list(df_raw.columns)}")

    # ── Step 2: Standardise column names ─────────────────────
    # Strip whitespace and convert to lowercase
    df_raw.columns = [c.strip().lower().replace(' ', '_') for c in df_raw.columns]

    # Strip leading/trailing whitespace and quotes from all values
    df_raw = df_raw.apply(lambda col: col.str.strip().str.strip('"') if col.dtype == object else col)

    # ── Step 3: Drop the 'index_name' column ─────────────────
    # This column contains "NIFTY 50" in every row — no information value
    if 'index_name' in df_raw.columns:
        df_raw = df_raw.drop(columns=['index_name'])
        if verbose:
            print(f"  Dropped column  : 'index_name' (constant 'NIFTY 50')")

    # ── Step 4: Parse the date column ────────────────────────
    if 'date' not in df_raw.columns:
        print(f"  ERROR: No 'date' column found in {filename}")
        print(f"  Available columns: {list(df_raw.columns)}")
        return None

    df_raw['date'] = df_raw['date'].apply(_parse_nse_date)

    # Count unparseable dates
    bad_dates = df_raw['date'].isna().sum()
    if bad_dates > 0:
        print(f"  WARNING: {bad_dates} date(s) could not be parsed → set to NaT")
        print(f"  Bad date rows:\n{df_raw[df_raw['date'].isna()]}")

    # ── Step 5: Convert OHLC columns to float ────────────────
    ohlc_cols = ['open', 'high', 'low', 'close']
    available_ohlc = [c for c in ohlc_cols if c in df_raw.columns]

    if verbose:
        print(f"  OHLC columns    : {available_ohlc}")

    for col in available_ohlc:
        df_raw[col] = pd.to_numeric(df_raw[col], errors='coerce')
        bad_vals = df_raw[col].isna().sum()
        if bad_vals > 0:
            print(f"  WARNING: {bad_vals} value(s) in '{col}' could not be converted to float")

    # ── Step 6: Sort chronologically (ascending) ──────────────
    # Raw files are in descending order (Dec→Jan). We reverse this.
    df_raw = df_raw.sort_values('date', ascending=True).reset_index(drop=True)

    # ── Step 7: Data quality checks ──────────────────────────

    # Check missing values
    missing = df_raw.isnull().sum()
    total_missing = missing.sum()
    if total_missing > 0:
        print(f"  MISSING VALUES:")
        for col, cnt in missing[missing > 0].items():
            print(f"    '{col}': {cnt} missing value(s)")
    else:
        if verbose:
            print(f"  Missing values  : 0 (clean)")

    # Check duplicate dates
    dup_dates = df_raw[df_raw['date'].duplicated(keep=False)]
    if len(dup_dates) > 0:
        print(f"  WARNING: {len(dup_dates)} duplicate date entries found:")
        print(f"{dup_dates[['date','close']].to_string()}")
    else:
        if verbose:
            print(f"  Duplicate dates : 0 (clean)")

    # Report date range
    valid_dates = df_raw['date'].dropna()
    if len(valid_dates) > 0:
        date_min = valid_dates.min().strftime("%d %b %Y")
        date_max = valid_dates.max().strftime("%d %b %Y")
        if verbose:
            print(f"  Date range      : {date_min} → {date_max}")
            print(f"  Trading days    : {len(df_raw)}")
            print(f"  Close range     : {df_raw['close'].min():.2f} → {df_raw['close'].max():.2f}")

    return df_raw


# ============================================================
# MAIN: Load all yearly files and combine
# ============================================================

def load_nifty50_data(data_dir=None, verbose=True):
    """
    Load all NIFTY 50 yearly CSV files and return a combined DataFrame.

    This function:
    1. Searches for all CSV files in the data directory
    2. Loads each file using _load_single_file()
    3. Combines them into one DataFrame
    4. Sorts by date (chronologically)
    5. Resets the index
    6. Performs final quality checks on the combined dataset
    7. Reports a summary

    Args:
        data_dir (str, optional): Path to folder containing CSV files.
                                  Defaults to DATA_DIR constant.
        verbose (bool): If True, print detailed loading report.

    Returns:
        pd.DataFrame: Combined clean DataFrame with columns:
                      date, open, high, low, close
                      Sorted chronologically. Index is 0-based integer.

    Raises:
        FileNotFoundError: If data_dir does not exist
        ValueError: If no CSV files are found
    """
    if data_dir is None:
        data_dir = DATA_DIR

    if verbose:
        print("=" * 60)
        print("NIFTY50 DATA LOADER")
        print("=" * 60)
        print(f"\nSearching for CSV files in:")
        print(f"  {data_dir}")

    # ── Find all CSV files ────────────────────────────────────
    if not os.path.exists(data_dir):
        raise FileNotFoundError(
            f"Data directory not found: {data_dir}\n"
            f"Please ensure your CSV files are in:\n{data_dir}"
        )

    csv_pattern = os.path.join(data_dir, "*.csv")
    csv_files = sorted(glob.glob(csv_pattern))

    if len(csv_files) == 0:
        raise ValueError(
            f"No CSV files found in: {data_dir}\n"
            f"Expected files named like: NIFTY 50_Historical_PR_*.csv"
        )

    if verbose:
        print(f"\nFound {len(csv_files)} CSV files:")
        for f in csv_files:
            size_kb = os.path.getsize(f) / 1024
            print(f"  {os.path.basename(f):60s} ({size_kb:.1f} KB)")

    # ── Load each file ────────────────────────────────────────
    if verbose:
        print(f"\n{'─' * 60}")
        print("LOADING INDIVIDUAL FILES")
        print(f"{'─' * 60}")

    all_dfs = []
    failed_files = []

    for filepath in csv_files:
        df_year = _load_single_file(filepath, verbose=verbose)

        if df_year is not None and len(df_year) > 0:
            all_dfs.append(df_year)
        else:
            failed_files.append(os.path.basename(filepath))
            print(f"  FAILED to load: {os.path.basename(filepath)}")

    if len(failed_files) > 0:
        print(f"\nWARNING: {len(failed_files)} file(s) failed to load:")
        for f in failed_files:
            print(f"  - {f}")

    if len(all_dfs) == 0:
        raise ValueError("No data was successfully loaded from any file.")

    # ── Combine all years ─────────────────────────────────────
    if verbose:
        print(f"\n{'─' * 60}")
        print("COMBINING ALL YEARS")
        print(f"{'─' * 60}")

    df_combined = pd.concat(all_dfs, ignore_index=True)

    # Sort the entire combined dataset chronologically
    df_combined = df_combined.sort_values('date', ascending=True).reset_index(drop=True)

    # ── Final quality checks on combined dataset ──────────────
    if verbose:
        print(f"\n{'─' * 60}")
        print("COMBINED DATASET QUALITY CHECK")
        print(f"{'─' * 60}")

    # Duplicate dates in combined dataset (e.g. year boundary overlap)
    dup_combined = df_combined[df_combined['date'].duplicated(keep=False)]
    if len(dup_combined) > 0:
        print(f"\nWARNING: {len(dup_combined)} duplicate dates in combined dataset:")
        print(dup_combined[['date', 'close']].to_string())
        print("Action: Keeping first occurrence of each duplicate date.")
        df_combined = df_combined.drop_duplicates(subset=['date'], keep='first')
        df_combined = df_combined.reset_index(drop=True)
    else:
        if verbose:
            print(f"  Duplicate dates in combined : 0 (clean)")

    # Total missing values
    total_missing_combined = df_combined.isnull().sum().sum()
    if total_missing_combined > 0:
        print(f"\nWARNING: {total_missing_combined} missing values in combined dataset:")
        print(df_combined.isnull().sum())
    else:
        if verbose:
            print(f"  Missing values in combined  : 0 (clean)")

    # ── Final summary ─────────────────────────────────────────
    if verbose:
        print(f"\n{'=' * 60}")
        print("LOADING COMPLETE — SUMMARY")
        print(f"{'=' * 60}")
        print(f"  Files loaded           : {len(all_dfs)} of {len(csv_files)}")
        print(f"  Total rows (trading days): {len(df_combined):,}")
        print(f"  Columns                : {list(df_combined.columns)}")
        print(f"  Date range             : {df_combined['date'].min().strftime('%d %b %Y')} → "
              f"{df_combined['date'].max().strftime('%d %b %Y')}")
        print(f"  Close price range      : {df_combined['close'].min():.2f} → "
              f"{df_combined['close'].max():.2f}")
        print(f"  Data types:\n{df_combined.dtypes.to_string()}")
        print(f"\n  First 3 rows:")
        print(df_combined.head(3).to_string())
        print(f"\n  Last 3 rows:")
        print(df_combined.tail(3).to_string())
        print(f"{'=' * 60}")

    return df_combined


# ============================================================
# SAVE: Write combined dataset to processed folder
# ============================================================

def save_processed(df, output_path=None, verbose=True):
    """
    Save the combined DataFrame to data/processed/nifty50_combined.csv

    IMPORTANT: Call this function ONLY after explicit user approval.
    This function is separated from load_nifty50_data() to ensure
    saving is a deliberate, approved action.

    Args:
        df (pd.DataFrame): The combined NIFTY 50 DataFrame
        output_path (str, optional): Custom output path.
                                     Defaults to COMBINED_CSV constant.
        verbose (bool): If True, print confirmation message.

    Returns:
        str: The path where the file was saved.
    """
    if output_path is None:
        output_path = COMBINED_CSV

    # Ensure the processed directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save to CSV with date formatted as YYYY-MM-DD
    df.to_csv(output_path, index=False, date_format='%Y-%m-%d')

    if verbose:
        size_kb = os.path.getsize(output_path) / 1024
        print(f"\n  Saved processed data to:")
        print(f"  {output_path}")
        print(f"  File size: {size_kb:.1f} KB")
        print(f"  Rows saved: {len(df):,}")

    return output_path


# ============================================================
# STANDALONE EXECUTION (for testing this module directly)
# ============================================================

if __name__ == "__main__":
    import sys
    # Add project root to path so 'src' imports work
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    print("Running data_loader.py in standalone mode...")
    print("This will LOAD the data but NOT save it.")
    print("Saving requires a separate explicit approval step.\n")

    df = load_nifty50_data(verbose=True)

    print(f"\nData loaded successfully into DataFrame.")
    print(f"Shape: {df.shape}")
    print(f"\nTo save: call save_processed(df) after approval.")

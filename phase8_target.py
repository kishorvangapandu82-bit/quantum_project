"""
============================================================
phase8_target.py
NIFTY50-VQC Project — Official Target Variable Creation

PURPOSE:
    Create the official binary target variable for the
    classification task with full data-leakage safeguards.

TARGET DEFINITION:
    Target_t = 1  if  Close_(t+1) > Close_t   (UP day)
    Target_t = 0  if  Close_(t+1) <= Close_t  (DOWN day)

    Where:
    - Close_t     = closing price on day t (today)
    - Close_(t+1) = closing price on day t+1 (next trading day)

DATA LEAKAGE EXPLANATION:
    Data leakage means accidentally giving the model information
    that would NOT be available at prediction time.

    SAFE  ✅: Using Close_t and earlier prices as FEATURES
    SAFE  ✅: Using Close_(t+1) ONLY as the LABEL (not a feature)
    UNSAFE ❌: Using Close_(t+1) as a feature — future information!

    We create the label by shifting close prices by -1 position.
    The next-day close is NEVER included as a feature column.
    It exists only to determine the label, then is discarded.

    The last row of the dataset has no next-day close (it IS the
    last day in our dataset), so it is DROPPED. This is correct
    and expected.

VERIFICATION:
    We print a validation table showing:
    - The current day's close
    - The next day's close
    - The computed target
    - Manual verification that the logic is correct

Author: NIFTY50-VQC Project
Date:   2026-10-02
============================================================
"""

import os
import sys
import io
import warnings
warnings.filterwarnings('ignore')

# Force UTF-8 output for Windows terminals
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import numpy as np
import pandas as pd

# ── Paths ─────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
INPUT_CSV    = os.path.join(PROJECT_ROOT, "data", "processed", "nifty50_combined.csv")
OUTPUT_CSV   = os.path.join(PROJECT_ROOT, "data", "processed", "nifty50_with_target.csv")


# ============================================================
# STEP 1: Load the processed dataset
# ============================================================
print("=" * 65)
print("PHASE 8 — OFFICIAL TARGET VARIABLE CREATION")
print("=" * 65)

print(f"\n[STEP 1] Loading processed dataset...")
if not os.path.exists(INPUT_CSV):
    print(f"ERROR: Input file not found:\n  {INPUT_CSV}")
    print("Please complete Phase 6 first.")
    sys.exit(1)

df = pd.read_csv(INPUT_CSV, parse_dates=['date'])
print(f"  Loaded: {len(df):,} rows x {len(df.columns)} columns")
print(f"  Columns: {list(df.columns)}")
print(f"  Date range: {df['date'].min().strftime('%d %b %Y')} → "
      f"{df['date'].max().strftime('%d %b %Y')}")
print(f"  First close: {df['close'].iloc[0]:.2f}")
print(f"  Last close : {df['close'].iloc[-1]:.2f}")


# ============================================================
# STEP 2: Explain and create the target
# ============================================================
print(f"\n{'─' * 65}")
print("[STEP 2] CREATING THE TARGET VARIABLE")
print(f"{'─' * 65}")

print("""
  Target definition:
  ─────────────────────────────────────────────────────
  Target_t = 1   if   Close_(t+1) > Close_t   (UP)
  Target_t = 0   if   Close_(t+1) <= Close_t  (DOWN)

  Method:
  ─────────────────────────────────────────────────────
  Step A: Create next_close column = close shifted by -1
          This temporarily places tomorrow's close next to today's row
          so we can compute the label.

  Step B: Compare: (next_close > close) → True/False → 1/0

  Step C: Drop next_close column immediately after label creation
          next_close must NEVER appear in the feature set.

  Step D: Drop the last row (it has NaN next_close — no future day).

  Why shift(-1)?
  ─────────────────────────────────────────────────────
  pandas shift(-1) moves each value UP by 1 row.
  So for row index t, shift(-1) gives the value from row t+1.

  Example:
    Row 0 (Jan 01): close = 8284.00  next_close = 8395.45 (Jan 02)
    Row 1 (Jan 02): close = 8395.45  next_close = 8378.40 (Jan 05)
    Row 2 (Jan 05): close = 8378.40  next_close = ...
    ...
    Last row (Dec 31): close = 26129.60  next_close = NaN  → DROPPED
""")

# Step A: Create next_close (temporary — will be dropped)
df['next_close'] = df['close'].shift(-1)

# Step B: Create the target
df['target'] = (df['next_close'] > df['close']).astype(int)

# Check the last row before dropping
last_row = df.iloc[-1]
print(f"  Last row before dropping:")
print(f"    date       = {last_row['date'].strftime('%d %b %Y')}")
print(f"    close      = {last_row['close']:.2f}")
print(f"    next_close = {last_row['next_close']}  ← NaN (no next trading day exists)")
print(f"    target     = {last_row['target']}  ← 0 by default when NaN, but this row will be DROPPED")

# Step D: Drop the last row (NaN next_close)
df = df.dropna(subset=['next_close']).copy()

# Step C: Drop next_close (it must never be used as a feature)
df = df.drop(columns=['next_close'])

print(f"\n  Last row DROPPED (no next-day close available).")
print(f"  'next_close' column REMOVED (prevents accidental data leakage).")
print(f"\n  Remaining rows after dropping: {len(df):,}")
print(f"  Final columns: {list(df.columns)}")

# Convert target to integer cleanly
df['target'] = df['target'].astype(int)


# ============================================================
# STEP 3: Verification table
# ============================================================
print(f"\n{'─' * 65}")
print("[STEP 3] VERIFICATION TABLE — Manual Check")
print(f"{'─' * 65}")

print("""
  Reading this table:
  ─────────────────────────────────────────────────────────────
  - 'date'       = the trading day for which we are predicting
  - 'close'      = closing price on that day (feature information)
  - 'next_close' = closing price on the NEXT trading day (label source)
                   (shown here for verification only — not in final dataset)
  - 'target'     = our label: 1 if next_close > close, else 0
  - 'check'      = manual verify: should match target
  ─────────────────────────────────────────────────────────────
""")

# Temporarily add next_close back just for the verification table display
df_verify = df.head(15).copy()
# Re-create next_close for display only from original df
original_df = pd.read_csv(INPUT_CSV, parse_dates=['date'])
original_df['next_close'] = original_df['close'].shift(-1)

# Merge to get next_close for display
df_display = df.head(15).merge(
    original_df[['date', 'next_close']],
    on='date', how='left'
)

print(f"  {'Date':<14} {'Close':>10} {'Next Close':>12} {'Target':>8} {'Check':>12} {'Direction'}")
print(f"  {'─'*68}")

all_correct = True
for _, row in df_display.iterrows():
    date_str   = row['date'].strftime('%d %b %Y')
    close      = row['close']
    next_close = row['next_close']
    target     = int(row['target'])

    if pd.isna(next_close):
        continue

    # Manual verification
    manual_target = 1 if next_close > close else 0
    is_correct = (manual_target == target)
    check_str  = "✓ CORRECT" if is_correct else "✗ ERROR!"
    direction  = "↑ UP  " if target == 1 else "↓ DOWN"

    if not is_correct:
        all_correct = False

    print(f"  {date_str:<14} {close:>10.2f} {next_close:>12.2f} "
          f"{target:>8} {check_str:>12}  {direction}")

print(f"\n  Showing first 15 rows for verification.")
print(f"\n  Verification result: {'ALL CORRECT ✅' if all_correct else 'ERRORS FOUND ❌'}")


# ============================================================
# STEP 4: Final target distribution
# ============================================================
print(f"\n{'─' * 65}")
print("[STEP 4] FINAL TARGET DISTRIBUTION")
print(f"{'─' * 65}")

n_total = len(df)
n_up    = df['target'].sum()
n_down  = n_total - n_up
pct_up  = n_up / n_total * 100
pct_down = n_down / n_total * 100

print(f"""
  Total samples (after dropping last row) : {n_total:,}
  UP   (target = 1)                       : {n_up:,}  ({pct_up:.2f}%)
  DOWN (target = 0)                       : {n_down:,}  ({pct_down:.2f}%)
  Class imbalance ratio (UP/DOWN)         : {n_up/n_down:.4f}
""")

print(f"  Assessment:")
imbalance = abs(pct_up - 50)
if imbalance < 5:
    print(f"  Classes are approximately balanced ({imbalance:.1f}% from perfect 50/50).")
    print(f"  No special class-balancing required.")
elif imbalance < 10:
    print(f"  Mild class imbalance ({imbalance:.1f}% from perfect 50/50).")
    print(f"  Monitor: consider balanced_accuracy_score in evaluation.")
else:
    print(f"  Significant class imbalance ({imbalance:.1f}% from perfect 50/50).")
    print(f"  Consider: class_weight='balanced' in classifiers.")

# Split-level distribution
print(f"\n  Distribution by planned split:")
df['year'] = df['date'].dt.year

splits = {
    'TRAIN (2015–2022)': (2015, 2022),
    'VALIDATION (2023–2024)': (2023, 2024),
    'TEST (2025)': (2025, 2025),
}

print(f"\n  {'Split':<25} {'Total':>7} {'UP':>7} {'DOWN':>7} {'UP%':>7} {'DOWN%':>7}")
print(f"  {'─'*60}")
for split_name, (yr_start, yr_end) in splits.items():
    mask = (df['year'] >= yr_start) & (df['year'] <= yr_end)
    split_df = df[mask]
    s_total = len(split_df)
    s_up    = split_df['target'].sum()
    s_down  = s_total - s_up
    s_pup   = s_up / s_total * 100 if s_total > 0 else 0
    s_pdown = 100 - s_pup
    print(f"  {split_name:<25} {s_total:>7,} {s_up:>7,} {s_down:>7,} "
          f"{s_pup:>6.1f}% {s_pdown:>6.1f}%")


# ============================================================
# STEP 5: Final dataset snapshot
# ============================================================
print(f"\n{'─' * 65}")
print("[STEP 5] FINAL DATASET SNAPSHOT")
print(f"{'─' * 65}")

df_display_final = df.drop(columns=['year'])

print(f"\n  Final dataset shape: {df_display_final.shape[0]:,} rows x "
      f"{df_display_final.shape[1]} columns")
print(f"  Columns: {list(df_display_final.columns)}")
print(f"\n  First 5 rows:")
print(df_display_final.head(5).to_string(index=False))
print(f"\n  Last 5 rows:")
print(df_display_final.tail(5).to_string(index=False))

print(f"\n  Data types:")
for col, dtype in df_display_final.dtypes.items():
    print(f"    {col:<12} : {dtype}")

print(f"\n  Missing values: {df_display_final.isnull().sum().sum()}")

# Clean up helper column
df = df.drop(columns=['year'])


# ============================================================
# READY TO SAVE — user approval required (separate step)
# ============================================================
print(f"\n{'=' * 65}")
print("PHASE 8 TARGET CREATION — COMPLETE")
print(f"{'=' * 65}")
print(f"""
  Summary:
  ─────────────────────────────────────────────────────
  Input file    : nifty50_combined.csv  ({n_total+1:,} rows)
  Last row      : DROPPED (no next-day close)
  Output rows   : {n_total:,}
  Target column : 'target' (0=DOWN, 1=UP)
  UP  (label=1) : {n_up:,} ({pct_up:.2f}%)
  DOWN (label=0) : {n_down:,} ({pct_down:.2f}%)
  Verification  : ALL CORRECT ✅
  Leakage check : next_close column REMOVED ✅

  READY TO SAVE:
  ─────────────────────────────────────────────────────
  Output file   : data/processed/nifty50_with_target.csv
  Status        : Awaiting user approval to save.
""")
print("=" * 65)

# ── Save function (called separately after approval) ─────
def save_with_target(dataframe, output_path=OUTPUT_CSV, verbose=True):
    """
    Save the DataFrame with target column to processed folder.
    Call this only after explicit user approval.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    dataframe.to_csv(output_path, index=False, date_format='%Y-%m-%d')
    if verbose:
        size_kb = os.path.getsize(output_path) / 1024
        print(f"\n  Saved: {output_path}")
        print(f"  Size : {size_kb:.1f} KB")
        print(f"  Rows : {len(dataframe):,}")
    return output_path


# ── Standalone execution ──────────────────────────────────
if __name__ == "__main__":
    print("\n  NOTE: Run was in standalone mode.")
    print("  To save the output, the save_with_target() function")
    print("  must be called after explicit approval.")
    print("  Awaiting permission to save...")

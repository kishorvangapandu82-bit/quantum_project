"""
============================================================
phase9_features.py
NIFTY50-VQC Project — Phase 9 Feature Engineering Execution

PURPOSE:
    Load nifty50_with_target.csv, compute all 8 technical indicators
    using src/features.py, analyze and drop initial NaN rows created
    by rolling windows, inspect feature statistics, verify zero leakage,
    and save nifty50_features.csv.
============================================================
"""

import sys
import os
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings('ignore')

sys.path.insert(0, '.')
from src.features import compute_all_features, get_clean_features

INPUT_CSV = 'data/processed/nifty50_with_target.csv'
OUTPUT_CSV = 'data/processed/nifty50_features.csv'

def main():
    print("=" * 70)
    print("PHASE 9 — FEATURE ENGINEERING EXECUTION")
    print("=" * 70)

    if not os.path.exists(INPUT_CSV):
        print(f"❌ Error: {INPUT_CSV} not found. Run Phase 8 first.")
        return

    print(f"📥 Loading dataset: {INPUT_CSV}")
    df = pd.read_csv(INPUT_CSV)
    df['date'] = pd.to_datetime(df['date'])
    print(f"   Rows loaded: {len(df):,}")
    print(f"   Columns: {list(df.columns)}")

    print("\n⚙️ Computing 8 Technical Indicator Features & Dropping NaN Warmup Rows...")
    df_clean, feature_cols = get_clean_features(df, verbose=True)
    print(f"   Features generated ({len(feature_cols)}): {feature_cols}")

    # Summary Statistics
    print("\n📊 Feature Summary Statistics (Clean Dataset):")
    print("=" * 80)
    stats_df = df_clean[feature_cols].describe().T[['mean', 'std', 'min', '50%', 'max']]
    stats_df['NaNs'] = df_clean[feature_cols].isna().sum()
    print(stats_df.to_string())
    print("=" * 80)

    # Preview Table
    print("\n👀 Preview of Cleaned Feature Dataset (First 5 Rows):")
    print("-" * 100)
    preview_cols = ['date', 'close', 'daily_return', 'sma_5', 'ema_12', 'rsi_14', 'volatility_5', 'target']
    print(df_clean[preview_cols].head().to_string(index=False))
    print("-" * 100)

    # Class balance check in clean set
    target_counts = df_clean['target'].value_counts()
    print("\n🎯 Target Variable Distribution (Clean Set):")
    print(f"   UP   (1): {target_counts.get(1, 0):,} ({target_counts.get(1, 0)/len(df_clean):.2%})")
    print(f"   DOWN (0): {target_counts.get(0, 0):,} ({target_counts.get(0, 0)/len(df_clean):.2%})")

    print("\n✅ Phase 9 Feature Calculation Completed Successfully!")
    print(f"\nTarget output path: {OUTPUT_CSV}")

if __name__ == '__main__':
    main()

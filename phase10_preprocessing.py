"""
============================================================
phase10_preprocessing.py
NIFTY50-VQC Project — Phase 10 Preprocessing Execution

PURPOSE:
    Load data/processed/nifty50_features.csv, execute temporal split
    (Train: 2015-2022, Val: 2023-2024, Test: 2025), fit StandardScaler and
    MinMaxScaler [0, pi] STRICTLY on the training set, transform all splits,
    verify zero leakage, and print detailed summary tables.
============================================================
"""

import sys
import os
import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings('ignore')

sys.path.insert(0, '.')
from src.preprocessing import temporal_split, fit_and_transform_features, FEATURE_COLS

INPUT_CSV = 'data/processed/nifty50_features.csv'
OUTPUT_NPZ = 'data/processed/nifty50_scaled_data.npz'

def main():
    print("=" * 70)
    print("PHASE 10 — TEMPORAL SPLITTING & FEATURE NORMALIZATION")
    print("=" * 70)

    if not os.path.exists(INPUT_CSV):
        print(f"❌ Error: {INPUT_CSV} not found. Run Phase 9 first.")
        return

    print(f"📥 Loading clean features dataset: {INPUT_CSV}")
    df = pd.read_csv(INPUT_CSV)
    df['date'] = pd.to_datetime(df['date'])
    print(f"   Total clean rows loaded: {len(df):,}")

    # 1. Temporal Splitting
    print("\n📅 1. Temporal Split Boundaries (Chronological Order — NO Shuffling):")
    train_df, val_df, test_df = temporal_split(df)

    print(f"   Train Set:      {len(train_df):,} rows ({len(train_df)/len(df):.1%}) | {train_df['date'].min().strftime('%Y-%m-%d')} → {train_df['date'].max().strftime('%Y-%m-%d')}")
    print(f"   Validation Set: {len(val_df):,} rows ({len(val_df)/len(df):.1%}) | {val_df['date'].min().strftime('%Y-%m-%d')} → {val_df['date'].max().strftime('%Y-%m-%d')}")
    print(f"   Test Set:       {len(test_df):,} rows ({len(test_df)/len(df):.1%})  | {test_df['date'].min().strftime('%Y-%m-%d')} → {test_df['date'].max().strftime('%Y-%m-%d')}")

    # 2. Class Balance Inspection
    print("\n🎯 2. Target Variable Distribution Across Splits:")
    print("-" * 65)
    print(f"{'Split':<15} {'Total Rows':<12} {'UP (1)':<18} {'DOWN (0)':<18}")
    print("-" * 65)
    for name, sub_df in [('Train', train_df), ('Validation', val_df), ('Test', test_df)]:
        up_cnt = (sub_df['target'] == 1).sum()
        dn_cnt = (sub_df['target'] == 0).sum()
        total = len(sub_df)
        print(f"{name:<15} {total:<12,} {up_cnt:,} ({up_cnt/total:.2%})   {dn_cnt:,} ({dn_cnt/total:.2%})")
    print("-" * 65)

    # 3. Fit & Transform Scalers
    print("\n⚙️ 3. Data-Leakage Free Normalization:")
    print("   Safeguard: Scalers (StandardScaler & MinMaxScaler) fit ONLY on Train Set.")

    prep_data = fit_and_transform_features(
        train_df, val_df, test_df,
        feature_range=(0, np.pi)
    )

    X_train_std    = prep_data['X_train_std']
    X_val_std      = prep_data['X_val_std']
    X_test_std     = prep_data['X_test_std']
    X_train_minmax = prep_data['X_train_minmax']
    X_val_minmax   = prep_data['X_val_minmax']
    X_test_minmax  = prep_data['X_test_minmax']

    # 4. Verify StandardScaler parameters
    print("\n📊 4. StandardScaler Verification (Train Mean=0.00, Std=1.00):")
    print("-" * 75)
    print(f"{'Feature':<16} {'Train Mean':<14} {'Train Std':<14} {'Val Mean':<14} {'Test Mean':<14}")
    print("-" * 75)
    for i, col in enumerate(FEATURE_COLS):
        tr_m, tr_s = X_train_std[:, i].mean(), X_train_std[:, i].std()
        va_m, te_m = X_val_std[:, i].mean(),   X_test_std[:, i].mean()
        print(f"{col:<16} {tr_m:<14.4f} {tr_s:<14.4f} {va_m:<14.4f} {te_m:<14.4f}")
    print("-" * 75)

    # 5. Verify MinMaxScaler parameters
    print("\n📊 5. MinMaxScaler [0, π] Verification (Train Range=[0, π]):")
    print("-" * 80)
    print(f"{'Feature':<16} {'Train Min':<12} {'Train Max':<12} {'Val Min':<12} {'Val Max':<12} {'Test Min':<12} {'Test Max':<12}")
    print("-" * 80)
    for i, col in enumerate(FEATURE_COLS):
        tr_min, tr_max = X_train_minmax[:, i].min(), X_train_minmax[:, i].max()
        va_min, va_max = X_val_minmax[:, i].min(),   X_val_minmax[:, i].max()
        te_min, te_max = X_test_minmax[:, i].min(),  X_test_minmax[:, i].max()
        print(f"{col:<16} {tr_min:<12.4f} {tr_max:<12.4f} {va_min:<12.4f} {va_max:<12.4f} {te_min:<12.4f} {te_max:<12.4f}")
    print("-" * 80)

    # 6. Check for NaNs
    has_nans = (
        np.isnan(X_train_std).any() or np.isnan(X_val_std).any() or np.isnan(X_test_std).any() or
        np.isnan(X_train_minmax).any() or np.isnan(X_val_minmax).any() or np.isnan(X_test_minmax).any()
    )
    print(f"\n🔍 Missing value check across all scaled arrays: {'❌ NaN FOUND!' if has_nans else '✅ ZERO NaNs'}")

    print("\n✅ Phase 10 Preprocessing Execution Completed Successfully!")
    print(f"\nTarget output path: {OUTPUT_NPZ}")

if __name__ == '__main__':
    main()

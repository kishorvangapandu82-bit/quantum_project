"""
============================================================
src/preprocessing.py
NIFTY50-VQC Project — Feature Scaling & Temporal Train/Val/Test Split

PURPOSE:
    1. Temporal (Date-Based) Train / Validation / Test Splitting.
       Strictly chronological — NO random shuffling.
       - Train:      2015-01-29 → 2022-12-30 (1,963 rows, 72.5%)
       - Validation: 2023-01-02 → 2024-12-31 (495 rows, 18.3%)
       - Test:       2025-01-01 → 2025-12-30 (248 rows, 9.2%)

    2. Data Leakage Safeguarded Normalization:
       - Scalers (StandardScaler and MinMaxScaler) are fitted ONLY on Training set.
       - Validation and Test sets are transformed using the training parameters.
       - Standardized features (z-scores) are used for Classical ML & PCA input.
       - MinMax scaled features [0, pi] are used for Quantum Angle Encoding.

Author: NIFTY50-VQC Project
Date:   2026-10-02
============================================================
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler


FEATURE_COLS = [
    'daily_return',
    'sma_5',
    'sma_20',
    'ema_12',
    'ema_26',
    'rsi_14',
    'volatility_5',
    'momentum_5'
]

TARGET_COL = 'target'


def temporal_split(df):
    """
    Split the DataFrame chronologically by date into Train, Validation, and Test sets.

    Split Boundaries:
        Train:      2015–2022 (inclusive)
        Validation: 2023–2024 (inclusive)
        Test:       2025       (inclusive)

    Args:
        df (pd.DataFrame): Must contain 'date' column as datetime or datetime-parsable string.

    Returns:
        tuple: (train_df, val_df, test_df)
    """
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])

    train_df = df[df['date'].dt.year <= 2022].reset_index(drop=True)
    val_df   = df[(df['date'].dt.year >= 2023) & (df['date'].dt.year <= 2024)].reset_index(drop=True)
    test_df  = df[df['date'].dt.year == 2025].reset_index(drop=True)

    return train_df, val_df, test_df


def fit_and_transform_features(train_df, val_df, test_df, feature_range=(0, np.pi)):
    """
    Fit scalers (StandardScaler & MinMaxScaler) ONLY on train_df and transform all splits.

    Data-Leakage Safeguard:
        The scalers calculate mean/std and min/max ONLY from the training set.
        Validation and Test sets are transformed using the learned training parameters.

    Args:
        train_df (pd.DataFrame): Training set containing FEATURE_COLS
        val_df   (pd.DataFrame): Validation set containing FEATURE_COLS
        test_df  (pd.DataFrame): Test set containing FEATURE_COLS
        feature_range (tuple): Range for MinMaxScaler (default: 0 to pi)

    Returns:
        dict: {
            'X_train_std':  np.ndarray,
            'X_val_std':    np.ndarray,
            'X_test_std':   np.ndarray,
            'X_train_minmax': np.ndarray,
            'X_val_minmax':   np.ndarray,
            'X_test_minmax':  np.ndarray,
            'y_train':      np.ndarray,
            'y_val':        np.ndarray,
            'y_test':       np.ndarray,
            'std_scaler':   fitted StandardScaler object,
            'minmax_scaler': fitted MinMaxScaler object,
            'feature_cols': list of feature names
        }
    """
    X_train = train_df[FEATURE_COLS].values
    X_val   = val_df[FEATURE_COLS].values
    X_test  = test_df[FEATURE_COLS].values

    y_train = train_df[TARGET_COL].values
    y_val   = val_df[TARGET_COL].values
    y_test  = test_df[TARGET_COL].values

    # 1. StandardScaler (Mean = 0, Std = 1 on train)
    std_scaler = StandardScaler()
    std_scaler.fit(X_train)

    X_train_std = std_scaler.transform(X_train)
    X_val_std   = std_scaler.transform(X_val)
    X_test_std  = std_scaler.transform(X_test)

    # 2. MinMaxScaler ([0, pi] on train)
    minmax_scaler = MinMaxScaler(feature_range=feature_range)
    minmax_scaler.fit(X_train)

    X_train_minmax = minmax_scaler.transform(X_train)
    X_val_minmax   = minmax_scaler.transform(X_val)
    X_test_minmax  = minmax_scaler.transform(X_test)

    return {
        'X_train_std':    X_train_std,
        'X_val_std':      X_val_std,
        'X_test_std':     X_test_std,
        'X_train_minmax': X_train_minmax,
        'X_val_minmax':   X_val_minmax,
        'X_test_minmax':  X_test_minmax,
        'y_train':        y_train,
        'y_val':          y_val,
        'y_test':         y_test,
        'std_scaler':     std_scaler,
        'minmax_scaler':  minmax_scaler,
        'feature_cols':   FEATURE_COLS
    }

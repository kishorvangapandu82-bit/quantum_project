"""
============================================================
src/features.py
NIFTY50-VQC Project — Technical Feature Engineering

PURPOSE:
    Compute 8 technical indicator features from NIFTY 50
    OHLC data. All features use ONLY information available
    on or before the prediction date — no future leakage.

FEATURES COMPUTED:
    1. daily_return   — (Close_t - Close_{t-1}) / Close_{t-1}
    2. sma_5          — 5-day Simple Moving Average of Close
    3. sma_20         — 20-day Simple Moving Average of Close
    4. ema_12         — 12-day Exponential Moving Average of Close
    5. ema_26         — 26-day Exponential Moving Average of Close
    6. rsi_14         — 14-day Relative Strength Index
    7. volatility_5   — 5-day rolling std of daily_return
    8. momentum_5     — Close_t - Close_{t-5}

WHY ROLLING WINDOWS CREATE NaN VALUES:
    Features that use a rolling window of N days cannot be
    computed for the first (N-1) rows because there is not
    enough historical data yet.

    Example: SMA_20 requires 20 days of Close prices.
    Rows 0–18 (first 19 rows) will have NaN for sma_20.

    This is NOT a data error — it is mathematically correct.
    These NaN rows are DROPPED before model training.
    We report exactly which rows are dropped and why.

LEAKAGE SAFEGUARDS:
    - All features use only Close_t and earlier prices
    - No feature uses Close_{t+1} or any future information
    - The 'target' column is passed through unchanged
    - The 'next_close' column is never present (dropped in Phase 8)

Author: NIFTY50-VQC Project
Date:   2026-10-02
============================================================
"""

import numpy as np
import pandas as pd


# ============================================================
# FEATURE 1: Daily Return
# ============================================================

def compute_daily_return(df):
    """
    Compute the daily percentage return of the closing price.

    Formula:
        daily_return_t = (Close_t - Close_{t-1}) / Close_{t-1}

    What it represents:
        The fractional change in price from the previous trading day.
        A positive value means the price rose today.
        A negative value means the price fell today.

    Why it might be informative:
        Strong positive returns may indicate momentum.
        Strong negative returns may indicate selling pressure.
        Recent return direction can correlate with next-day direction.

    Limitations:
        - Does not capture magnitude in an absolute sense
        - Can be noisy — one bad day does not predict the next
        - Sensitive to extreme events (e.g., COVID crash days)

    NaN values:
        - Row 0 only (no previous day exists)
        - All other rows: computed normally

    Args:
        df (pd.DataFrame): Must contain 'close' column, sorted by date ascending.

    Returns:
        pd.Series: Daily return values (float), first value is NaN.
    """
    return df['close'].pct_change()


# ============================================================
# FEATURE 2 & 3: Simple Moving Average (SMA)
# ============================================================

def compute_sma(df, window):
    """
    Compute Simple Moving Average (SMA) of the closing price.

    Formula:
        SMA_t = (Close_t + Close_{t-1} + ... + Close_{t-N+1}) / N

    What it represents:
        The average closing price over the last N trading days.
        It smooths out daily noise to show the underlying trend.

    Why two SMAs (5 and 20)?
        SMA_5  = short-term trend (about 1 trading week)
        SMA_20 = medium-term trend (about 1 trading month)

        The relationship between short and long SMAs can signal:
        - SMA_5 > SMA_20: price is trending up (bullish)
        - SMA_5 < SMA_20: price is trending down (bearish)
        This crossover is used by traders as a basic signal.

    Limitations:
        - Lagging indicator — reacts to past prices, not future
        - Does not predict reversals
        - More sensitive to recent prices for smaller windows

    NaN values:
        First (N-1) rows will be NaN.
        SMA_5:  first 4 rows are NaN
        SMA_20: first 19 rows are NaN

    Args:
        df (pd.DataFrame): Must contain 'close' column.
        window (int): Rolling window size (e.g., 5 or 20).

    Returns:
        pd.Series: SMA values (float).
    """
    return df['close'].rolling(window=window, min_periods=window).mean()


# ============================================================
# FEATURE 4 & 5: Exponential Moving Average (EMA)
# ============================================================

def compute_ema(df, span):
    """
    Compute Exponential Moving Average (EMA) of the closing price.

    Formula:
        EMA_t = Close_t * alpha + EMA_{t-1} * (1 - alpha)
        where alpha = 2 / (span + 1)

    What it represents:
        Like SMA, but gives MORE weight to recent prices.
        The most recent day has the highest weight.
        Older days have exponentially smaller weights.

    Why two EMAs (12 and 26)?
        EMA_12 = faster EMA (reacts quickly to price changes)
        EMA_26 = slower EMA (more stable, reacts slowly)

        MACD (a popular indicator) = EMA_12 - EMA_26
        When EMA_12 > EMA_26: upward momentum
        When EMA_12 < EMA_26: downward momentum

    Difference from SMA:
        EMA responds faster to recent price changes.
        SMA treats all days in the window equally.

    Limitations:
        - Also a lagging indicator
        - Can produce false signals in choppy/sideways markets

    NaN values:
        First row is NaN (no previous EMA to use).
        Subsequent rows use exponential smoothing from row 1.
        In practice, pandas uses min_periods=1 for EMA by default,
        so NaN only appears for row 0.

    Args:
        df (pd.DataFrame): Must contain 'close' column.
        span (int): EMA span (e.g., 12 or 26).

    Returns:
        pd.Series: EMA values (float).
    """
    return df['close'].ewm(span=span, adjust=False).mean()


# ============================================================
# FEATURE 6: Relative Strength Index (RSI)
# ============================================================

def compute_rsi(df, window=14):
    """
    Compute the Relative Strength Index (RSI).

    Formula:
        Step 1: daily_change = Close_t - Close_{t-1}
        Step 2: gain_t = max(daily_change, 0)  (only positive changes)
                loss_t = max(-daily_change, 0) (only negative changes, positive value)
        Step 3: avg_gain = rolling mean of gain over window days
                avg_loss = rolling mean of loss over window days
        Step 4: RS = avg_gain / avg_loss
        Step 5: RSI = 100 - (100 / (1 + RS))

    What it represents:
        RSI is a momentum oscillator that ranges from 0 to 100.
        - RSI > 70: market is considered OVERBOUGHT
          (price rose a lot recently — may reverse downward)
        - RSI < 30: market is considered OVERSOLD
          (price fell a lot recently — may bounce upward)
        - RSI near 50: balanced momentum

    Why it might be informative:
        Extreme RSI values historically correlate with reversals.
        However, in strong trends RSI can stay extreme for long periods.

    Limitations:
        - RSI alone is not a reliable predictor
        - Overbought/oversold thresholds (30/70) are rules of thumb
        - In strong bull markets, RSI can stay above 70 for months

    NaN values:
        First (window) rows = NaN (not enough history for avg_gain/loss).
        RSI_14: first 14 rows are NaN.

    Args:
        df (pd.DataFrame): Must contain 'close' column.
        window (int): RSI period (default = 14).

    Returns:
        pd.Series: RSI values in range [0, 100].
    """
    delta = df['close'].diff()
    gain  = delta.clip(lower=0)
    loss  = (-delta).clip(lower=0)

    avg_gain = gain.rolling(window=window, min_periods=window).mean()
    avg_loss = loss.rolling(window=window, min_periods=window).mean()

    # Avoid division by zero — if avg_loss is 0, RSI = 100
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))

    # When avg_loss is exactly 0 (all gains), RSI = 100
    rsi = rsi.fillna(100)

    return rsi


# ============================================================
# FEATURE 7: Volatility (Rolling Standard Deviation)
# ============================================================

def compute_volatility(df, window=5):
    """
    Compute rolling volatility as the standard deviation of daily returns.

    Formula:
        daily_return_t = (Close_t - Close_{t-1}) / Close_{t-1}
        volatility_t   = std(daily_return_{t-N+1}, ..., daily_return_t)

    What it represents:
        How much the price has been fluctuating recently.
        High volatility = large, unpredictable price swings.
        Low volatility  = stable, predictable price moves.

    Why it might be informative:
        - High volatility periods (e.g., COVID crash) are often followed
          by more high volatility — "volatility clustering"
        - Low volatility periods may precede breakouts
        - Volatility can help models learn different market regimes

    Limitations:
        - Backward-looking — measures recent variance, not future
        - 5-day window may be too short to be stable
        - Does not indicate direction, only magnitude

    NaN values:
        daily_return has 1 NaN (row 0).
        volatility_5 needs 5 returns → first 5 rows are NaN
        (rows 0–4 will be NaN in the final dataset).

    Args:
        df (pd.DataFrame): Must contain 'close' column.
        window (int): Rolling window (default = 5 days).

    Returns:
        pd.Series: Volatility values (float, always >= 0).
    """
    daily_return = df['close'].pct_change()
    return daily_return.rolling(window=window, min_periods=window).std()


# ============================================================
# FEATURE 8: Momentum
# ============================================================

def compute_momentum(df, window=5):
    """
    Compute price momentum as the difference between current and past close.

    Formula:
        momentum_5_t = Close_t - Close_{t-5}

    What it represents:
        How much the price has moved (in absolute index points) over
        the last 5 trading days.
        Positive = price is higher than 5 days ago (upward momentum)
        Negative = price is lower than 5 days ago (downward momentum)

    Why it might be informative:
        Momentum is one of the most documented effects in finance.
        Assets that have been rising recently tend to continue rising
        (momentum effect) — though this effect is not guaranteed
        and can reverse.

    Difference from daily_return:
        daily_return = 1-day change (fraction)
        momentum_5   = 5-day change (absolute points)
        They capture similar but complementary information.

    Limitations:
        - Absolute value depends on index level (8000 vs 26000)
        - May need normalisation relative to price level
        - Short momentum windows are noisy

    NaN values:
        First 5 rows are NaN (no price from 5 days ago).

    Args:
        df (pd.DataFrame): Must contain 'close' column.
        window (int): Look-back period (default = 5 days).

    Returns:
        pd.Series: Momentum values (float, can be positive or negative).
    """
    return df['close'].diff(window)


# ============================================================
# MAIN: Compute all features together
# ============================================================

def compute_all_features(df, verbose=True):
    """
    Compute all 8 technical features and return an enriched DataFrame.

    This function:
    1. Validates the input DataFrame has required columns
    2. Computes each of the 8 features
    3. Reports NaN counts per feature (expected from rolling windows)
    4. Identifies the minimum rows to drop (due to warmup period)
    5. Returns the enriched DataFrame WITH NaN rows still present
       (caller is responsible for dropping NaN rows)

    IMPORTANT: NaN rows are NOT dropped inside this function.
    This is intentional — we let the caller decide when and how
    to drop them, ensuring the date alignment is preserved for
    train/val/test splitting.

    Args:
        df (pd.DataFrame): Must contain columns: date, open, high, low, close, target
                           Sorted chronologically (oldest first).
        verbose (bool): If True, print detailed feature report.

    Returns:
        pd.DataFrame: Original columns + 8 new feature columns.
                      May contain NaN values in first ~20 rows.

    Feature columns added:
        daily_return, sma_5, sma_20, ema_12, ema_26,
        rsi_14, volatility_5, momentum_5
    """
    # Validate input
    required_cols = ['date', 'close', 'target']
    missing_cols  = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Input DataFrame missing required columns: {missing_cols}")

    if verbose:
        print(f"\n  Input DataFrame shape: {df.shape}")
        print(f"  Date range: {df['date'].min().strftime('%d %b %Y')} → "
              f"{df['date'].max().strftime('%d %b %Y')}")
        print(f"\n  Computing features...")

    df = df.copy()

    # ── Compute each feature ──────────────────────────────────
    feature_definitions = [
        ('daily_return', lambda d: compute_daily_return(d),
         "Daily return: (Close_t - Close_{t-1}) / Close_{t-1}"),
        ('sma_5',        lambda d: compute_sma(d, window=5),
         "SMA_5: 5-day simple moving average of Close"),
        ('sma_20',       lambda d: compute_sma(d, window=20),
         "SMA_20: 20-day simple moving average of Close"),
        ('ema_12',       lambda d: compute_ema(d, span=12),
         "EMA_12: 12-day exponential moving average of Close"),
        ('ema_26',       lambda d: compute_ema(d, span=26),
         "EMA_26: 26-day exponential moving average of Close"),
        ('rsi_14',       lambda d: compute_rsi(d, window=14),
         "RSI_14: 14-day Relative Strength Index"),
        ('volatility_5', lambda d: compute_volatility(d, window=5),
         "Volatility_5: 5-day rolling std of daily returns"),
        ('momentum_5',   lambda d: compute_momentum(d, window=5),
         "Momentum_5: Close_t - Close_{t-5}"),
    ]

    feature_names = []
    for feat_name, feat_fn, feat_desc in feature_definitions:
        df[feat_name] = feat_fn(df)
        feature_names.append(feat_name)
        if verbose:
            nan_count = df[feat_name].isna().sum()
            val_range = (f"{df[feat_name].min():.4f} to {df[feat_name].max():.4f}"
                         if not df[feat_name].isna().all() else "all NaN")
            print(f"    [OK] {feat_name:<15} NaN rows: {nan_count:<4} "
                  f"Range: {val_range}")

    # ── NaN analysis ──────────────────────────────────────────
    if verbose:
        print(f"\n  NaN Analysis (expected from rolling window warmup):")
        print(f"  {'Feature':<15} {'NaN Count':<12} {'Why'}")
        print(f"  {'─'*60}")

        nan_reasons = {
            'daily_return': 'needs 1 previous day',
            'sma_5':        'needs 5 previous days',
            'sma_20':       'needs 20 previous days  ← MAX',
            'ema_12':       'uses all history (exponential)',
            'ema_26':       'uses all history (exponential)',
            'rsi_14':       'needs 14 previous days',
            'volatility_5': 'needs 5 previous returns',
            'momentum_5':   'needs 5 previous days',
        }

        for feat_name in feature_names:
            nan_count = df[feat_name].isna().sum()
            reason    = nan_reasons.get(feat_name, '')
            print(f"  {feat_name:<15} {nan_count:<12} {reason}")

    # Find the row where all features become non-NaN
    first_valid_idx = df[feature_names].dropna().index[0]
    rows_to_drop    = df.index.get_loc(first_valid_idx)

    if verbose:
        print(f"\n  Rows that will be dropped (NaN warmup period):")
        print(f"  First {rows_to_drop} rows (index 0 to {rows_to_drop-1})")
        print(f"  Date range of dropped rows: "
              f"{df['date'].iloc[0].strftime('%d %b %Y')} → "
              f"{df['date'].iloc[rows_to_drop-1].strftime('%d %b %Y')}")
        print(f"\n  Reason: SMA_20 requires 20 days of history.")
        print(f"  The first valid row is: "
              f"{df['date'].iloc[rows_to_drop].strftime('%d %b %Y')}")
        print(f"\n  Remaining rows after dropping: "
              f"{len(df) - rows_to_drop:,}")

    return df, feature_names


# ============================================================
# CONVENIENCE: Compute features and drop NaN rows
# ============================================================

def get_clean_features(df, verbose=True):
    """
    Compute all features and return a clean DataFrame with NaN rows dropped.

    This is the main entry point for the pipeline.
    Call this after loading nifty50_with_target.csv.

    Returns:
        tuple: (df_clean, feature_names)
            df_clean     — DataFrame with all features, no NaN rows
            feature_names — list of 8 feature column names
    """
    df_with_features, feature_names = compute_all_features(df, verbose=verbose)

    # Drop rows where any feature is NaN
    n_before = len(df_with_features)
    df_clean = df_with_features.dropna(subset=feature_names).copy()
    df_clean = df_clean.reset_index(drop=True)
    n_after  = len(df_clean)
    n_dropped = n_before - n_after

    if verbose:
        print(f"\n  Rows dropped (NaN warmup): {n_dropped}")
        print(f"  Rows remaining (clean)   : {n_after:,}")
        print(f"  Final date range: "
              f"{df_clean['date'].min().strftime('%d %b %Y')} → "
              f"{df_clean['date'].max().strftime('%d %b %Y')}")

    return df_clean, feature_names

"""
============================================================
phase19_limitations.py
NIFTY50-VQC Project — Phase 19 Non-Stationarity & Limitations Analysis

PURPOSE:
    Analyze financial market non-stationarity, volatility shifts, and return distributions
    across Train (2015–2022), Validation (2023–2024), and Test (2025) periods.
    Generate fig13_regime_shifts.png and summarize limitations.
============================================================
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')

INPUT_CSV = 'data/processed/nifty50_features.csv'
FIG_DIR   = 'results/figures'
TAB_DIR   = 'results/tables'

os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(TAB_DIR, exist_ok=True)


def main():
    print("=" * 75)
    print("PHASE 19 — FINANCIAL MARKET NON-STATIONARITY & LIMITATIONS ANALYSIS")
    print("=" * 75)

    if not os.path.exists(INPUT_CSV):
        print(f"❌ Error: {INPUT_CSV} not found.")
        return

    df = pd.read_csv(INPUT_CSV)
    df['date'] = pd.to_datetime(df['date'])

    # Add Period Column
    def get_period(year):
        if year <= 2022:
            return 'Train (2015-2022)'
        elif year <= 2024:
            return 'Validation (2023-2024)'
        else:
            return 'Test (2025)'

    df['period'] = df['date'].dt.year.apply(get_period)

    # 1. Summary Statistics per Period
    print("\n📊 1. Volatility and Return Distribution Across Periods:")
    print("-" * 80)
    print(f"{'Period':<25} {'Mean Daily Return':<20} {'Daily Volatility (Std)':<24} {'Annualized Vol':<18}")
    print("-" * 80)

    stats = []
    for name, group in df.groupby('period', sort=False):
        mean_ret = group['daily_return'].mean()
        std_ret  = group['daily_return'].std()
        ann_vol  = std_ret * np.sqrt(252)
        print(f"{name:<25} {mean_ret:<20.4%} {std_ret:<24.4%} {ann_vol:<18.2%}")
        stats.append({
            'Period': name,
            'Mean_Daily_Return': f"{mean_ret:.4%}",
            'Daily_Volatility': f"{std_ret:.4%}",
            'Annualized_Volatility': f"{ann_vol:.2%}"
        })
    print("-" * 80)

    # Save summary
    pd.DataFrame(stats).to_csv(os.path.join(TAB_DIR, 'table7_market_regime_statistics.csv'), index=False)

    # 2. Plot Regime Shifts Figure
    print("\n🎨 2. Generating Regime Shifts Figure...")
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # Rolling Volatility (20-day)
    df['rolling_vol_20'] = df['daily_return'].rolling(20).std() * np.sqrt(252)

    sns.lineplot(data=df, x='date', y='rolling_vol_20', hue='period', palette=['#1f77b4', '#ff7f0e', '#2ca02c'], ax=axes[0], linewidth=1.5)
    axes[0].set_title('20-Day Rolling Annualized Volatility (%) Across Splits', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Annualized Volatility', fontsize=10, fontweight='bold')
    axes[0].grid(True, linestyle=':', alpha=0.6)

    # NIFTY 50 Index Price
    sns.lineplot(data=df, x='date', y='close', hue='period', palette=['#1f77b4', '#ff7f0e', '#2ca02c'], ax=axes[1], linewidth=1.8, legend=False)
    axes[1].set_title('NIFTY 50 Index Closing Price Journey (2015–2025)', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Index Level (Points)', fontsize=10, fontweight='bold')
    axes[1].set_xlabel('Date', fontsize=10, fontweight='bold')
    axes[1].grid(True, linestyle=':', alpha=0.6)

    plt.suptitle('Phase 19 — Financial Market Non-Stationarity & Regime Shift Analysis', fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    fig_path = os.path.join(FIG_DIR, 'fig13_regime_shifts.png')
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   Saved plot: {fig_path}")

    print("\n✅ Phase 19 Limitations Analysis Completed Successfully!")

if __name__ == '__main__':
    main()

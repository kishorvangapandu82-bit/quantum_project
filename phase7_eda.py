"""
============================================================
phase7_eda.py
NIFTY50-VQC Project — Exploratory Data Analysis

PURPOSE:
    Perform comprehensive exploratory analysis on the combined
    NIFTY 50 dataset (2015–2025) and save publication-quality
    plots to results/figures/.

    This script is READ-ONLY with respect to input data.
    It creates new plot files but never modifies any data.

WHAT THIS SCRIPT PRODUCES:
    1. Console: Full statistical summary of the dataset
    2. Console: Trading days per year
    3. Console: Preliminary target class distribution (UP vs DOWN)
    4. Plot:    eda_01_price_trend.png    — NIFTY 50 price history
    5. Plot:    eda_02_returns_dist.png   — Daily returns distribution
    6. Plot:    eda_03_annual_returns.png — Year-by-year returns bar chart

IMPORTANT NOTE:
    The target variable (UP/DOWN) computed here is PRELIMINARY
    and for EDA purposes only.
    The official target will be created in Phase 8 with
    full data-leakage safeguards.

Author: NIFTY50-VQC Project
Date:   2026-10-02
============================================================
"""

import os
import sys
import warnings
warnings.filterwarnings('ignore')

# Force UTF-8 output for Windows terminals
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend — saves to file without display
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib.dates as mdates
import seaborn as sns

# ── Paths ─────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
PROCESSED_CSV = os.path.join(PROJECT_ROOT, "data", "processed", "nifty50_combined.csv")
FIGURES_DIR   = os.path.join(PROJECT_ROOT, "results", "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

# ── Plot style ────────────────────────────────────────────
plt.rcParams.update({
    'figure.dpi': 150,
    'savefig.dpi': 150,
    'font.family': 'DejaVu Sans',
    'font.size': 11,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'axes.titleweight': 'bold',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': '--',
    'figure.facecolor': '#FAFAFA',
    'axes.facecolor': '#FAFAFA',
    'lines.linewidth': 1.5,
    'legend.framealpha': 0.8,
})

# ── Colour palette ────────────────────────────────────────
C_BLUE   = '#2563EB'
C_GREEN  = '#16A34A'
C_RED    = '#DC2626'
C_ORANGE = '#EA580C'
C_PURPLE = '#7C3AED'
C_GREY   = '#6B7280'


# ============================================================
# STEP 1: Load the processed dataset
# ============================================================
print("=" * 65)
print("PHASE 7 — EXPLORATORY DATA ANALYSIS")
print("=" * 65)

print(f"\n[STEP 1] Loading processed dataset...")
if not os.path.exists(PROCESSED_CSV):
    print(f"ERROR: Processed CSV not found at:\n  {PROCESSED_CSV}")
    print("Please complete Phase 6 first (run data_loader.py).")
    sys.exit(1)

df = pd.read_csv(PROCESSED_CSV, parse_dates=['date'])
print(f"  Loaded: {len(df):,} rows x {len(df.columns)} columns")
print(f"  File  : {PROCESSED_CSV}")


# ============================================================
# STEP 2: Basic dataset properties
# ============================================================
print(f"\n{'─' * 65}")
print("[STEP 2] DATASET PROPERTIES")
print(f"{'─' * 65}")

print(f"\n  Shape             : {df.shape[0]:,} rows x {df.shape[1]} columns")
print(f"  Columns           : {list(df.columns)}")
print(f"\n  Data Types:")
for col, dtype in df.dtypes.items():
    print(f"    {col:<10} : {dtype}")

print(f"\n  Date range        : {df['date'].min().strftime('%d %b %Y')} → "
      f"{df['date'].max().strftime('%d %b %Y')}")
print(f"  Total trading days: {len(df):,}")
print(f"  Calendar span     : ~{(df['date'].max() - df['date'].min()).days / 365.25:.1f} years")

# Missing values
missing = df.isnull().sum()
print(f"\n  Missing Values:")
for col, cnt in missing.items():
    status = "CLEAN" if cnt == 0 else f"WARNING: {cnt} missing!"
    print(f"    {col:<10} : {cnt}  ({status})")

# Duplicate dates
dup_count = df['date'].duplicated().sum()
print(f"\n  Duplicate Dates   : {dup_count}  ({'CLEAN' if dup_count == 0 else 'WARNING!'})")


# ============================================================
# STEP 3: Statistical summary
# ============================================================
print(f"\n{'─' * 65}")
print("[STEP 3] STATISTICAL SUMMARY OF OHLC PRICES")
print(f"{'─' * 65}")

summary = df[['open', 'high', 'low', 'close']].describe()
print(f"\n{summary.round(2).to_string()}")

print(f"\n  Interpretation:")
print(f"  - NIFTY 50 ranged from {df['close'].min():.2f} to {df['close'].max():.2f} "
      f"over the 11-year period")
print(f"  - Mean closing price: {df['close'].mean():.2f}")
print(f"  - Std of closing price: {df['close'].std():.2f} "
      f"(high std is expected for a growing index)")


# ============================================================
# STEP 4: Trading days per year
# ============================================================
print(f"\n{'─' * 65}")
print("[STEP 4] TRADING DAYS PER YEAR")
print(f"{'─' * 65}")

df['year'] = df['date'].dt.year
yearly_counts = df.groupby('year').size().reset_index(name='trading_days')
yearly_open   = df.groupby('year')['close'].first().reset_index(name='year_open_close')
yearly_close  = df.groupby('year')['close'].last().reset_index(name='year_end_close')

print(f"\n  {'Year':<6} {'Trading Days':<15} {'Start Close':<15} {'End Close':<15} {'Annual Return'}")
print(f"  {'─'*65}")

annual_returns = []
for _, row in yearly_counts.iterrows():
    yr = row['year']
    td = row['trading_days']
    yr_start = df[df['year'] == yr]['close'].iloc[0]
    yr_end   = df[df['year'] == yr]['close'].iloc[-1]
    ann_ret  = ((yr_end - yr_start) / yr_start) * 100
    annual_returns.append({'year': yr, 'trading_days': td,
                           'start_close': yr_start, 'end_close': yr_end,
                           'annual_return_pct': ann_ret})
    arrow = "↑" if ann_ret >= 0 else "↓"
    print(f"  {yr:<6} {td:<15} {yr_start:<15.2f} {yr_end:<15.2f} "
          f"{arrow} {ann_ret:+.1f}%")

df_annual = pd.DataFrame(annual_returns)
total_ret = ((df['close'].iloc[-1] - df['close'].iloc[0]) / df['close'].iloc[0]) * 100
print(f"\n  Total return (Jan 2015 → Dec 2025): {total_ret:+.1f}%")
print(f"  Starting close (01 Jan 2015) : {df['close'].iloc[0]:.2f}")
print(f"  Ending close  (31 Dec 2025) : {df['close'].iloc[-1]:.2f}")


# ============================================================
# STEP 5: Daily returns
# ============================================================
print(f"\n{'─' * 65}")
print("[STEP 5] DAILY RETURNS ANALYSIS")
print(f"{'─' * 65}")

# Calculate daily returns: (Close_t - Close_{t-1}) / Close_{t-1}
df['daily_return'] = df['close'].pct_change()

# Drop the first NaN (no previous day for day 1)
returns = df['daily_return'].dropna()

print(f"\n  Total daily returns computed : {len(returns):,}")
print(f"  Mean daily return            : {returns.mean()*100:+.4f}%")
print(f"  Std of daily returns         : {returns.std()*100:.4f}%")
print(f"  Min daily return (worst day) : {returns.min()*100:+.2f}%")
print(f"  Max daily return (best day)  : {returns.max()*100:+.2f}%")
print(f"  Median daily return          : {returns.median()*100:+.4f}%")
print(f"  Skewness                     : {returns.skew():.4f}")
print(f"  Kurtosis (excess)            : {returns.kurtosis():.4f}")

# Identify best and worst days
worst_day = df.loc[df['daily_return'].idxmin(), ['date', 'daily_return', 'close']]
best_day  = df.loc[df['daily_return'].idxmax(), ['date', 'daily_return', 'close']]

print(f"\n  Worst trading day: {worst_day['date'].strftime('%d %b %Y')} | "
      f"Return: {worst_day['daily_return']*100:+.2f}% | Close: {worst_day['close']:.2f}")
print(f"  Best  trading day: {best_day['date'].strftime('%d %b %Y')} | "
      f"Return: {best_day['daily_return']*100:+.2f}% | Close: {best_day['close']:.2f}")


# ============================================================
# STEP 6: Preliminary target class distribution
# ============================================================
print(f"\n{'─' * 65}")
print("[STEP 6] PRELIMINARY TARGET CLASS DISTRIBUTION")
print(f"{'─' * 65}")

print("""
  NOTE: This is a PRELIMINARY analysis for EDA purposes only.
  The official target variable will be created in Phase 8
  with full data-leakage safeguards.

  Definition: Target_t = 1 if Close_(t+1) > Close_t
                         0 otherwise
""")

# Shift close prices to create next-day target
# We do NOT use this for training — just for EDA understanding
df_temp = df.copy()
df_temp['next_close'] = df_temp['close'].shift(-1)
df_temp['target_prelim'] = (df_temp['next_close'] > df_temp['close']).astype(int)
df_temp = df_temp.dropna(subset=['next_close'])  # Drop last row

n_total = len(df_temp)
n_up    = df_temp['target_prelim'].sum()
n_down  = n_total - n_up
pct_up  = n_up / n_total * 100
pct_down = n_down / n_total * 100

print(f"  Total labelled days : {n_total:,}")
print(f"  UP   (label = 1)    : {n_up:,}  ({pct_up:.1f}%)")
print(f"  DOWN (label = 0)    : {n_down:,}  ({pct_down:.1f}%)")
print(f"\n  Class imbalance ratio (UP/DOWN): {pct_up/pct_down:.3f}")

if abs(pct_up - 50) < 5:
    print(f"\n  Assessment: Classes are approximately balanced (within 5% of 50/50).")
elif abs(pct_up - 50) < 10:
    print(f"\n  Assessment: Mild class imbalance detected (within 10% of 50/50).")
    print(f"  This should be monitored but may not require special handling.")
else:
    print(f"\n  Assessment: Significant class imbalance detected.")
    print(f"  Consider using balanced accuracy or class weights in models.")

# Year-by-year class distribution
print(f"\n  Year-by-year class distribution:")
print(f"  {'Year':<6} {'UP':<8} {'DOWN':<8} {'UP%':<8} {'DOWN%'}")
print(f"  {'─'*45}")
for yr in sorted(df_temp['year'].unique()):
    yr_data = df_temp[df_temp['year'] == yr]
    yr_up   = yr_data['target_prelim'].sum()
    yr_down = len(yr_data) - yr_up
    yr_pct_up = yr_up / len(yr_data) * 100
    print(f"  {yr:<6} {yr_up:<8} {yr_down:<8} {yr_pct_up:<8.1f}% {100-yr_pct_up:.1f}%")


# ============================================================
# PLOT 1: NIFTY 50 Price History (2015–2025)
# ============================================================
print(f"\n{'─' * 65}")
print("[STEP 7] Generating Plot 1 — NIFTY 50 Price History...")
print(f"{'─' * 65}")

fig, axes = plt.subplots(2, 1, figsize=(14, 9),
                          gridspec_kw={'height_ratios': [3, 1], 'hspace': 0.08})

# ── Top panel: Price ──────────────────────────────────────
ax1 = axes[0]
ax1.plot(df['date'], df['close'], color=C_BLUE, linewidth=1.2, label='NIFTY 50 Close')
ax1.fill_between(df['date'], df['close'], df['close'].min() * 0.95,
                 alpha=0.08, color=C_BLUE)

# Highlight key events
covid_start = pd.Timestamp('2020-01-20')
covid_low   = pd.Timestamp('2020-03-23')
covid_end   = pd.Timestamp('2020-12-31')
ax1.axvspan(covid_start, covid_end, alpha=0.07, color=C_RED, label='COVID-19 Period')
ax1.annotate('COVID-19\nCrash (Mar 2020)', xy=(covid_low, 7610),
             xytext=(pd.Timestamp('2018-06-01'), 7000),
             fontsize=9, color=C_RED,
             arrowprops=dict(arrowstyle='->', color=C_RED, lw=1.2),
             ha='center')

# Train/Val/Test split lines
train_end  = pd.Timestamp('2022-12-31')
val_end    = pd.Timestamp('2024-12-31')
ax1.axvline(x=train_end, color=C_ORANGE, linestyle='--', linewidth=1.5, alpha=0.8)
ax1.axvline(x=val_end,   color=C_PURPLE, linestyle='--', linewidth=1.5, alpha=0.8)

ax1.text(pd.Timestamp('2016-06-01'), ax1.get_ylim()[1] if ax1.get_ylim()[1] > 0 else 25000,
         'TRAIN\n(2015–2022)', ha='center', fontsize=9, color=C_ORANGE,
         bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7))
ax1.text(pd.Timestamp('2023-07-01'), 23000,
         'VAL\n(2023–24)', ha='center', fontsize=9, color=C_PURPLE,
         bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7))
ax1.text(pd.Timestamp('2025-07-01'), 21000,
         'TEST\n(2025)', ha='center', fontsize=9, color=C_GREEN,
         bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7))

ax1.set_title('NIFTY 50 Daily Closing Price — 2015 to 2025\n'
              '(with Train / Validation / Test Split boundaries)',
              fontsize=13, fontweight='bold', pad=12)
ax1.set_ylabel('Index Value (Points)', fontsize=11)
ax1.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f'{x:,.0f}'))
ax1.legend(loc='upper left', fontsize=9)
ax1.set_xlim(df['date'].min(), df['date'].max())

# ── Bottom panel: Daily returns ───────────────────────────
ax2 = axes[1]
colors_ret = [C_GREEN if r >= 0 else C_RED for r in df['daily_return'].fillna(0)]
ax2.bar(df['date'], df['daily_return'] * 100, color=colors_ret, width=2, alpha=0.6)
ax2.axhline(y=0, color='black', linewidth=0.8, alpha=0.5)
ax2.set_ylabel('Daily Return (%)', fontsize=10)
ax2.set_xlabel('Date', fontsize=11)
ax2.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f'{x:+.0f}%'))
ax2.set_xlim(df['date'].min(), df['date'].max())
ax2.xaxis.set_major_locator(mdates.YearLocator())
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

plt.tight_layout()
plot1_path = os.path.join(FIGURES_DIR, "eda_01_price_trend.png")
fig.savefig(plot1_path, bbox_inches='tight', facecolor='#FAFAFA')
plt.close(fig)
print(f"  Saved: {plot1_path}")


# ============================================================
# PLOT 2: Daily Returns Distribution
# ============================================================
print(f"\n[STEP 8] Generating Plot 2 — Daily Returns Distribution...")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# ── Left: Histogram with KDE ──────────────────────────────
ax = axes[0]
returns_clean = df['daily_return'].dropna() * 100
ax.hist(returns_clean, bins=80, color=C_BLUE, alpha=0.65,
        edgecolor='white', linewidth=0.3, density=True)
returns_clean.plot.kde(ax=ax, color=C_ORANGE, linewidth=2.0, label='KDE')

# Normal distribution overlay for comparison
from scipy import stats
mu, sigma = returns_clean.mean(), returns_clean.std()
x_range = np.linspace(returns_clean.min(), returns_clean.max(), 300)
ax.plot(x_range, stats.norm.pdf(x_range, mu, sigma),
        color=C_RED, linewidth=1.5, linestyle='--', label=f'Normal(μ={mu:.3f}, σ={sigma:.3f})')

ax.axvline(x=0, color='black', linewidth=1.0, alpha=0.6, linestyle='-')
ax.axvline(x=mu, color=C_ORANGE, linewidth=1.5, linestyle=':', label=f'Mean = {mu:.4f}%')
ax.set_title('Distribution of Daily Returns\n(Histogram + KDE vs Normal Distribution)',
             fontweight='bold')
ax.set_xlabel('Daily Return (%)')
ax.set_ylabel('Density')
ax.legend(fontsize=9)

stats_text = (f"n = {len(returns_clean):,}\n"
              f"Mean = {mu:+.4f}%\n"
              f"Std = {sigma:.4f}%\n"
              f"Skew = {returns_clean.skew():.3f}\n"
              f"Kurt = {returns_clean.kurtosis():.3f}")
ax.text(0.97, 0.97, stats_text, transform=ax.transAxes,
        fontsize=9, va='top', ha='right',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.85))

# ── Right: Q-Q Plot ───────────────────────────────────────
ax2 = axes[1]
(osm, osr), (slope, intercept, r) = stats.probplot(returns_clean, dist='norm')
ax2.scatter(osm, osr, color=C_BLUE, alpha=0.35, s=6, label='Observed')
ax2.plot(osm, slope * np.array(osm) + intercept,
         color=C_RED, linewidth=1.5, label='Normal reference line')
ax2.set_title('Q-Q Plot: Daily Returns vs Normal Distribution\n'
              '(Deviation from line = departure from normality)',
              fontweight='bold')
ax2.set_xlabel('Theoretical Quantiles (Normal)')
ax2.set_ylabel('Sample Quantiles (Daily Returns)')
ax2.legend(fontsize=9)
ax2.text(0.05, 0.95, f'R² = {r**2:.4f}', transform=ax2.transAxes,
         fontsize=10, va='top',
         bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.85))

plt.tight_layout()
plot2_path = os.path.join(FIGURES_DIR, "eda_02_returns_dist.png")
fig.savefig(plot2_path, bbox_inches='tight', facecolor='#FAFAFA')
plt.close(fig)
print(f"  Saved: {plot2_path}")


# ============================================================
# PLOT 3: Annual Returns Bar Chart
# ============================================================
print(f"\n[STEP 9] Generating Plot 3 — Annual Returns...")

fig, ax = plt.subplots(figsize=(12, 5))

bar_colors = [C_GREEN if r >= 0 else C_RED for r in df_annual['annual_return_pct']]
bars = ax.bar(df_annual['year'], df_annual['annual_return_pct'],
              color=bar_colors, edgecolor='white', linewidth=0.8, width=0.65, alpha=0.85)

# Add value labels on bars
for bar, val in zip(bars, df_annual['annual_return_pct']):
    y_pos = val + 0.5 if val >= 0 else val - 1.5
    ax.text(bar.get_x() + bar.get_width() / 2, y_pos,
            f'{val:+.1f}%', ha='center', va='bottom' if val >= 0 else 'top',
            fontsize=9, fontweight='bold', color='#1F2937')

ax.axhline(y=0, color='black', linewidth=1.0, alpha=0.7)
ax.set_title('NIFTY 50 Annual Returns (2015–2025)\n'
             'Green = positive year | Red = negative year',
             fontweight='bold')
ax.set_xlabel('Year')
ax.set_ylabel('Annual Return (%)')
ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f'{x:+.0f}%'))
ax.set_xticks(df_annual['year'])

# Shade train / val / test regions
ax.axvspan(2014.5, 2022.5, alpha=0.06, color=C_ORANGE, label='Train (2015–2022)')
ax.axvspan(2022.5, 2024.5, alpha=0.06, color=C_PURPLE, label='Validation (2023–2024)')
ax.axvspan(2024.5, 2025.5, alpha=0.06, color=C_GREEN,  label='Test (2025)')
ax.legend(fontsize=9)

plt.tight_layout()
plot3_path = os.path.join(FIGURES_DIR, "eda_03_annual_returns.png")
fig.savefig(plot3_path, bbox_inches='tight', facecolor='#FAFAFA')
plt.close(fig)
print(f"  Saved: {plot3_path}")


# ============================================================
# FINAL SUMMARY
# ============================================================
print(f"\n{'=' * 65}")
print("PHASE 7 EDA — COMPLETE SUMMARY")
print(f"{'=' * 65}")
print(f"""
  Dataset Overview:
  ─────────────────────────────────────────────────────
  Total trading days    : {len(df):,}
  Date range            : 01 Jan 2015 → 31 Dec 2025
  Columns               : date, open, high, low, close
  Missing values        : 0
  Duplicate dates       : 0

  Price Summary:
  ─────────────────────────────────────────────────────
  Starting close (Jan 2015) : {df['close'].iloc[0]:>10,.2f}
  Ending close  (Dec 2025)  : {df['close'].iloc[-1]:>10,.2f}
  Total return (11 years)   : {total_ret:>+9.1f}%
  All-time high close       : {df['close'].max():>10,.2f}
  All-time low  close       : {df['close'].min():>10,.2f}

  Daily Returns:
  ─────────────────────────────────────────────────────
  Mean daily return    : {returns.mean()*100:+.4f}%
  Std daily return     : {returns.std()*100:.4f}%
  Skewness             : {returns.skew():.4f}
  Excess Kurtosis      : {returns.kurtosis():.4f}
  (Kurtosis > 0 = fat tails — common in financial data)

  Target Distribution (Preliminary):
  ─────────────────────────────────────────────────────
  UP days   (label=1)  : {n_up:,}  ({pct_up:.1f}%)
  DOWN days (label=0)  : {n_down:,}  ({pct_down:.1f}%)

  Plots Saved:
  ─────────────────────────────────────────────────────
  eda_01_price_trend.png    — NIFTY 50 price + split markers
  eda_02_returns_dist.png   — Returns histogram + Q-Q plot
  eda_03_annual_returns.png — Year-by-year bar chart
  Location: results/figures/
""")
print("=" * 65)

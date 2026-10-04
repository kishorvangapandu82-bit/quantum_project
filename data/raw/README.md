# data/raw/ — Raw NIFTY 50 CSV Files

This directory contains the original, unmodified NIFTY 50 daily price data.

## Contents

- One CSV file per year: 2015.csv through 2025.csv (11 files total)
- Source: National Stock Exchange of India (NSE) historical data
- Format: Daily OHLCV data

## Column Descriptions

| Column | Description |
|--------|-------------|
| Date | Trading date (DD-Mon-YYYY or YYYY-MM-DD format) |
| Open | Opening index level |
| High | Intraday high |
| Low | Intraday low |
| Close | Closing index level (used for all calculations) |
| Volume | Number of shares traded |
| Shares Traded | Shares traded (may duplicate Volume) |
| Turnover | Total turnover value |

## Important Rules

> **NEVER modify files in this directory.**
> Raw data is the immutable source of truth. All transformations write to data/processed/.
> If raw files are corrupted or changed, pipeline results cannot be reproduced.

## Loading Raw Files

`python
from src.data_loader import load_nifty50_data
df = load_nifty50_data(data_dir='data/raw')
`

---

*NIFTY50-VQC Project | data/raw/ | Last updated: 2026-10-03*

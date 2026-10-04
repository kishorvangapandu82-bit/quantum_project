# Research Methodology
## Classifying Future Market Direction Using a Variational Quantum Classifier (VQC)

> **Status:** COMPLETE — All experiments executed and results recorded (2026-10-03).
> **Important:** Only actual experimental results are inserted into this document. No results are fabricated or assumed.

---

## 1. Problem Statement

This study investigates whether a Variational Quantum Classifier (VQC) can classify the next-day price direction of the NIFTY 50 index using historical market features. The classification task is binary:

- **Class 1 (UP):** The next trading day's closing price is strictly higher than the current day's closing price.
- **Class 0 (DOWN):** The next trading day's closing price is equal to or lower than the current day's closing price.

This is a classification problem, not a price prediction problem. The model outputs a direction label, not a price value.

---

## 2. Research Question

> *Can a Variational Quantum Classifier classify the next-day direction of the NIFTY 50 market using historical market features, and how does its performance compare with conventional machine-learning models?*

The comparison is objective and numerical. No model is ranked or labelled as superior.

---

## 3. Dataset

- **Index:** NIFTY 50 (National Stock Exchange of India, 50-stock index)
- **Period:** 2015-01-01 to 2025-12-31 (pending data inspection)
- **Frequency:** Daily (trading days only)
- **Source:** User-provided CSV files (one per year)
- **Expected columns:** Date, Open, High, Low, Close, Volume, Shares Traded, Turnover

- Total clean rows: 2,706 (after 19-day warmup dropped)
- Date range: 2015-01-29 to 2025-12-30
- Yearly files: 11 (one per year, 2015-2025)

---

## 4. Data Preprocessing


- Standardise column names across yearly CSV files
- Parse and validate date column
- Convert OHLCV columns to numeric
- Detect and handle missing values
- Detect and handle duplicate dates
- Sort chronologically
- Combine yearly files into one dataset

Raw files are never modified. A processed file is saved separately.

---

## 5. Target Construction

The binary target variable is defined as:

```
Target_t = 1   if   Close_(t+1) > Close_t
Target_t = 0   otherwise
```

**Data leakage prevention:**
- Features on day `t` use ONLY information available up to and including day `t`
- The closing price of day `t+1` is ONLY used to construct the label, never as a feature
- The final row (last trading day) is dropped because no next-day close exists

---

## 6. Feature Engineering


Eight initial features computed from OHLCV data:

| Feature | Formula | Window |
|---------|---------|--------|
| Daily Return | (Close_t - Close_{t-1}) / Close_{t-1} | 1 day |
| SMA_5 | Simple Moving Average | 5 days |
| SMA_20 | Simple Moving Average | 20 days |
| EMA_12 | Exponential Moving Average | 12 days |
| EMA_26 | Exponential Moving Average | 26 days |
| RSI_14 | Relative Strength Index | 14 days |
| Volatility | Rolling standard deviation of returns | 5 days |
| Momentum_5 | Close_t - Close_{t-5} | 5 days |

All features use only past and current information — no future information is used.

---

## 7. Train / Validation / Test Strategy

Chronological splitting is used. Random shuffling of time-series data is NOT performed, as it would introduce data leakage by allowing future data to inform past predictions.

| Split | Period | Purpose | Rows | UP % | DOWN % |
|-------|--------|---------|------|------|--------|
| Training | 2015–2022 | Fit all models | 1,963 (72.5%) | 53.49% | 46.51% |
| Validation | 2023–2024 | Model selection and hyperparameter decisions | 495 (18.3%) | 56.16% | 43.84% |
| Test | 2025 | Final unbiased evaluation (touched only once) | 248 (9.2%) | 50.00% | 50.00% |

---

## 8. Classical Baselines Results (2025 Test Set)

Three classical ML models were trained on the 2015–2022 Training set, hyperparameter-tuned on Validation (2023–2024), and evaluated on the 2025 Test set:

| Model | Best Hyperparameters | Test Accuracy | Precision | Recall | F1-Score | ROC-AUC | Balanced Acc |
|-------|----------------------|---------------|-----------|--------|----------|---------|--------------|
| **Logistic Regression** | `C=0.001` | **50.00%** | 0.5000 | 0.9919 | 0.6649 | 0.5223 | 50.00% |
| **SVM (Linear)** | `kernel='linear', C=0.1` | **50.00%** | 0.5000 | 1.0000 | 0.6667 | 0.4725 | 50.00% |
| **Random Forest** | `n_estimators=50, max_depth=3` | **52.42%** | 0.5469 | 0.2823 | 0.3723 | 0.5306 | 52.42% |

All models were evaluated strictly on standardized features without data leakage.

---

## 9. Feature Scaling

**StandardScaler** (from scikit-learn) is applied:

```
z = (x - mean) / std
```

**Critical rule:** The scaler is fit ONLY on the training set. The learned mean and standard deviation are then applied to transform the validation and test sets. Fitting the scaler on the full dataset would constitute data leakage.

---

## 10. Dimensionality Reduction (PCA)

Principal Component Analysis (PCA) is applied to the quantum pipeline only.

- **Input:** 8 standardised features
- **Output:** 4 principal components
- **Rationale:** Each principal component maps to one qubit. Fewer dimensions reduce quantum simulation complexity.

PCA is fit ONLY on the training set. The learned transformation is applied to validation and test sets.

- PC1 (Q0): 50.08% | PC2 (Q1): 24.13% | PC3 (Q2): 13.17% | PC4 (Q3): 7.51% | Cumulative: 94.89%

---

## 11. Quantum Feature Encoding


Each of the 4 principal components is encoded into one qubit using a feature map. The feature map translates classical numerical values into quantum rotation angles applied to qubits.

---

## 12. VQC Architecture

- **Qubits:** 4 (Q0, Q1, Q2, Q3)
- **Feature Map:** `ZZFeatureMap` (`zz_feature_map`, 1 repetition, linear entanglement)
- **Ansatz:** `RealAmplitudes` (`real_amplitudes`, 1 repetition, linear entanglement)
- **Trainable Parameters:** 8 variational parameters ($\theta_0 \dots \theta_7$)
- **Optimizer:** `SPSA` (Simultaneous Perturbation Stochastic Approximation)
- **Iterations:** 40
- **Simulator:** `StatevectorSampler` (Qiskit Primitives)

---

## 13. VQC Training & Hyperparameter Selection

Architecture tuning was performed on the Validation set across 5 candidate configurations:

| Config ID | Feature Map | Ansatz | Reps | Optimizer | Val Accuracy | Val F1-Score |
|-----------|-------------|--------|------|-----------|--------------|--------------|
| VQC-1 | ZZ | RealAmplitudes | 1 | COBYLA | 51.31% | 0.6119 |
| VQC-2 | ZZ | RealAmplitudes | 2 | COBYLA | 46.87% | 0.3869 |
| VQC-3 | Z | RealAmplitudes | 1 | COBYLA | 51.52% | 0.5312 |
| VQC-4 | ZZ | EfficientSU2 | 1 | COBYLA | 46.06% | 0.3472 |
| **VQC-5 (Selected)** | **ZZ** | **RealAmplitudes** | **1** | **SPSA** | **53.74%** | **0.6264** |

---

## 14. Evaluation Metrics

All models were evaluated strictly on the **2025 Test Set** (248 trading days) using:
- Accuracy, Precision, Recall, F1-Score, ROC-AUC, and Balanced Accuracy.

---

## 15. Experimental Environment

- Python 3.11.9, Qiskit 2.5.2, Qiskit Machine Learning 0.9.1
- Windows 11 Home 64-bit, Intel Core i7-13650HX
- Simulator: Qiskit `StatevectorSampler`

---

## 16. Results — Master Model Comparison (2025 Test Set)

| Model Name | Accuracy | Precision | Recall | F1-Score | ROC-AUC | Balanced Acc |
|------------|----------|-----------|--------|----------|---------|--------------|
| **VQC (4 Qubits)** | **48.39%** | 0.4908 | 0.8629 | 0.6257 | 0.4906 | 48.39% |
| **Logistic Regression** | **50.00%** | 0.5000 | 0.9919 | **0.6649** | 0.5223 | 50.00% |
| **SVM (Linear)** | **50.00%** | 0.5000 | 1.0000 | **0.6667** | 0.4725 | 50.00% |
| **Random Forest** | **52.42%** | **0.5469** | 0.2823 | 0.3723 | **0.5306** | **52.42%** |

*All results recorded from empirical execution without data leakage.*

---

## 17. Limitations


- Market non-stationarity: Statistical properties of financial time series change over time
- Financial noise: Daily price movements contain substantial random variation
- Limited feature set: Only 8 technical indicators used
- No transaction costs modelled
- No trading strategy backtest performed
- Quantum simulation only — not tested on real quantum hardware
- Small qubit count (4) due to simulation constraints
- Results are specific to the NIFTY 50 index and the 2015–2025 period
- Past performance does not guarantee future performance

---

## 18. Conclusion

Empirical evaluation on the 2025 Test Set (248 trading days) shows no statistically significant performance difference between the 4-Qubit VQC and classical baselines. All models operate near random-chance accuracy. McNemar test confirms no model is statistically superior (all p-values > 0.50). Market non-stationarity (volatility regime shift: 17.52% training to 11.81% test) partially explains the difficulty. Results are consistent with the Efficient Market Hypothesis on short-term index prediction.

---

**Important Disclaimer:**  
This is an academic machine-learning and quantum machine-learning classification experiment. It is NOT a financial advisory system. Predictions should not be used for actual trading decisions.

---

*Last updated: 2026-10-03 (All phases complete)*

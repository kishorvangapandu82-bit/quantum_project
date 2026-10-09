# Empirical Evaluation of Variational Quantum Classifiers (VQC) for Daily Stock Market Direction Prediction: A Comparative Study on NIFTY 50 (2015–2025)

---

## Executive Abstract

Predicting daily stock market direction is a classic challenge in financial engineering due to high noise-to-signal ratios, non-stationarity, and regime shifts. This study presents a rigorous empirical evaluation of a **4-Qubit Variational Quantum Classifier (VQC)** for binary next-day price direction classification ($y_t \in \{0, 1\}$) on the NIFTY 50 index using 11 years of historical daily market data (2015–2025; 2,706 trading days). 

Using a strict chronological split (Training: 2015–2022, 1,963 days; Validation: 2023–2024, 495 days; Test: 2025, 248 days) and data-leakage-free feature scaling, eight technical indicators were reduced via Principal Component Analysis (PCA) to 4 principal components capturing **94.89% cumulative variance**. A 4-qubit quantum architecture featuring a `ZZFeatureMap` for non-linear feature mapping and a `RealAmplitudes` variational ansatz trained via SPSA optimization achieved **48.39% accuracy** and **0.6257 F1-score** on the untouched 2025 Test Set. 

Benchmarking against classical machine learning models (Logistic Regression: 50.00%, Random Forest: 52.42%) and McNemar's statistical significance test ($p = 0.6636$) reveals **no statistically significant advantage** for current 4-qubit VQC models over classical baselines in daily market direction forecasting.

---

## 1. Introduction & Problem Formulation

Predicting whether a financial index will close higher or lower on the next trading day is formulated as a binary classification problem:

$$\text{Target}_t = \begin{cases} 1 & \text{if } \text{Close}_{t+1} > \text{Close}_t \\ 0 & \text{otherwise} \end{cases}$$

### Data Leakage Safeguards:
1. Features on day $t$ use **only** information available on or before day $t$.
2. MinMaxScaler ($[0, \pi]$) and StandardScaler parameters are fit **strictly on the Training set (2015–2022)**.
3. No future information is introduced during feature calculation or PCA transformation.

---

## 2. Dataset & Feature Engineering

The dataset comprises 11 consolidated yearly files (2,706 clean rows after 19-day SMA warmup alignment).

### Technical Feature Set:
1. `daily_return`: $(Close_t - Close_{t-1}) / Close_{t-1}$
2. `sma_5`: 5-day Simple Moving Average
3. `sma_20`: 20-day Simple Moving Average
4. `ema_12`: 12-day Exponential Moving Average
5. `ema_26`: 26-day Exponential Moving Average
6. `rsi_14`: 14-day Relative Strength Index
7. `volatility_5`: 5-day rolling return standard deviation
8. `momentum_5`: 5-day price change ($Close_t - Close_{t-5}$)

---

## 3. Dimensionality Reduction & Quantum Encoding

To map 8 technical features onto a 4-qubit simulator, PCA was fitted on standardized training features ($X_{\text{train\_std}}$):

| Component | Qubit Mapping | Explained Variance (%) | Cumulative Variance (%) | Primary Loadings |
| :--- | :--- | :--- | :--- | :--- |
| **PC1** | **Qubit 0** | **50.08%** | 50.08% | `sma_5`, `sma_20`, `ema_12`, `ema_26` (Trend) |
| **PC2** | **Qubit 1** | **24.13%** | 74.21% | `momentum_5`, `rsi_14`, `daily_return` (Momentum) |
| **PC3** | **Qubit 2** | **13.17%** | 87.38% | `volatility_5`, `daily_return` (Volatility Spike) |
| **PC4** | **Qubit 3** | **7.51%** | **94.89%** | `volatility_5`, `rsi_14` (Oscillator Divergence) |

PCA components were linearly scaled to $[0, \pi]$ for direct Pauli-Z rotation angle encoding ($RY(\theta)$).

---

## 4. Quantum Circuit Architecture & VQC Model

- **Qubits:** 4 (Q0, Q1, Q2, Q3)
- **Feature Map:** `ZZFeatureMap` (2-qubit entangling gates $U_{\text{ZZ}}(\theta_i, \theta_j) = e^{-i(\pi - x_i)(\pi - x_j) Z_i Z_j / 2}$)
- **Variational Ansatz:** `RealAmplitudes` (8 trainable rotation angles $\theta_0 \dots \theta_7$ with linear nearest-neighbor CZ entanglers)
- **Optimizer:** Simultaneous Perturbation Stochastic Approximation (SPSA, 40 iterations)
- **Simulator Primitive:** Qiskit `StatevectorSampler`

---

## 5. Master Experimental Results (2025 Test Set — 248 Days)

| Model Name | Accuracy | Precision | Recall | F1-Score | ROC-AUC | Balanced Acc |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Variational Quantum Classifier (VQC)** | **48.39%** | 0.4908 | 0.8629 | **0.6257** | **0.4906** | 48.39% |
| **Logistic Regression** | **50.00%** | 0.5000 | 0.9919 | **0.6649** | **0.5223** | 50.00% |
| **Random Forest Classifier** | **52.42%** | **0.5469** | 0.2823 | **0.3723** | **0.5306** | **52.42%** |

---

## 6. Statistical Significance & Error Analysis

### McNemar's Test ($p$-value Matrix):
- **VQC vs Logistic Regression:** $p = 0.6636$ (Not Statistically Significant)
- **VQC vs Random Forest:** $p = 0.5668$ (Not Statistically Significant)

### Error Breakdown:
- **VQC:** High Recall (86.29%) but Low Specificity (8.06%), showing a systematic bias toward predicting UP days due to the positive drift of the index during training.
- **Random Forest:** Moderate Accuracy (52.42%) but low recall (28.23%), predicting conservative DOWN movements.

---

## 7. Limitations & Market Non-Stationarity

1. **Market Regime Shift:** Annualized market volatility dropped from **17.52%** during training (2015–2022) to **11.81%** in the 2025 test period.
2. **Qubit Count Constraint:** 4 qubits restrict feature space resolution, requiring PCA compression that strips high-frequency intraday signals.
3. **No Quantum Advantage Observed:** Under NISQ simulation constraints without error correction, shallow VQCs do not exceed classical baselines on noisy financial time series.

---

## 8. Conclusion

This project delivers a complete, reproducible Qiskit 2.x and PyData research baseline. While VQCs demonstrate functional convergence and classification capabilities, classical models perform comparably or slightly better on low-dimensional technical features of NIFTY 50 data.

---

*Report generated automatically by NIFTY50-VQC Research Suite on 2026-10-03.*

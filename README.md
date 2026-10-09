# NIFTY50-VQC: Variational Quantum Classifier for Stock Market Direction Prediction

> **Empirical evaluation of a 4-Qubit Variational Quantum Classifier (VQC) on NIFTY 50 daily direction classification (2015-2025)**

---

## Project Summary

This project investigates whether a **Variational Quantum Classifier (VQC)** can classify the next-day price direction of the NIFTY 50 index, and how it compares against classical ML baselines.

- **Classification task:** Binary — UP (1) or DOWN (0) next-day direction
- **Dataset:** NIFTY 50 index, 2015-2025, 2,706 clean trading days
- **Quantum model:** 4-Qubit VQC with ZZFeatureMap + RealAmplitudes ansatz (SPSA optimizer)
- **Classical baselines:** Logistic Regression, Random Forest

**Main finding:** No statistically significant advantage for 4-qubit VQC over classical methods under NISQ simulation constraints (McNemar p > 0.50 for all model pairs).

---

## Directory Structure

```
quantum_project/
|-- README.md                         <- This file
|-- requirements.txt                  <- Full pinned dependencies
|-- NIFTY50_DATASET.zip               <- Raw data archive (2015-2025)
|
|-- data/
|   |-- README.md                     <- Data directory guide
|   |-- raw/                          <- Original yearly NIFTY50 CSVs (2015-2025)
|   |   |-- README.md
|   |-- processed/                    <- Cleaned CSVs + NPZ arrays
|       |-- README.md
|       |-- nifty50_combined.csv      <- All years merged
|       |-- nifty50_with_target.csv   <- With binary UP/DOWN label
|       |-- nifty50_features.csv      <- With 8 technical indicators
|       |-- nifty50_scaled_data.npz   <- Scaled train/val/test arrays
|       |-- nifty50_pca_4d.npz        <- 4D PCA quantum-encoded arrays
|
|-- src/
|   |-- README.md                     <- Source package guide
|   |-- __init__.py
|   |-- data_loader.py                <- Load & combine yearly CSVs
|   |-- features.py                   <- 8 technical indicators
|   |-- preprocessing.py             <- Temporal split + StandardScaler/MinMaxScaler
|   |-- pca_reduction.py             <- PCA 8D->4D + [0,pi] quantum angle encoding
|   |-- classical_models.py          <- LR, Random Forest
|   |-- vqc_model.py                 <- VQCClassifierWrapper (Qiskit 2.x)
|   |-- quantum_circuit.py           <- ZZFeatureMap, RealAmplitudes, circuit diagram
|
|-- Phase Scripts (run in order):
|   |-- phase10_preprocessing.py     <- Temporal split & feature scaling
|   |-- phase11_classical_models.py  <- Classical baselines
|   |-- phase12_pca.py               <- PCA dimensionality reduction
|   |-- phase15_vqc.py               <- VQC tuning & training
|   |-- phase18_statistical_analysis.py <- McNemar statistical tests
|   |-- phase19_limitations.py       <- Market regime shift analysis
|
|-- results/
|   |-- README.md                    <- Results guide
|   |-- figures/                     <- 13 publication-quality PNG plots
|   |   |-- README.md
|   |-- tables/                      <- 6 CSV result tables
|       |-- README.md
|
|-- docs/
    |-- final_report.md              <- Full research report
    |-- methodology.md               <- Detailed experimental methodology
    |-- experiment_log.md            <- Phase-by-phase experiment log (EXP-001 to EXP-016)
    |-- environment_info.txt         <- System & package version info
```

---

## Experimental Results (2025 Test Set — 248 Trading Days)

| Model                   | Accuracy | Precision | Recall | F1-Score | ROC-AUC | Balanced Acc |
|-------------------------|----------|-----------|--------|----------|---------|--------------|
| VQC (4 Qubits, ZZ+SPSA) | 48.39%  | 0.4908    | 0.8629 | 0.6257   | 0.4906  | 48.39%       |
| Logistic Regression      | 50.00%  | 0.5000    | 0.9919 | 0.6649   | 0.5223  | 50.00%       |
| Random Forest            | 52.42%  | 0.5469    | 0.2823 | 0.3723   | 0.5306  | 52.42%       |

**McNemar Statistical Significance:** All p-values > 0.50 — no model is statistically superior to any other.

---

## Setup & Reproduction

`ash
# Step 1: Create virtual environment
python -m venv .venv
.venv\Scripts\activate

# Step 2: Install all dependencies (pinned versions)
pip install -r requirements.txt

# Step 3: Run pipeline in order (each phase depends on the previous)
.venv\Scripts\python.exe phase10_preprocessing.py
.venv\Scripts\python.exe phase11_classical_models.py
.venv\Scripts\python.exe phase12_pca.py
.venv\Scripts\python.exe phase15_vqc.py           # Slow: ~10-30 min on CPU
.venv\Scripts\python.exe phase18_statistical_analysis.py
.venv\Scripts\python.exe phase19_limitations.py
`

> **Note:** Phase 15 (VQC training) trains on 1,963 samples with 40 SPSA iterations.
> Estimated runtime: 10-30 minutes on a modern CPU. No GPU required.

---

## Environment

| Component | Version / Spec |
|-----------|---------------|
| Python | 3.11.9 |
| Qiskit | 2.5.2 |
| qiskit-machine-learning | 0.9.1 |
| qiskit-aer | 0.17.2 (CPU only) |
| scikit-learn | 1.9.1 |
| numpy | 2.4.6 |
| pandas | 3.0.6 |
| OS | Windows 11, 64-bit |
| CPU | Intel Core i7-13650HX |
| RAM | 16 GB |
| GPU | NVIDIA RTX 5050 (not used — no qiskit-aer-gpu on Windows) |
| Random Seed | 42 |

---

## Key Findings

1. **VQC performance:** 48.39% accuracy — below all classical baselines
2. **PCA efficiency:** 4 components capture 94.89% of original variance
3. **VQC bias:** Systematic bias toward predicting UP days (Recall: 86.29%, Specificity: 8.06%)
4. **Regime shift:** Annualised volatility dropped from 17.52% (training) to 11.81% (test) — a 33% reduction
5. **No quantum advantage** under current 4-qubit NISQ simulation constraints
6. **Statistical conclusion:** All McNemar test p-values > 0.50 — performance differences are statistical noise

---

## PCA Component Summary

| Component | Qubit | Variance Explained | Cumulative | Primary Loadings |
|-----------|-------|--------------------|-----------|-----------------|
| PC1 | Q0 | 50.08% | 50.08% | sma_5, sma_20, ema_12, ema_26 (Trend) |
| PC2 | Q1 | 24.13% | 74.21% | momentum_5, rsi_14, daily_return (Momentum) |
| PC3 | Q2 | 13.17% | 87.38% | volatility_5, daily_return (Volatility) |
| PC4 | Q3 |  7.51% | 94.89% | volatility_5, rsi_14 (Oscillator) |

---

## VQC Architecture

`
4-Qubit Circuit: ZZFeatureMap (reps=1, linear entanglement) + RealAmplitudes ansatz (reps=1)
Trainable parameters: 8 variational rotation angles (theta_0 ... theta_7)
Optimizer: SPSA (Simultaneous Perturbation Stochastic Approximation), 40 iterations
Simulator: Qiskit StatevectorSampler (CPU, exact simulation)
`

---

## Disclaimer

This is an academic ML/QML classification experiment. It is **NOT** a financial advisory system.
Predictions and results should **not** be used for actual trading decisions.

---

*NIFTY50-VQC Research Project | Completed: 2026-10-03*
*Python 3.11.9 | Qiskit 2.5.2 | qiskit-machine-learning 0.9.1*

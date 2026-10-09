# src/ — Python Source Package

This directory contains all reusable Python modules for the NIFTY50-VQC project.
Every module enforces strict **data-leakage-free** conventions: scalers, PCA, and
encoders are always fitted on the Training set only and then applied to Val/Test.

---

## Module Reference

### data_loader.py
Loads and combines all 11 yearly NIFTY 50 CSV files into a single clean DataFrame.
- Standardises column names to lowercase snake_case
- Parses and validates date column
- Converts OHLCV to numeric, removes duplicates, sorts chronologically

### features.py
Computes 8 technical indicators from OHLCV data (no future leakage).

| Feature | Formula | NaN warmup |
|---------|---------|-----------|
| daily_return | (Close_t - Close_{t-1}) / Close_{t-1} | 1 row |
| sma_5 | 5-day Simple Moving Average | 4 rows |
| sma_20 | 20-day Simple Moving Average | 19 rows (maximum) |
| ema_12 | 12-day Exponential Moving Average | ~0 (EWM) |
| ema_26 | 26-day Exponential Moving Average | ~0 (EWM) |
| rsi_14 | 14-day Relative Strength Index | 13 rows |
| volatility_5 | 5-day rolling std of daily returns | 4 rows |
| momentum_5 | Close_t - Close_{t-5} | 5 rows |

Key functions:
- compute_all_features(df) — adds features, reports NaN counts
- get_clean_features(df) — adds features + drops NaN warmup rows

### preprocessing.py
Temporal train/val/test splitting and feature normalisation.

- **Split boundaries:**
  - Train:      2015-01-29 to 2022-12-30 (1,963 rows)
  - Validation: 2023-01-02 to 2024-12-31 (495 rows)
  - Test:       2025-01-01 to 2025-12-30 (248 rows)
- **StandardScaler:** fit on Train ONLY, mean=0 std=1 (for classical ML and PCA input)
- **MinMaxScaler [0, pi]:** fit on Train ONLY (for direct quantum angle encoding)

Key functions:
- 	emporal_split(df) — returns (train_df, val_df, test_df)
- it_and_transform_features(...) — returns dict of all scaled arrays

### pca_reduction.py
PCA dimensionality reduction: 8 standardised features -> 4 principal components (one per qubit).

- PCA fitted on Train ONLY (2015-2022)
- MinMaxScaler [0, pi] fitted on Train PCA output ONLY
- 94.89% cumulative variance retained across 4 components

Key functions:
- it_and_transform_pca(X_train_std, X_val_std, X_test_std) — returns dict with quantum-ready arrays

### classical_models.py
Trains two classical ML baselines with validation-based hyperparameter tuning.

| Model | Hyperparameter grid | Best config (2025 Test) |
|-------|--------------------|-----------------------|
| Logistic Regression | C in {0.001, 0.01, 0.1, 1.0, 100.0} | C=0.001, Acc=50.00% |
| Random Forest | n_estimators x max_depth x min_samples_split | n=50, depth=3, Acc=52.42% |

Key functions:
- train_logistic_regression(X_train, y_train, X_val, y_val) -> (model, params, val_metrics)
- train_random_forest(...) -> (model, params, val_metrics)
- evaluate_model(model, X, y_true) -> dict with accuracy, precision, recall, f1, roc_auc, bal_acc, cm

### vqc_model.py
Wraps Qiskit Machine Learning VQC with training loss tracking and evaluation metrics.

- Supports optimizers: COBYLA, SPSA, ADAM
- Loss history captured via optimizer callback
- Full evaluation metrics matching classical model interface

Key class: VQCClassifierWrapper(num_qubits, feature_map_type, ansatz_type, reps, optimizer_name, maxiter)

Selected configuration (VQC-5):
- Feature Map: ZZFeatureMap (reps=1)
- Ansatz: RealAmplitudes (reps=1)
- Optimizer: SPSA (40 iterations)
- Result: Acc=48.39%, F1=0.6257, AUC=0.4906 on 2025 Test Set

### quantum_circuit.py
Builds Qiskit quantum circuit components.

Key functions:
- uild_feature_map(num_qubits, reps, map_type) — ZZFeatureMap or ZFeatureMap
- uild_ansatz(num_qubits, reps, ansatz_type) — RealAmplitudes or EfficientSU2
- plot_quantum_circuit(feature_map, ansatz, save_path) — saves circuit diagram PNG

---

## Data Leakage Policy (CRITICAL)

> All scalers (StandardScaler, MinMaxScaler) and PCA are **fit ONLY on the Training set (2015-2022)**.
> Validation and Test sets are transformed using the training parameters only.
> Violating this policy would constitute data leakage and invalidate all results.

---

*NIFTY50-VQC Project | src/ package | Last updated: 2026-10-03*

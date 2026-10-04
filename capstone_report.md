# NIFTY50-VQC: Variational Quantum Classifier for Stock Market Direction Prediction
## Quantum Capstone Project — Full Report

**Name:** *(Your Full Name)*
**Roll Number:** *(Your Roll Number)*
**Institution:** *(Your Institution Name)*
**Date:** October 2026

---

## Abstract

This report presents an empirical investigation into whether a **Variational Quantum Classifier (VQC)** can classify the next-day price direction of the **NIFTY 50 stock index** (UP or DOWN) using historical market data.

The study uses 10 years of NIFTY 50 daily OHLCV data (2015–2025), with 8 engineered technical indicators reduced to 4 dimensions via Principal Component Analysis (PCA) for quantum encoding. The primary algorithm is a **4-Qubit VQC** built with Qiskit 2.x, using a `ZZFeatureMap` for feature encoding and `RealAmplitudes` as the variational ansatz, optimized via SPSA (Simultaneous Perturbation Stochastic Approximation) for 40 iterations.

Evaluation on a strictly held-out 2025 test set (248 trading days) showed VQC achieved **48.39% accuracy**, compared to classical baselines: Logistic Regression (50.00%), SVM/Linear (50.00%), and Random Forest (52.42%). McNemar statistical significance testing confirmed that **no model is statistically superior** to any other (all p-values > 0.50). The study concludes that the 4-Qubit VQC, under NISQ simulation constraints, does not demonstrate a measurable quantum advantage over classical ML methods for this financial classification task.

---

## Problem Statement

### What is the Problem?
The **NIFTY 50** is the flagship stock market index of the National Stock Exchange of India, composed of 50 large-cap stocks. Predicting short-term market direction (whether the index will close higher or lower the next trading day) is one of the most challenging tasks in computational finance.

Classical machine learning models have been widely applied to this problem but consistently struggle to significantly beat random chance (~50%) due to the noisy, non-stationary nature of financial time series. **Quantum Machine Learning (QML)**, specifically Variational Quantum Classifiers (VQCs), has emerged as a promising research direction — but its effectiveness on real-world Indian market data has not been empirically studied.

This project empirically tests: *Can a Variational Quantum Classifier classify the next-day price direction of the NIFTY 50 index, and how does it compare against conventional ML models?*

### What is the Input?
- **Raw Data:** 11 CSV files (one per year, 2015–2025) containing NIFTY 50 daily trading data
- **Columns:** Date, Open, High, Low, Close, Volume, Shares Traded, Turnover
- **After engineering:** 8 technical indicators computed from OHLCV — Daily Return, SMA_5, SMA_20, EMA_12, EMA_26, RSI_14, Volatility_5, Momentum_5
- **For quantum model:** 4 principal components (from PCA) scaled to angle range [0, π]
- **Total clean samples:** 2,706 trading days (after 19-day feature warmup dropped)

### What is the Expected Output?
- **Binary classification label** for each trading day:
  - **1 (UP):** The next trading day's closing price is strictly higher than the current day's closing price
  - **0 (DOWN):** The next trading day's closing price is equal to or lower than the current day's closing price
- **Performance metrics:** Accuracy, Precision, Recall, F1-Score, ROC-AUC, Balanced Accuracy
- **Statistical significance:** McNemar test p-values for all model pairings

### Limitations
- Market non-stationarity: Statistical properties of financial time series change over time (e.g., volatility regime shifts)
- Financial noise: Daily price movements contain substantial random variation that no model can fully capture
- Limited feature set: Only 8 technical indicators; no fundamental, macroeconomic, or sentiment data used
- No transaction costs or slippage modelled
- No trading strategy backtesting performed
- Quantum simulation only — not tested on real quantum hardware
- Small qubit count (4 qubits) due to CPU simulation constraints
- Results are specific to the NIFTY 50 index for the 2015–2025 period

---

## Objectives

### How Are We Solving the Problem?

The solution follows a rigorous, end-to-end machine learning pipeline with strict data-leakage safeguards:

1. **Data Collection and Cleaning:** Combine 11 yearly NIFTY 50 CSV files, standardize column names, parse dates, handle missing values, remove duplicates, and sort chronologically.

2. **Target Construction:** Define a binary target variable based on next-day price direction, using only past information as features (no look-ahead bias).

3. **Feature Engineering:** Compute 8 technical indicators from OHLCV data using only rolling windows of past data.

4. **Temporal Train/Val/Test Split:** Split the dataset chronologically (no shuffling) to preserve time-series integrity and prevent future data from informing past predictions.

5. **Feature Scaling:** Apply StandardScaler (fit only on training set) to normalize all 8 features.

6. **PCA Dimensionality Reduction:** Reduce 8 standardized features to 4 principal components (capturing 94.89% of variance) for quantum encoding compatibility.

7. **Quantum Angle Encoding:** Scale PCA components to the [0, π] range using MinMaxScaler, mapping each component to one qubit rotation angle.

8. **VQC Training:** Train the VQC using ZZFeatureMap + RealAmplitudes circuit with SPSA optimization on 1,963 training samples.

9. **Architecture Tuning:** Test 5 VQC configurations on the validation set and select the best by F1-Score.

10. **Final Evaluation:** Evaluate all models (VQC + 3 classical baselines) on the 2025 test set (touched only once). Apply McNemar tests for statistical significance.

### Which Algorithm is Used?
- **Primary:** Variational Quantum Classifier (VQC) — a parameterized quantum circuit trained via classical optimization
- **Classical Baselines:** Logistic Regression, Support Vector Machine (Linear kernel), Random Forest

### How Does the Algorithm Analyze Complexity?
- **Time:** VQC requires O(T × N × 2ⁿ) operations — exponential in qubit count n, making classical simulation impractical beyond ~30 qubits.
- **Space:** The quantum statevector requires O(2ⁿ) complex amplitudes — 16 for 4 qubits, growing exponentially.
- Classical models have polynomial complexity, giving them a practical advantage in current NISQ-era simulation.

---

## Proposed Algorithm

### Clear Explanation of the Algorithm

A **Variational Quantum Classifier (VQC)** is a hybrid quantum-classical machine learning model that runs on parameterized quantum circuits. It consists of two components:

**1. Feature Map (ZZFeatureMap):**
The feature map encodes classical data into quantum states using a series of Hadamard gates and entangling ZZ-rotations. For a 4-dimensional input vector x = [x₀, x₁, x₂, x₃]:
- Each component xᵢ is encoded into qubit Qᵢ via rotation gates: `Rz(2xᵢ)`
- Pairs of qubits are entangled with CNOT gates and cross-product rotations: `Rz(2xᵢxⱼ)`, capturing feature interactions in quantum space.
- This creates a **quantum feature space** that is computationally hard to simulate classically (the source of potential quantum advantage).

**2. Ansatz (RealAmplitudes):**
The ansatz is the "learnable" part of the circuit — a sequence of trainable single-qubit Ry rotations and CNOT entangling gates:
- 8 trainable parameters: θ₀, θ₁, ..., θ₇
- These parameters are optimized to minimize classification loss.

**3. Measurement and Classification:**
After the circuit executes, all 4 qubits are measured. The probability of measuring the "UP" bitstring (e.g., `|1⟩`) is used as the classification confidence score.

**4. Optimizer (SPSA):**
SPSA approximates the gradient using only two circuit evaluations per iteration (not full gradient computation), making it robust to quantum noise.

---

### Flowchart

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                   NIFTY50-VQC Pipeline Flowchart                            │
└─────────────────────────────────────────────────────────────────────────────┘

  [Raw CSV Files: 2015–2025]
           │
           ▼
  [Data Cleaning & Merging]
  (Standardize columns, parse dates, remove nulls/duplicates)
           │
           ▼
  [Feature Engineering]
  (Compute 8 technical indicators: SMA, EMA, RSI, Volatility, Momentum, Return)
           │
           ▼
  [Target Construction]
  (Target_t = 1 if Close_(t+1) > Close_t, else 0)
           │
           ▼
  [Temporal Split: No shuffling]
  ┌───────────────┬──────────────────┬──────────────────┐
  │  TRAIN        │  VALIDATION      │  TEST            │
  │  2015–2022    │  2023–2024       │  2025            │
  │  1,963 rows   │  495 rows        │  248 rows        │
  └───────────────┴──────────────────┴──────────────────┘
           │
           ▼
  [StandardScaler: fit on TRAIN only]
           │
           ├──────────────────────────────────────────────────┐
           │ Quantum Pipeline                                  │ Classical Pipeline
           ▼                                                   ▼
  [PCA: 8D → 4D]                                    [Logistic Regression]
  (fit on TRAIN only, 94.89% variance retained)     [SVM (Linear)]
           │                                          [Random Forest]
           ▼                                                   │
  [Quantum Angle Encoding: [0, π]]                            │
  (MinMaxScaler, fit on TRAIN PCA only)                       │
           │                                                   │
           ▼                                                   │
  [VQC Architecture Tuning on VALIDATION]                     │
  (5 configs tested: VQC-1 to VQC-5)                         │
           │                                                   │
           ▼                                                   │
  [Train Optimal VQC (VQC-5) on Full TRAIN]                  │
  (ZZFeatureMap + RealAmplitudes + SPSA, 40 iterations)      │
           │                                                   │
           └──────────────────┬────────────────────────────────┘
                              ▼
                [FINAL EVALUATION: 2025 TEST SET]
                (Accuracy, Precision, Recall, F1, ROC-AUC)
                              │
                              ▼
                [McNemar Statistical Significance Tests]
                              │
                              ▼
                      [Results & Analysis]
```

---

### Step-by-Step Process

**Phase 1 — Data Preparation (phase10_preprocessing.py):**
1. Load 11 yearly CSV files; merge into one DataFrame sorted by date
2. Engineer 8 technical indicators (19-day warmup → drop early rows)
3. Construct binary target variable (next-day direction)
4. Temporal split: Train (2015–2022), Val (2023–2024), Test (2025)
5. Fit StandardScaler on training set; transform all three splits
6. Save arrays to `data/processed/nifty50_scaled_data.npz`

**Phase 2 — Classical Baselines (phase11_classical_models.py):**
1. Grid-search hyperparameters on validation set for LR, SVM, RF
2. Train each model on full training set with best hyperparameters
3. Evaluate on 2025 test set; save metrics and figures

**Phase 3 — PCA for Quantum (phase12_pca.py):**
1. Fit PCA (4 components) on X_train_std only
2. Apply MinMaxScaler [0, π] on PCA outputs (quantum angle encoding)
3. Save to `data/processed/nifty50_pca_4d.npz`

**Phase 4 — VQC Training (phase15_vqc.py):**
1. Load quantum-encoded data
2. Test 5 VQC configurations on a 400-sample training subsample + validation set
3. Select best config by validation F1-Score → VQC-5 (ZZ + RealAmplitudes + SPSA)
4. Train optimal VQC on full 1,963 training samples (40 SPSA iterations)
5. Evaluate on 2025 test set; generate circuit diagram, loss curve, confusion matrix, ROC plot

**Phase 5 — Statistical Analysis (phase18_statistical_analysis.py):**
1. Collect predictions from all 4 models on 2025 test set
2. Perform McNemar tests for all 6 pairings
3. Report contingency tables and p-values

---

### Applications of the VQC Algorithm
- **Finance:** Market direction classification, credit risk scoring, fraud detection
- **Healthcare:** Disease classification from medical imaging features
- **Cybersecurity:** Network intrusion detection
- **Chemistry:** Molecular property prediction (the original motivation for VQC)
- **General:** Any binary/multi-class classification where quantum feature maps may offer advantage over classical kernels

---

## Pseudocode

```
ALGORITHM: Variational Quantum Classifier (VQC) — Full Pipeline

═══════════════════════════════════════════════════════════════
PHASE A: DATA PREPARATION
═══════════════════════════════════════════════════════════════

FUNCTION prepare_data(csv_files):
    D = CONCAT(csv_files)                         // Merge 11 yearly files
    D = SORT(D, by='Date')                        // Chronological order
    D = CLEAN(D)                                   // Remove nulls, duplicates

    // Feature Engineering
    FOR each row t in D:
        D[t]['daily_return']  = (Close_t - Close_{t-1}) / Close_{t-1}
        D[t]['sma_5']         = mean(Close[t-4:t+1])
        D[t]['sma_20']        = mean(Close[t-19:t+1])
        D[t]['ema_12']        = EMA(Close, span=12)[t]
        D[t]['ema_26']        = EMA(Close, span=26)[t]
        D[t]['rsi_14']        = RSI(Close, period=14)[t]
        D[t]['volatility_5']  = std(daily_return[t-4:t+1])
        D[t]['momentum_5']    = Close_t - Close_{t-5}
        D[t]['target']        = 1 if Close_{t+1} > Close_t else 0

    DROP first 19 rows (warmup)
    DROP last row (no next-day close)

    // Temporal Split
    TRAIN = D[Date <= 2022-12-31]                 // 1,963 samples
    VAL   = D[2023-01-01 <= Date <= 2024-12-31]  //   495 samples
    TEST  = D[Date >= 2025-01-01]                 //   248 samples

    // Feature Scaling (fit ONLY on TRAIN)
    scaler = StandardScaler.fit(TRAIN.features)
    X_train_std = scaler.transform(TRAIN.features)
    X_val_std   = scaler.transform(VAL.features)
    X_test_std  = scaler.transform(TEST.features)

    RETURN X_train_std, X_val_std, X_test_std, y_train, y_val, y_test

═══════════════════════════════════════════════════════════════
PHASE B: PCA AND QUANTUM ENCODING
═══════════════════════════════════════════════════════════════

FUNCTION quantum_encode(X_train_std, X_val_std, X_test_std):
    pca = PCA(n_components=4).fit(X_train_std)    // Fit on TRAIN only
    X_train_pca = pca.transform(X_train_std)
    X_val_pca   = pca.transform(X_val_std)
    X_test_pca  = pca.transform(X_test_std)

    // Scale to quantum angle range [0, pi]
    angle_scaler = MinMaxScaler(feature_range=(0, pi)).fit(X_train_pca)
    X_train_q = angle_scaler.transform(X_train_pca)  // Shape: (1963, 4)
    X_val_q   = angle_scaler.transform(X_val_pca)    // Shape: (495, 4)
    X_test_q  = angle_scaler.transform(X_test_pca)   // Shape: (248, 4)

    RETURN X_train_q, X_val_q, X_test_q

═══════════════════════════════════════════════════════════════
PHASE C: VQC TRAINING
═══════════════════════════════════════════════════════════════

FUNCTION build_vqc_circuit(num_qubits=4):
    feature_map = ZZFeatureMap(feature_dimension=4, reps=1)
    ansatz      = RealAmplitudes(num_qubits=4, reps=1)
    RETURN feature_map, ansatz

FUNCTION train_vqc(X_train_q, y_train, feature_map, ansatz):
    theta = RANDOM_INIT(size=8)                   // 8 variational parameters
    optimizer = SPSA(maxiter=40)

    FOR iteration = 1 TO 40:
        delta = RANDOM_PERTURBATION()              // SPSA: stochastic gradient
        loss_plus  = CROSS_ENTROPY(PREDICT(X_train_q, theta + c*delta), y_train)
        loss_minus = CROSS_ENTROPY(PREDICT(X_train_q, theta - c*delta), y_train)
        gradient   = (loss_plus - loss_minus) / (2 * c)
        theta      = theta - alpha * gradient      // Update parameters

    RETURN theta_optimal

FUNCTION predict_vqc(X, theta, feature_map, ansatz):
    predictions = []
    FOR each sample x in X:
        circuit = feature_map(x) + ansatz(theta)   // Build full circuit
        statevector = STATEVECTOR_SAMPLER.run(circuit)
        prob_up = |amplitude of |1111>|^2          // Probability of UP class
        label   = 1 IF prob_up >= 0.5 ELSE 0
        predictions.APPEND(label)
    RETURN predictions

═══════════════════════════════════════════════════════════════
PHASE D: EVALUATION
═══════════════════════════════════════════════════════════════

FUNCTION evaluate(y_true, y_pred, y_prob):
    accuracy           = correct_predictions / total_predictions
    precision          = TP / (TP + FP)
    recall             = TP / (TP + FN)
    f1_score           = 2 * precision * recall / (precision + recall)
    roc_auc            = AREA_UNDER_ROC_CURVE(y_true, y_prob)
    balanced_accuracy  = (sensitivity + specificity) / 2
    RETURN all_metrics

FUNCTION mcnemar_test(y_pred_A, y_pred_B, y_true):
    // Compare models A and B
    b = COUNT(A correct, B wrong)
    c = COUNT(A wrong, B correct)
    chi2 = (|b - c| - 1)^2 / (b + c)
    p_value = CHI2_DISTRIBUTION_PVALUE(chi2, df=1)
    RETURN p_value

═══════════════════════════════════════════════════════════════
MAIN EXECUTION
═══════════════════════════════════════════════════════════════

X_train_std, X_val_std, X_test_std, y_train, y_val, y_test = prepare_data(csv_files)
X_train_q, X_val_q, X_test_q = quantum_encode(X_train_std, X_val_std, X_test_std)
feature_map, ansatz = build_vqc_circuit()
theta = train_vqc(X_train_q, y_train, feature_map, ansatz)
y_pred_vqc = predict_vqc(X_test_q, theta, feature_map, ansatz)
metrics_vqc = evaluate(y_test, y_pred_vqc, y_prob_vqc)
```

---

## Implementation

### Programming Language Used
**Python 3.11.9** — chosen for its rich ecosystem of quantum computing and machine learning libraries.

### Tools Used

| Category | Tool | Version | Purpose |
|----------|------|---------|---------|
| Quantum Framework | Qiskit | 2.5.2 | Quantum circuit building, simulation |
| QML Library | qiskit-machine-learning | 0.9.1 | VQC implementation |
| Quantum Simulator | Qiskit Aer (StatevectorSampler) | 0.17.2 | Exact CPU-based simulation |
| Classical ML | scikit-learn | 1.9.1 | LR, SVM, RF, PCA, StandardScaler |
| Data Handling | pandas | 3.0.6 | CSV loading, DataFrame operations |
| Numerical Computing | numpy | 2.4.6 | Array operations, NPZ file I/O |
| Visualization | matplotlib, seaborn | latest | Plots, heatmaps, ROC curves |
| IDE | VS Code + Python Extension | latest | Development environment |
| Hardware | Intel Core i7-13650HX, 16 GB RAM, Windows 11 | — | CPU-only simulation |

### Code Explanation

The project is organized into **modular phase scripts**, each building on the previous:

**`src/pca_reduction.py` — PCA and Quantum Encoding Module:**
```python
def fit_and_transform_pca(X_train_std, X_val_std, X_test_std, n_components=4):
    # PCA fit ONLY on training data (leakage safeguard)
    pca = PCA(n_components=4, random_state=42)
    pca.fit(X_train_std)

    # Transform all splits using training PCA
    X_train_pca = pca.transform(X_train_std)  # Shape: (1963, 4)
    X_val_pca   = pca.transform(X_val_std)    # Shape: (495, 4)
    X_test_pca  = pca.transform(X_test_std)   # Shape: (248, 4)

    # Scale PCA components to quantum angle range [0, pi]
    quantum_scaler = MinMaxScaler(feature_range=(0, np.pi))
    quantum_scaler.fit(X_train_pca)  # Fit on train PCA ONLY

    X_train_quantum = quantum_scaler.transform(X_train_pca)
    X_val_quantum   = quantum_scaler.transform(X_val_pca)
    X_test_quantum  = quantum_scaler.transform(X_test_pca)

    return X_train_quantum, X_val_quantum, X_test_quantum
```

**`phase15_vqc.py` — VQC Training and Comparison:**
```python
from src.vqc_model import VQCClassifierWrapper

# Test 5 VQC architecture configurations on Validation set
configs = [
    {'id': 'VQC-1', 'map': 'zz', 'ansatz': 'real_amplitudes', 'reps': 1, 'opt': 'cobyla'},
    {'id': 'VQC-5', 'map': 'zz', 'ansatz': 'real_amplitudes', 'reps': 1, 'opt': 'spsa'},
    # ... (5 configurations total)
]

# Select best by validation F1-Score
best_config = configs[argmax([val_f1 for each config])]  # VQC-5 selected

# Train optimal VQC on full training set (1,963 samples, 40 SPSA iterations)
optimal_vqc = VQCClassifierWrapper(
    num_qubits=4,
    feature_map_type='zz',
    ansatz_type='real_amplitudes',
    reps=1,
    optimizer_name='spsa',
    maxiter=40,
    random_state=42
)
optimal_vqc.fit(X_train_quantum, y_train)

# Final evaluation on 2025 test set
vqc_test_metrics = optimal_vqc.evaluate(X_test_quantum, y_test)
```

**`phase18_statistical_analysis.py` — McNemar Tests:**
```python
from statsmodels.stats.contingency_tables import mcnemar

# Build contingency table for each model pair
for (name_A, pred_A), (name_B, pred_B) in all_pairs:
    contingency = build_mcnemar_table(y_test, pred_A, pred_B)
    result = mcnemar(contingency, exact=True)
    print(f"{name_A} vs {name_B}: p-value = {result.pvalue:.4f}")
```

### Attached Code Files
The complete implementation is available in the project repository under:
- `src/` — Reusable modules (data loader, features, PCA, VQC model, quantum circuit)
- `phase10_preprocessing.py` — Data cleaning and scaling
- `phase11_classical_models.py` — Classical baselines
- `phase12_pca.py` — PCA dimensionality reduction
- `phase15_vqc.py` — VQC training, tuning, and master comparison
- `phase18_statistical_analysis.py` — McNemar tests
- `phase19_limitations.py` — Market regime analysis

---

## Website (Optional)

### Technologies Used
- **HTML5** — Page structure and semantic markup
- **CSS3** — Styling, animations, responsive layout
- **JavaScript (Vanilla)** — Interactive prediction interface and dynamic charts
- **Python Flask / HTTP Server** — Backend prediction API (`prediction_server.py`)

### Tools Used
- **VS Code** — Development
- **Python 3.11.9 + Flask** — REST API server for model predictions
- **Browser:** Chrome / Edge for testing

### Explanation of Design
The website (`website/index.html`) provides a **live stock direction prediction interface**:
1. Users input values for the 8 technical indicators (Daily Return, SMA_5, SMA_20, EMA_12, EMA_26, RSI_14, Volatility_5, Momentum_5)
2. The frontend sends these values to the Python prediction server (`prediction_server.py`)
3. The server applies the trained VQC model and classical baselines
4. The website displays:
   - Prediction result (UP / DOWN) with confidence scores
   - Comparison of all 4 model predictions
   - Real-time probability gauge visualization

### Screenshots
> *(Add screenshots of your website here from the `website/` directory)*

---

## Results

### Input Given
- **Training Set:** 1,963 samples (2015–2022), 4 PCA-encoded quantum features, binary labels
- **Validation Set:** 495 samples (2023–2024) — used only for hyperparameter/architecture selection
- **Test Set:** 248 samples (2025) — touched exactly once for final evaluation

**Best VQC Configuration Selected (VQC-5):**
| Parameter | Value |
|-----------|-------|
| Feature Map | ZZFeatureMap (reps=1, linear entanglement) |
| Ansatz | RealAmplitudes (reps=1, linear entanglement) |
| Optimizer | SPSA (maxiter=40) |
| Trainable Parameters | 8 (θ₀ ... θ₇) |
| Validation Accuracy | 53.74% |
| Validation F1-Score | 0.6264 |

### Output Obtained — Master Model Comparison (2025 Test Set — 248 Trading Days)

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC | Balanced Acc |
|-------|----------|-----------|--------|----------|---------|--------------|
| **VQC (4-Qubit, ZZ+SPSA)** | **48.39%** | 0.4908 | **0.8629** | 0.6257 | 0.4906 | 48.39% |
| Logistic Regression (C=0.001) | 50.00% | 0.5000 | 0.9919 | **0.6649** | 0.5223 | 50.00% |
| SVM Linear (C=0.1) | 50.00% | 0.5000 | **1.0000** | **0.6667** | 0.4725 | 50.00% |
| **Random Forest (50 trees, depth=3)** | **52.42%** | **0.5469** | 0.2823 | 0.3723 | **0.5306** | **52.42%** |

### McNemar Statistical Significance (Pairwise)

| Model Pair | p-value | Interpretation |
|------------|---------|---------------|
| VQC vs Logistic Regression | > 0.50 | No significant difference |
| VQC vs SVM | > 0.50 | No significant difference |
| VQC vs Random Forest | > 0.50 | No significant difference |
| LR vs SVM | > 0.50 | No significant difference |
| LR vs Random Forest | > 0.50 | No significant difference |
| SVM vs Random Forest | > 0.50 | No significant difference |

**All p-values > 0.50 → Performance differences are statistical noise, not genuine model superiority.**

### Algorithm Performance Analysis

**VQC-Specific Observations:**
- **Systematic UP bias:** VQC predicted UP on 86.29% of test days (Recall = 86.29%) but only correctly identified DOWN days 8.06% of the time (Specificity = 8.06%)
- **Root cause:** Training set class imbalance (53.49% UP vs 46.51% DOWN) + SPSA optimizer convergence to UP-dominant predictions

**Market Regime Shift:**
| Period | Annualized Volatility | Class Balance |
|--------|-----------------------|---------------|
| Training (2015–2022) | 17.52% | 53.49% UP / 46.51% DOWN |
| Test (2025) | 11.81% | 50.00% UP / 50.00% DOWN |

The **33% reduction in volatility** from training to test period represents a regime shift that likely contributed to all models performing near random chance.

### Time and Space Complexity

**Time Complexity:**

| Algorithm | Training Time Complexity | Actual Runtime (approx.) |
|-----------|--------------------------|--------------------------|
| VQC (4-qubit, SPSA 40 iter) | O(T × N × 2ⁿ) | 10–30 minutes |
| Logistic Regression | O(N × d × iterations) | < 1 second |
| SVM (Linear) | O(N² × d) | < 5 seconds |
| Random Forest | O(k × N × log N × d) | < 10 seconds |

Where: T = 40 iterations, N = 1,963 training samples, n = 4 qubits, d = 8 features, k = 50 trees.

> **VQC bottleneck:** Quantum simulation is exponential in qubit count — each circuit evaluation requires simulating a 2⁴ = 16-dimensional complex statevector.

**Space Complexity:**

| Component | Space | Value for this project |
|-----------|-------|----------------------|
| VQC Statevector | O(2ⁿ) | 16 complex amplitudes |
| VQC Parameters | O(p) | 8 floats |
| PCA Components | O(k × d) | 4 × 8 = 32 floats |
| Full Training Data (VQC) | O(N × 2ⁿ) | 1,963 × 16 |
| Random Forest | O(k × N × depth) | ~50 × 1,963 × 3 |

**Generated Figures:**
- `fig9_quantum_circuit.png` — 4-qubit VQC circuit diagram
- `fig10_vqc_training_loss.png` — SPSA training loss curve (40 iterations)
- `fig11_vqc_confusion_matrix.png` — VQC confusion matrix on 2025 test set
- `fig12_quantum_vs_classical_roc.png` — Master ROC curves (all 4 models)
- `fig13_regime_shifts.png` — Volatility regime shift analysis
- `fig7_pca_variance.png` — Scree plot of PCA explained variance
- `fig8_pca_loadings.png` — PCA component loading heatmap
- `fig4_confusion_matrices.png` — All classical model confusion matrices
- `fig5_roc_curves.png` — Classical model ROC curves

---

## Conclusion

This project successfully implemented and empirically evaluated a **4-Qubit Variational Quantum Classifier (VQC)** for binary stock market direction classification on 10 years of NIFTY 50 data. The problem was approached with rigorous scientific methodology: strict temporal data splitting, no data leakage, architecture tuning on a held-out validation set, and final evaluation on a strictly untouched 2025 test set.

**The problem was solved as follows:**
- The full pipeline — from raw CSV ingestion to quantum feature encoding, VQC training, classical baseline comparison, and statistical significance testing — was implemented and executed completely.
- All 5 VQC architecture configurations were systematically evaluated, and the best (VQC-5: ZZFeatureMap + RealAmplitudes + SPSA) was identified and trained on the full 1,963-sample training set.
- Final evaluation was performed on 248 previously unseen trading days from 2025.

**The algorithm worked as designed, but revealed a key finding:** Under current NISQ-era simulation constraints (4 qubits, CPU-only), the VQC (48.39% accuracy) does not outperform classical baselines (50–52% accuracy). McNemar statistical tests confirm that **none of the 4 models is statistically superior** to any other — all p-values exceed 0.50. Performance differences are within the range of statistical noise.

**Key contributions:**
1. First empirical VQC study on 10 years of NIFTY 50 daily data
2. Rigorous no-leakage pipeline with temporal splits and test-set isolation
3. Comprehensive 5-config VQC architecture tuning
4. Statistical validation via McNemar tests (not just accuracy comparison)
5. Market regime shift analysis explaining test-set difficulty

**Future directions:** Scaling to 8+ qubits (on real quantum hardware), incorporating fundamental/sentiment data, exploring quantum kernel methods (QSVM), and testing on multiple market indices.

This work contributes to the growing body of empirical evidence on near-term quantum machine learning and demonstrates both the feasibility and current limitations of VQC-based financial classification.

> **Important Disclaimer:** This is an academic machine-learning and quantum computing research experiment. It is NOT a financial advisory system. The predictions and results presented in this report should NOT be used for actual investment or trading decisions.

---

*NIFTY50-VQC Research Project | Completed: 2026-10-03*
*Python 3.11.9 | Qiskit 2.5.2 | qiskit-machine-learning 0.9.1 | scikit-learn 1.9.1*
*Dataset: NIFTY 50 Index (NSE India) | 2015–2025 | 2,706 Trading Days*

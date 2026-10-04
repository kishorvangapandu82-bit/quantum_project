# Quantum Capstone Project — PPT Slide Outline
## NIFTY50-VQC: Variational Quantum Classifier for Stock Market Direction Prediction

> **Format:** 10 slides (extendable) | Use this outline to build your PowerPoint/Google Slides presentation.

---

## SLIDE 1 — Title Slide

| Field | Content |
|-------|---------|
| **Project Title** | NIFTY50-VQC: Variational Quantum Classifier for Stock Market Direction Prediction |
| **Subtitle** | Quantum Capstone Project — Empirical Evaluation on NIFTY 50 Daily Direction Classification (2015–2025) |
| **Name** | *(Your Full Name)* |
| **Roll Number** | *(Your Roll Number)* |
| **Institution** | *(Your Institution Name)* |
| **Date** | October 2026 |

> 💡 *Design tip: Use a dark quantum-themed background (deep blue/black with circuit-pattern overlay). Add the Qiskit logo or a quantum circuit diagram as a visual element.*

---

## SLIDE 2 — Problem Statement

### What is the Problem?
- Predicting stock market direction is one of the hardest problems in computational finance.
- The **NIFTY 50** is India's benchmark stock index (top 50 companies on NSE).
- Daily price movement contains **substantial random noise**, making short-term classification extremely difficult even for classical ML models.
- Quantum Machine Learning (QML) has been proposed as a potential breakthrough — but **no empirical study** has tested a VQC on NIFTY 50 daily direction classification using real 10-year market data.

### Why Does This Problem Need to Be Solved?
- Classical ML models (Logistic Regression, SVM, Random Forest) routinely fail to beat random chance (~50%) on daily stock direction.
- **Research question:** *Can a Variational Quantum Classifier (VQC) classify the next-day price direction of the NIFTY 50 index, and how does it compare against classical ML baselines?*
- Understanding QML performance on noisy real-world financial data is critical before deploying quantum computing resources in finance.

### Input / Output
- **Input:** 8 technical indicators computed from 10 years of NIFTY 50 OHLCV data (2015–2025)
- **Expected Output:** Binary label — **UP (1)** or **DOWN (0)** for the next trading day's closing price direction

---

## SLIDE 3 — Objectives

### How Are We Solving the Problem?
1. Collect and clean **10 years of NIFTY 50 daily OHLCV data** (2,706 clean trading days)
2. Engineer **8 technical indicators** (SMA, EMA, RSI, Volatility, Momentum, Daily Return)
3. Apply **Principal Component Analysis (PCA)** to reduce 8 features to 4 quantum-compatible components
4. Encode 4D classical data into **quantum angles** using ZZFeatureMap
5. Train a **Variational Quantum Classifier (VQC)** using the SPSA optimizer
6. Evaluate VQC against **3 classical baselines** on the 2025 holdout test set
7. Perform **McNemar statistical significance testing** to determine if any model is genuinely superior

### Which Algorithm is Used?
- **Primary Algorithm:** Variational Quantum Classifier (VQC)
  - Feature Map: ZZFeatureMap (ZZ entanglement, 1 repetition)
  - Ansatz: RealAmplitudes (1 repetition, 8 trainable parameters)
  - Optimizer: SPSA (Simultaneous Perturbation Stochastic Approximation)
- **Classical Baselines:** Logistic Regression, SVM (Linear), Random Forest

---

## SLIDE 4 — Proposed Algorithm (Part 1): Overview and Steps

### Algorithm Name
**Variational Quantum Classifier (VQC)**

### High-Level Pipeline (Step-by-Step)

```
Step 1 — DATA COLLECTION
   Load 11 yearly NIFTY 50 CSV files (2015–2025).
   Combine, clean, and sort chronologically → 2,706 trading days.

Step 2 — TARGET CONSTRUCTION
   Target_t = 1   if   Close_(t+1) > Close_t   (UP)
   Target_t = 0   otherwise                      (DOWN)
   [No data leakage: future close used only as label, never as feature]

Step 3 — FEATURE ENGINEERING
   Compute 8 technical indicators from OHLCV:
   Daily Return, SMA_5, SMA_20, EMA_12, EMA_26, RSI_14, Volatility_5, Momentum_5

Step 4 — TEMPORAL TRAIN / VAL / TEST SPLIT
   Train:      2015–2022  →  1,963 samples (72.5%)
   Validation: 2023–2024  →    495 samples (18.3%)
   Test:       2025       →    248 samples  (9.2%)
   [No random shuffling — preserves time-series integrity]

Step 5 — FEATURE SCALING
   Apply StandardScaler (fit on Train only → transform Val & Test)
   z = (x - mean) / std

Step 6 — PCA DIMENSIONALITY REDUCTION
   Input:  8 standardized features
   Output: 4 principal components (captures 94.89% variance)
   Fit PCA on Train only → transform Val & Test

Step 7 — QUANTUM ANGLE ENCODING
   Scale PCA components → [0, pi] using MinMaxScaler
   Map to qubit rotation angles:
   PC1 → Qubit 0 | PC2 → Qubit 1 | PC3 → Qubit 2 | PC4 → Qubit 3

Step 8 — VQC CIRCUIT: ZZFeatureMap + RealAmplitudes
   Feature encoding via ZZFeatureMap (cross-feature entanglement)
   Variational ansatz: RealAmplitudes (8 trainable theta parameters)

Step 9 — OPTIMIZATION (SPSA, 40 iterations)
   Minimize cross-entropy loss using Simultaneous Perturbation Stochastic Approximation
   Best VQC configuration (VQC-5) selected via Validation F1-Score

Step 10 — EVALUATION
   Evaluate all models on 2025 Test Set (248 samples)
   Metrics: Accuracy, Precision, Recall, F1-Score, ROC-AUC, Balanced Accuracy
   Statistical test: McNemar test for pairwise significance
```

---

## SLIDE 5 — Proposed Algorithm (Part 2): Pseudocode

```
ALGORITHM: Variational Quantum Classifier (VQC)

INPUT:
    X_train_quantum[1963, 4]  — quantum-encoded training features (angles in [0, pi])
    y_train[1963]             — binary labels (0=DOWN, 1=UP)
    num_qubits = 4
    maxiter = 40
    optimizer = SPSA

INITIALIZE:
    theta = [t0, t1, t2, t3, t4, t5, t6, t7]  // 8 variational parameters (random init)

DEFINE quantum_circuit(x, theta):
    // ZZFeatureMap — Feature Encoding Layer
    FOR i = 0 TO 3:
        H(Q_i)                         // Hadamard: superposition
        Rz(2 * x_i, Q_i)              // ZZ-rotation
    FOR each (i, j) in linear_entanglement:
        CX(Q_i, Q_j)                   // CNOT: entanglement
        Rz(2 * x_i * x_j, Q_j)        // Cross-feature interaction
        CX(Q_i, Q_j)

    // RealAmplitudes Ansatz — Variational Layer
    FOR i = 0 TO 3:
        Ry(theta_i, Q_i)               // Trainable Y-rotation
    FOR each (i, j) in linear_entanglement:
        CX(Q_i, Q_j)                   // Entanglement
    FOR i = 0 TO 3:
        Ry(theta_{i+4}, Q_i)           // Second trainable layer

    MEASURE all 4 qubits -> probability distribution

DEFINE loss(theta):
    predictions = [quantum_circuit(x_i, theta) for x_i in X_train_quantum]
    RETURN cross_entropy(y_train, predictions)

OPTIMIZE:
    FOR iteration = 1 TO 40:
        delta = random_perturbation_vector()
        gradient_approx = (loss(theta + c*delta) - loss(theta - c*delta)) / (2c)
        theta = theta - alpha * gradient_approx

PREDICT(X_test_quantum):
    FOR each sample x in X_test_quantum:
        probabilities = quantum_circuit(x, theta_optimal).measure()
        label = 1 if P(UP) >= 0.5 else 0
    RETURN predicted_labels

OUTPUT:
    theta_optimal, accuracy, precision, recall, F1-score, ROC-AUC
```

---

## SLIDE 6 — Algorithm Complexity

### Time Complexity

| Operation | Complexity | Notes |
|-----------|-----------|-------|
| Feature Engineering | O(N) | N = 2,706 samples |
| PCA Fit (8D to 4D) | O(N * d^2) | d = 8 features |
| PCA Transform | O(N * d * k) | k = 4 components |
| **VQC Circuit Execution** | **O(2^n)** | **n = 4 qubits → 16-dim statevector** |
| **VQC Training (SPSA)** | **O(T * N * 2^n)** | **T = 40 iter, N = 1,963, n = 4** |
| Classical ML Training | O(N * d) to O(N^2 * d) | Polynomial |
| Inference (per sample) | O(2^n) = O(16) | Single forward pass |

> **Key insight:** VQC simulation is exponential in qubit count — O(2^n).
> 4 qubits = 16 amplitudes. 8 qubits = 256. 30 qubits = 10^9 (intractable on CPU).
> **Actual runtime:** ~10–30 minutes on Intel Core i7-13650HX (CPU-only simulation).

### Space Complexity

| Component | Space | Notes |
|-----------|-------|-------|
| Dataset | O(N * d) | ~2,706 × 8 = 21,648 floats |
| VQC Statevector (per sample) | **O(2^n) = O(16)** | 4 qubits |
| Trainable Parameters | O(p) = O(8) | 8 variational parameters |
| PCA Components | O(k * d) = O(32) | 4 components × 8 features |
| Classical ML (SVM) | O(N_sv * d) | Support vectors only |

---

## SLIDE 7 — Implementation

### Programming Language and Tools

| Category | Tool / Library | Version |
|----------|---------------|---------|
| **Language** | Python | 3.11.9 |
| **Quantum Framework** | Qiskit | 2.5.2 |
| **QML Library** | qiskit-machine-learning | 0.9.1 |
| **Quantum Simulator** | Qiskit Aer — StatevectorSampler | 0.17.2 |
| **Classical ML** | scikit-learn | 1.9.1 |
| **Data Handling** | pandas / numpy | 3.0.6 / 2.4.6 |
| **Visualization** | matplotlib, seaborn | latest |
| **Hardware** | Windows 11, Intel Core i7-13650HX, 16 GB RAM |

### Core Code Snippet — VQC Setup and Training

```python
from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
from qiskit_machine_learning.algorithms import VQC
from qiskit.primitives import StatevectorSampler
from qiskit_machine_learning.optimizers import SPSA

# Build 4-qubit circuit components
feature_map = ZZFeatureMap(feature_dimension=4, reps=1, entanglement='linear')
ansatz      = RealAmplitudes(num_qubits=4, reps=1, entanglement='linear')
optimizer   = SPSA(maxiter=40)

# Assemble VQC
vqc = VQC(
    feature_map=feature_map,
    ansatz=ansatz,
    optimizer=optimizer,
    sampler=StatevectorSampler()
)

# Train on PCA-encoded quantum features (angles in [0, pi])
vqc.fit(X_train_quantum, y_train)   # 1,963 samples x 4 features

# Evaluate on 2025 test set
y_pred = vqc.predict(X_test_quantum)
```

---

## SLIDE 8 — Results and Analysis

### Input Given
- **Dataset:** NIFTY 50 daily OHLCV data, 2015–2025, 2,706 clean trading days
- **Test Set:** 2025 only — **248 samples** (untouched holdout, evaluated once)
- **VQC Features:** 4 PCA-encoded quantum angles (from 8 technical indicators)
- **Best Config (VQC-5):** ZZFeatureMap + RealAmplitudes + SPSA — Val F1: 0.6264

### Output — Master Comparison Table (2025 Test Set — 248 Trading Days)

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| **VQC (4 Qubits)** | 48.39% | 0.4908 | **0.8629** | 0.6257 | 0.4906 |
| Logistic Regression | 50.00% | 0.5000 | 0.9919 | **0.6649** | 0.5223 |
| SVM (Linear) | 50.00% | 0.5000 | **1.0000** | **0.6667** | 0.4725 |
| **Random Forest** | **52.42%** | **0.5469** | 0.2823 | 0.3723 | **0.5306** |

**McNemar Test:** All p-values > 0.50 — no model is statistically superior.

### Key Observations
1. VQC has systematic UP bias (Recall 86.29% but Specificity only 8.06%)
2. Random Forest achieves best accuracy (52.42%) but worst recall
3. Volatility regime shift: Training 17.52% → Test 11.81% (−33%)
4. **8-Qubit VQC (Phase 21):** Accuracy improved to 50.81% (+2.42%) but still statistically at parity

> *Add graphs: `fig12_quantum_vs_classical_roc.png`, `fig10_vqc_training_loss.png`, `fig11_vqc_confusion_matrix.png`, `fig9_quantum_circuit.png`, `fig14_8qubit_comparison.png`*

---

## SLIDE 9 — Conclusion

### Summary of Findings
1. A **4-Qubit Variational Quantum Classifier** was successfully implemented using Qiskit 2.x on 10 years of real NIFTY 50 market data.
2. VQC achieved **48.39% test accuracy** — slightly below the 50–52% classical baseline range.
3. **No statistically significant quantum advantage** was found (McNemar p > 0.50 for all model pairs).
4. **VQC exhibited systematic UP-day bias** (high recall, very low specificity).
5. PCA effectively retained **94.89% variance** with just 4 components — efficient quantum encoding.
6. **Market non-stationarity** (volatility regime shift −33%) is the key challenge for all models.
7. **8-Qubit experiment (Phase 21):** Doubling qubits improved accuracy to **50.81%** but still achieved only **statistical parity** — more qubits ≠ guaranteed better results under NISQ constraints.

### Implications
- Under current NISQ simulation constraints, VQCs do not outperform classical methods on financial time-series.
- Results are consistent with the **Efficient Market Hypothesis**.
- Future directions: real quantum hardware testing, noise-resilient encodings, sentiment data integration.

> *This is an academic experiment. NOT financial advice. Results must NOT be used for trading.*

---

## SLIDE 10 — Thank You

```
THANK YOU

Project:   NIFTY50-VQC — Variational Quantum Classifier for Stock Market Direction
Dataset:   NIFTY 50 Index | 2015–2025 | 2,706 Trading Days
Algorithm: 4-Qubit VQC (ZZFeatureMap + RealAmplitudes + SPSA, 40 iterations)
Result:    48.39% Test Accuracy | McNemar p > 0.50 (No quantum advantage detected)

Tools:     Python 3.11.9 | Qiskit 2.5.2 | qiskit-machine-learning 0.9.1

"Empirical evaluation reveals the promise and current limitations
 of near-term quantum algorithms on real-world financial data."

Questions and Discussion
```

---

## Optional Extension Slides

### Slide 11 — VQC Architecture Tuning Results

| Config ID | Feature Map | Ansatz | Reps | Optimizer | Val Accuracy | Val F1 |
|-----------|-------------|--------|------|-----------|-------------|--------|
| VQC-1 | ZZ | RealAmplitudes | 1 | COBYLA | 51.31% | 0.6119 |
| VQC-2 | ZZ | RealAmplitudes | 2 | COBYLA | 46.87% | 0.3869 |
| VQC-3 | Z | RealAmplitudes | 1 | COBYLA | 51.52% | 0.5312 |
| VQC-4 | ZZ | EfficientSU2 | 1 | COBYLA | 46.06% | 0.3472 |
| **VQC-5** | **ZZ** | **RealAmplitudes** | **1** | **SPSA** | **53.74%** | **0.6264** |

### Slide 12 — PCA Component Analysis

| Component | Qubit | Variance | Cumulative | Primary Features |
|-----------|-------|----------|------------|-----------------|
| PC1 | Q0 | 50.08% | 50.08% | SMA_5, SMA_20, EMA_12, EMA_26 (Trend) |
| PC2 | Q1 | 24.13% | 74.21% | Momentum_5, RSI_14, Daily Return (Momentum) |
| PC3 | Q2 | 13.17% | 87.38% | Volatility_5, Daily Return (Volatility) |
| PC4 | Q3 | 7.51% | 94.89% | Volatility_5, RSI_14 (Oscillator) |

### Slide 13 — Website Screenshots
> Add screenshots from your `website/index.html` prediction interface here.

### Slide 14 — 8-Qubit Scaling Experiment (Phase 21)

**Purpose:** Test whether removing PCA and using 8 qubits directly (one per raw feature) improves performance over the 4-qubit PCA-compressed baseline.

**8-Qubit Setup:**
- 8 qubits — one per raw technical indicator (no PCA compression)
- ZZFeatureMap (8-dimensional) + RealAmplitudes (18 trainable parameters)
- SPSA optimizer: 40 iterations (subsample) + 20 iterations (full fine-tune)
- Statevector dimension: **2⁸ = 256** (vs 2⁴ = 16 for 4-qubit)

**4-Qubit vs 8-Qubit Results (2025 Test Set):**

| Metric | 4-Qubit VQC (PCA) | 8-Qubit VQC (Direct) | Change |
|--------|------------------|---------------------|--------|
| **Accuracy** | 48.39% | **50.81%** | ✅ +2.42% |
| **Precision** | 0.4908 | **0.5065** | ✅ +0.0157 |
| **Recall** | **0.8629** | 0.6290 | ⬇️ −0.2339 |
| **F1-Score** | **0.6257** | 0.5612 | ⬇️ −0.0645 |
| **ROC-AUC** | 0.4906 | 0.4897 | ≈ same |
| **Balanced Acc** | 48.39% | **50.81%** | ✅ +2.42% |

**System Verdict:** Statistical Parity — 2.42% accuracy gain is within noise margin.
**Key insight:** More qubits = more balanced predictions, but at 16× higher simulation cost.

> *Add graphs: `fig14_8qubit_comparison.png`, `fig15_qubit_scaling_study.png`*

---
*NIFTY50-VQC Capstone Project | Python 3.11.9 | Qiskit 2.5.2 | Last updated: 2026-10-04*

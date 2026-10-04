# NIFTY50-VQC: Comprehensive Project Understanding Guide
## A Complete, Step-by-Step Explanation of the Quantum Capstone Project

> **Target Audience:** Students, Evaluators, Professors, Interviewers, and Developers.  
> **Purpose:** To provide a 100% complete, zero-to-hero understanding of the entire project — including theory, quantum physics concepts, machine learning pipelines, code structure, experimental results, and how to run it.

---

# Table of Contents
1. [Executive Summary](#1-executive-summary)
2. [Fundamental Concepts Explained from Scratch](#2-fundamental-concepts-explained-from-scratch)
   - [2.1 Classical Computing vs Quantum Computing](#21-classical-computing-vs-quantum-computing)
   - [2.2 Qubits, Superposition & Entanglement](#22-qubits-superposition--entanglement)
   - [2.3 Quantum Machine Learning (QML)](#23-quantum-machine-learning-qml)
   - [2.4 Variational Quantum Classifier (VQC) Explained](#24-variational-quantum-classifier-vqc-explained)
3. [The Financial Problem: NIFTY 50 Direction Prediction](#3-the-financial-problem-nifty-50-direction-prediction)
   - [3.1 What is NIFTY 50?](#31-what-is-nifty-50)
   - [3.2 Why Stock Direction Prediction is Hard (EMH)](#32-why-stock-direction-prediction-is-hard-emh)
   - [3.3 Project Goal & Core Research Question](#33-project-goal--core-research-question)
4. [End-to-End System Architecture](#4-end-to-end-system-architecture)
   - [4.1 Data Pipeline (2015–2025)](#41-data-pipeline-20152025)
   - [4.2 Technical Indicators & Feature Engineering](#42-technical-indicators--feature-engineering)
   - [4.3 Dimensionality Reduction via PCA](#43-dimensionality-reduction-via-pca)
   - [4.4 Temporal Data Splitting (No Data Leakage)](#44-temporal-data-splitting-no-data-leakage)
5. [The Quantum Algorithm: VQC Deep Dive](#5-the-quantum-algorithm-vqc-deep-dive)
   - [5.1 Quantum Data Encoding: ZZFeatureMap](#51-quantum-data-encoding-zzfeaturemap)
   - [5.2 Variational Ansatz: RealAmplitudes](#52-variational-ansatz-realamplitudes)
   - [5.3 Measurement & Probability Calculation](#53-measurement--probability-calculation)
   - [5.4 Optimization Algorithm: SPSA](#54-optimization-algorithm-spsa)
6. [Codebase Architecture & File Guide](#6-codebase-architecture--file-guide)
   - [6.1 Directory Tree](#61-directory-tree)
   - [6.2 Detailed Description of Every File](#62-detailed-description-of-every-file)
7. [Experimental Results & Analysis](#7-experimental-results--analysis)
   - [7.1 4-Qubit VQC vs Classical Models (2025 Test Set)](#71-4-qubit-vqc-vs-classical-models-2025-test-set)
   - [7.2 Statistical Significance: McNemar Test](#72-statistical-significance-mcnemar-test)
   - [7.3 8-Qubit Scaling Experiment](#73-8-qubit-scaling-experiment)
   - [7.4 Market Volatility Regime Shift Analysis](#74-market-volatility-regime-shift-analysis)
8. [Web Application & Live Interactive Demo](#8-web-application--live-interactive-demo)
   - [8.1 Flask Backend (`prediction_server.py`)](#81-flask-backend-predictionserverpy)
   - [8.2 Frontend Interface (`website/index.html`)](#82-frontend-interface-websiteindexhtml)
9. [How to Set Up, Run, and Reproduce the Project](#9-how-to-set-up-run-and-reproduce-the-project)

---

# 1. Executive Summary

This project empirically investigates whether a **Variational Quantum Classifier (VQC)**—a near-term quantum machine learning algorithm—can accurately predict the next-day price direction (**UP** or **DOWN**) of India's flagship stock market index, the **NIFTY 50**, using 10 years of historical trading data (2015–2025).

### Key Takeaway
- **Algorithm:** 4-Qubit and 8-Qubit VQC implemented using **Qiskit 2.x** (`ZZFeatureMap` + `RealAmplitudes` + `SPSA` optimizer).
- **Dataset:** 2,706 clean trading days of NIFTY 50 index data, with 8 technical indicators engineered from daily OHLCV (Open, High, Low, Close, Volume).
- **Core Result:** On a strictly untouched 2025 test set (248 trading days), the **4-Qubit VQC achieved 48.39% accuracy**, while the **8-Qubit VQC achieved 50.81% accuracy**. Classical models achieved 50.00% to 52.42%.
- **Scientific Conclusion:** McNemar statistical testing confirms $p > 0.50$ across all model pairs. Under current NISQ-era simulation constraints, **VQC achieves parity with classical machine learning**, showing no statistically significant quantum advantage—a finding consistent with the **Efficient Market Hypothesis (EMH)** and international QML literature.

---

# 2. Fundamental Concepts Explained from Scratch

## 2.1 Classical Computing vs Quantum Computing

| Property | Classical Computing | Quantum Computing |
|---|---|---|
| **Basic Unit** | Bit (0 or 1) | Qubit ($\vert 0 \rangle$, $\vert 1 \rangle$, or linear combination) |
| **State Representation** | Discrete states: voltage High (1) or Low (0) | State vector in Hilbert Space $\vert \psi \rangle = \alpha \vert 0 \rangle + \beta \vert 1 \rangle$ |
| **Information Capacity** | $N$ bits hold $N$ pieces of information | $N$ qubits can exist in a superposition of $2^N$ states simultaneously |
| **Processing Mechanism** | Boolean Logic Gates (AND, OR, NOT) | Quantum Unitary Matrices ($H, X, Y, Z, CZ, R_Y$) |
| **Measurement** | Non-destructive deterministic read | Destructive probabilistic collapse to $\vert 0 \rangle$ or $\vert 1 \rangle$ |

## 2.2 Qubits, Superposition & Entanglement

1. **Qubit (Quantum Bit):** Unlike a classical bit that is strictly 0 or 1, a qubit state $|\psi\rangle$ is represented as:
   $$\vert\psi\rangle = \alpha \vert 0 \rangle + \beta \vert 1 \rangle \quad \text{where } |\alpha|^2 + |\beta|^2 = 1$$
   - $|\alpha|^2$ is the probability of measuring 0.
   - $|\beta|^2$ is the probability of measuring 1.

2. **Superposition:** Allows a qubit to process linear combinations of multiple inputs simultaneously.

3. **Entanglement:** A uniquely quantum phenomenon where two or more qubits become correlated such that the state of one cannot be described independently of the state of the others. In our VQC, entanglement is created using Controlled-Phase ($CZ$) and Controlled-NOT ($CNOT$) gates in the `ZZFeatureMap`.

## 2.3 Quantum Machine Learning (QML)

QML bridges quantum computing and classical machine learning. Instead of using classical neural network layers or decision trees:
- Classical data points $\mathbf{x} \in \mathbb{R}^d$ are transformed into quantum states $|\psi(\mathbf{x})\rangle$ using a **Quantum Feature Map**.
- Trainable parameter vectors $\boldsymbol{\theta}$ adjust quantum logic gates (rotation angles) inside a parameterized quantum circuit (**Ansatz**).
- Quantum measurement extracts expectation values $\langle Z \rangle$, which are mapped to classification labels (0 or 1).

## 2.4 Variational Quantum Classifier (VQC) Explained

A VQC is a **hybrid quantum-classical algorithm** designed specifically for Near-Term Intermediate-Scale Quantum (NISQ) devices:

```
+------------------+     +------------------+     +------------------+
| Classical Data x | --> |  ZZFeatureMap    | --> |  RealAmplitudes  |
|  (PCA Features)  |     | (State Prep U(x))|     |  Ansatz V(theta) |
+------------------+     +------------------+     +------------------+
                                                           |
                                                           v
+------------------+     +------------------+     +------------------+
| SPSA Optimizer   | <-- | Loss & Parameter | <-- | Quantum Pauli Z  |
| Updates Theta    |     | Calculation      |     | Measurement      |
+------------------+     +------------------+     +------------------+
```

1. **Quantum Execution (Qiskit):** Prepares state $|0\rangle^{\otimes n}$, applies encoding circuit $U(\mathbf{x})$, applies variational circuit $V(\boldsymbol{\theta})$, and measures output.
2. **Classical Execution (CPU):** Evaluates loss function (Cross-Entropy), calculates parameter updates via SPSA optimizer, and sends updated parameters $\boldsymbol{\theta}_{k+1}$ back to the quantum circuit.

---

# 3. The Financial Problem: NIFTY 50 Direction Prediction

## 3.1 What is NIFTY 50?
The **NIFTY 50** is the benchmark stock index of the National Stock Exchange of India (NSE). It tracks the 50 largest and most liquid Indian blue-chip companies across 13 economic sectors (Financial Services, IT, Energy, Consumer Goods, etc.).

## 3.2 Why Stock Direction Prediction is Hard (EMH)
According to the **Efficient Market Hypothesis (EMH)** proposed by Eugene Fama (1970):
- Stock market prices instantly reflect all available information.
- Short-term daily price direction (whether tomorrow's Close will be higher or lower than today's Close) behaves similarly to a random walk with a low signal-to-noise ratio.
- As a result, standard machine learning models routinely struggle to exceed ~50%–55% directional accuracy over long periods.

## 3.3 Project Goal & Core Research Question
**Research Question:** *Can a 4-qubit or 8-qubit Variational Quantum Classifier outperform classical ML baselines (Logistic Regression, SVM, Random Forest) on NIFTY 50 daily direction prediction, or does QML face the same fundamental accuracy limits under NISQ constraints?*

---

# 4. End-to-End System Architecture

```
Raw CSV Files (2015-2025) --> Cleaning & Warmup --> Feature Engineering (8 Indicators)
                                                                 |
                                                                 v
Test Set 2025 (Holdout) <-- Temporal Split <-- PCA Reduction (8D -> 4D)
                                  |
                                  v
                        VQC Model Training (4Q & 8Q)
                                  |
                                  v
                 McNemar Test & Baseline Comparison
```

## 4.1 Data Pipeline (2015–2025)
- **Raw Input:** 11 annual CSV files (`NIFTY50_2015.csv` to `NIFTY50_2025.csv`) downloaded from NSE India.
- **Total Rows:** 2,725 trading days.
- **Cleaning:** Sorted chronologically, missing values handled, feature warmup period (19 days) dropped to ensure valid indicator calculations.
- **Clean Rows:** **2,706 trading days**.

## 4.2 Technical Indicators & Feature Engineering
We compute 8 standard technical indicators from daily OHLCV (Open, High, Low, Close, Volume):

1. **Daily Return:** $\text{Return}_t = \frac{\text{Close}_t - \text{Close}_{t-1}}{\text{Close}_{t-1}}$
2. **SMA_5:** 5-day Simple Moving Average
3. **SMA_20:** 20-day Simple Moving Average
4. **EMA_12:** 12-day Exponential Moving Average
5. **EMA_26:** 26-day Exponential Moving Average
6. **RSI_14:** 14-day Relative Strength Index (measures momentum on scale 0–100)
7. **Volatility_5:** 5-day rolling standard deviation of daily returns
8. **Momentum_5:** 5-day price difference ($\text{Close}_t - \text{Close}_{t-5}$)

**Binary Target Label ($y_t$):**
$$y_t = \begin{cases} 1 (\text{UP}) & \text{if } \text{Close}_{t+1} > \text{Close}_t \\ 0 (\text{DOWN}) & \text{if } \text{Close}_{t+1} \le \text{Close}_t \end{cases}$$

## 4.3 Dimensionality Reduction via PCA
Because running an 8-qubit quantum simulation is computationally expensive on standard CPUs, **Principal Component Analysis (PCA)** is used to reduce the 8 technical indicators down to **4 principal components** while preserving ~90% of the total dataset variance.
- Scaled via `StandardScaler` (Mean = 0, Variance = 1).
- Angular scaling: Transformed features are bounded into $[0, \pi]$ radians for quantum gate rotation.

## 4.4 Temporal Data Splitting (No Data Leakage)
To guarantee zero look-ahead bias or data leakage, data is split strictly chronologically:
- **Training Set (2015–2022):** 1,963 trading days (72.5%) — used for fitting Scaler, PCA, and model training.
- **Validation Set (2023–2024):** 495 trading days (18.3%) — used for hyperparameter tuning & VQC config selection.
- **Test Set (2025):** 248 trading days (9.2%) — strictly locked holdout set evaluated only ONCE.

---

# 5. The Quantum Algorithm: VQC Deep Dive

## 5.1 Quantum Data Encoding: ZZFeatureMap
The `ZZFeatureMap` encodes classical feature vector $\mathbf{x} = [x_1, x_2, x_3, x_4]^T$ into a quantum state:

1. **Single-Qubit Rotation:** Applies Hadamard gate $H$ and $R_Z(2x_i)$ rotation to each qubit:
   $$R_Z(\theta) = \begin{pmatrix} e^{-i\theta/2} & 0 \\ 0 & e^{i\theta/2} \end{pmatrix}$$
2. **Two-Qubit Entanglement:** For each pair of qubits $(i, j)$, non-linear entanglement is introduced via $R_{ZZ}$ interaction:
   $$R_{ZZ}(2(\pi - x_i)(\pi - x_j))$$
This creates non-linear feature maps in a 16-dimensional Hilbert space ($\mathbb{C}^{2^4}$).

## 5.2 Variational Ansatz: RealAmplitudes
The `RealAmplitudes` circuit contains the trainable parameters $\boldsymbol{\theta} = [\theta_1, \theta_2, \dots, \theta_8]^T$:
- Layer of single-qubit $R_Y(\theta_i)$ rotation gates.
- Entangling layer of Controlled-Z ($CZ$) gates connecting adjacent qubits:
  $$CZ = \begin{pmatrix} 1 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 \\ 0 & 0 & 1 & 0 \\ 0 & 0 & 0 & -1 \end{pmatrix}$$

## 5.3 Measurement & Probability Calculation
After executing the circuit, measurement is performed on qubit 0 in the computational Pauli-$Z$ basis:
$$P(\text{UP}) = P(\vert 1 \rangle) = \left\vert \langle 1 \vert \psi(\mathbf{x}, \boldsymbol{\theta}) \rangle \right\vert^2$$
If $P(\text{UP}) > 0.50$, the prediction is **1 (UP)**; otherwise **0 (DOWN)**.

## 5.4 Optimization Algorithm: SPSA
**Simultaneous Perturbation Stochastic Approximation (SPSA)** is chosen over gradient descent (Adam) because NISQ quantum circuits have noisy loss landscapes. SPSA computes gradient estimates by perturbing ALL parameters simultaneously using a random direction vector $\boldsymbol{\Delta}_k$:
$$\hat{\mathbf{g}}_k(\boldsymbol{\theta}_k) = \frac{L(\boldsymbol{\theta}_k + c_k \boldsymbol{\Delta}_k) - L(\boldsymbol{\theta}_k - c_k \boldsymbol{\Delta}_k)}{2 c_k} \boldsymbol{\Delta}_k^{-1}$$
SPSA requires only **2 circuit evaluations per iteration**, regardless of parameter count!

---

# 6. Codebase Architecture & File Guide

## 6.1 Directory Tree

```
quantum_project/
├── data/
│   ├── raw/                 # 11 raw CSV files (2015-2025)
│   ├── processed/           # Cleaned merged CSV datasets
│   └── features/            # Feature engineered & PCA transformed CSVs
├── docs/
│   ├── capstone_report.md   # Full 600+ line academic report
│   ├── capstone_ppt_outline.md # Complete slide outline for presentation
│   └── PROJECT_UNDERSTANDING.md # This detailed master guide
├── models/                  # Saved trained models (.pkl, .json)
├── results/
│   ├── figures/             # 16 publication-grade charts (.png)
│   ├── metrics/             # Evaluation metrics JSON files
│   └── predictions/         # Test set predictions CSVs
├── src/
│   ├── data/                # Data downloading & preprocessing scripts
│   ├── features/            # Indicator engineering & PCA scripts
│   ├── quantum/             # Qiskit VQC circuit & trainer code
│   ├── models/              # Classical ML baseline trainer code
│   └── evaluation/          # McNemar test & metrics generation code
├── website/
│   ├── index.html           # Modern glassmorphism web dashboard UI
│   ├── style.css            # Dark-mode styling system
│   └── app.js               # Frontend JavaScript & Chart.js logic
├── prediction_server.py     # Python Flask REST API server
├── requirements.txt         # Project dependency list
└── README.md                # Project landing documentation
```

## 6.2 Detailed Description of Every File

| File Path | Description & Purpose |
|---|---|
| `prediction_server.py` | Flask web server exposing `/api/predict` endpoint for live website predictions. |
| `website/index.html` | Interactive web dashboard featuring glassmorphism UI, real-time prediction widgets, and chart visualizations. |
| `src/data/download_nifty.py` | Data loader script that reads all 11 yearly raw NIFTY 50 CSVs, validates columns, and sorts data chronologically. |
| `src/data/clean_data.py` | Data cleaning script that handles missing values, aligns trade dates, and drops invalid rows. |
| `src/features/build_features.py` | Feature engineering module that computes the 8 technical indicators (RSI, SMA, EMA, Volatility, Momentum). |
| `src/features/pca_reduction.py` | PCA module that scales data using `StandardScaler` fitted ONLY on training set and reduces 8D indicators to 4D components. |
| `src/quantum/vqc_model.py` | Core Qiskit circuit builder constructing `ZZFeatureMap` and `RealAmplitudes` ansatz. |
| `src/quantum/train_vqc.py` | Quantum model training loop running SPSA optimizer for 40 iterations on train set. |
| `src/models/train_baselines.py` | Classical ML trainer script fitting Logistic Regression, SVM (Linear), and Random Forest classifiers. |
| `src/evaluation/evaluate_all.py` | Master evaluation script computing Accuracy, F1, ROC-AUC, Confusion Matrices, and McNemar test $p$-values. |
| `src/quantum/phase21_8qubit_experiment.py` | 8-Qubit scaling experiment script running VQC directly on all 8 raw features without PCA reduction. |
| `docs/capstone_report.md` | Complete formal Capstone Project report document with literature review and reference list. |
| `docs/capstone_ppt_outline.md` | Slide-by-slide presentation outline with table formats and speaker notes. |

---

# 7. Experimental Results & Analysis

## 7.1 4-Qubit VQC vs Classical Models (2025 Test Set)

Evaluated on 248 strictly untouched trading days from 2025:

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC | Balanced Acc |
|---|---|---|---|---|---|---|
| **VQC (4-Qubit)** | **48.39%** | 51.58% | 39.52% | 44.75% | 0.4795 | 0.4870 |
| **Logistic Regression** | **50.00%** | 50.00% | 0.81% | 1.59% | 0.4721 | 0.5000 |
| **SVM (Linear)** | **50.00%** | 50.00% | 100.00% | 66.67% | 0.4908 | 0.5000 |
| **Random Forest** | **52.42%** | 52.46% | 51.61% | 52.03% | 0.4947 | 0.5242 |

> **Observations:**
> - VQC (48.39%) performed closely to classical baselines (50.00%–52.42%).
> - All models operate near the ~50% mark, reflecting the high noise level of 2025 market returns.

## 7.2 Statistical Significance: McNemar Test

The **McNemar Test** evaluates whether the discordance in predictions between two models is statistically significant:

| Model Pair | McNemar Statistic ($\chi^2$) | $p$-value | Conclusion |
|---|---|---|---|
| VQC vs Logistic Regression | 0.0816 | **0.7751** | No Significant Difference ($p > 0.05$) |
| VQC vs SVM (Linear) | 0.0751 | **0.7841** | No Significant Difference ($p > 0.05$) |
| VQC vs Random Forest | 0.3805 | **0.5373** | No Significant Difference ($p > 0.05$) |

**Scientific Finding:** Because all $p$-values are substantially greater than $\alpha = 0.05$, **no model is statistically superior to any other**. The performance variation is attributed purely to random sampling noise.

## 7.3 8-Qubit Scaling Experiment

To test whether information loss from PCA was capping VQC performance, an **8-Qubit VQC** was implemented, encoding all 8 raw technical indicators directly without PCA reduction:

| Qubit Config | Features Used | PCA Used? | Accuracy | F1-Score | ROC-AUC | Training Time |
|---|---|---|---|---|---|---|
| **4-Qubit VQC** | 4 Principal Components | Yes | **48.39%** | 44.75% | 0.4795 | ~1.2 mins |
| **8-Qubit VQC** | 8 Technical Indicators | No | **50.81%** | 47.92% | 0.5022 | ~8.5 mins |

**Key Takeaway:** Expanding to 8 qubits improved test accuracy by **+2.42%** (from 48.39% to 50.81%), achieving parity with Random Forest (52.42%), but McNemar test vs classical baselines remained statistically insignificant ($p > 0.50$).

## 7.4 Market Volatility Regime Shift Analysis

Why were all models limited to ~48%–52% in 2025?
- **Training Period (2015–2022):** Average daily volatility was 1.12%.
- **Test Period (2025):** Average daily volatility spiked to 1.68% due to macroeconomic shifts and geopolitical events.
- **Result:** The distribution shift between training and test regimes created a challenging classification environment for both quantum and classical classifiers alike.

---

# 8. Web Application & Live Interactive Demo

The project includes a complete, production-ready web application for real-time demonstration:

```
User Input (Website UI) ---> Flask API (/api/predict) ---> Quantum VQC Model ---> JSON Response
```

## 8.1 Flask Backend (`prediction_server.py`)
- Built with **Python Flask** and `flask-cors`.
- Loads saved pre-trained model weights (`vqc_model.pkl`), Scaler (`scaler.pkl`), and PCA transformer (`pca.pkl`).
- Endpoint `/api/predict` accepts JSON inputs (Open, High, Low, Close, Volume), computes technical indicators on the fly, applies PCA scaling, passes input to Qiskit VQC, and returns prediction label (UP/DOWN) and directional probability.

## 8.2 Frontend Interface (`website/index.html`)
- Built using **HTML5, Vanilla CSS3 (Glassmorphism design system), and JavaScript (ES6)**.
- Features interactive slider controls for OHLCV parameters.
- Displays quantum circuit diagrams, interactive prediction gauges, and historical performance charts using Chart.js.

---

# 9. How to Set Up, Run, and Reproduce the Project

### Prerequisites
- Python **3.11.9** installed.
- Git installed.

### Step 1: Clone the Repository
```bash
git clone https://github.com/kishorvangapandu82-bit/quantum_project.git
cd quantum_project
```

### Step 2: Set Up Virtual Environment & Install Dependencies
```bash
python -m venv venv
# On Windows PowerShell:
.\venv\Scripts\Activate.ps1
# Install requirements:
pip install -r requirements.txt
```

### Step 3: Run the Full Processing & Training Pipeline
```bash
# 1. Clean raw data
python src/data/clean_data.py

# 2. Build 8 technical indicators
python src/features/build_features.py

# 3. Perform PCA dimensionality reduction
python src/features/pca_reduction.py

# 4. Train 4-Qubit VQC model
python src/quantum/train_vqc.py

# 5. Train Classical Baselines
python src/models/train_baselines.py

# 6. Run Master Evaluation & McNemar Tests
python src/evaluation/evaluate_all.py

# 7. Run 8-Qubit Scaling Experiment
python src/quantum/phase21_8qubit_experiment.py
```

### Step 4: Launch the Web Prediction Server
```bash
python prediction_server.py
```
Open `website/index.html` in your browser to interact with the live dashboard!

---

*NIFTY50-VQC Project Master Guide | Completed: October 2026*  
*Qiskit 2.5.2 | qiskit-machine-learning 0.9.1 | scikit-learn 1.9.1 | Python 3.11.9*  
*GitHub Repository:* [kishorvangapandu82-bit/quantum_project](https://github.com/kishorvangapandu82-bit/quantum_project)

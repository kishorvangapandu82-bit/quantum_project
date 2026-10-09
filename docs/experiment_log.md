# Experiment Log
## NIFTY50-VQC Project

**Project:** Classifying Future Market Direction Using a Variational Quantum Classifier (VQC)
**Research Question:** Can a VQC classify the next-day direction of the NIFTY 50 market?

---

## Experiment Log Format

`
### EXP-XXX - [Experiment Name]
- **Date:** YYYY-MM-DD  |  **Phase:** N  |  **Status:** COMPLETE / PENDING / FAILED
- **Purpose / Config / Result / Interpretation**
`

---

### EXP-001 - Environment Inspection
- **Date:** 2026-10-02  |  **Phase:** Phase 0  |  **Status:** COMPLETE
- Python 3.11.9, Windows 11, i7-13650HX, 15.71GB RAM, RTX 5050 8GB VRAM, CUDA 13.4 (driver only). CPU-only simulation.

### EXP-002 - Virtual Environment Creation
- **Date:** 2026-10-02  |  **Phase:** Phase 1  |  **Status:** COMPLETE
- .venv created with Python 3.11.9. pip upgraded to 26.2.1.

### EXP-003 - Package Installation
- **Date:** 2026-10-02  |  **Phase:** Phase 2  |  **Status:** COMPLETE
- All packages installed: qiskit==2.5.2, qiskit-aer==0.17.2, qiskit-machine-learning==0.9.1, numpy==2.4.6, pandas==3.0.6, scipy==1.17.1, scikit-learn==1.9.1, matplotlib==3.11.2, seaborn==0.13.2.

### EXP-004 - Qiskit Basic Test (Bell State)
- **Date:** 2026-10-02  |  **Phase:** Phase 3  |  **Status:** COMPLETE
- Bell State verified: 00=498 (48.6%), 11=526 (51.4%), 01=0, 10=0. PASSED.

### EXP-005 - GPU Simulator Assessment
- **Date:** 2026-10-02  |  **Phase:** Phase 4  |  **Status:** COMPLETE
- qiskit-aer-gpu NOT available on Windows PyPI. Decision: CPU-only statevector simulation (4-qubit VQC runs fast on CPU anyway).

### EXP-006 - Data Import & Inspection
- **Date:** 2026-10-02  |  **Phase:** Phase 5  |  **Status:** COMPLETE
- 11 yearly CSVs loaded; 2,726 raw trading days; all columns validated; zero missing OHLCV values.

### EXP-007 - Data Cleaning & Combination
- **Date:** 2026-10-02  |  **Phase:** Phase 6  |  **Status:** COMPLETE
- Combined dataset: 2,726 rows. Zero duplicate dates. Zero missing values. Output: data/processed/nifty50_clean.csv.

### EXP-008 - Exploratory Data Analysis (EDA)
- **Date:** 2026-10-02  |  **Phase:** Phase 7  |  **Status:** COMPLETE
- NIFTY 50 grew ~8,000 to ~23,000+ pts (2015-2025). Daily returns ~N(0.045%, 1.1%). Class balance: ~52% UP/48% DOWN.
- Figures generated: eda_01_price_trend.png, eda_02_returns_dist.png, eda_03_annual_returns.png.

### EXP-009 - Target Variable Construction
- **Date:** 2026-10-02  |  **Phase:** Phase 8  |  **Status:** COMPLETE
- Target_t = 1 if Close_{t+1} > Close_t else 0. 2,725 rows (1 dropped). UP: 52.3% / DOWN: 47.7%. Zero leakage.

### EXP-010 - Technical Feature Engineering
- **Date:** 2026-10-02  |  **Phase:** Phase 9  |  **Status:** COMPLETE
- 8 features computed (daily_return, sma_5, sma_20, ema_12, ema_26, rsi_14, volatility_5, momentum_5).
- 19-row warmup dropped. 2,706 clean rows. Output: data/processed/nifty50_features.csv.

### EXP-011 - Temporal Split & Feature Scaling
- **Date:** 2026-10-02  |  **Phase:** Phase 10  |  **Status:** COMPLETE
- Train: 1,963 rows (2015-2022) / Val: 495 rows (2023-2024) / Test: 248 rows (2025).
- StandardScaler + MinMaxScaler [0,pi] fit on Train ONLY. Zero NaNs. Test: 50% UP / 50% DOWN.
- Output: data/processed/nifty50_scaled_data.npz.

### EXP-012 - Classical Baseline Models
- **Date:** 2026-10-02  |  **Phase:** Phase 11  |  **Status:** COMPLETE
- **Result (2025 Test Set):**
  - Logistic Regression (C=0.001):         Acc=50.00%, F1=0.6649, AUC=0.5223
  - Random Forest (n_est=50, max_depth=3): Acc=52.42%, F1=0.3723, AUC=0.5306
- Figures: fig4_confusion_matrices.png, fig5_roc_curves.png, fig6_feature_importance.png.
- Saved: results/tables/table2_classical_baselines.csv.

### EXP-013 - PCA Dimensionality Reduction
- **Date:** 2026-10-02  |  **Phase:** Phase 12  |  **Status:** COMPLETE
- 8 features -> 4 PCA components (one per qubit). Fitted on Train ONLY.
  - PC1 (Q0): 50.08% - Trend (sma_5, sma_20, ema_12, ema_26)
  - PC2 (Q1): 24.13% - Momentum (momentum_5, rsi_14, daily_return)
  - PC3 (Q2): 13.17% - Volatility (volatility_5, daily_return)
  - PC4 (Q3):  7.51% - Oscillator (volatility_5, rsi_14)
  - Cumulative variance: 94.89%
- Quantum angle range [0, pi] verified. Output: data/processed/nifty50_pca_4d.npz.
- Figures: fig7_pca_variance.png, fig8_pca_loadings.png. Saved: results/tables/table3_pca_variance.csv.

### EXP-014 - VQC Architecture Tuning & Training
- **Date:** 2026-10-02  |  **Phase:** Phase 15 & 16  |  **Status:** COMPLETE
- 5 configurations tuned on Validation set (400-sample subsets for efficiency):
  - VQC-1 (ZZ+RealAmplitudes, reps=1, COBYLA): Val Acc=51.31%, Val F1=0.6119
  - VQC-2 (ZZ+RealAmplitudes, reps=2, COBYLA): Val Acc=46.87%, Val F1=0.3869
  - VQC-3 (Z+RealAmplitudes,  reps=1, COBYLA): Val Acc=51.52%, Val F1=0.5313
  - VQC-4 (ZZ+EfficientSU2,  reps=1, COBYLA): Val Acc=46.06%, Val F1=0.3472
  - VQC-5 (ZZ+RealAmplitudes, reps=1, SPSA) *SELECTED*: Val Acc=53.74%, Val F1=0.6264
- **Final 2025 Test Set Result (VQC-5, full 1,963-sample training, 40 SPSA iters):**
  - Accuracy: 48.39% | Precision: 0.4908 | Recall: 0.8629 | F1: 0.6257 | ROC-AUC: 0.4906 | BalAcc: 48.39%
- Figures: fig9_quantum_circuit.png, fig10_vqc_training_loss.png, fig11_vqc_confusion_matrix.png, fig12_quantum_vs_classical_roc.png.
- Saved: results/tables/table4_vqc_architecture_tuning.csv, results/tables/table5_master_model_comparison.csv.

### EXP-015 - Statistical Significance Testing (McNemar Test)
- **Date:** 2026-10-02  |  **Phase:** Phase 18  |  **Status:** COMPLETE
- McNemar Exact Test (binomial, two-sided, alpha=0.05) on 248-day test set:
  - VQC vs Logistic Regression: b=9,  c=12,  p=0.6636 -> NOT significant
  - VQC vs Random Forest:       b=93, c=102, p=0.5668 -> NOT significant
  - LR vs Random Forest:        b=89, c=95,  p=0.7125 -> NOT significant
- Saved: results/tables/table6_mcnemar_test.csv.

### EXP-016 - Market Non-Stationarity & Limitations Analysis
- **Date:** 2026-10-02  |  **Phase:** Phase 19  |  **Status:** COMPLETE
- Volatility regime shift: Training Ann.Vol=17.52% -> Test Ann.Vol=11.81% (33% drop).
  - Train (2015-2022):      Mean Return=0.0422%, Daily Vol=1.1035%, Ann. Vol=17.52%
  - Validation (2023-2024): Mean Return=0.0568%, Daily Vol=0.7601%, Ann. Vol=12.07%
  - Test (2025):            Mean Return=0.0401%, Daily Vol=0.7442%, Ann. Vol=11.81%
- Figure: fig13_regime_shifts.png. Saved: results/tables/table7_market_regime_statistics.csv.

---

## Master Results Summary (2025 Test Set - 248 Trading Days)

| Model                    | Accuracy | Precision | Recall | F1-Score | ROC-AUC | Balanced Acc |
|--------------------------|----------|-----------|--------|----------|---------|--------------|
| VQC (4 Qubits, ZZ+SPSA)  | 48.39%   | 0.4908    | 0.8629 | 0.6257   | 0.4906  | 48.39%       |
| Logistic Regression       | 50.00%   | 0.5000    | 0.9919 | 0.6649   | 0.5223  | 50.00%       |
| Random Forest             | 52.42%   | 0.5469    | 0.2823 | 0.3723   | 0.5306  | 52.42%       |

**Statistical Conclusion:** No model demonstrates statistically significant superiority. All McNemar p-values > 0.50. All models perform near random-chance on 2025 NIFTY 50.

---

*All results recorded here are actual experimental findings - not fabricated.*
*Last updated: 2026-10-03 (All phases complete - EXP-001 through EXP-016)*

# src/__init__.py
# ============================================================
# NIFTY50-VQC Project — Source Package
# ============================================================
#
# This file makes the 'src' directory a Python package.
# It allows importing modules like:
#   from src.data_loader import load_nifty50_data
#   from src.features import compute_all_features, get_clean_features
#   from src.preprocessing import temporal_split, fit_and_transform_features
#   from src.pca_reduction import fit_and_transform_pca
#   from src.classical_models import train_logistic_regression, train_random_forest
#   from src.vqc_model import VQCClassifierWrapper
#   from src.quantum_circuit import build_feature_map, build_ansatz, plot_quantum_circuit
#
# Modules in this package:
#   data_loader.py      — loads and combines yearly NIFTY50 CSV files
#   features.py         — computes 8 technical indicators (daily_return, SMA, EMA, RSI, volatility, momentum)
#   preprocessing.py    — StandardScaler + MinMaxScaler fit on Train ONLY; temporal train/val/test split
#   pca_reduction.py    — PCA (8D -> 4D) + MinMaxScaler [0, pi] for quantum angle encoding
#   classical_models.py — Logistic Regression, Random Forest with validation-tuned hyperparameters
#   vqc_model.py        — VQCClassifierWrapper: 4-qubit VQC with COBYLA/SPSA/ADAM optimizers
#   quantum_circuit.py  — Qiskit circuit library: ZZFeatureMap, ZFeatureMap, RealAmplitudes, EfficientSU2
#
# Project Status: ALL PHASES COMPLETE (2026-10-03)
#   - Data: 2,706 clean NIFTY50 rows (2015-2025)
#   - Split: Train=1963 / Val=495 / Test=248 (chronological, no shuffle)
#   - Best Model: Random Forest (52.42% Test Acc) | VQC: 48.39%
#   - Statistical result: No significant difference between models (McNemar p > 0.50)
# ============================================================

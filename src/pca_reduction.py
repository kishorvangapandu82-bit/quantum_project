"""
============================================================
src/pca_reduction.py
NIFTY50-VQC Project — PCA Dimensionality Reduction (8D → 4D for Qubits)

PURPOSE:
    Reduce 8 standardized features to 4 Principal Components (PC1..PC4)
    matching the 4-qubit hardware/simulator layout of our VQC model.

LEAKAGE SAFEGUARD:
    - PCA fitted STRICTLY on X_train_std ONLY (2015–2022).
    - MinMaxScaler ([0, pi]) for Quantum Encoding fitted STRICTLY on X_train_pca.
    - Transform applied to Validation and Test sets.

QUANTUM MAPPING RATIONALE:
    Each principal component corresponds 1-to-1 with one qubit in a 4-qubit quantum circuit:
    - PC1 → Qubit 0
    - PC2 → Qubit 1
    - PC3 → Qubit 2
    - PC4 → Qubit 3

Author: NIFTY50-VQC Project
Date:   2026-10-02
============================================================
"""

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler


def fit_and_transform_pca(X_train_std, X_val_std, X_test_std, n_components=4, feature_range=(0, np.pi)):
    """
    Fit PCA on standardized training data, transform all splits, and scale to quantum range [0, pi].

    Args:
        X_train_std (np.ndarray): Standardized training features (1963, 8)
        X_val_std   (np.ndarray): Standardized validation features (495, 8)
        X_test_std  (np.ndarray): Standardized test features (248, 8)
        n_components (int): Target dimensions (default: 4 components for 4 qubits)
        feature_range (tuple): Target scale for angle encoding (default: 0 to pi)

    Returns:
        dict: {
            'X_train_pca':        Unscaled 4D PCA components for Train,
            'X_val_pca':          Unscaled 4D PCA components for Val,
            'X_test_pca':         Unscaled 4D PCA components for Test,
            'X_train_quantum':    Scaled [0, pi] 4D components for Train,
            'X_val_quantum':      Scaled [0, pi] 4D components for Val,
            'X_test_quantum':     Scaled [0, pi] 4D components for Test,
            'pca':                Fitted PCA object,
            'quantum_scaler':     Fitted MinMaxScaler object,
            'explained_variance': Array of explained variance ratio per component,
            'cumulative_variance': Array of cumulative explained variance ratio,
            'components':         Component loading matrix (4, 8)
        }
    """
    # 1. Fit PCA strictly on Training set
    pca = PCA(n_components=n_components, random_state=42)
    pca.fit(X_train_std)

    # 2. Transform standardized features into principal components
    X_train_pca = pca.transform(X_train_std)
    X_val_pca   = pca.transform(X_val_std)
    X_test_pca  = pca.transform(X_test_std)

    # 3. Fit Quantum Angle Scaler [0, pi] strictly on training PCA components
    quantum_scaler = MinMaxScaler(feature_range=feature_range)
    quantum_scaler.fit(X_train_pca)

    # 4. Transform PCA components to angle range [0, pi]
    X_train_quantum = quantum_scaler.transform(X_train_pca)
    X_val_quantum   = quantum_scaler.transform(X_val_pca)
    X_test_quantum  = quantum_scaler.transform(X_test_pca)

    explained_variance  = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance)

    return {
        'X_train_pca':        X_train_pca,
        'X_val_pca':          X_val_pca,
        'X_test_pca':         X_test_pca,
        'X_train_quantum':    X_train_quantum,
        'X_val_quantum':      X_val_quantum,
        'X_test_quantum':     X_test_quantum,
        'pca':                pca,
        'quantum_scaler':     quantum_scaler,
        'explained_variance': explained_variance,
        'cumulative_variance': cumulative_variance,
        'components':         pca.components_
    }

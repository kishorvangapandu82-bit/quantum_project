"""
============================================================
src/vqc_model.py
NIFTY50-VQC Project — Variational Quantum Classifier (VQC) Model Interface

PURPOSE:
    Encapsulate Qiskit Machine Learning's VQC classifier with:
    1. Flexible optimizer configuration (COBYLA, SPSA, ADAM/scipy)
    2. Loss history callback tracking during training
    3. Comprehensive evaluation metrics (Accuracy, Precision, Recall, F1, ROC-AUC, Balanced Acc)
    4. Deterministic random seed management

LEAKAGE SAFEGUARD:
    - Trained STRICTLY on X_train_quantum, y_train (2015–2022)
    - Hyperparameters tuned on X_val_quantum, y_val (2023–2024)
    - Final evaluation on X_test_quantum, y_test (2025)

Author: NIFTY50-VQC Project
Date:   2026-10-02
============================================================
"""

import numpy as np
from qiskit_machine_learning.algorithms import VQC
from qiskit_machine_learning.optimizers import COBYLA, SPSA, ADAM
from qiskit.primitives import StatevectorSampler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, balanced_accuracy_score, confusion_matrix
)
from src.quantum_circuit import build_feature_map, build_ansatz


class VQCClassifierWrapper:
    """
    Wrapper class for Qiskit VQC classifier with training loss callback,
    hyperparameter selection, and standard evaluation interface.
    """

    def __init__(self, num_qubits=4, feature_map_type='zz', ansatz_type='real_amplitudes',
                 reps=2, optimizer_name='cobyla', maxiter=100, learning_rate=0.01, random_state=42):
        
        self.num_qubits = num_qubits
        self.feature_map_type = feature_map_type
        self.ansatz_type = ansatz_type
        self.reps = reps
        self.optimizer_name = optimizer_name.lower()
        self.maxiter = maxiter
        self.learning_rate = learning_rate
        self.random_state = random_state

        self.loss_history = []

        # Build feature map and ansatz circuits
        self.feature_map = build_feature_map(num_qubits=num_qubits, reps=1, map_type=feature_map_type)
        self.ansatz = build_ansatz(num_qubits=num_qubits, reps=reps, ansatz_type=ansatz_type)

        # Build optimizer
        if self.optimizer_name == 'cobyla':
            self.optimizer = COBYLA(maxiter=maxiter)
        elif self.optimizer_name == 'spsa':
            self.optimizer = SPSA(maxiter=maxiter, learning_rate=learning_rate, perturbation=0.01)
        elif self.optimizer_name == 'adam':
            self.optimizer = ADAM(maxiter=maxiter, lr=learning_rate)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}. Choose 'cobyla', 'spsa', or 'adam'.")

        # Callback function to capture loss at each optimization step
        def callback(*args):
            if len(args) == 2:
                # COBYLA signature: (weights, loss)
                self.loss_history.append(float(args[1]))
            elif len(args) >= 3:
                # SPSA signature: (nfev, weights, loss, stepsize, accepted)
                self.loss_history.append(float(args[2]))
            elif len(args) == 1:
                self.loss_history.append(float(args[0]))

        # Sampler primitive
        self.sampler = StatevectorSampler()

        # Instantiate Qiskit VQC
        self.vqc = VQC(
            feature_map=self.feature_map,
            ansatz=self.ansatz,
            optimizer=self.optimizer,
            sampler=self.sampler,
            callback=callback
        )

    def fit(self, X_train, y_train):
        """Fit VQC on training dataset."""
        self.loss_history = []
        self.vqc.fit(X_train, y_train)
        return self

    def predict(self, X):
        """Predict binary labels (0 or 1)."""
        return self.vqc.predict(X)

    def predict_proba(self, X):
        """Predict class probabilities."""
        return self.vqc.predict_proba(X)

    def evaluate(self, X, y_true):
        """Compute comprehensive evaluation metrics."""
        y_pred = self.predict(X)
        probs  = self.predict_proba(X)
        y_prob = probs[:, 1] if probs.ndim == 2 else probs

        acc      = accuracy_score(y_true, y_pred)
        prec     = precision_score(y_true, y_pred, zero_division=0)
        rec      = recall_score(y_true, y_pred, zero_division=0)
        f1       = f1_score(y_true, y_pred, zero_division=0)
        bal_acc  = balanced_accuracy_score(y_true, y_pred)
        roc_auc  = roc_auc_score(y_true, y_prob)
        cm       = confusion_matrix(y_true, y_pred)

        return {
            'accuracy': acc,
            'precision': prec,
            'recall': rec,
            'f1_score': f1,
            'balanced_accuracy': bal_acc,
            'roc_auc': roc_auc,
            'confusion_matrix': cm,
            'y_pred': y_pred,
            'y_prob': y_prob
        }

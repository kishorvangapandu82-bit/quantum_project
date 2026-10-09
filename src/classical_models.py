"""
============================================================
src/classical_models.py
NIFTY50-VQC Project — Classical Baseline Machine Learning Models

PURPOSE:
    Implement, train, tune, and evaluate two classical ML baselines:
    1. Logistic Regression (Linear baseline)
    2. Random Forest Classifier (Ensemble decision trees)

LEAKAGE SAFEGUARD:
    - Trained ONLY on X_train_std, y_train (2015–2022)
    - Hyperparameter selection guided by X_val_std, y_val (2023–2024)
    - Final unbiased evaluation on X_test_std, y_test (2025)

METRICS COMPUTED:
    - Accuracy, Precision, Recall, F1-Score, ROC-AUC, Balanced Accuracy
    - Confusion Matrix (TN, FP, FN, TP)

Author: NIFTY50-VQC Project
Date:   2026-10-02
============================================================
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, balanced_accuracy_score, confusion_matrix, roc_curve
)


def evaluate_model(model, X, y_true):
    """
    Compute comprehensive metrics for a classification model.

    Args:
        model: Trained scikit-learn model object
        X (np.ndarray): Feature matrix
        y_true (np.ndarray): True ground truth labels (0 or 1)

    Returns:
        dict: Metric names mapped to float values and confusion matrix arrays
    """
    y_pred = model.predict(X)

    # Probability estimates for ROC-AUC
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X)[:, 1]
    elif hasattr(model, "decision_function"):
        y_prob = model.decision_function(X)
        # Normalize decision function to [0, 1] range for ROC-AUC
        y_prob = (y_prob - y_prob.min()) / (y_prob.max() - y_prob.min() + 1e-12)
    else:
        y_prob = y_pred

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


def train_logistic_regression(X_train, y_train, X_val, y_val, random_state=42):
    """
    Train Logistic Regression with hyperparameter tuning over C values on Validation set.

    Args:
        X_train, y_train: Training features and labels
        X_val, y_val: Validation features and labels
        random_state (int): Random seed for reproducibility

    Returns:
        tuple: (best_model, best_params, val_metrics)
    """
    c_candidates = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
    best_model = None
    best_score = -1.0
    best_params = {}

    for C in c_candidates:
        model = LogisticRegression(C=C, random_state=random_state, max_iter=1000)
        model.fit(X_train, y_train)
        val_metrics = evaluate_model(model, X_val, y_val)

        # Select model based on Validation F1-Score
        if val_metrics['f1_score'] > best_score:
            best_score = val_metrics['f1_score']
            best_model = model
            best_params = {'C': C}

    best_val_metrics = evaluate_model(best_model, X_val, y_val)
    return best_model, best_params, best_val_metrics


def train_random_forest(X_train, y_train, X_val, y_val, random_state=42):
    """
    Train Random Forest with hyperparameter tuning over n_estimators, max_depth, min_samples_split.

    Args:
        X_train, y_train: Training features and labels
        X_val, y_val: Validation features and labels
        random_state (int): Random seed for reproducibility

    Returns:
        tuple: (best_model, best_params, val_metrics)
    """
    best_model = None
    best_score = -1.0
    best_params = {}

    param_grid = [
        {'n_estimators': 50,  'max_depth': 3, 'min_samples_split': 5},
        {'n_estimators': 100, 'max_depth': 3, 'min_samples_split': 5},
        {'n_estimators': 100, 'max_depth': 5, 'min_samples_split': 2},
        {'n_estimators': 200, 'max_depth': 5, 'min_samples_split': 5},
        {'n_estimators': 100, 'max_depth': None, 'min_samples_split': 10},
    ]

    for params in param_grid:
        model = RandomForestClassifier(**params, random_state=random_state)
        model.fit(X_train, y_train)
        val_metrics = evaluate_model(model, X_val, y_val)

        if val_metrics['f1_score'] > best_score:
            best_score = val_metrics['f1_score']
            best_model = model
            best_params = params

    best_val_metrics = evaluate_model(best_model, X_val, y_val)
    return best_model, best_params, best_val_metrics

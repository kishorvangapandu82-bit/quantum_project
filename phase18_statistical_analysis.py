"""
============================================================
phase18_statistical_analysis.py
NIFTY50-VQC Project — Phase 18 Statistical Significance & Error Analysis

PURPOSE:
    Perform McNemar's test for paired nominal data to test if performance
    differences between VQC and classical baselines are statistically significant.
    Compute confusion matrix breakdown metrics (Sensitivity, Specificity, FPR, FNR).
============================================================
"""

import sys
import os
import numpy as np
import pandas as pd
from scipy.stats import binomtest
import warnings

warnings.filterwarnings('ignore')

sys.path.insert(0, '.')
from src.classical_models import (
    train_logistic_regression,
    train_random_forest,
    evaluate_model
)
from src.vqc_model import VQCClassifierWrapper

INPUT_PCA_NPZ = 'data/processed/nifty50_pca_4d.npz'
INPUT_STD_NPZ = 'data/processed/nifty50_scaled_data.npz'
TAB_DIR       = 'results/tables'

os.makedirs(TAB_DIR, exist_ok=True)


def run_mcnemar_test(y_true, y_pred1, y_pred2, model1_name, model2_name):
    """
    Compute McNemar's Exact Test statistic and p-value between two models.
    Uses binomial test under null hypothesis p = 0.5 for discordant pairs.
    """
    correct1 = (y_pred1 == y_true)
    correct2 = (y_pred2 == y_true)

    b = np.sum(correct1 & ~correct2)  # Model 1 correct, Model 2 wrong
    c = np.sum(~correct1 & correct2)  # Model 1 wrong, Model 2 correct

    n_discordant = b + c
    if n_discordant == 0:
        p_value = 1.0
    else:
        res = binomtest(b, n_discordant, p=0.5, alternative='two-sided')
        p_value = res.pvalue

    is_sig = p_value < 0.05
    return {
        'Comparison': f"{model1_name} vs {model2_name}",
        'b_m1_right_m2_wrong': int(b),
        'c_m1_wrong_m2_right': int(c),
        'p_value': float(p_value),
        'Significant_at_0.05': 'YES' if is_sig else 'NO (No Statistically Significant Difference)'
    }


def log_print(msg):
    print(msg)
    sys.stdout.flush()

def main():
    log_print("=" * 75)
    log_print("PHASE 18 — STATISTICAL SIGNIFICANCE & ERROR ANALYSIS")
    log_print("=" * 75)

    data_pca = np.load(INPUT_PCA_NPZ, allow_pickle=True)
    X_train_q, X_val_q, X_test_q = data_pca['X_train_quantum'], data_pca['X_val_quantum'], data_pca['X_test_quantum']
    y_train, y_val, y_test = data_pca['y_train'], data_pca['y_val'], data_pca['y_test']

    data_std = np.load(INPUT_STD_NPZ, allow_pickle=True)
    X_train_std, X_val_std, X_test_std = data_std['X_train_std'], data_std['X_val_std'], data_std['X_test_std']

    # Fit VQC
    print("⚙️ Evaluating VQC predictions...")
    vqc = VQCClassifierWrapper(num_qubits=4, feature_map_type='zz', ansatz_type='real_amplitudes', reps=1, optimizer_name='spsa', maxiter=40, random_state=42)
    vqc.fit(X_train_q, y_train)
    y_pred_vqc = vqc.predict(X_test_q)

    # Fit Classical
    print("⚙️ Evaluating Classical baselines predictions...")
    lr, _, _  = train_logistic_regression(X_train_std, y_train, X_val_std, y_val)
    rf, _, _  = train_random_forest(X_train_std, y_train, X_val_std, y_val)

    y_pred_lr  = lr.predict(X_test_std)
    y_pred_rf  = rf.predict(X_test_std)

    # Run McNemar tests
    print("\n📊 1. McNemar's Test for Paired Classifiers:")
    print("-" * 80)
    tests = [
        run_mcnemar_test(y_test, y_pred_vqc, y_pred_lr,  "VQC", "Logistic Regression"),
        run_mcnemar_test(y_test, y_pred_vqc, y_pred_rf,  "VQC", "Random Forest"),
        run_mcnemar_test(y_test, y_pred_lr,  y_pred_rf,  "Logistic Regression", "Random Forest")
    ]

    mcnemar_df = pd.DataFrame(tests)
    print(mcnemar_df.to_string(index=False))
    print("-" * 80)

    # Save table
    mcnemar_csv = os.path.join(TAB_DIR, 'table6_mcnemar_test.csv')
    mcnemar_df.to_csv(mcnemar_csv, index=False)
    print(f"\n💾 McNemar test summary saved: {mcnemar_csv}")

    # Detailed Error Breakdown
    print("\n🔍 2. Detailed Confusion Matrix Error Breakdown:")
    print("-" * 85)
    print(f"{'Model':<30} {'TN':<6} {'FP':<6} {'FN':<6} {'TP':<6} {'Sensitivity':<14} {'Specificity':<14}")
    print("-" * 85)

    models_preds = [
        ('VQC (4 Qubits)', y_pred_vqc),
        ('Logistic Regression', y_pred_lr),
        ('Random Forest', y_pred_rf)
    ]

    for name, y_pred in models_preds:
        cm = evaluate_model(lr, X_test_std, y_test)['confusion_matrix'] if 'Logistic' in name else evaluate_model(vqc, X_test_q, y_test)['confusion_matrix'] if 'VQC' in name else evaluate_model(rf, X_test_std, y_test)['confusion_matrix']
        tn, fp, fn, tp = cm.ravel()
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0
        print(f"{name:<30} {tn:<6} {fp:<6} {fn:<6} {tp:<6} {sens:<14.2%} {spec:<14.2%}")
    print("-" * 85)

    print("\n✅ Phase 18 Statistical Analysis Completed Successfully!")

if __name__ == '__main__':
    main()

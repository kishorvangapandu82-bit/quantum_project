"""
============================================================
phase11_classical_models.py
NIFTY50-VQC Project — Phase 11 Classical Models Execution

PURPOSE:
    Load data/processed/nifty50_scaled_data.npz, train and tune
    three classical baselines (Logistic Regression, SVM, Random Forest),
    evaluate performance on Train, Validation, and Test sets, generate
    comparison tables and publication-quality figures (Confusion Matrices,
    ROC Curves, Feature Importances).
============================================================
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')

sys.path.insert(0, '.')
from src.classical_models import (
    evaluate_model,
    train_logistic_regression,
    train_random_forest
)

INPUT_NPZ = 'data/processed/nifty50_scaled_data.npz'
FIG_DIR   = 'results/figures'
TAB_DIR   = 'results/tables'

os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(TAB_DIR, exist_ok=True)


def plot_confusion_matrices(results, feature_cols):
    """Generate side-by-side confusion matrix heatmaps for classical models."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    model_names = ['Logistic Regression', 'Random Forest']

    for i, name in enumerate(model_names):
        cm = results[name]['test_metrics']['confusion_matrix']
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=axes[i],
                    xticklabels=['DOWN (0)', 'UP (1)'],
                    yticklabels=['DOWN (0)', 'UP (1)'])
        axes[i].set_title(f"{name}\nTest Accuracy: {results[name]['test_metrics']['accuracy']:.2%}", fontsize=12, fontweight='bold')
        axes[i].set_xlabel("Predicted Label", fontsize=10)
        axes[i].set_ylabel("True Label", fontsize=10)

    plt.suptitle("Phase 11 — Test Set Confusion Matrices (Classical Baselines)", fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig_path = os.path.join(FIG_DIR, "fig4_confusion_matrices.png")
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   Saved plot: {fig_path}")


def plot_roc_curves(results):
    """Generate combined ROC curve plot for classical models on Test set."""
    from sklearn.metrics import roc_curve, auc

    plt.figure(figsize=(8, 6))
    colors = ['#1f77b4', '#2ca02c']
    model_names = ['Logistic Regression', 'Random Forest']

    for i, name in enumerate(model_names):
        y_prob = results[name]['test_metrics']['y_prob']
        y_true = results[name]['y_test']
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = results[name]['test_metrics']['roc_auc']
        plt.plot(fpr, tpr, color=colors[i], lw=2.5, label=f"{name} (AUC = {roc_auc:.3f})")

    plt.plot([0, 1], [0, 1], color='navy', lw=1.5, linestyle='--', label='Random Chance (AUC = 0.500)')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=11, fontweight='bold')
    plt.ylabel('True Positive Rate (Sensitivity)', fontsize=11, fontweight='bold')
    plt.title('Phase 11 — Test Set ROC Curves (Classical Baselines)', fontsize=13, fontweight='bold', pad=12)
    plt.legend(loc='lower right', fontsize=10, frameon=True, facecolor='white', framealpha=0.9)
    plt.grid(True, linestyle=':', alpha=0.6)

    fig_path = os.path.join(FIG_DIR, "fig5_roc_curves.png")
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   Saved plot: {fig_path}")


def plot_feature_importance(lr_model, rf_model, feature_cols):
    """Generate bar plot showing Logistic Regression coefficients and Random Forest feature importances."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Logistic Regression Coefficients
    lr_coefs = lr_model.coef_[0]
    sns.barplot(x=lr_coefs, y=feature_cols, ax=axes[0], palette='vlag')
    axes[0].set_title('Logistic Regression Coefficients', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Coefficient Value (Directional Impact)', fontsize=10)
    axes[0].axvline(0, color='black', linestyle='--', linewidth=0.8)

    # Random Forest Importance
    rf_imp = rf_model.feature_importances_
    sorted_idx = np.argsort(rf_imp)
    sns.barplot(x=rf_imp[sorted_idx], y=[feature_cols[i] for i in sorted_idx], ax=axes[1], palette='Blues_r')
    axes[1].set_title('Random Forest Feature Importances', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Gini Importance (Relative Weight)', fontsize=10)

    plt.suptitle("Phase 11 — Classical Feature Importance Analysis", fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig_path = os.path.join(FIG_DIR, "fig6_feature_importance.png")
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   Saved plot: {fig_path}")


def main():
    print("=" * 75)
    print("PHASE 11 — CLASSICAL MACHINE LEARNING BASELINE MODELS")
    print("=" * 75)

    if not os.path.exists(INPUT_NPZ):
        print(f"❌ Error: {INPUT_NPZ} not found. Run Phase 10 first.")
        return

    print(f"📥 Loading preprocessed data: {INPUT_NPZ}")
    data = np.load(INPUT_NPZ, allow_pickle=True)

    X_train_std = data['X_train_std']
    X_val_std   = data['X_val_std']
    X_test_std  = data['X_test_std']

    y_train = data['y_train']
    y_val   = data['y_val']
    y_test  = data['y_test']

    feature_cols = list(data['feature_cols'])

    print(f"   Train samples: {X_train_std.shape[0]:,}")
    print(f"   Val samples:   {X_val_std.shape[0]:,}")
    print(f"   Test samples:  {X_test_std.shape[0]:,}")
    print(f"   Features ({len(feature_cols)}): {feature_cols}")

    results = {}

    # 1. Logistic Regression
    print("\n🔹 Training & Tuning 1/3: Logistic Regression...")
    lr_model, lr_params, lr_val = train_logistic_regression(X_train_std, y_train, X_val_std, y_val)
    lr_tr   = evaluate_model(lr_model, X_train_std, y_train)
    lr_te   = evaluate_model(lr_model, X_test_std, y_test)
    print(f"   Best Parameters: {lr_params}")
    print(f"   Test Accuracy:   {lr_te['accuracy']:.2%} | F1-Score: {lr_te['f1_score']:.4f} | ROC-AUC: {lr_te['roc_auc']:.4f}")

    results['Logistic Regression'] = {
        'model': lr_model,
        'params': lr_params,
        'train_metrics': lr_tr,
        'val_metrics': lr_val,
        'test_metrics': lr_te,
        'y_test': y_test
    }

    # 2. Random Forest
    print("\n🔹 Training & Tuning 2/2: Random Forest Classifier...")
    rf_model, rf_params, rf_val = train_random_forest(X_train_std, y_train, X_val_std, y_val)
    rf_tr   = evaluate_model(rf_model, X_train_std, y_train)
    rf_te   = evaluate_model(rf_model, X_test_std, y_test)
    print(f"   Best Parameters: {rf_params}")
    print(f"   Test Accuracy:   {rf_te['accuracy']:.2%} | F1-Score: {rf_te['f1_score']:.4f} | ROC-AUC: {rf_te['roc_auc']:.4f}")

    results['Random Forest'] = {
        'model': rf_model,
        'params': rf_params,
        'train_metrics': rf_tr,
        'val_metrics': rf_val,
        'test_metrics': rf_te,
        'y_test': y_test
    }

    # Generate Figures
    print("\n🎨 Generating Publication-Quality Figures...")
    plot_confusion_matrices(results, feature_cols)
    plot_roc_curves(results)
    plot_feature_importance(lr_model, rf_model, feature_cols)

    # Master Results Table
    print("\n📊 MASTER CLASSICAL BASELINES RESULTS TABLE (TEST SET - 2025):")
    print("=" * 95)
    print(f"{'Model':<25} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'ROC-AUC':<12} {'Bal Acc':<12}")
    print("=" * 95)

    rows = []
    for name, res in results.items():
        tm = res['test_metrics']
        print(f"{name:<25} {tm['accuracy']:<12.2%} {tm['precision']:<12.4f} {tm['recall']:<12.4f} {tm['f1_score']:<12.4f} {tm['roc_auc']:<12.4f} {tm['balanced_accuracy']:<12.4f}")
        rows.append({
            'Model': name,
            'Accuracy': f"{tm['accuracy']:.4f}",
            'Precision': f"{tm['precision']:.4f}",
            'Recall': f"{tm['recall']:.4f}",
            'F1_Score': f"{tm['f1_score']:.4f}",
            'ROC_AUC': f"{tm['roc_auc']:.4f}",
            'Balanced_Accuracy': f"{tm['balanced_accuracy']:.4f}"
        })
    print("=" * 95)

    # Save summary table CSV
    csv_path = os.path.join(TAB_DIR, 'table2_classical_baselines.csv')
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"\n💾 Results table saved: {csv_path}")

    print("\n✅ Phase 11 Classical Models Execution Completed Successfully!")

if __name__ == '__main__':
    main()

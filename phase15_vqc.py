"""
============================================================
phase15_vqc.py
NIFTY50-VQC Project — Phase 15 & 16 VQC Training & Benchmark Comparison

PURPOSE:
    Load 4D PCA quantum features (nifty50_pca_4d.npz), tune VQC architecture
    configurations (Feature Map, Ansatz, Reps, Optimizer) on Validation set,
    train optimal VQC model, evaluate on 2025 Test set, and generate
    master comparison figures and tables comparing VQC vs Classical Baselines.
============================================================
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve
import warnings

warnings.filterwarnings('ignore')

sys.path.insert(0, '.')
from src.vqc_model import VQCClassifierWrapper
from src.quantum_circuit import plot_quantum_circuit
from src.classical_models import (
    train_logistic_regression,
    train_svm,
    train_random_forest,
    evaluate_model
)

INPUT_PCA_NPZ = 'data/processed/nifty50_pca_4d.npz'
INPUT_STD_NPZ = 'data/processed/nifty50_scaled_data.npz'
FIG_DIR       = 'results/figures'
TAB_DIR       = 'results/tables'

os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(TAB_DIR, exist_ok=True)


def log_print(msg):
    """Print message and immediately flush stdout."""
    print(msg)
    sys.stdout.flush()


def plot_vqc_loss_curve(loss_history, save_path):
    """Plot VQC training loss over optimization iterations."""
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(loss_history) + 1), loss_history, color='#882255', linewidth=2.2, label='Training Loss')
    plt.xlabel('Optimization Iteration', fontsize=11, fontweight='bold')
    plt.ylabel('Loss Value', fontsize=11, fontweight='bold')
    plt.title('Phase 16 — Variational Quantum Classifier (VQC) Training Loss Curve', fontsize=13, fontweight='bold', pad=12)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(loc='upper right', fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    log_print(f"   Saved plot: {save_path}")


def plot_vqc_confusion_matrix(cm, acc, save_path):
    """Plot VQC Test confusion matrix heatmap."""
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Purples', cbar=False,
                xticklabels=['DOWN (0)', 'UP (1)'],
                yticklabels=['DOWN (0)', 'UP (1)'])
    plt.title(f"VQC Test Set Confusion Matrix\nAccuracy: {acc:.2%}", fontsize=12, fontweight='bold', pad=12)
    plt.xlabel("Predicted Label", fontsize=10, fontweight='bold')
    plt.ylabel("True Label", fontsize=10, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    log_print(f"   Saved plot: {save_path}")


def plot_master_roc_comparison(all_models_results, save_path):
    """Plot master ROC curves comparing VQC against Classical Baselines on 2025 Test Set."""
    plt.figure(figsize=(8, 6))
    colors = {
        'Variational Quantum Classifier (VQC)': '#882255',
        'Logistic Regression': '#1f77b4',
        'Support Vector Machine (SVM)': '#ff7f0e',
        'Random Forest Classifier': '#2ca02c'
    }

    for name, res in all_models_results.items():
        y_prob = res['test_metrics']['y_prob']
        y_true = res['y_test']
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc_val = res['test_metrics']['roc_auc']
        color = colors.get(name, '#333333')
        lw = 3.0 if 'Quantum' in name else 2.0
        plt.plot(fpr, tpr, color=color, lw=lw, label=f"{name} (AUC = {auc_val:.3f})")

    plt.plot([0, 1], [0, 1], color='navy', lw=1.5, linestyle='--', label='Random Chance (AUC = 0.500)')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=11, fontweight='bold')
    plt.ylabel('True Positive Rate (Sensitivity)', fontsize=11, fontweight='bold')
    plt.title('Phase 17 — Master ROC Curves: VQC vs Classical Baselines (Test 2025)', fontsize=13, fontweight='bold', pad=12)
    plt.legend(loc='lower right', fontsize=9.5, frameon=True, facecolor='white', framealpha=0.9)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    log_print(f"   Saved master ROC plot: {save_path}")


def main():
    log_print("=" * 75)
    log_print("PHASE 15 & 16 — VARIATIONAL QUANTUM CLASSIFIER (VQC) TRAINING & COMPARISON")
    log_print("=" * 75)

    if not os.path.exists(INPUT_PCA_NPZ) or not os.path.exists(INPUT_STD_NPZ):
        log_print("❌ Error: Processed NPZ files missing. Run earlier phases first.")
        return

    log_print(f"📥 Loading 4D Quantum PCA features: {INPUT_PCA_NPZ}")
    data_pca = np.load(INPUT_PCA_NPZ, allow_pickle=True)
    X_train_q = data_pca['X_train_quantum']
    X_val_q   = data_pca['X_val_quantum']
    X_test_q  = data_pca['X_test_quantum']
    y_train   = data_pca['y_train']
    y_val     = data_pca['y_val']
    y_test    = data_pca['y_test']

    data_std = np.load(INPUT_STD_NPZ, allow_pickle=True)
    X_train_std = data_std['X_train_std']
    X_val_std   = data_std['X_val_std']
    X_test_std  = data_std['X_test_std']

    log_print(f"   Train samples: {X_train_q.shape[0]:,} | Val: {X_val_q.shape[0]:,} | Test: {X_test_q.shape[0]:,}")

    # Subsample for architecture tuning efficiency
    np.random.seed(42)
    sub_idx = np.random.choice(len(X_train_q), size=min(400, len(X_train_q)), replace=False)
    X_train_sub = X_train_q[sub_idx]
    y_train_sub = y_train[sub_idx]

    # 1. Architecture Hyperparameter Tuning on Validation Set
    log_print("\n🔬 1. Tuning VQC Architectures on Validation Set...")
    log_print("-" * 80)
    log_print(f"{'Config ID':<10} {'Feature Map':<14} {'Ansatz':<16} {'Reps':<6} {'Optimizer':<10} {'Val Acc':<10} {'Val F1':<10}")
    log_print("-" * 80)

    configs = [
        {'id': 'VQC-1', 'map': 'zz', 'ansatz': 'real_amplitudes', 'reps': 1, 'opt': 'cobyla', 'maxiter': 25},
        {'id': 'VQC-2', 'map': 'zz', 'ansatz': 'real_amplitudes', 'reps': 2, 'opt': 'cobyla', 'maxiter': 25},
        {'id': 'VQC-3', 'map': 'z',  'ansatz': 'real_amplitudes', 'reps': 1, 'opt': 'cobyla', 'maxiter': 25},
        {'id': 'VQC-4', 'map': 'zz', 'ansatz': 'efficient_su2',    'reps': 1, 'opt': 'cobyla', 'maxiter': 25},
        {'id': 'VQC-5', 'map': 'zz', 'ansatz': 'real_amplitudes', 'reps': 1, 'opt': 'spsa',   'maxiter': 25},
    ]

    tuning_results = []
    best_vqc_config = None
    best_val_f1 = -1.0

    for cfg in configs:
        model_wrap = VQCClassifierWrapper(
            num_qubits=4,
            feature_map_type=cfg['map'],
            ansatz_type=cfg['ansatz'],
            reps=cfg['reps'],
            optimizer_name=cfg['opt'],
            maxiter=cfg['maxiter'],
            random_state=42
        )
        model_wrap.fit(X_train_sub, y_train_sub)
        val_m = model_wrap.evaluate(X_val_q, y_val)

        tuning_results.append({
            'Config_ID': cfg['id'],
            'Feature_Map': cfg['map'].upper(),
            'Ansatz': cfg['ansatz'],
            'Reps': cfg['reps'],
            'Optimizer': cfg['opt'].upper(),
            'Val_Accuracy': val_m['accuracy'],
            'Val_F1_Score': val_m['f1_score'],
            'Val_ROC_AUC': val_m['roc_auc']
        })

        log_print(f"{cfg['id']:<10} {cfg['map'].upper():<14} {cfg['ansatz']:<16} {cfg['reps']:<6} {cfg['opt'].upper():<10} {val_m['accuracy']:<10.2%} {val_m['f1_score']:<10.4f}")

        if val_m['f1_score'] > best_val_f1:
            best_val_f1 = val_m['f1_score']
            best_vqc_config = cfg

    log_print("-" * 80)
    log_print(f"⭐ Optimal VQC Architecture Selected: {best_vqc_config['id']} (Feature Map: {best_vqc_config['map'].upper()}, Ansatz: {best_vqc_config['ansatz']}, Reps: {best_vqc_config['reps']}, Optimizer: {best_vqc_config['opt'].upper()})")

    pd.DataFrame(tuning_results).to_csv(os.path.join(TAB_DIR, 'table4_vqc_architecture_tuning.csv'), index=False)

    # 2. Train Optimal VQC Model on Full Training Set
    log_print(f"\n🚀 2. Training Optimal VQC Model ({best_vqc_config['id']}) on Full Training Set (1,963 samples)...")
    optimal_vqc = VQCClassifierWrapper(
        num_qubits=4,
        feature_map_type=best_vqc_config['map'],
        ansatz_type=best_vqc_config['ansatz'],
        reps=best_vqc_config['reps'],
        optimizer_name=best_vqc_config['opt'],
        maxiter=40,
        random_state=42
    )
    optimal_vqc.fit(X_train_q, y_train)

    vqc_train_metrics = optimal_vqc.evaluate(X_train_q, y_train)
    vqc_val_metrics   = optimal_vqc.evaluate(X_val_q, y_val)
    vqc_test_metrics  = optimal_vqc.evaluate(X_test_q, y_test)

    log_print(f"   Train Accuracy: {vqc_train_metrics['accuracy']:.2%} | Val Accuracy: {vqc_val_metrics['accuracy']:.2%} | Test Accuracy: {vqc_test_metrics['accuracy']:.2%}")
    log_print(f"   Test F1-Score:  {vqc_test_metrics['f1_score']:.4f}  | Test ROC-AUC: {vqc_test_metrics['roc_auc']:.4f}  | Test Bal Acc: {vqc_test_metrics['balanced_accuracy']:.4f}")

    # Plot Circuit Diagram & Figures
    log_print("\n🎨 Generating VQC Figures...")
    plot_quantum_circuit(optimal_vqc.feature_map, optimal_vqc.ansatz, save_path=os.path.join(FIG_DIR, 'fig9_quantum_circuit.png'))
    plot_vqc_loss_curve(optimal_vqc.loss_history, save_path=os.path.join(FIG_DIR, 'fig10_vqc_training_loss.png'))
    plot_vqc_confusion_matrix(vqc_test_metrics['confusion_matrix'], vqc_test_metrics['accuracy'], save_path=os.path.join(FIG_DIR, 'fig11_vqc_confusion_matrix.png'))

    # 3. Fit Classical Baselines for Comparison
    log_print("\n⚔️ 3. Fitting Classical Baseline Models for Master Comparison...")
    lr_model, _, _   = train_logistic_regression(X_train_std, y_train, X_val_std, y_val)
    svm_model, _, _  = train_svm(X_train_std, y_train, X_val_std, y_val)
    rf_model, _, _   = train_random_forest(X_train_std, y_train, X_val_std, y_val)

    all_models_results = {
        'Variational Quantum Classifier (VQC)': {
            'test_metrics': vqc_test_metrics,
            'y_test': y_test
        },
        'Logistic Regression': {
            'test_metrics': evaluate_model(lr_model, X_test_std, y_test),
            'y_test': y_test
        },
        'Support Vector Machine (SVM)': {
            'test_metrics': evaluate_model(svm_model, X_test_std, y_test),
            'y_test': y_test
        },
        'Random Forest Classifier': {
            'test_metrics': evaluate_model(rf_model, X_test_std, y_test),
            'y_test': y_test
        }
    }

    plot_master_roc_comparison(all_models_results, save_path=os.path.join(FIG_DIR, 'fig12_quantum_vs_classical_roc.png'))

    # 4. Master Comparison Table
    log_print("\n🏆 MASTER MODEL COMPARISON TABLE (2025 TEST SET - 248 TRADING DAYS):")
    log_print("=" * 100)
    log_print(f"{'Model Name':<38} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'ROC-AUC':<12} {'Bal Acc':<12}")
    log_print("=" * 100)

    master_rows = []
    for name, res in all_models_results.items():
        tm = res['test_metrics']
        log_print(f"{name:<38} {tm['accuracy']:<12.2%} {tm['precision']:<12.4f} {tm['recall']:<12.4f} {tm['f1_score']:<12.4f} {tm['roc_auc']:<12.4f} {tm['balanced_accuracy']:<12.4f}")
        master_rows.append({
            'Model_Name': name,
            'Accuracy': f"{tm['accuracy']:.4f}",
            'Precision': f"{tm['precision']:.4f}",
            'Recall': f"{tm['recall']:.4f}",
            'F1_Score': f"{tm['f1_score']:.4f}",
            'ROC_AUC': f"{tm['roc_auc']:.4f}",
            'Balanced_Accuracy': f"{tm['balanced_accuracy']:.4f}"
        })
    log_print("=" * 100)

    master_csv_path = os.path.join(TAB_DIR, 'table5_master_model_comparison.csv')
    pd.DataFrame(master_rows).to_csv(master_csv_path, index=False)
    log_print(f"\n💾 Master comparison table saved: {master_csv_path}")

    log_print("\n✅ Phase 15 & 16 VQC Training & Full Comparison Completed Successfully!")

if __name__ == '__main__':
    main()

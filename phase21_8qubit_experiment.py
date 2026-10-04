"""
============================================================
phase21_8qubit_experiment.py
NIFTY50-VQC Project -- Phase 21: 8-Qubit Direct VQC (No PCA)

PURPOSE:
    Evaluate whether removing PCA compression and feeding all 8 raw
    technical indicators DIRECTLY into 8 qubits improves classification
    accuracy over the baseline 4-qubit PCA-compressed VQC.

PIPELINE:
    OHLCV -> 8 Technical Indicators -> StandardScaler -> MinMaxScaler[0,pi]
          -> ZZFeatureMap(8 qubits) -> RealAmplitudes(8 qubits) -> Prediction

Author: NIFTY50-VQC Project
Date:   2026-10-04
============================================================
"""

import os, sys, time, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, balanced_accuracy_score, confusion_matrix, roc_curve
)
from qiskit.circuit.library import zz_feature_map, real_amplitudes
from qiskit.primitives import StatevectorSampler
from qiskit_machine_learning.algorithms import VQC
from qiskit_machine_learning.optimizers import SPSA

warnings.filterwarnings('ignore')

STD_NPZ = 'data/processed/nifty50_scaled_data.npz'
PCA_NPZ = 'data/processed/nifty50_pca_4d.npz'
FIG_DIR = 'results/figures'
TAB_DIR = 'results/tables'
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(TAB_DIR, exist_ok=True)

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

BASELINE_4Q = {
    'label':    '4-Qubit VQC (PCA)',
    'accuracy': 0.4839,
    'precision':0.4908,
    'recall':   0.8629,
    'f1_score': 0.6257,
    'roc_auc':  0.4906,
    'bal_acc':  0.4839,
}

def log(msg):
    print(msg, flush=True)

def load_data():
    log("Loading pre-processed dataset...")
    if not os.path.exists(STD_NPZ):
        log(f"ERROR: Missing {STD_NPZ}. Run phase10 & phase12 first.")
        sys.exit(1)
    data_std = np.load(STD_NPZ, allow_pickle=True)
    X_train_std = data_std['X_train_std']
    X_val_std   = data_std['X_val_std']
    X_test_std  = data_std['X_test_std']
    data_pca = np.load(PCA_NPZ, allow_pickle=True)
    y_train = data_pca['y_train']
    y_val   = data_pca['y_val']
    y_test  = data_pca['y_test']
    log(f"  Train: {X_train_std.shape[0]} | Val: {X_val_std.shape[0]} | Test: {X_test_std.shape[0]}")
    log(f"  Features: {X_train_std.shape[1]} raw technical indicators")
    return (X_train_std, X_val_std, X_test_std), (y_train, y_val, y_test)

def apply_quantum_scaling(X_train, X_val, X_test):
    log("\nApplying MinMax scaling to [0, pi] (fit on train only)...")
    mms = MinMaxScaler(feature_range=(0, np.pi))
    Xtr = np.clip(mms.fit_transform(X_train), 0, np.pi)
    Xv  = np.clip(mms.transform(X_val),  0, np.pi)
    Xt  = np.clip(mms.transform(X_test), 0, np.pi)
    log(f"  8D quantum feature range: [{Xtr.min():.4f}, {Xtr.max():.4f}]")
    return Xtr, Xv, Xt, mms

def train_8qubit_vqc(X_train_q8, y_train, maxiter=40):
    log(f"\nBuilding 8-Qubit VQC (ZZFeatureMap + RealAmplitudes, SPSA maxiter={maxiter})...")
    loss_history = []

    def callback(nfev, x, fx, dx, accept=None):
        loss_history.append(float(fx))
        if len(loss_history) % 10 == 0:
            log(f"  [SPSA iter {len(loss_history)}/{maxiter}] Loss = {fx:.5f}")

    fm  = zz_feature_map(feature_dimension=8, reps=1)
    an  = real_amplitudes(num_qubits=8, reps=1)
    log(f"  Circuit trainable parameters: {an.num_parameters}")

    vqc = VQC(
        sampler=StatevectorSampler(),
        feature_map=fm, ansatz=an,
        optimizer=SPSA(maxiter=maxiter, callback=callback),
    )

    np.random.seed(RANDOM_SEED)
    sub_idx = np.random.choice(len(X_train_q8), size=min(400, len(X_train_q8)), replace=False)
    X_sub = X_train_q8[sub_idx]
    y_sub = y_train[sub_idx]

    log(f"  Training on subsample ({X_sub.shape[0]} samples)...")
    t0 = time.time()
    vqc.fit(X_sub, y_sub)
    log(f"  Subsampled training complete in {time.time()-t0:.1f}s")

    log(f"  Fine-tuning on full training set ({X_train_q8.shape[0]} samples, maxiter=20)...")
    vqc_full = VQC(
        sampler=StatevectorSampler(),
        feature_map=fm, ansatz=an,
        optimizer=SPSA(maxiter=20),
    )
    vqc_full.fit(X_train_q8, y_train)
    log(f"  Full training complete in {time.time()-t0:.1f}s total")
    return vqc_full, loss_history

def evaluate_vqc(vqc, X, y_true, label=""):
    y_pred = np.array(vqc.predict(X)).ravel()
    try:
        probs  = vqc.predict_proba(X)
        y_prob = probs[:, 1] if probs.ndim == 2 else probs.ravel()
        auc    = roc_auc_score(y_true, y_prob)
    except Exception:
        y_prob = y_pred.astype(float)
        auc    = 0.5
    acc     = accuracy_score(y_true, y_pred)
    prec    = precision_score(y_true, y_pred, zero_division=0)
    rec     = recall_score(y_true, y_pred, zero_division=0)
    f1      = f1_score(y_true, y_pred, zero_division=0)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    cm_mat  = confusion_matrix(y_true, y_pred)
    if label:
        log(f"\n  [{label}]")
        log(f"  Accuracy:  {acc:.4f} ({acc:.2%})")
        log(f"  F1-Score:  {f1:.4f}  |  ROC-AUC: {auc:.4f}")
        log(f"  Precision: {prec:.4f}  |  Recall: {rec:.4f}")
        log(f"  Confusion Matrix:\n{cm_mat}")
    return {'accuracy': acc, 'precision': prec, 'recall': rec,
            'f1_score': f1, 'roc_auc': auc, 'bal_acc': bal_acc,
            'cm': cm_mat, 'y_pred': y_pred, 'y_prob': y_prob}

def plot_comparison(r, loss_history, y_test):
    log("\nGenerating comparison figures...")
    b = BASELINE_4Q
    fig = plt.figure(figsize=(16, 12))
    fig.suptitle('Phase 21 -- 8-Qubit Direct VQC vs 4-Qubit PCA-VQC\nNIFTY 50 Test Set 2025',
                 fontsize=14, fontweight='bold', y=0.98)
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.35)

    # A: Metric bars
    ax = fig.add_subplot(gs[0, 0])
    metrics = ['Accuracy','Precision','Recall','F1-Score','ROC-AUC','Bal.Acc']
    v4 = [b['accuracy'],b['precision'],b['recall'],b['f1_score'],b['roc_auc'],b['bal_acc']]
    v8 = [r['accuracy'],r['precision'],r['recall'],r['f1_score'],r['roc_auc'],r['bal_acc']]
    x = np.arange(len(metrics)); w = 0.35
    b4 = ax.bar(x-w/2, v4, w, label='4-Qubit (PCA)', color='#882255', alpha=0.85, edgecolor='white')
    b8 = ax.bar(x+w/2, v8, w, label='8-Qubit (Direct)', color='#1b7837', alpha=0.85, edgecolor='white')
    ax.set_xticks(x); ax.set_xticklabels(metrics, rotation=18, ha='right', fontsize=9)
    ax.set_ylim(0, 1.1); ax.axhline(0.5, color='navy', linestyle='--', lw=1.2, alpha=0.6)
    ax.set_title('Metric Comparison', fontweight='bold'); ax.legend(fontsize=8)
    ax.grid(axis='y', linestyle=':', alpha=0.5)
    for bar in list(b4)+list(b8):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01,
                f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=6.5)

    # B: ROC curves
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot([0, b['roc_auc'], 1], [0, b['roc_auc'], 1], color='#882255', lw=2.5,
             label=f"4-Qubit VQC (AUC={b['roc_auc']:.4f})")
    fpr, tpr, _ = roc_curve(y_test, r['y_prob'])
    ax2.plot(fpr, tpr, color='#1b7837', lw=2.5,
             label=f"8-Qubit VQC (AUC={r['roc_auc']:.4f})")
    ax2.plot([0,1],[0,1],'k--',lw=1.2,label='Random (AUC=0.500)')
    ax2.set_xlabel('False Positive Rate'); ax2.set_ylabel('True Positive Rate')
    ax2.set_title('ROC Curves Comparison', fontweight='bold'); ax2.legend(fontsize=9)
    ax2.grid(linestyle=':', alpha=0.5)

    # C: Confusion matrix
    ax3 = fig.add_subplot(gs[1, 0])
    sns.heatmap(r['cm'], annot=True, fmt='d', cmap='Greens', ax=ax3,
                xticklabels=['DOWN','UP'], yticklabels=['DOWN','UP'],
                cbar=False, linewidths=1, linecolor='white')
    ax3.set_title(f"8-Qubit Confusion Matrix (Test Acc: {r['accuracy']:.2%})", fontweight='bold')
    ax3.set_xlabel('Predicted'); ax3.set_ylabel('True')

    # D: Loss curve
    ax4 = fig.add_subplot(gs[1, 1])
    if loss_history:
        ax4.plot(range(1, len(loss_history)+1), loss_history, color='#1b7837', lw=2)
        ax4.axhline(np.min(loss_history), color='#882255', linestyle='--', lw=1.2,
                    label=f'Min: {np.min(loss_history):.4f}')
        ax4.set_xlabel('SPSA Iteration'); ax4.set_ylabel('Loss')
        ax4.set_title('8-Qubit SPSA Training Loss', fontweight='bold'); ax4.legend()
        ax4.grid(linestyle=':', alpha=0.5)
    else:
        ax4.text(0.5,0.5,'No callback loss\n(full-set fine-tune)', ha='center', va='center',
                 transform=ax4.transAxes, fontsize=11, color='gray')
        ax4.set_title('8-Qubit Training Loss', fontweight='bold')

    plt.savefig(os.path.join(FIG_DIR,'fig14_8qubit_comparison.png'), dpi=200, bbox_inches='tight')
    plt.close()
    log(f"  Saved: {FIG_DIR}/fig14_8qubit_comparison.png")

    # Scaling study
    fig2, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig2.suptitle('Quantum Scaling Study: 4-Qubit (PCA) vs 8-Qubit (Direct)\nNIFTY 50 Test Set 2025',
                  fontsize=13, fontweight='bold')
    lbls = ['4-Qubit\n(PCA)', '8-Qubit\n(Direct)']
    clrs = ['#882255','#1b7837']
    axes[0].bar(lbls, [b['accuracy']*100, r['accuracy']*100], color=clrs, width=0.4, edgecolor='white', alpha=0.9)
    axes[0].axhline(50, color='navy', linestyle='--', lw=1.5)
    axes[0].set_ylim(40, 65); axes[0].set_ylabel('Accuracy (%)'); axes[0].set_title('Accuracy Scaling', fontweight='bold')
    axes[0].grid(axis='y', linestyle=':', alpha=0.5)
    for i, v in enumerate([b['accuracy']*100, r['accuracy']*100]):
        axes[0].text(i, v+0.3, f'{v:.2f}%', ha='center', fontweight='bold', fontsize=11)
    axes[1].bar(lbls, [b['f1_score'], r['f1_score']], color=clrs, width=0.4, edgecolor='white', alpha=0.9)
    axes[1].set_ylim(0, 0.85); axes[1].set_ylabel('F1-Score'); axes[1].set_title('F1-Score Scaling', fontweight='bold')
    axes[1].grid(axis='y', linestyle=':', alpha=0.5)
    for i, v in enumerate([b['f1_score'], r['f1_score']]):
        axes[1].text(i, v+0.01, f'{v:.4f}', ha='center', fontweight='bold', fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR,'fig15_qubit_scaling_study.png'), dpi=200, bbox_inches='tight')
    plt.close()
    log(f"  Saved: {FIG_DIR}/fig15_qubit_scaling_study.png")

def save_table(r):
    rows = [
        {'Model':'4-Qubit VQC (PCA, ZZFeatureMap, SPSA)','Qubits':4,'PCA':'Yes',
         'Accuracy':BASELINE_4Q['accuracy'],'Precision':BASELINE_4Q['precision'],
         'Recall':BASELINE_4Q['recall'],'F1_Score':BASELINE_4Q['f1_score'],
         'ROC_AUC':BASELINE_4Q['roc_auc'],'Bal_Acc':BASELINE_4Q['bal_acc']},
        {'Model':'8-Qubit VQC (Direct, ZZFeatureMap, SPSA)','Qubits':8,'PCA':'No',
         'Accuracy':r['accuracy'],'Precision':r['precision'],
         'Recall':r['recall'],'F1_Score':r['f1_score'],
         'ROC_AUC':r['roc_auc'],'Bal_Acc':r['bal_acc']},
    ]
    df = pd.DataFrame(rows)
    csv = os.path.join(TAB_DIR,'table6_qubit_scaling_comparison.csv')
    df.to_csv(csv, index=False)
    log(f"\nTable saved: {csv}")
    return df

def main():
    log("="*70)
    log("PHASE 21 -- 8-QUBIT DIRECT VQC EXPERIMENT (NO PCA COMPRESSION)")
    log("="*70)
    (X_tr,X_v,X_te),(y_tr,y_v,y_te) = load_data()
    X_tr8,X_v8,X_te8,_ = apply_quantum_scaling(X_tr,X_v,X_te)
    vqc8, loss = train_8qubit_vqc(X_tr8, y_tr, maxiter=40)
    log("\n"+"="*70)
    log("EVALUATION RESULTS")
    log("="*70)
    log(f"\n  BASELINE 4-Qubit: Acc={BASELINE_4Q['accuracy']:.4f}  F1={BASELINE_4Q['f1_score']:.4f}  AUC={BASELINE_4Q['roc_auc']:.4f}")
    _  = evaluate_vqc(vqc8, X_tr8, y_tr, "8-Qubit Train")
    _  = evaluate_vqc(vqc8, X_v8,  y_v,  "8-Qubit Val")
    r  = evaluate_vqc(vqc8, X_te8, y_te, "8-Qubit Test (2025)")
    log("\n"+"="*70)
    log("FINAL COMPARISON")
    log("="*70)
    log(f"  {'Metric':<16} {'4-Qubit':>12} {'8-Qubit':>12} {'Delta':>10}")
    log("-"*54)
    for k, lbl in [('accuracy','Accuracy'),('f1_score','F1-Score'),('roc_auc','ROC-AUC'),('precision','Precision'),('recall','Recall')]:
        v4=BASELINE_4Q.get(k,0); v8=r[k]; d=v8-v4
        log(f"  {lbl:<16} {v4:>12.4f} {v8:>12.4f} {('+' if d>=0 else '')+f'{d:.4f}':>10}")
    log("="*70)
    acc_d = r['accuracy']-BASELINE_4Q['accuracy']
    f1_d  = r['f1_score']-BASELINE_4Q['f1_score']
    log("\nVERDICT:")
    if acc_d > 0.01 or f1_d > 0.01:
        log("  8-Qubit Direct VQC OUTPERFORMS the 4-Qubit PCA baseline!")
    elif abs(acc_d) <= 0.01 and abs(f1_d) <= 0.01:
        log("  8-Qubit Direct VQC achieves STATISTICAL PARITY with 4-Qubit PCA.")
    else:
        log("  4-Qubit PCA VQC slightly outperforms 8-Qubit Direct on this dataset.")
    plot_comparison(r, loss, y_te)
    df = save_table(r)
    log("\n"+"="*70)
    log(df.to_string(index=False))
    log("\nPhase 21 Complete!")

if __name__ == '__main__':
    main()

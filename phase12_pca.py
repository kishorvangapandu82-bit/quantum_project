"""
============================================================
phase12_pca.py
NIFTY50-VQC Project — Phase 12 PCA Execution & Analysis

PURPOSE:
    Execute PCA reduction from 8 features to 4 principal components,
    fit strictly on training set, map components to [0, pi] for quantum
    angle encoding, analyze explained variance, and plot figures.
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
from src.pca_reduction import fit_and_transform_pca

INPUT_NPZ = 'data/processed/nifty50_scaled_data.npz'
OUTPUT_NPZ = 'data/processed/nifty50_pca_4d.npz'
FIG_DIR   = 'results/figures'
TAB_DIR   = 'results/tables'

os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(TAB_DIR, exist_ok=True)


def plot_pca_variance(explained_var, cumulative_var):
    """Plot individual and cumulative explained variance ratio per principal component."""
    fig, ax1 = plt.subplots(figsize=(9, 5))

    pcs = [f"PC{i+1}" for i in range(len(explained_var))]
    x = np.arange(len(pcs))

    # Bar chart for individual variance
    bars = ax1.bar(x, explained_var * 100, color='#2b5c8f', alpha=0.85, width=0.5, label='Individual Variance (%)')
    ax1.set_xlabel('Principal Components (Qubit Mapping: PC1→Q0, PC2→Q1, PC3→Q2, PC4→Q3)', fontsize=10, fontweight='bold')
    ax1.set_ylabel('Individual Explained Variance (%)', fontsize=10, fontweight='bold', color='#2b5c8f')
    ax1.set_xticks(x)
    ax1.set_xticklabels(pcs, fontsize=11, fontweight='bold')
    ax1.set_ylim(0, 100)

    # Annotate bars
    for bar, var in zip(bars, explained_var):
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2.0, yval + 1.5, f"{var:.1%}", ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Line chart for cumulative variance
    ax2 = ax1.twinx()
    line = ax2.plot(x, cumulative_var * 100, color='#d95f02', marker='o', linewidth=2.5, markersize=8, label='Cumulative Variance (%)')
    ax2.set_ylabel('Cumulative Explained Variance (%)', fontsize=10, fontweight='bold', color='#d95f02')
    ax2.set_ylim(0, 105)

    # Annotate cumulative line points
    for i, cum_var in enumerate(cumulative_var):
        ax2.annotate(f"{cum_var:.1%}", (x[i], cum_var * 100), textcoords="offset points", xytext=(0, 10), ha='center', fontsize=10, fontweight='bold', color='#d95f02')

    plt.title('Phase 12 — PCA Explained Variance Ratio (8D Features → 4 Qubits)', fontsize=13, fontweight='bold', pad=15)
    ax1.grid(True, linestyle=':', alpha=0.5)

    plt.tight_layout()
    fig_path = os.path.join(FIG_DIR, 'fig7_pca_variance.png')
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   Saved plot: {fig_path}")


def plot_pca_loadings(components, feature_cols):
    """Plot heatmap of feature loadings for each principal component."""
    plt.figure(figsize=(10, 5))
    pc_names = [f"PC{i+1} (Qubit {i})" for i in range(components.shape[0])]

    loadings_df = pd.DataFrame(components, index=pc_names, columns=feature_cols)

    sns.heatmap(loadings_df, annot=True, fmt='.3f', cmap='coolwarm', center=0, cbar=True, linewidths=0.5)
    plt.title('Phase 12 — PCA Component Feature Loadings (Feature Contributions)', fontsize=13, fontweight='bold', pad=12)
    plt.xlabel('Original Technical Features', fontsize=11, fontweight='bold')
    plt.ylabel('Principal Component', fontsize=11, fontweight='bold')
    plt.xticks(rotation=45, ha='right')

    plt.tight_layout()
    fig_path = os.path.join(FIG_DIR, 'fig8_pca_loadings.png')
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   Saved plot: {fig_path}")


def main():
    print("=" * 75)
    print("PHASE 12 — PCA DIMENSIONALITY REDUCTION (8D FEATURES → 4 QUBITS)")
    print("=" * 75)

    if not os.path.exists(INPUT_NPZ):
        print(f"❌ Error: {INPUT_NPZ} not found. Run Phase 10 first.")
        return

    print(f"📥 Loading preprocessed standardized features: {INPUT_NPZ}")
    data = np.load(INPUT_NPZ, allow_pickle=True)

    X_train_std = data['X_train_std']
    X_val_std   = data['X_val_std']
    X_test_std  = data['X_test_std']

    y_train = data['y_train']
    y_val   = data['y_val']
    y_test  = data['y_test']

    feature_cols = list(data['feature_cols'])

    print(f"   Original feature dimensions: {X_train_std.shape[1]} features")
    print(f"   Target quantum dimensions:   4 principal components (4 qubits)")

    # Execute PCA fit and transform
    print("\n⚙️ Fitting PCA strictly on Training set and transforming all splits...")
    pca_results = fit_and_transform_pca(X_train_std, X_val_std, X_test_std, n_components=4, feature_range=(0, np.pi))

    exp_var = pca_results['explained_variance']
    cum_var = pca_results['cumulative_variance']
    components = pca_results['components']

    X_train_quantum = pca_results['X_train_quantum']
    X_val_quantum   = pca_results['X_val_quantum']
    X_test_quantum  = pca_results['X_test_quantum']

    print("\n📊 Explained Variance Ratio per Component:")
    print("-" * 65)
    print(f"{'Component':<12} {'Qubit Target':<15} {'Explained Var':<18} {'Cumulative Var':<18}")
    print("-" * 65)
    for i in range(len(exp_var)):
        print(f"PC{i+1:<10} Qubit {i:<13} {exp_var[i]:<18.2%} {cum_var[i]:<18.2%}")
    print("-" * 65)

    print("\n📊 Feature Loadings Matrix (Contributions to PCs):")
    print("=" * 85)
    loadings_df = pd.DataFrame(components, index=[f"PC{i+1}" for i in range(4)], columns=feature_cols)
    print(loadings_df.to_string())
    print("=" * 85)

    # Save summary table CSV
    tab_path = os.path.join(TAB_DIR, 'table3_pca_variance.csv')
    pd.DataFrame({
        'Component': [f'PC{i+1}' for i in range(4)],
        'Qubit': [f'Qubit {i}' for i in range(4)],
        'Explained_Variance_Pct': [f'{v:.4%}' for v in exp_var],
        'Cumulative_Variance_Pct': [f'{c:.4%}' for c in cum_var]
    }).to_csv(tab_path, index=False)
    print(f"\n💾 Summary table saved: {tab_path}")

    # Generate figures
    print("\n🎨 Generating Figures...")
    plot_pca_variance(exp_var, cum_var)
    plot_pca_loadings(components, feature_cols)

    # Range and NaN Checks
    print("\n🔍 Quantum Angle Range Verification ([0, π] on Train):")
    print("-" * 75)
    print(f"{'Component':<12} {'Train Min':<12} {'Train Max':<12} {'Val Min':<12} {'Val Max':<12} {'Test Min':<12} {'Test Max':<12}")
    print("-" * 75)
    for i in range(4):
        tr_min, tr_max = X_train_quantum[:, i].min(), X_train_quantum[:, i].max()
        va_min, va_max = X_val_quantum[:, i].min(),   X_val_quantum[:, i].max()
        te_min, te_max = X_test_quantum[:, i].min(),  X_test_quantum[:, i].max()
        print(f"PC{i+1:<10} {tr_min:<12.4f} {tr_max:<12.4f} {va_min:<12.4f} {va_max:<12.4f} {te_min:<12.4f} {te_max:<12.4f}")
    print("-" * 75)

    has_nans = np.isnan(X_train_quantum).any() or np.isnan(X_val_quantum).any() or np.isnan(X_test_quantum).any()
    print(f"\nMissing value check: {'❌ NaN FOUND!' if has_nans else '✅ ZERO NaNs'}")

    print("\n✅ Phase 12 PCA Execution Completed Successfully!")
    print(f"\nTarget output path: {OUTPUT_NPZ}")

if __name__ == '__main__':
    main()

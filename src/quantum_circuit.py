"""
============================================================
src/quantum_circuit.py
NIFTY50-VQC Project — Quantum Feature Map & Ansatz Architecture

PURPOSE:
    Construct 4-qubit Quantum Feature Maps and Variational Ansatzes
    using Qiskit 2.x circuit functions:
    1. Feature Map: ZZFeatureMap (zz_feature_map) or ZFeatureMap (z_feature_map)
       Encodes 4 classical principal components into qubit state rotation angles.
    2. Ansatz: RealAmplitudes (real_amplitudes) or EfficientSU2 (efficient_su2)
       Parametrized quantum circuit with trainable rotation gates (RY) and entangling gates (CZ / CNOT).

QUANTUM HARDWARE SPECIFICATIONS:
    - Qubits: 4 (Q0, Q1, Q2, Q3)
    - Entanglement Strategy: Linear (nearest-neighbor entanglement)
    - Rotation Gates: RY(theta), RZ(theta)
    - Simulator Backend: StatevectorSampler (Qiskit Primitives)

Author: NIFTY50-VQC Project
Date:   2026-10-02
============================================================
"""

import matplotlib.pyplot as plt
import numpy as np
from qiskit.circuit.library import zz_feature_map, z_feature_map, real_amplitudes, efficient_su2
from qiskit import QuantumCircuit


def build_feature_map(num_qubits=4, reps=1, entanglement='linear', map_type='zz'):
    """
    Construct Quantum Feature Map circuit for encoding classical data into quantum states.

    Args:
        num_qubits (int): Number of qubits (default: 4)
        reps (int): Repetitions / depth of feature map
        entanglement (str): Entanglement pattern ('linear', 'full', 'circular')
        map_type (str): 'zz' or 'z'

    Returns:
        QuantumCircuit: Qiskit QuantumCircuit representing the feature map
    """
    if map_type == 'zz':
        return zz_feature_map(feature_dimension=num_qubits, reps=reps, entanglement=entanglement)
    elif map_type == 'z':
        return z_feature_map(feature_dimension=num_qubits, reps=reps)
    else:
        raise ValueError(f"Unknown map_type: {map_type}. Choose 'zz' or 'z'.")


def build_ansatz(num_qubits=4, reps=2, entanglement='linear', ansatz_type='real_amplitudes'):
    """
    Construct Parametrized Variational Quantum Ansatz circuit.

    Args:
        num_qubits (int): Number of qubits (default: 4)
        reps (int): Depth / repetitions of variational layers
        entanglement (str): Entanglement pattern ('linear', 'full')
        ansatz_type (str): 'real_amplitudes' or 'efficient_su2'

    Returns:
        QuantumCircuit: Qiskit QuantumCircuit representing the variational ansatz
    """
    if ansatz_type == 'real_amplitudes':
        return real_amplitudes(num_qubits=num_qubits, reps=reps, entanglement=entanglement)
    elif ansatz_type == 'efficient_su2':
        return efficient_su2(num_qubits=num_qubits, reps=reps, entanglement=entanglement)
    else:
        raise ValueError(f"Unknown ansatz_type: {ansatz_type}. Choose 'real_amplitudes' or 'efficient_su2'.")


def plot_quantum_circuit(feature_map, ansatz, save_path=None):
    """
    Draw and save a publication-quality quantum circuit diagram showing Feature Map + Ansatz.

    Args:
        feature_map (QuantumCircuit): Feature Map circuit
        ansatz (QuantumCircuit): Variational Ansatz circuit
        save_path (str): File path to save circuit image

    Returns:
        plt.Figure: Matplotlib figure object
    """
    combined_qc = QuantumCircuit(feature_map.num_qubits)
    combined_qc.compose(feature_map, inplace=True)
    combined_qc.barrier()
    combined_qc.compose(ansatz, inplace=True)

    fig = combined_qc.draw(output='mpl', style='iqp')
    plt.title("Phase 13 — VQC Architecture: 4-Qubit ZZFeatureMap + RealAmplitudes Ansatz", fontsize=12, fontweight='bold', pad=15)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"   Saved quantum circuit plot: {save_path}")

    plt.close()
    return fig

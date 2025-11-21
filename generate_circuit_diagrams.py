"""
Generate Clean Circuit Diagrams for All Quantum Approaches
Creates publication-ready circuit visualizations
"""

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
import matplotlib.pyplot as plt
import os

os.makedirs('circuits/final', exist_ok=True)

print("="*80)
print("Generating Circuit Diagrams for Paper/Documentation")
print("="*80)

# ============================================================================
# 1. Q-SVM & Q-KNN Quantum Feature Map (SAME CIRCUIT)
# ============================================================================

def quantum_feature_map_qsvm(n_qubits=6):
    """
    Quantum feature map used by both Q-SVM and Q-KNN
    This is the ONLY quantum part - creates quantum states from classical data
    """
    qc = QuantumCircuit(n_qubits)

    # Example feature values
    x = [0.5, 0.3, 0.7, 0.2, 0.9, 0.4][:n_qubits]

    qc.barrier(label='Layer 1: RY Encoding')
    for i in range(n_qubits):
        qc.ry(np.pi * x[i], i)

    qc.barrier(label='Layer 2: CZ Entanglement')
    for i in range(n_qubits - 1):
        qc.cz(i, i + 1)

    qc.barrier(label='Layer 3: Feature Interactions')
    for i in range(n_qubits):
        qc.rz(np.pi * x[i] * x[(i+1) % n_qubits], i)

    qc.barrier(label='Layer 4: Star Entanglement')
    for i in range(1, n_qubits):
        qc.cz(0, i)

    qc.barrier(label='Layer 5: Re-encoding')
    for i in range(n_qubits):
        qc.ry(np.pi/2 * x[i], i)

    qc.barrier(label='Layer 6: Ring Closure')
    if n_qubits > 2:
        qc.cz(n_qubits - 1, 0)

    return qc

# Generate for 4 and 6 qubits
for n_q in [4, 6]:
    print(f"\n[1] Creating Q-SVM/Q-KNN feature map ({n_q} qubits)...")
    qc = quantum_feature_map_qsvm(n_q)

    fig = qc.draw('mpl', style='iqp', fold=-1, scale=0.9)
    plt.suptitle(f'Q-SVM & Q-KNN: Quantum Feature Map ({n_q} Qubits)\n'
                 f'Encodes classical data → quantum state |ψ(x)⟩',
                 fontsize=13, fontweight='bold', y=0.98)

    # Add explanation
    plt.figtext(0.5, 0.02,
                'Usage:\n'
                '• Q-SVM: K(x,y) = |⟨ψ(x)|ψ(y)⟩|² → Quantum kernel for SVM\n'
                '• Q-KNN: d(x,y) = √(1 - K(x,y)) → Quantum distance for k-NN',
                ha='center', fontsize=9, family='monospace',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))

    plt.savefig(f'circuits/final/qsvm_qknn_feature_map_{n_q}qubits.png',
                dpi=300, bbox_inches='tight', facecolor='white')
    print(f"   Saved: circuits/final/qsvm_qknn_feature_map_{n_q}qubits.png")
    plt.close()

# ============================================================================
# 2. VQC (Variational Quantum Classifier) - Full Quantum
# ============================================================================

def vqc_complete_circuit(n_qubits=6):
    """
    Complete VQC circuit: Feature Map + Variational Classifier
    This is end-to-end quantum classification
    """
    qc = QuantumCircuit(n_qubits)
    x = [0.5, 0.3, 0.7, 0.2, 0.9, 0.4][:n_qubits]

    # Part 1: Feature encoding (same as Q-SVM)
    qc.barrier(label='Feature Map')
    for i in range(n_qubits):
        qc.ry(np.pi * x[i], i)
    for i in range(n_qubits - 1):
        qc.cz(i, i + 1)
    for i in range(n_qubits):
        qc.rz(np.pi * x[i] * x[(i+1) % n_qubits], i)

    # Part 2: Variational classifier (trainable!)
    n_layers = 2
    params = ParameterVector('θ', n_qubits * n_layers * 2)

    param_idx = 0
    for layer in range(n_layers):
        qc.barrier(label=f'Variational Layer {layer+1}')

        # Trainable rotations
        for i in range(n_qubits):
            qc.ry(params[param_idx], i)
            param_idx += 1
            qc.rz(params[param_idx], i)
            param_idx += 1

        # Entanglement
        for i in range(n_qubits - 1):
            qc.cx(i, i + 1)
        if n_qubits > 2:
            qc.cx(n_qubits - 1, 0)

    # Measurement
    qc.barrier(label='Measurement')
    qc.measure_all()

    return qc

print(f"\n[2] Creating VQC complete circuit (6 qubits)...")
qc_vqc = vqc_complete_circuit(6)

fig = qc_vqc.draw('mpl', style='iqp', fold=-1, scale=0.7)
plt.suptitle('VQC: Full Quantum Classifier (6 Qubits)\n'
             'Feature Map + Trainable Variational Circuit + Measurement',
             fontsize=13, fontweight='bold', y=0.98)

plt.figtext(0.5, 0.02,
            'Architecture:\n'
            '• Feature Map: Encodes classical data (same as Q-SVM/Q-KNN)\n'
            '• Variational Layers: TRAINABLE parameters (θ) optimized via gradient descent\n'
            '• Measurement: Quantum measurement determines classification',
            ha='center', fontsize=9, family='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.3))

plt.savefig('circuits/final/vqc_complete_6qubits.png',
            dpi=300, bbox_inches='tight', facecolor='white')
print(f"   Saved: circuits/final/vqc_complete_6qubits.png")
plt.close()

# ============================================================================
# 3. Comparison Diagram
# ============================================================================

print(f"\n[3] Creating algorithm comparison diagram...")

fig, axes = plt.subplots(3, 1, figsize=(14, 12))

# Q-SVM circuit
qc_qsvm = quantum_feature_map_qsvm(4)
qc_qsvm.draw('mpl', ax=axes[0], style='iqp', fold=-1)
axes[0].set_title('Q-SVM: Quantum Kernel Method\n'
                  'K(x,y) = |⟨ψ(x)|ψ(y)⟩|² → Precomputed kernel for classical SVM',
                  fontsize=12, fontweight='bold', pad=10)

# Q-KNN circuit (same as Q-SVM)
qc_qknn = quantum_feature_map_qsvm(4)
qc_qknn.draw('mpl', ax=axes[1], style='iqp', fold=-1)
axes[1].set_title('Q-KNN: Quantum Distance Method\n'
                  'd(x,y) = √(1 - |⟨ψ(x)|ψ(y)⟩|²) → Distance for k-NN classification',
                  fontsize=12, fontweight='bold', pad=10)

# VQC circuit (simplified view)
qc_vqc_simple = QuantumCircuit(4)
x = [0.5, 0.3, 0.7, 0.2]
for i in range(4):
    qc_vqc_simple.ry(np.pi * x[i], i)
for i in range(3):
    qc_vqc_simple.cz(i, i+1)

qc_vqc_simple.barrier(label='+ Variational Layers')
params = ParameterVector('θ', 8)
for i in range(4):
    qc_vqc_simple.ry(params[i], i)
    qc_vqc_simple.rz(params[i+4], i)

qc_vqc_simple.barrier(label='Measure')
qc_vqc_simple.measure_all()

qc_vqc_simple.draw('mpl', ax=axes[2], style='iqp', fold=-1)
axes[2].set_title('VQC: End-to-End Quantum Classifier\n'
                  'Feature Map + TRAINABLE Circuit + Measurement → Direct classification',
                  fontsize=12, fontweight='bold', pad=10)

plt.suptitle('Quantum ML Approaches Comparison (4 Qubits)',
             fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig('circuits/final/quantum_approaches_comparison.png',
            dpi=300, bbox_inches='tight', facecolor='white')
print(f"   Saved: circuits/final/quantum_approaches_comparison.png")
plt.close()

print("\n" + "="*80)
print("✅ Circuit diagrams generated!")
print("="*80)
print(f"\nGenerated files in circuits/final/:")
print(f"  • qsvm_qknn_feature_map_4qubits.png")
print(f"  • qsvm_qknn_feature_map_6qubits.png")
print(f"  • vqc_complete_6qubits.png")
print(f"  • quantum_approaches_comparison.png")

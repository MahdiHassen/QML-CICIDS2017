"""
Visualize Quantum Circuits with Layer Labels
Q-SVM and Q-KNN use the same 6-layer quantum feature map
"""

import numpy as np
from qiskit import QuantumCircuit
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import os

# Create output directory
os.makedirs('circuits/labeled', exist_ok=True)

def quantum_feature_map_labeled(n_qubits, show_example_values=False):
    """
    Deep 6-layer quantum feature map with entanglement
    Used by both Q-SVM (kernel) and Q-KNN (distance)
    """
    qc = QuantumCircuit(n_qubits)

    # Use example feature values for demonstration
    x = np.random.rand(n_qubits) if show_example_values else [0.5] * n_qubits

    # LAYER 1: RY Encoding
    qc.barrier(label='Layer 1')
    for i in range(n_qubits):
        qc.ry(np.pi * x[i], i)

    # LAYER 2: Linear Chain Entanglement
    qc.barrier(label='Layer 2')
    for i in range(n_qubits - 1):
        qc.cz(i, i + 1)

    # LAYER 3: Feature Interactions (KEY!)
    qc.barrier(label='Layer 3')
    for i in range(n_qubits):
        qc.rz(np.pi * x[i] * x[(i+1) % n_qubits], i)

    # LAYER 4: Star Pattern Entanglement
    qc.barrier(label='Layer 4')
    for i in range(1, n_qubits):
        qc.cz(0, i)

    # LAYER 5: Re-encoding
    qc.barrier(label='Layer 5')
    for i in range(n_qubits):
        qc.ry(np.pi/2 * x[i], i)

    # LAYER 6: Ring Closure
    qc.barrier(label='Layer 6')
    if n_qubits > 2:
        qc.cz(n_qubits-1, 0)

    return qc

# Generate circuits for different qubit counts
qubit_counts = [4, 6, 8, 12]

for n_qubits in qubit_counts:
    print(f"Generating {n_qubits}-qubit circuit diagram...")

    # Create circuit
    qc = quantum_feature_map_labeled(n_qubits, show_example_values=False)

    # Draw circuit
    fig = qc.draw('mpl', style='iqp', fold=-1, scale=0.8)
    plt.suptitle(f'Quantum Feature Map: {n_qubits} Qubits\n'
                 f'Used by Q-SVM (kernel) and Q-KNN (distance)',
                 fontsize=14, fontweight='bold', y=0.98)

    # Save
    filename = f'circuits/labeled/quantum_circuit_{n_qubits}qubits.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {filename}")

# Create a detailed annotated version for 6 qubits
print("\nGenerating detailed annotated 6-qubit circuit...")
qc = quantum_feature_map_labeled(6, show_example_values=False)

fig = qc.draw('mpl', style='iqp', fold=-1, scale=1.0)

# Add detailed annotations
plt.suptitle('Quantum Feature Map Architecture (6 Qubits)\n'
             'Q-SVM uses quantum kernel: K(x,y) = |⟨ψ(x)|ψ(y)⟩|²\n'
             'Q-KNN uses quantum distance: d(x,y) = √(1 - K(x,y))',
             fontsize=12, fontweight='bold', y=0.98)

plt.figtext(0.02, 0.02,
            'Layer 1: RY Encoding - RY(π·x[i])\n'
            'Layer 2: Linear Chain - CZ(i, i+1)\n'
            'Layer 3: Feature Interactions - RZ(π·x[i]·x[i+1]) ⭐\n'
            'Layer 4: Star Pattern - CZ(0, i)\n'
            'Layer 5: Re-encoding - RY(π/2·x[i])\n'
            'Layer 6: Ring Closure - CZ(n-1, 0)',
            fontsize=9, family='monospace',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))

plt.savefig('circuits/labeled/quantum_circuit_annotated_6qubits.png',
            dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print("  Saved: circuits/labeled/quantum_circuit_annotated_6qubits.png")

# Create comparison diagram showing Q-SVM vs Q-KNN usage
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

# Q-SVM usage
qc1 = quantum_feature_map_labeled(4)
qc1.draw('mpl', ax=ax1, style='iqp', fold=-1)
ax1.set_title('Q-SVM: Quantum Kernel Method\n'
              'K(x,y) = |⟨ψ(x)|ψ(y)⟩|² → Precomputed kernel for SVM',
              fontsize=12, fontweight='bold', pad=10)

# Q-KNN usage
qc2 = quantum_feature_map_labeled(4)
qc2.draw('mpl', ax=ax2, style='iqp', fold=-1)
ax2.set_title('Q-KNN: Quantum Distance Method\n'
              'd(x,y) = √(1 - |⟨ψ(x)|ψ(y)⟩|²) → Distance for k-NN classification',
              fontsize=12, fontweight='bold', pad=10)

plt.suptitle('Same Quantum Circuit - Different Usage',
             fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig('circuits/labeled/qsvm_vs_qknn_comparison.png',
            dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print("  Saved: circuits/labeled/qsvm_vs_qknn_comparison.png")

# Create a single comprehensive diagram with all layer details
print("\nGenerating comprehensive layer breakdown...")
qc = quantum_feature_map_labeled(6)

# Get circuit depth and gate counts
depth = qc.depth()
gate_counts = qc.count_ops()

fig = qc.draw('mpl', style='iqp', fold=-1, scale=1.2)

title_text = (
    'Complete Quantum Feature Map Architecture\n'
    f'Circuit Depth: {depth} | Gates: {sum(gate_counts.values())} | '
    f'RY: {gate_counts.get("ry", 0)} | CZ: {gate_counts.get("cz", 0)} | RZ: {gate_counts.get("rz", 0)}'
)

plt.suptitle(title_text, fontsize=11, fontweight='bold', y=0.98)

# Add comprehensive legend
legend_text = """
╔═══════════════════════════════════════════════════════════╗
║  6-LAYER QUANTUM FEATURE MAP ARCHITECTURE                 ║
╠═══════════════════════════════════════════════════════════╣
║  Layer 1: RY Encoding                                     ║
║    • RY(π·x[i]) rotation on each qubit                    ║
║    • Encodes classical features into quantum states       ║
║                                                            ║
║  Layer 2: Linear Chain Entanglement                       ║
║    • CZ gates between adjacent qubits: CZ(i, i+1)         ║
║    • Creates nearest-neighbor correlations                ║
║                                                            ║
║  Layer 3: Feature Interactions ⭐ (KEY INNOVATION)         ║
║    • RZ(π·x[i]·x[i+1]) - product of neighboring features  ║
║    • Captures non-linear feature relationships            ║
║                                                            ║
║  Layer 4: Star Pattern Entanglement                       ║
║    • CZ(0, i) connects first qubit to all others          ║
║    • Creates long-range quantum correlations              ║
║                                                            ║
║  Layer 5: Re-encoding                                     ║
║    • RY(π/2·x[i]) reinforces quantum state encoding       ║
║    • Adds additional expressivity                         ║
║                                                            ║
║  Layer 6: Ring Closure                                    ║
║    • CZ(n-1, 0) connects last qubit to first              ║
║    • Completes circular topology                          ║
╚═══════════════════════════════════════════════════════════╝

Usage:
• Q-SVM: Computes kernel K(x,y) = |⟨ψ(x)|ψ(y)⟩|²
• Q-KNN: Computes distance d(x,y) = √(1 - K(x,y))
"""

plt.figtext(0.5, -0.15, legend_text,
            fontsize=7, family='monospace', ha='center',
            bbox=dict(boxstyle='round', facecolor='lightyellow',
                     alpha=0.8, edgecolor='black', linewidth=2))

plt.savefig('circuits/labeled/quantum_circuit_comprehensive.png',
            dpi=300, bbox_inches='tight', facecolor='white', pad_inches=0.5)
plt.close()
print("  Saved: circuits/labeled/quantum_circuit_comprehensive.png")

print("\n" + "="*70)
print("CIRCUIT VISUALIZATION COMPLETE!")
print("="*70)
print("\nGenerated diagrams:")
print("  • Individual circuits: 4, 6, 8, 12 qubits")
print("  • Annotated 6-qubit diagram")
print("  • Q-SVM vs Q-KNN comparison")
print("  • Comprehensive layer breakdown")
print(f"\nAll saved in: circuits/labeled/")
print("\nKey Architecture Features:")
print("  ✓ 6 layers with multi-scale entanglement")
print("  ✓ Feature interactions via RZ(π·x[i]·x[i+1])")
print("  ✓ Linear + Star + Ring entanglement patterns")
print("  ✓ Same circuit used by both Q-SVM and Q-KNN")

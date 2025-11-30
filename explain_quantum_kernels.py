"""
Comprehensive Explanation of Quantum Kernels and Quantum Distance
for Q-SVM and Q-KNN
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patches as mpatches

print("="*80)
print(" QUANTUM MACHINE LEARNING: KERNELS & DISTANCE EXPLAINED")
print("="*80)

# Create comprehensive visualization
fig = plt.figure(figsize=(18, 12))
gs = fig.add_gridspec(3, 3, hspace=0.4, wspace=0.3)

# ============================================================================
# 1. Quantum Feature Map Visualization
# ============================================================================
ax1 = fig.add_subplot(gs[0, :])
ax1.set_xlim(0, 10)
ax1.set_ylim(0, 6)
ax1.axis('off')
ax1.set_title('Step 1: Quantum Feature Map - Classical Data → Quantum State',
              fontsize=14, fontweight='bold', pad=20)

# Classical data
classical_box = FancyBboxPatch((0.5, 2), 1.5, 2, boxstyle="round,pad=0.1",
                               edgecolor='blue', facecolor='lightblue', linewidth=2)
ax1.add_patch(classical_box)
ax1.text(1.25, 3, 'Classical\nData\nx = [x₁, x₂, ..., xₙ]',
         ha='center', va='center', fontsize=10, fontweight='bold')

# Arrow 1
arrow1 = FancyArrowPatch((2, 3), (3.5, 3), arrowstyle='->', mutation_scale=30,
                        linewidth=3, color='purple')
ax1.add_patch(arrow1)
ax1.text(2.75, 3.5, 'Quantum\nCircuit', ha='center', fontsize=9,
         bbox=dict(boxstyle='round', facecolor='lavender'))

# Quantum circuit representation
circuit_box = FancyBboxPatch((3.5, 1.5), 2.5, 3, boxstyle="round,pad=0.1",
                            edgecolor='purple', facecolor='#f0e6ff', linewidth=2)
ax1.add_patch(circuit_box)
ax1.text(4.75, 4, '6-Layer Quantum Circuit', ha='center', fontweight='bold', fontsize=10)
ax1.text(4.75, 3.4, '1. RY(π·xᵢ) - Encode', ha='center', fontsize=8)
ax1.text(4.75, 3.0, '2. CZ gates - Entangle', ha='center', fontsize=8)
ax1.text(4.75, 2.6, '3. RZ(π·xᵢ·xⱼ) - Interact', ha='center', fontsize=8)
ax1.text(4.75, 2.2, '4. Star entanglement', ha='center', fontsize=8)
ax1.text(4.75, 1.8, '5. Re-encode', ha='center', fontsize=8)

# Arrow 2
arrow2 = FancyArrowPatch((6, 3), (7.5, 3), arrowstyle='->', mutation_scale=30,
                        linewidth=3, color='green')
ax1.add_patch(arrow2)
ax1.text(6.75, 3.5, 'Execute', ha='center', fontsize=9,
         bbox=dict(boxstyle='round', facecolor='lightgreen'))

# Quantum state
state_box = FancyBboxPatch((7.5, 2), 2, 2, boxstyle="round,pad=0.1",
                          edgecolor='green', facecolor='lightgreen', linewidth=2)
ax1.add_patch(state_box)
ax1.text(8.5, 3, 'Quantum State\n|ψ(x)⟩\n(2ⁿ complex numbers)',
         ha='center', va='center', fontsize=10, fontweight='bold')

# Dimension indicator
ax1.text(8.5, 0.8, 'For n=6 qubits: |ψ⟩ lives in 2⁶ = 64 dimensional space!',
         ha='center', fontsize=9, style='italic',
         bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))

# ============================================================================
# 2. Q-SVM Quantum Kernel
# ============================================================================
ax2 = fig.add_subplot(gs[1, 0:2])
ax2.set_xlim(0, 10)
ax2.set_ylim(0, 6)
ax2.axis('off')
ax2.set_title('Q-SVM: Quantum Kernel Approach', fontsize=13, fontweight='bold', pad=15)

# Two quantum states
state1_box = FancyBboxPatch((0.5, 3.5), 1.5, 1.5, boxstyle="round,pad=0.1",
                           edgecolor='blue', facecolor='lightblue', linewidth=2)
ax2.add_patch(state1_box)
ax2.text(1.25, 4.25, '|ψ(x)⟩', ha='center', va='center', fontsize=12, fontweight='bold')

state2_box = FancyBboxPatch((0.5, 1), 1.5, 1.5, boxstyle="round,pad=0.1",
                           edgecolor='red', facecolor='lightcoral', linewidth=2)
ax2.add_patch(state2_box)
ax2.text(1.25, 1.75, '|ψ(y)⟩', ha='center', va='center', fontsize=12, fontweight='bold')

# Inner product
arrow3 = FancyArrowPatch((2.2, 4.25), (3.5, 3.5), arrowstyle='->', mutation_scale=20,
                        linewidth=2, color='purple')
ax2.add_patch(arrow3)
arrow4 = FancyArrowPatch((2.2, 1.75), (3.5, 2.5), arrowstyle='->', mutation_scale=20,
                        linewidth=2, color='purple')
ax2.add_patch(arrow4)

inner_box = FancyBboxPatch((3.5, 2.2), 2, 1.6, boxstyle="round,pad=0.1",
                          edgecolor='purple', facecolor='plum', linewidth=2)
ax2.add_patch(inner_box)
ax2.text(4.5, 3.3, 'Inner Product', ha='center', fontweight='bold', fontsize=11)
ax2.text(4.5, 2.8, '⟨ψ(x)|ψ(y)⟩', ha='center', fontsize=14)

# Kernel value
arrow5 = FancyArrowPatch((5.5, 3), (6.5, 3), arrowstyle='->', mutation_scale=25,
                        linewidth=3, color='green')
ax2.add_patch(arrow5)
ax2.text(6, 3.5, '|·|²', ha='center', fontsize=10,
         bbox=dict(boxstyle='round', facecolor='lightyellow'))

kernel_box = FancyBboxPatch((6.5, 2.2), 2.5, 1.6, boxstyle="round,pad=0.1",
                           edgecolor='darkgreen', facecolor='lightgreen', linewidth=3)
ax2.add_patch(kernel_box)
ax2.text(7.75, 3.4, 'QUANTUM KERNEL', ha='center', fontweight='bold', fontsize=11)
ax2.text(7.75, 2.8, 'K(x,y) = |⟨ψ(x)|ψ(y)⟩|²', ha='center', fontsize=12)

# Properties
ax2.text(4.5, 0.8, '• Range: [0, 1]', ha='center', fontsize=9)
ax2.text(4.5, 0.4, '• K(x,x) = 1 (identical points)', ha='center', fontsize=9)
ax2.text(7.75, 0.8, '• K(x,y) ≈ 1: Similar in quantum space', ha='center', fontsize=9)
ax2.text(7.75, 0.4, '• K(x,y) ≈ 0: Different in quantum space', ha='center', fontsize=9)

# ============================================================================
# 3. Q-KNN Quantum Distance
# ============================================================================
ax3 = fig.add_subplot(gs[1, 2])
ax3.set_xlim(0, 5)
ax3.set_ylim(0, 6)
ax3.axis('off')
ax3.set_title('Q-KNN: Distance', fontsize=12, fontweight='bold', pad=10)

# Start with kernel
kernel_box2 = FancyBboxPatch((0.5, 4), 4, 1.2, boxstyle="round,pad=0.1",
                            edgecolor='green', facecolor='lightgreen', linewidth=2)
ax3.add_patch(kernel_box2)
ax3.text(2.5, 4.6, 'K(x,y) = |⟨ψ(x)|ψ(y)⟩|²', ha='center', fontsize=11)

# Arrow down
arrow6 = FancyArrowPatch((2.5, 4), (2.5, 3), arrowstyle='->', mutation_scale=25,
                        linewidth=3, color='orange')
ax3.add_patch(arrow6)
ax3.text(3.5, 3.5, 'd = √(1-K)', ha='center', fontsize=9,
         bbox=dict(boxstyle='round', facecolor='lightyellow'))

# Distance
dist_box = FancyBboxPatch((0.5, 1.5), 4, 1.2, boxstyle="round,pad=0.1",
                         edgecolor='darkorange', facecolor='#ffe6cc', linewidth=3)
ax3.add_patch(dist_box)
ax3.text(2.5, 2.3, 'QUANTUM DISTANCE', ha='center', fontweight='bold', fontsize=11)
ax3.text(2.5, 1.9, 'd(x,y) = √(1 - K(x,y))', ha='center', fontsize=10)

# Properties
ax3.text(2.5, 0.8, '• Range: [0, 1]', ha='center', fontsize=8)
ax3.text(2.5, 0.4, '• d(x,x) = 0 (same point)', ha='center', fontsize=8)
ax3.text(2.5, 0.0, '• d(x,y) ≈ 1: Very different', ha='center', fontsize=8)

# ============================================================================
# 4. Mathematical Comparison Table
# ============================================================================
ax4 = fig.add_subplot(gs[2, :])
ax4.axis('off')
ax4.set_title('Mathematical Comparison: Classical vs Quantum', fontsize=13, fontweight='bold', pad=15)

# Create table data
table_data = [
    ['', 'Classical SVM (RBF)', 'Classical K-NN', 'Q-SVM', 'Q-KNN'],
    ['Method', 'Kernel-based', 'Distance-based', 'Quantum Kernel', 'Quantum Distance'],
    ['Feature Map', 'Implicit (RBF)', 'None (original)', 'Quantum Circuit', 'Quantum Circuit'],
    ['Formula', 'K(x,y)=exp(-γ‖x-y‖²)', 'd(x,y)=‖x-y‖', 'K(x,y)=|⟨ψ(x)|ψ(y)⟩|²', 'd(x,y)=√(1-K(x,y))'],
    ['Output', 'Kernel matrix', 'Distances', 'Kernel matrix', 'Distance matrix'],
    ['Space', 'Infinite-dim (implicit)', 'Original space', '2ⁿ-dim Hilbert space', '2ⁿ-dim Hilbert space'],
    ['Computation', 'Classical', 'Classical', 'Quantum circuit', 'Quantum circuit'],
]

# Draw table
cell_height = 0.7
cell_widths = [1.5, 3, 3, 3, 3]
start_y = 5

colors = ['#e3f2fd', '#fff3e0', '#fff3e0', '#e8f5e9', '#fff9c4']

for i, row in enumerate(table_data):
    y = start_y - i * cell_height
    x = 0
    for j, (cell, width) in enumerate(zip(row, cell_widths)):
        if i == 0:  # Header
            color = '#1976d2' if j == 0 else colors[j-1] if j > 0 else 'lightgray'
            text_color = 'white' if j == 0 else 'black'
            fontweight = 'bold'
        elif j == 0:  # Row labels
            color = '#1976d2'
            text_color = 'white'
            fontweight = 'bold'
        else:
            color = colors[j-1]
            text_color = 'black'
            fontweight = 'normal'

        rect = FancyBboxPatch((x, y), width, cell_height,
                             boxstyle="round,pad=0.02",
                             edgecolor='gray', facecolor=color, linewidth=1)
        ax4.add_patch(rect)

        ax4.text(x + width/2, y + cell_height/2, cell,
                ha='center', va='center', fontsize=8.5,
                fontweight=fontweight, color=text_color, wrap=True)
        x += width

ax4.set_xlim(-0.5, sum(cell_widths) + 0.5)
ax4.set_ylim(-1, 6.5)

plt.suptitle('Quantum Machine Learning: How Q-SVM and Q-KNN Work',
            fontsize=16, fontweight='bold', y=0.98)

plt.savefig('quantum_kernel_explanation.png', dpi=300, bbox_inches='tight')
print("✓ Saved: quantum_kernel_explanation.png")

# ============================================================================
# Print Detailed Explanation
# ============================================================================
print("\n" + "="*80)
print(" DETAILED MATHEMATICAL EXPLANATION")
print("="*80)

print("\n" + "─"*80)
print(" STEP 1: QUANTUM FEATURE MAP (SHARED BY BOTH Q-SVM & Q-KNN)")
print("─"*80)

print("""
The quantum feature map Φ: ℝⁿ → ℋ₂ⁿ transforms classical data into quantum states:

    x = [x₁, x₂, ..., xₙ] → |ψ(x)⟩

The 6-layer circuit creates a highly expressive mapping:

    Layer 1: RY(π·xᵢ) for i=1..n
             Encodes classical values as qubit rotation angles

    Layer 2: CZ gates (linear chain)
             Creates entanglement between adjacent qubits

    Layer 3: RZ(π·xᵢ·xⱼ) - KEY INNOVATION!
             Encodes pairwise feature interactions
             This is what makes the kernel so powerful

    Layer 4: Star pattern CZ gates
             Entangles all qubits with the first qubit

    Layer 5: RY(π/2·xᵢ)
             Re-encodes features with different rotation

    Layer 6: Ring closure CZ(n-1, 0)
             Creates periodic boundary entanglement

Result: |ψ(x)⟩ is a 2ⁿ-dimensional complex vector (for n qubits)
        For n=6: |ψ⟩ has 64 complex components!
""")

print("\n" + "─"*80)
print(" STEP 2a: Q-SVM QUANTUM KERNEL")
print("─"*80)

print("""
Q-SVM uses the quantum kernel to measure similarity:

    K(x, y) = |⟨ψ(x)|ψ(y)⟩|²

Breaking it down:

    1. Get quantum states: |ψ(x)⟩ and |ψ(y)⟩

    2. Compute inner product: ⟨ψ(x)|ψ(y)⟩ = Σᵢ ψ*(x)ᵢ · ψ(y)ᵢ
       This is a complex number measuring overlap

    3. Take squared magnitude: K(x,y) = |⟨ψ(x)|ψ(y)⟩|²
       Now it's a real number in [0, 1]

Interpretation:
    • K(x,y) = 1: Perfect quantum similarity (states are identical)
    • K(x,y) = 0: Orthogonal states (maximally different)
    • K(x,y) ≈ 1: Similar network traffic patterns
    • K(x,y) ≈ 0: Different network traffic patterns

This kernel is then used in classical SVM:

    svm = SVC(kernel='precomputed')
    svm.fit(K_train, y_train)  # K_train is the quantum kernel matrix

The quantum circuit implicitly creates a feature space that may be
impossible to compute classically!
""")

print("\n" + "─"*80)
print(" STEP 2b: Q-KNN QUANTUM DISTANCE")
print("─"*80)

print("""
Q-KNN converts the quantum kernel into a distance metric:

    d(x, y) = √(1 - K(x,y)) = √(1 - |⟨ψ(x)|ψ(y)⟩|²)

Why this formula?

    • If K(x,y) = 1 (identical): d(x,y) = √(1-1) = 0 ✓
    • If K(x,y) = 0 (orthogonal): d(x,y) = √(1-0) = 1 ✓

This is called the "fidelity-based distance" and is a proper metric!

The K-NN algorithm then uses this distance:

    for each test point x_test:
        1. Compute d(x_test, x_train) for all training points
        2. Find k=5 nearest neighbors
        3. Take majority vote of their labels

Key insight: Q-KNN uses the SAME quantum circuit as Q-SVM,
             just transforms the kernel differently!
""")

print("\n" + "─"*80)
print(" WHY QUANTUM KERNELS CAN BE POWERFUL")
print("─"*80)

print("""
Classical RBF kernel:  K(x,y) = exp(-γ‖x-y‖²)
    • Maps to infinite-dimensional space (implicitly)
    • But limited to Gaussian-like similarity

Quantum kernel:  K(x,y) = |⟨ψ(x)|ψ(y)⟩|²
    • Maps to 2ⁿ-dimensional Hilbert space (explicitly)
    • Can capture complex, non-linear feature interactions
    • Entanglement creates correlations impossible classically
    • Layer 3 (feature interactions) enables quantum advantage

Potential advantages:
    1. Exponential feature space (2ⁿ dimensions)
    2. Feature interactions through entanglement
    3. Can potentially learn patterns inaccessible to classical kernels

Reality check (for CICIDS2017):
    • Classical K-NN: 98.12% F1 ✓
    • Quantum Q-KNN: 97.38% F1 (slightly lower)

Why? Network intrusion data may not have the structure that
benefits from quantum feature maps. Quantum advantage is
problem-dependent!
""")

print("\n" + "─"*80)
print(" COMPUTATIONAL COST")
print("─"*80)

print("""
Classical K-NN distance: O(n) per pair
Quantum K-NN distance: O(2ⁿ) per pair (statevector simulation)

This is why quantum models are MUCH slower:
    • Classical K-NN: 0.09s for 8,444 samples
    • Quantum Q-KNN: 34.55s for same dataset (380× slower!)

On real quantum hardware:
    • Would be faster than simulation
    • But current NISQ devices are noisy
    • Need error correction for reliable results
""")

print("\n" + "="*80)
print(" SUMMARY")
print("="*80)

print("""
Q-SVM:  Uses quantum kernel K(x,y) = |⟨ψ(x)|ψ(y)⟩|² in SVM algorithm
Q-KNN:  Uses quantum distance d(x,y) = √(1-K(x,y)) in K-NN algorithm

Both:   Share the SAME quantum circuit for feature mapping
        Differ only in post-processing of quantum inner products

The 6-layer quantum circuit creates rich, entangled feature
representations that can potentially outperform classical methods
for certain problems.
""")

print("\n" + "="*80)

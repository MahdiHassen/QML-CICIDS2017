"""
Detailed Explanation of the 6-Layer Quantum Circuit
How the Q-SVM/Q-KNN Feature Map Works
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch, FancyArrowPatch
import matplotlib.patches as mpatches

print("="*80)
print(" HOW THE 6-LAYER QUANTUM CIRCUIT WORKS")
print("="*80)

# Create detailed layer-by-layer explanation
fig = plt.figure(figsize=(20, 14))

# Define example data point
x_example = [0.8, 0.3, 0.6, 0.9, 0.2, 0.7]
n_qubits = 6

print(f"\nExample data point: x = {x_example}")
print(f"Number of qubits: {n_qubits}")
print(f"Quantum state dimensionality: 2^{n_qubits} = {2**n_qubits} complex numbers\n")

# ============================================================================
# LAYER-BY-LAYER BREAKDOWN
# ============================================================================

layer_descriptions = []

# Layer 1
layer_descriptions.append({
    'name': 'Layer 1: RY Encoding',
    'code': 'for i in range(n):\n    qc.ry(π * x[i], i)',
    'gates': 'RY gates',
    'what': 'Amplitude Encoding',
    'why': 'Encodes classical feature values as quantum rotation angles',
    'details': [
        'RY gate rotates qubit around Y-axis',
        'Angle = π × x[i] (feature value scaled by π)',
        'Maps x ∈ [0,1] to rotation ∈ [0,π]',
        'Creates superposition: |0⟩ → cos(θ/2)|0⟩ + sin(θ/2)|1⟩',
        'Example: x[0]=0.8 → RY(0.8π) on q₀'
    ],
    'math': 'RY(θ) = [[cos(θ/2), -sin(θ/2)], [sin(θ/2), cos(θ/2)]]',
    'color': '#c2185b'
})

# Layer 2
layer_descriptions.append({
    'name': 'Layer 2: Linear Chain Entanglement',
    'code': 'for i in range(n-1):\n    qc.cz(i, i+1)',
    'gates': 'CZ gates',
    'what': 'Nearest-Neighbor Entanglement',
    'why': 'Creates correlations between adjacent qubits',
    'details': [
        'CZ = Controlled-Z gate',
        'Entangles qubit i with qubit i+1',
        'Creates connections: q₀↔q₁, q₁↔q₂, ..., q₄↔q₅',
        'Linear topology (chain)',
        'Enables information flow between features'
    ],
    'math': 'CZ = diag(1, 1, 1, -1) — flips phase if both qubits are |1⟩',
    'color': '#1976d2'
})

# Layer 3
layer_descriptions.append({
    'name': 'Layer 3: Feature Interactions ⭐ KEY INNOVATION!',
    'code': 'for i in range(n):\n    qc.rz(π * x[i] * x[(i+1)%n], i)',
    'gates': 'RZ gates',
    'what': 'Pairwise Feature Interactions',
    'why': 'Encodes products of features (non-linear interactions)',
    'details': [
        'RZ rotates around Z-axis',
        'Angle = π × x[i] × x[i+1] (product of neighboring features!)',
        'Circular: x[5] interacts with x[0]',
        'This creates QUADRATIC feature interactions',
        'Example: x[0]=0.8, x[1]=0.3 → RZ(0.24π) on q₀',
        'This is what gives quantum potential advantage!'
    ],
    'math': 'RZ(θ) = [[e^(-iθ/2), 0], [0, e^(iθ/2)]]',
    'color': '#00acc1'
})

# Layer 4
layer_descriptions.append({
    'name': 'Layer 4: Star Entanglement',
    'code': 'for i in range(1, n):\n    qc.cz(0, i)',
    'gates': 'CZ gates',
    'what': 'Hub-and-Spoke Entanglement',
    'why': 'Connects first qubit to all others (global correlation)',
    'details': [
        'Qubit 0 becomes the "hub"',
        'All other qubits connect to q₀',
        'Creates star topology: q₀ ↔ {q₁, q₂, q₃, q₄, q₅}',
        'Enables long-range feature correlations',
        'Complements the linear chain from Layer 2'
    ],
    'math': 'Same CZ gate as Layer 2, different connectivity pattern',
    'color': '#1976d2'
})

# Layer 5
layer_descriptions.append({
    'name': 'Layer 5: Re-encoding',
    'code': 'for i in range(n):\n    qc.ry(π/2 * x[i], i)',
    'gates': 'RY gates',
    'what': 'Second Encoding Layer',
    'why': 'Adds additional expressivity with different rotation angles',
    'details': [
        'Same RY gates as Layer 1',
        'But with HALF the rotation angle (π/2 instead of π)',
        'Example: x[0]=0.8 → RY(0.4π) on q₀',
        'Allows circuit to "fine-tune" the encoding',
        'Increases expressivity of the feature map'
    ],
    'math': 'RY(θ) with θ = π/2 × x[i] (half of Layer 1)',
    'color': '#c2185b'
})

# Layer 6
layer_descriptions.append({
    'name': 'Layer 6: Ring Closure',
    'code': 'if n > 2:\n    qc.cz(n-1, 0)',
    'gates': 'CZ gate',
    'what': 'Periodic Boundary Condition',
    'why': 'Connects last qubit back to first (completes the ring)',
    'details': [
        'Single CZ gate: q₅ ↔ q₀',
        'Creates periodic boundary',
        'Combined with Layer 2, forms a RING topology',
        'All qubits now in a circular arrangement',
        'Enables rotation-invariant feature representation'
    ],
    'math': 'CZ(5, 0) — closes the chain into a ring',
    'color': '#1976d2'
})

# ============================================================================
# Create Visual Explanation
# ============================================================================

n_layers = len(layer_descriptions)
for i, layer in enumerate(layer_descriptions):
    ax = plt.subplot(n_layers, 1, i+1)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 2)
    ax.axis('off')

    # Title
    ax.text(5, 1.8, layer['name'], ha='center', fontsize=14, fontweight='bold')

    # Left box: Code
    code_box = FancyBboxPatch((0.2, 0.3), 2.5, 1.3, boxstyle="round,pad=0.05",
                              edgecolor=layer['color'], facecolor='#f5f5f5', linewidth=2)
    ax.add_patch(code_box)
    ax.text(1.45, 1.45, 'Code:', ha='center', fontsize=10, fontweight='bold')
    ax.text(1.45, 0.9, layer['code'], ha='center', va='center', fontsize=8,
            family='monospace', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Middle box: What & Why
    purpose_box = FancyBboxPatch((3, 0.3), 3.5, 1.3, boxstyle="round,pad=0.05",
                                edgecolor=layer['color'], facecolor='lightyellow', linewidth=2)
    ax.add_patch(purpose_box)
    ax.text(4.75, 1.45, f"What: {layer['what']}", ha='center', fontsize=9, fontweight='bold')
    ax.text(4.75, 1.15, f"Why: {layer['why']}", ha='center', fontsize=8, style='italic')
    ax.text(4.75, 0.7, f"Gates: {layer['gates']}", ha='center', fontsize=8,
            bbox=dict(boxstyle='round', facecolor=layer['color'], alpha=0.3))

    # Right box: Math
    math_box = FancyBboxPatch((6.7, 0.3), 3, 1.3, boxstyle="round,pad=0.05",
                             edgecolor=layer['color'], facecolor='lightblue', linewidth=2)
    ax.add_patch(math_box)
    ax.text(8.2, 1.45, 'Mathematics:', ha='center', fontsize=10, fontweight='bold')
    ax.text(8.2, 0.9, layer['math'], ha='center', va='center', fontsize=7,
            family='monospace', wrap=True)

plt.tight_layout()
plt.savefig('circuit_layers_explained.png', dpi=300, bbox_inches='tight')
print("✓ Saved: circuit_layers_explained.png")

# ============================================================================
# Print Detailed Explanation
# ============================================================================

print("\n" + "="*80)
print(" DETAILED LAYER-BY-LAYER EXPLANATION")
print("="*80)

for i, layer in enumerate(layer_descriptions, 1):
    print(f"\n{'='*80}")
    print(f" {layer['name']}")
    print(f"{'='*80}")

    print(f"\nCode:")
    print(f"  {layer['code']}")

    print(f"\nPurpose: {layer['what']}")
    print(f"Why: {layer['why']}")

    print(f"\nDetails:")
    for detail in layer['details']:
        print(f"  • {detail}")

    print(f"\nMathematics:")
    print(f"  {layer['math']}")

# ============================================================================
# Overall Circuit Properties
# ============================================================================

print("\n" + "="*80)
print(" OVERALL CIRCUIT PROPERTIES")
print("="*80)

print(f"""
Number of Qubits: {n_qubits}
Number of Layers: 6
Total Quantum State Dimension: 2^{n_qubits} = {2**n_qubits}

Gate Count for {n_qubits} qubits:
  • RY gates (Layer 1): {n_qubits}
  • CZ gates (Layer 2): {n_qubits - 1}
  • RZ gates (Layer 3): {n_qubits}
  • CZ gates (Layer 4): {n_qubits - 1}
  • RY gates (Layer 5): {n_qubits}
  • CZ gates (Layer 6): 1
  ────────────────────────────
  Total gates: {n_qubits + (n_qubits-1) + n_qubits + (n_qubits-1) + n_qubits + 1}

Circuit Depth: ~23 (depends on qubit connectivity)

Entanglement Structure:
  • Linear chain (Layer 2): Creates nearest-neighbor correlations
  • Star pattern (Layer 4): Creates hub-based global correlations
  • Ring closure (Layer 6): Creates periodic boundaries
  • Combined: Rich, multi-scale entanglement pattern
""")

# ============================================================================
# Example Computation
# ============================================================================

print("\n" + "="*80)
print(" EXAMPLE: APPLYING CIRCUIT TO DATA")
print("="*80)

print(f"\nInput data: x = {x_example}")
print(f"\nInitial state: |000000⟩ (all qubits in |0⟩)")

print("\n" + "─"*80)
print("After Layer 1 (RY Encoding):")
print("─"*80)
for i, xi in enumerate(x_example):
    angle = np.pi * xi
    print(f"  q{i}: RY({angle:.4f}) = RY({xi}π)")
print("  → Each qubit now in superposition based on feature value")

print("\n" + "─"*80)
print("After Layer 2 (Chain Entanglement):")
print("─"*80)
print("  CZ gates: q0↔q1, q1↔q2, q2↔q3, q3↔q4, q4↔q5")
print("  → Qubits are now ENTANGLED (can't describe them independently!)")

print("\n" + "─"*80)
print("After Layer 3 (Feature Interactions):")
print("─"*80)
for i in range(n_qubits):
    j = (i + 1) % n_qubits
    product = x_example[i] * x_example[j]
    angle = np.pi * product
    print(f"  q{i}: RZ({angle:.4f}) = RZ({x_example[i]:.1f} × {x_example[j]:.1f} × π)")
print("  → Encoded pairwise feature products (QUADRATIC interactions)")

print("\n" + "─"*80)
print("After Layer 4 (Star Entanglement):")
print("─"*80)
print("  CZ gates: q0↔q1, q0↔q2, q0↔q3, q0↔q4, q0↔q5")
print("  → All qubits now connected to q0 (global structure)")

print("\n" + "─"*80)
print("After Layer 5 (Re-encoding):")
print("─"*80)
for i, xi in enumerate(x_example):
    angle = np.pi * xi / 2
    print(f"  q{i}: RY({angle:.4f}) = RY({xi}π/2)")
print("  → Additional fine-tuning of amplitudes")

print("\n" + "─"*80)
print("After Layer 6 (Ring Closure):")
print("─"*80)
print("  CZ gate: q5↔q0")
print("  → Completes the ring topology")

print("\n" + "─"*80)
print("Final Quantum State:")
print("─"*80)
print(f"  |ψ(x)⟩ = a₀|000000⟩ + a₁|000001⟩ + ... + a₆₃|111111⟩")
print(f"  where aᵢ are complex numbers with |a₀|² + |a₁|² + ... + |a₆₃|² = 1")
print(f"\n  This state lives in a {2**n_qubits}-dimensional complex Hilbert space!")
print(f"  It encodes the original {n_qubits} classical features in a highly")
print(f"  non-linear, entangled quantum representation.")

# ============================================================================
# Why This Design?
# ============================================================================

print("\n" + "="*80)
print(" WHY THIS SPECIFIC 6-LAYER DESIGN?")
print("="*80)

print("""
1. Layer 1 (RY Encoding):
   → Standard amplitude encoding technique
   → Maps classical data to quantum amplitudes

2. Layer 2 (Linear Entanglement):
   → Creates local correlations between features
   → Simple, physically realizable on most quantum hardware

3. Layer 3 (Feature Interactions) - THE KEY!:
   → Encodes x[i] × x[i+1] products
   → This is what classical kernels struggle with
   → Creates QUADRATIC feature space
   → Potential source of quantum advantage

4. Layer 4 (Star Entanglement):
   → Adds global, long-range correlations
   → Complements local correlations from Layer 2
   → Creates richer entanglement structure

5. Layer 5 (Re-encoding):
   → Adds expressivity without adding entanglement
   → Different rotation angles from Layer 1
   → Allows more flexible feature representation

6. Layer 6 (Ring Closure):
   → Creates translational symmetry
   → All features treated "equally" (no first/last bias)
   → Enables rotation-invariant representations

Result: A highly expressive, well-connected quantum feature map
        that can potentially capture patterns classical methods miss!
""")

# ============================================================================
# Comparison
# ============================================================================

print("\n" + "="*80)
print(" CLASSICAL vs QUANTUM FEATURE SPACE")
print("="*80)

print("""
Classical Data (6 features):
  x = [x₁, x₂, x₃, x₄, x₅, x₆] ∈ ℝ⁶

Classical RBF Kernel:
  Maps to infinite-dimensional space (implicitly)
  But features are K(x,y) = exp(-γ||x-y||²)
  → Gaussian similarity only

Quantum Feature Map:
  Maps to 2⁶ = 64-dimensional complex Hilbert space
  Features include:
    • Individual features: x₁, x₂, ..., x₆ (Layer 1)
    • Pairwise products: x₁×x₂, x₂×x₃, ..., x₆×x₁ (Layer 3)
    • Complex phases and entanglement (Layers 2,4,6)
    • Additional rotations (Layer 5)
  → Rich, non-linear feature interactions!

This is why quantum kernels can be powerful:
  They efficiently encode complex feature interactions
  that would require many polynomial features classically.
""")

print("\n" + "="*80)
print(" END OF EXPLANATION")
print("="*80)

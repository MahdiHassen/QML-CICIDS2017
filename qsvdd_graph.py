import numpy as np
from qiskit import QuantumCircuit
import matplotlib.pyplot as plt

def build_exact_target_circuit():
    n_qubits = 4
    qc = QuantumCircuit(n_qubits)
    
    def add_entanglement(circuit):
        # Connect every qubit to every other qubit
        for i in range(n_qubits):
            for j in range(i + 1, n_qubits):
                circuit.cz(i, j)
        circuit.barrier() # Visual separator

    # --- Layer 1: Ry(π/2) ---
    for i in range(n_qubits):
        qc.ry(np.pi / 2, i)
    
    # --- Layer 2: Entanglement ---
    add_entanglement(qc)

    # --- Layer 3: Rz(π/4) ---
    for i in range(n_qubits):
        qc.rz(np.pi / 4, i)

    # --- Layer 4: Entanglement ---
    add_entanglement(qc)

    # --- Layer 5: Ry(π/4) ---
    for i in range(n_qubits):
        qc.ry(np.pi / 4, i)

    # --- Layer 6: Entanglement ---
    add_entanglement(qc)

    # --- Layer 7: Ry(π/4) ---
    for i in range(n_qubits):
        qc.ry(np.pi / 4, i)

    return qc

# Generate and draw the circuit
qc = build_exact_target_circuit()
qc.draw('mpl', style='iqp') 
plt.title("Q-SVDD Feature Map", fontsize=14, fontweight='bold', pad=20)
plt.show()
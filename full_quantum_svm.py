"""
Full Quantum SVM: Variational Quantum Classifier (VQC)
Network Intrusion Detection on CICIDS2017

This implements a FULLY quantum classifier where:
1. Quantum feature map encodes data
2. Quantum variational circuit performs classification
3. Training optimizes quantum circuit parameters
4. Prediction uses quantum measurements

Compare this to hybrid approach where only kernel is quantum.
"""

import pandas as pd
import numpy as np
import time
import os
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.metrics import accuracy_score, f1_score
from imblearn.under_sampling import RandomUnderSampler
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter, ParameterVector
from qiskit_aer import AerSimulator
from qiskit.primitives import Sampler
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Setup paths
RESULTS_DIR = 'quantum_models/results'
VIZ_DIR = f'{RESULTS_DIR}/visualizations'
os.makedirs(VIZ_DIR, exist_ok=True)

print("="*80)
print("FULL QUANTUM SVM: Variational Quantum Classifier")
print("="*80)
print("\nThis is a FULLY quantum approach where both feature encoding")
print("AND classification happen in quantum circuits!\n")

# ============================================================================
# Quantum Circuits
# ============================================================================
sim = AerSimulator()

def quantum_feature_map(x, n_qubits):
    """Encode classical data into quantum state (same as before)"""
    qc = QuantumCircuit(n_qubits)

    # Layer 1: RY encoding
    for i in range(n_qubits):
        qc.ry(np.pi * x[i], i)

    # Layer 2: Entanglement
    for i in range(n_qubits - 1):
        qc.cz(i, i + 1)

    # Layer 3: Feature interactions
    for i in range(n_qubits):
        qc.rz(np.pi * x[i] * x[(i+1) % n_qubits], i)

    return qc

def variational_circuit(n_qubits, n_layers=2):
    """
    Trainable quantum circuit for classification.
    This is the KEY difference - these parameters are LEARNED!
    """
    qc = QuantumCircuit(n_qubits)

    # Create trainable parameters
    num_params = n_qubits * n_layers * 2  # 2 rotations per qubit per layer
    params = ParameterVector('θ', num_params)

    param_idx = 0
    for layer in range(n_layers):
        # Rotation layer (trainable)
        for i in range(n_qubits):
            qc.ry(params[param_idx], i)
            param_idx += 1
            qc.rz(params[param_idx], i)
            param_idx += 1

        # Entanglement layer
        for i in range(n_qubits - 1):
            qc.cx(i, i + 1)
        if n_qubits > 2:
            qc.cx(n_qubits - 1, 0)  # Ring

    return qc, params

def create_full_quantum_circuit(x, n_qubits, n_layers=2):
    """
    Complete quantum circuit: Feature Map + Variational Classifier
    This is the FULL quantum model!
    """
    # Part 1: Feature encoding (fixed)
    qc = quantum_feature_map(x, n_qubits)

    # Part 2: Variational classifier (trainable)
    var_qc, params = variational_circuit(n_qubits, n_layers)
    qc.compose(var_qc, inplace=True)

    # Measurement on first qubit (determines class)
    qc.measure_all()

    return qc, params

class VariationalQuantumClassifier:
    """
    Full Quantum SVM using Variational Quantum Circuits

    Unlike hybrid Q-SVM:
    - No classical SVM training
    - No precomputed kernels
    - Direct quantum circuit optimization
    """

    def __init__(self, n_qubits, n_layers=2, learning_rate=0.1, n_epochs=50):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.num_params = n_qubits * n_layers * 2

        # Initialize random parameters
        self.params = np.random.uniform(-np.pi, np.pi, self.num_params)

        self.training_history = []

    def quantum_predict_proba(self, x, params):
        """
        Run quantum circuit and measure probability of class 1
        This is pure quantum computation!
        """
        # Create parameterized circuit
        qc, param_vec = create_full_quantum_circuit(x, self.n_qubits, self.n_layers)

        # Bind parameters (use assign_parameters in Qiskit)
        param_dict = {param_vec[i]: params[i] for i in range(len(params))}
        qc_bound = qc.assign_parameters(param_dict)

        # Run circuit
        result = sim.run(qc_bound, shots=1000).result()
        counts = result.get_counts()

        # Measure first qubit: '1' at end = class 1, '0' = class 0
        prob_1 = 0
        total_shots = sum(counts.values())
        for bitstring, count in counts.items():
            if bitstring[-1] == '1':  # First qubit (rightmost in Qiskit convention)
                prob_1 += count

        return prob_1 / total_shots

    def compute_loss(self, X, y, params):
        """
        Loss function: Mean squared error between prediction and true label
        """
        loss = 0
        for xi, yi in zip(X, y):
            pred_prob = self.quantum_predict_proba(xi, params)
            loss += (pred_prob - yi)**2
        return loss / len(X)

    def compute_gradient(self, X, y, params, epsilon=0.01):
        """
        Numerical gradient estimation (parameter shift rule would be better)
        """
        grad = np.zeros_like(params)

        for i in range(len(params)):
            # Forward difference
            params_plus = params.copy()
            params_plus[i] += epsilon
            loss_plus = self.compute_loss(X, y, params_plus)

            params_minus = params.copy()
            params_minus[i] -= epsilon
            loss_minus = self.compute_loss(X, y, params_minus)

            grad[i] = (loss_plus - loss_minus) / (2 * epsilon)

        return grad

    def fit(self, X_train, y_train, X_val=None, y_val=None, batch_size=50):
        """
        Train the quantum circuit using gradient descent
        This is QUANTUM TRAINING!
        """
        # Convert to numpy arrays to avoid pandas indexing issues
        y_train = np.array(y_train)
        if y_val is not None:
            y_val = np.array(y_val)

        print(f"\n[Training Full Quantum Classifier]")
        print(f"  Qubits: {self.n_qubits}")
        print(f"  Layers: {self.n_layers}")
        print(f"  Parameters: {self.num_params}")
        print(f"  Training samples: {len(X_train)}")

        for epoch in range(self.n_epochs):
            epoch_start = time.time()

            # Use a batch for faster training
            batch_indices = np.random.choice(len(X_train),
                                           min(batch_size, len(X_train)),
                                           replace=False)
            X_batch = X_train[batch_indices]
            y_batch = y_train[batch_indices]

            # Compute loss and gradient
            loss = self.compute_loss(X_batch, y_batch, self.params)
            grad = self.compute_gradient(X_batch, y_batch, self.params)

            # Gradient descent update
            self.params -= self.learning_rate * grad

            # Evaluate on validation set
            if X_val is not None and epoch % 5 == 0:
                val_preds = self.predict(X_val[:50])  # Small subset for speed
                val_acc = accuracy_score(y_val[:50], val_preds)
                val_f1 = f1_score(y_val[:50], val_preds)

                epoch_time = time.time() - epoch_start
                print(f"  Epoch {epoch:2d}/{self.n_epochs}: "
                      f"Loss={loss:.4f}, Val_Acc={val_acc:.4f}, "
                      f"Val_F1={val_f1:.4f}, Time={epoch_time:.1f}s")

                self.training_history.append({
                    'epoch': epoch,
                    'loss': loss,
                    'val_acc': val_acc,
                    'val_f1': val_f1
                })

        return self

    def predict_proba(self, X):
        """Predict probabilities for all samples"""
        probs = []
        for x in X:
            prob = self.quantum_predict_proba(x, self.params)
            probs.append(prob)
        return np.array(probs)

    def predict(self, X):
        """Predict class labels"""
        probs = self.predict_proba(X)
        return (probs > 0.5).astype(int)

# ============================================================================
# Load and Prepare Dataset
# ============================================================================
print("\n[1] Loading dataset...")
df = pd.read_csv('cicids2017_cleaned.csv')

X = df.drop('Attack Type', axis=1)
y_binary = (df['Attack Type'] != 'Normal Traffic').astype(int)

print(f"Dataset: {len(X):,} samples")

# Use SMALLER sample for VQC (it's slow to train)
print("\n[2] Balancing dataset (SMALLER for VQC)...")
initial_sample = 5000  # Much smaller than hybrid approach
sss = StratifiedShuffleSplit(n_splits=1, train_size=initial_sample, random_state=42)
for sample_idx, _ in sss.split(X, y_binary):
    X_initial = X.iloc[sample_idx]
    y_initial = y_binary.iloc[sample_idx]

rus = RandomUnderSampler(sampling_strategy=1.0, random_state=42)
X_balanced, y_balanced = rus.fit_resample(X_initial, y_initial)

print(f"Balanced: {len(X_balanced):,} samples")
print("(Note: Smaller dataset due to VQC training cost)")

# ============================================================================
# Train Full Quantum Classifier
# ============================================================================
n_qubits = 4  # Start with 4 qubits
results = []

print(f"\n{'='*80}")
print(f"Training Full Quantum SVM with {n_qubits} qubits")
print(f"{'='*80}")

# Feature selection
selector = SelectKBest(score_func=mutual_info_classif, k=n_qubits)
X_selected = selector.fit_transform(X_balanced, y_balanced)

selected_features = X.columns.values[selector.get_support(indices=True)]
print(f"Features: {', '.join(selected_features)}")

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X_selected, y_balanced, test_size=0.25, random_state=42, stratify=y_balanced
)

# Scale to [0,1]
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train VQC
print("\n[Training Variational Quantum Classifier]")
t0 = time.time()

vqc = VariationalQuantumClassifier(
    n_qubits=n_qubits,
    n_layers=1,  # Reduced to 1 layer for speed
    learning_rate=0.15,
    n_epochs=15  # Reduced for speed
)

vqc.fit(X_train_scaled, y_train, X_val=X_test_scaled, y_val=y_test, batch_size=20)

train_time = time.time() - t0

# Evaluate
print("\n[Evaluating on test set...]")
t0 = time.time()
y_pred = vqc.predict(X_test_scaled)
pred_time = time.time() - t0

vqc_acc = accuracy_score(y_test, y_pred)
vqc_f1 = f1_score(y_test, y_pred)

print(f"\n{'='*80}")
print(f"FULL QUANTUM SVM RESULTS")
print(f"{'='*80}")
print(f"Accuracy: {vqc_acc:.4f}")
print(f"F1-Score: {vqc_f1:.4f}")
print(f"Training Time: {train_time:.2f}s")
print(f"Prediction Time: {pred_time:.2f}s")

# ============================================================================
# Compare with Hybrid Approach
# ============================================================================
print(f"\n{'='*80}")
print("COMPARISON: Full Quantum vs Hybrid Approach")
print(f"{'='*80}")

# Train hybrid Q-SVM for comparison
print("\n[Training Hybrid Q-SVM for comparison...]")

from sklearn.svm import SVC

def get_statevector(x, n_qubits):
    qc = quantum_feature_map(x, n_qubits)
    qc.save_statevector()
    result = sim.run(qc).result()
    return result.get_statevector(qc)

def quantum_kernel_matrix(XA, XB, n_qubits):
    sv_A = np.array([get_statevector(xa, n_qubits) for xa in XA])
    sv_B = np.array([get_statevector(xb, n_qubits) for xb in XB])
    inner = sv_A.conj() @ sv_B.T
    return np.abs(inner)**2

t0 = time.time()
K_train = quantum_kernel_matrix(X_train_scaled, X_train_scaled, n_qubits)
K_test = quantum_kernel_matrix(X_test_scaled, X_train_scaled, n_qubits)

hybrid_qsvm = SVC(kernel='precomputed', C=10.0)
hybrid_qsvm.fit(K_train, y_train)
y_pred_hybrid = hybrid_qsvm.predict(K_test)
hybrid_time = time.time() - t0

hybrid_acc = accuracy_score(y_test, y_pred_hybrid)
hybrid_f1 = f1_score(y_test, y_pred_hybrid)

print(f"\nHybrid Q-SVM: Acc={hybrid_acc:.4f}, F1={hybrid_f1:.4f}, Time={hybrid_time:.2f}s")
print(f"Full Quantum: Acc={vqc_acc:.4f}, F1={vqc_f1:.4f}, Time={train_time:.2f}s")

# ============================================================================
# Visualizations
# ============================================================================
print(f"\n[Creating visualizations...]")

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Full Quantum SVM Analysis', fontsize=16, fontweight='bold')

# Training history
if vqc.training_history:
    history = vqc.training_history
    epochs = [h['epoch'] for h in history]
    losses = [h['loss'] for h in history]
    val_f1s = [h['val_f1'] for h in history]

    ax1.plot(epochs, losses, 'o-', color='#2E7D32', linewidth=2, markersize=6)
    ax1.set_xlabel('Epoch', fontweight='bold')
    ax1.set_ylabel('Loss', fontweight='bold')
    ax1.set_title('Training Loss (Quantum Circuit Optimization)')
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, val_f1s, 's-', color='#1B5E20', linewidth=2, markersize=6)
    ax2.set_xlabel('Epoch', fontweight='bold')
    ax2.set_ylabel('Validation F1-Score', fontweight='bold')
    ax2.set_title('Validation Performance')
    ax2.grid(True, alpha=0.3)

# Comparison
models = ['Full Quantum\n(VQC)', 'Hybrid\n(Q-Kernel + Classical SVM)']
f1_scores = [vqc_f1, hybrid_f1]
times = [train_time, hybrid_time]

x_pos = np.arange(len(models))
bars = ax3.bar(x_pos, f1_scores, color=['#2E7D32', '#1976D2'], alpha=0.7, edgecolor='black', linewidth=2)
ax3.set_xticks(x_pos)
ax3.set_xticklabels(models)
ax3.set_ylabel('F1-Score', fontweight='bold')
ax3.set_title('Performance Comparison')
ax3.grid(True, alpha=0.3, axis='y')

# Add value labels
for bar in bars:
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.4f}', ha='center', va='bottom', fontweight='bold')

# Time comparison
bars2 = ax4.bar(x_pos, times, color=['#2E7D32', '#1976D2'], alpha=0.7, edgecolor='black', linewidth=2)
ax4.set_xticks(x_pos)
ax4.set_xticklabels(models)
ax4.set_ylabel('Training Time (seconds)', fontweight='bold')
ax4.set_title('Computational Cost')
ax4.set_yscale('log')
ax4.grid(True, alpha=0.3, which='both')

# Add value labels
for bar in bars2:
    height = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.1f}s', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig(f'{VIZ_DIR}/full_quantum_svm_analysis.png', dpi=300, bbox_inches='tight')
print(f"Saved: {VIZ_DIR}/full_quantum_svm_analysis.png")

# ============================================================================
# Save Results
# ============================================================================
print(f"\n[Saving results...]")

with open(f'{RESULTS_DIR}/full_quantum_svm_results.txt', 'w') as f:
    f.write("Full Quantum SVM Results - Variational Quantum Classifier\n")
    f.write("="*80 + "\n\n")

    f.write("Architecture:\n")
    f.write(f"  Qubits: {n_qubits}\n")
    f.write(f"  Variational Layers: {vqc.n_layers}\n")
    f.write(f"  Trainable Parameters: {vqc.num_params}\n")
    f.write(f"  Training Epochs: {vqc.n_epochs}\n")
    f.write(f"  Learning Rate: {vqc.learning_rate}\n\n")

    f.write("Dataset:\n")
    f.write(f"  Total Samples: {len(X_balanced):,}\n")
    f.write(f"  Training Samples: {len(X_train):,}\n")
    f.write(f"  Test Samples: {len(X_test):,}\n")
    f.write(f"  Features: {', '.join(selected_features)}\n\n")

    f.write("Results:\n")
    f.write(f"  Accuracy: {vqc_acc:.4f}\n")
    f.write(f"  F1-Score: {vqc_f1:.4f}\n")
    f.write(f"  Training Time: {train_time:.2f}s\n")
    f.write(f"  Prediction Time: {pred_time:.2f}s\n\n")

    f.write("Comparison with Hybrid Approach:\n")
    f.write(f"  Hybrid Q-SVM F1: {hybrid_f1:.4f}\n")
    f.write(f"  Full Quantum F1: {vqc_f1:.4f}\n")
    f.write(f"  Difference: {(vqc_f1 - hybrid_f1)*100:+.2f}%\n\n")

    f.write("Key Differences:\n")
    f.write("  Full Quantum VQC:\n")
    f.write("    ✓ Feature encoding in quantum circuit\n")
    f.write("    ✓ Classification in quantum circuit\n")
    f.write("    ✓ Parameters trained via gradient descent\n")
    f.write("    ✓ Fully quantum except optimization loop\n\n")

    f.write("  Hybrid Q-SVM:\n")
    f.write("    ✓ Feature encoding in quantum circuit\n")
    f.write("    ✗ Classification uses classical SVM\n")
    f.write("    ✗ No trainable quantum parameters\n")
    f.write("    ✗ Only kernel is quantum\n")

print(f"Saved: {RESULTS_DIR}/full_quantum_svm_results.txt")

print("\n" + "="*80)
print("FULL QUANTUM SVM TRAINING COMPLETE!")
print("="*80)
print(f"\nThis implementation uses VARIATIONAL QUANTUM CIRCUITS")
print(f"where both encoding AND classification are quantum!")
print(f"\nResults saved in: {RESULTS_DIR}/")

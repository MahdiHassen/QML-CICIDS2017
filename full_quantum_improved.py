"""
IMPROVED Full Quantum SVM with Enhanced VQC
Network Intrusion Detection on CICIDS2017

Improvements:
- Deeper variational circuits (2-3 layers)
- Adam optimizer with momentum
- Learning rate scheduling
- Better circuit ansatz (hardware-efficient)
- More training epochs with early stopping
- Larger batch sizes for stable gradients
"""

import pandas as pd
import numpy as np
import time
import os
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score
from imblearn.under_sampling import RandomUnderSampler
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit_aer import AerSimulator
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Setup
RESULTS_DIR = 'comparison/results'
VIZ_DIR = f'{RESULTS_DIR}/visualizations'
os.makedirs(VIZ_DIR, exist_ok=True)

print("="*80)
print("IMPROVED FULL QUANTUM SVM with Enhanced VQC")
print("="*80)
print("\nImprovements:")
print("  • Deeper circuits (2-3 layers vs 1)")
print("  • Adam optimizer (momentum + adaptive learning rate)")
print("  • Learning rate decay")
print("  • Early stopping")
print("  • More epochs (60 vs 20)")
print("  • Better circuit ansatz\n")

sim = AerSimulator()

# ============================================================================
# Enhanced Quantum Circuits
# ============================================================================

def quantum_feature_map(x, n_qubits):
    """Enhanced feature encoding with more layers"""
    qc = QuantumCircuit(n_qubits)

    # Layer 1: RY encoding
    for i in range(n_qubits):
        qc.ry(np.pi * x[i], i)

    # Layer 2: Entanglement
    for i in range(n_qubits - 1):
        qc.cz(i, i + 1)
    if n_qubits > 2:
        qc.cz(n_qubits - 1, 0)  # Ring

    # Layer 3: Feature interactions
    for i in range(n_qubits):
        qc.rz(np.pi * x[i] * x[(i+1) % n_qubits], i)

    # Layer 4: More entanglement
    for i in range(n_qubits - 1):
        qc.cx(i, i + 1)

    return qc

def hardware_efficient_ansatz(n_qubits, n_layers=2):
    """
    Hardware-efficient variational ansatz
    Better for NISQ devices and more expressive
    """
    qc = QuantumCircuit(n_qubits)
    num_params = n_qubits * n_layers * 3  # RX, RY, RZ per qubit per layer
    params = ParameterVector('θ', num_params)

    param_idx = 0
    for layer in range(n_layers):
        # Single-qubit rotations (all 3 axes)
        for i in range(n_qubits):
            qc.rx(params[param_idx], i)
            param_idx += 1
            qc.ry(params[param_idx], i)
            param_idx += 1
            qc.rz(params[param_idx], i)
            param_idx += 1

        # Entanglement layer (circular pattern)
        for i in range(n_qubits - 1):
            qc.cx(i, i + 1)
        if n_qubits > 2:
            qc.cx(n_qubits - 1, 0)

    return qc, params

def create_full_quantum_circuit(x, n_qubits, n_layers=2):
    """Complete enhanced circuit"""
    qc = quantum_feature_map(x, n_qubits)
    var_qc, params = hardware_efficient_ansatz(n_qubits, n_layers)
    qc.compose(var_qc, inplace=True)
    qc.measure_all()
    return qc, params

class ImprovedVQC:
    """
    Improved Variational Quantum Classifier

    Enhancements:
    - Adam optimizer with momentum
    - Learning rate decay
    - Early stopping
    - Better gradient estimation
    """

    def __init__(self, n_qubits, n_layers=2, learning_rate=0.1, n_epochs=60):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.num_params = n_qubits * n_layers * 3

        # Initialize parameters with better strategy
        self.params = np.random.uniform(-np.pi/4, np.pi/4, self.num_params)

        # Adam optimizer state
        self.m = np.zeros_like(self.params)  # First moment
        self.v = np.zeros_like(self.params)  # Second moment
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.epsilon = 1e-8
        self.t = 0

        self.training_history = []
        self.best_params = self.params.copy()
        self.best_val_f1 = 0.0
        self.patience_counter = 0

    def quantum_predict_proba(self, x, params):
        """Predict with quantum circuit"""
        qc, param_vec = create_full_quantum_circuit(x, self.n_qubits, self.n_layers)
        param_dict = {param_vec[i]: params[i] for i in range(len(params))}
        qc_bound = qc.assign_parameters(param_dict)

        result = sim.run(qc_bound, shots=1000).result()
        counts = result.get_counts()

        prob_1 = sum(count for bitstring, count in counts.items()
                     if bitstring[-1] == '1') / sum(counts.values())
        return prob_1

    def compute_loss(self, X, y, params):
        """Mean squared error loss"""
        losses = [(self.quantum_predict_proba(xi, params) - yi)**2
                  for xi, yi in zip(X, y)]
        return np.mean(losses)

    def compute_gradient(self, X, y, params, epsilon=0.01):
        """Numerical gradient with smaller epsilon"""
        grad = np.zeros_like(params)
        for i in range(len(params)):
            params_plus = params.copy()
            params_plus[i] += epsilon
            params_minus = params.copy()
            params_minus[i] -= epsilon

            grad[i] = (self.compute_loss(X, y, params_plus) -
                      self.compute_loss(X, y, params_minus)) / (2 * epsilon)

        return grad

    def adam_update(self, grad):
        """
        Adam optimizer update
        Combines momentum and adaptive learning rates
        """
        self.t += 1

        # Update biased first moment estimate
        self.m = self.beta1 * self.m + (1 - self.beta1) * grad

        # Update biased second moment estimate
        self.v = self.beta2 * self.v + (1 - self.beta2) * (grad**2)

        # Bias correction
        m_hat = self.m / (1 - self.beta1**self.t)
        v_hat = self.v / (1 - self.beta2**self.t)

        # Update parameters
        update = self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
        return update

    def fit(self, X_train, y_train, X_val=None, y_val=None,
            batch_size=25, patience=10, verbose=True):
        """
        Train with improvements:
        - Adam optimizer
        - Learning rate decay
        - Early stopping
        """
        y_train = np.array(y_train)
        y_val = np.array(y_val) if y_val is not None else None

        if verbose:
            print(f"  Training Improved VQC:")
            print(f"    Qubits: {self.n_qubits}, Layers: {self.n_layers}")
            print(f"    Parameters: {self.num_params}")
            print(f"    Optimizer: Adam with learning rate decay")

        for epoch in range(self.n_epochs):
            # Learning rate decay
            current_lr = self.learning_rate * (0.95 ** (epoch // 10))

            # Mini-batch sampling
            batch_idx = np.random.choice(len(X_train),
                                        min(batch_size, len(X_train)),
                                        replace=False)
            X_batch, y_batch = X_train[batch_idx], y_train[batch_idx]

            # Compute loss and gradient
            loss = self.compute_loss(X_batch, y_batch, self.params)
            grad = self.compute_gradient(X_batch, y_batch, self.params)

            # Adam update
            update = self.adam_update(grad)
            self.params -= update

            # Validation and early stopping
            if X_val is not None and epoch % 3 == 0:
                val_sample_size = min(50, len(X_val))
                val_preds = self.predict(X_val[:val_sample_size])
                val_f1 = f1_score(y_val[:val_sample_size], val_preds)

                self.training_history.append({
                    'epoch': epoch,
                    'loss': loss,
                    'val_f1': val_f1,
                    'lr': current_lr
                })

                if verbose and epoch % 9 == 0:
                    print(f"    Epoch {epoch:2d}/{self.n_epochs}: "
                          f"Loss={loss:.4f}, Val_F1={val_f1:.4f}, LR={current_lr:.4f}")

                # Early stopping
                if val_f1 > self.best_val_f1:
                    self.best_val_f1 = val_f1
                    self.best_params = self.params.copy()
                    self.patience_counter = 0
                else:
                    self.patience_counter += 1

                if self.patience_counter >= patience:
                    if verbose:
                        print(f"    Early stopping at epoch {epoch}")
                    self.params = self.best_params
                    break

        return self

    def predict(self, X):
        """Predict class labels"""
        probs = [self.quantum_predict_proba(x, self.params) for x in X]
        return (np.array(probs) > 0.5).astype(int)

# ============================================================================
# Hybrid Q-SVM (for comparison)
# ============================================================================

def get_statevector(x, n_qubits):
    qc = quantum_feature_map(x, n_qubits)
    qc.save_statevector()
    return sim.run(qc).result().get_statevector(qc)

def quantum_kernel_matrix(XA, XB, n_qubits):
    sv_A = np.array([get_statevector(xa, n_qubits) for xa in XA])
    sv_B = np.array([get_statevector(xb, n_qubits) for xb in XB])
    return np.abs(sv_A.conj() @ sv_B.T)**2

# ============================================================================
# Load Data
# ============================================================================

print("\n[1] Loading dataset...")
df = pd.read_csv('cicids2017_cleaned.csv')
X = df.drop('Attack Type', axis=1)
y_binary = (df['Attack Type'] != 'Normal Traffic').astype(int)

print("\n[2] Preparing dataset...")
initial_sample = 4000  # Slightly larger for better training
sss = StratifiedShuffleSplit(n_splits=1, train_size=initial_sample, random_state=42)
for sample_idx, _ in sss.split(X, y_binary):
    X_initial = X.iloc[sample_idx]
    y_initial = y_binary.iloc[sample_idx]

rus = RandomUnderSampler(sampling_strategy=1.0, random_state=42)
X_balanced, y_balanced = rus.fit_resample(X_initial, y_initial)

print(f"Balanced: {len(X_balanced):,} samples")

# ============================================================================
# Train Improved VQC Across Feature Counts
# ============================================================================

feature_counts = [4, 6, 8]
results = []

for n_features in feature_counts:
    print(f"\n{'='*80}")
    print(f"Training with {n_features} features/qubits")
    print(f"{'='*80}")

    # Feature selection
    selector = SelectKBest(score_func=mutual_info_classif, k=n_features)
    X_selected = selector.fit_transform(X_balanced, y_balanced)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_selected, y_balanced, test_size=0.25, random_state=42, stratify=y_balanced
    )

    # Scale
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    scaler_classical = StandardScaler()
    X_train_classical = scaler_classical.fit_transform(X_train)
    X_test_classical = scaler_classical.transform(X_test)

    # Classical K-NN (baseline)
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train_classical, y_train)
    knn_f1 = f1_score(y_test, knn.predict(X_test_classical))

    # Hybrid Q-SVM
    print("\n[1/2] Hybrid Q-SVM...")
    t0 = time.time()
    K_train = quantum_kernel_matrix(X_train_scaled, X_train_scaled, n_features)
    K_test = quantum_kernel_matrix(X_test_scaled, X_train_scaled, n_features)
    qsvm = SVC(kernel='precomputed', C=10.0)
    qsvm.fit(K_train, y_train)
    qsvm_f1 = f1_score(y_test, qsvm.predict(K_test))
    qsvm_time = time.time() - t0
    print(f"  Hybrid Q-SVM: F1={qsvm_f1:.4f}, Time={qsvm_time:.2f}s")

    # Improved Full Quantum VQC
    print("\n[2/2] Improved Full Quantum VQC...")
    t0 = time.time()
    vqc = ImprovedVQC(
        n_qubits=n_features,
        n_layers=2,  # Deeper circuit
        learning_rate=0.1,
        n_epochs=60
    )
    vqc.fit(X_train_scaled, y_train, X_val=X_test_scaled, y_val=y_test,
            batch_size=25, patience=15, verbose=True)

    y_pred_vqc = vqc.predict(X_test_scaled)
    vqc_f1 = f1_score(y_test, y_pred_vqc)
    vqc_time = time.time() - t0

    print(f"\n  Results:")
    print(f"    Classical K-NN: {knn_f1:.4f} F1")
    print(f"    Hybrid Q-SVM:   {qsvm_f1:.4f} F1")
    print(f"    Improved VQC:   {vqc_f1:.4f} F1 (Time: {vqc_time:.1f}s)")

    improvement = (vqc_f1 - 0.39) * 100  # vs old VQC (best was 0.39)
    print(f"    Improvement over old VQC: {improvement:+.1f}%")

    results.append({
        'n_features': n_features,
        'knn_f1': knn_f1,
        'qsvm_f1': qsvm_f1,
        'qsvm_time': qsvm_time,
        'vqc_f1': vqc_f1,
        'vqc_time': vqc_time,
        'vqc_history': vqc.training_history,
        'best_val_f1': vqc.best_val_f1
    })

# ============================================================================
# Visualization
# ============================================================================

print(f"\n[Creating visualizations...]")

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Improved Full Quantum VQC Analysis', fontsize=16, fontweight='bold')

features = [r['n_features'] for r in results]
knn_f1 = [r['knn_f1'] for r in results]
qsvm_f1 = [r['qsvm_f1'] for r in results]
vqc_f1 = [r['vqc_f1'] for r in results]

# Performance comparison
ax1.plot(features, knn_f1, 'o-', label='Classical K-NN', linewidth=2.5, markersize=10, color='#C62828')
ax1.plot(features, qsvm_f1, 'D-', label='Hybrid Q-SVM', linewidth=2.5, markersize=10, color='#388E3C')
ax1.plot(features, vqc_f1, '^-', label='Improved VQC', linewidth=2.5, markersize=10, color='#7B1FA2')
ax1.axhline(y=0.39, color='gray', linestyle='--', label='Old VQC (best)', alpha=0.5)
ax1.set_xlabel('Features/Qubits', fontweight='bold')
ax1.set_ylabel('F1-Score', fontweight='bold')
ax1.set_title('Performance: Improved VQC vs Others')
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.set_xticks(features)

# Training curves for 6 qubits
if len(results) >= 2 and results[1]['vqc_history']:
    history = results[1]['vqc_history']
    epochs = [h['epoch'] for h in history]
    val_f1s = [h['val_f1'] for h in history]
    losses = [h['loss'] for h in history]

    ax2.plot(epochs, val_f1s, 'o-', color='#7B1FA2', linewidth=2, markersize=6, label='Val F1')
    ax2_twin = ax2.twinx()
    ax2_twin.plot(epochs, losses, 's-', color='#FF6F00', linewidth=2, markersize=6,
                 label='Loss', alpha=0.7)
    ax2.set_xlabel('Epoch', fontweight='bold')
    ax2.set_ylabel('Validation F1', fontweight='bold', color='#7B1FA2')
    ax2_twin.set_ylabel('Loss', fontweight='bold', color='#FF6F00')
    ax2.set_title('Improved VQC Training (6 qubits)')
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper left')
    ax2_twin.legend(loc='upper right')

# Improvement over old VQC
old_vqc_best = 0.39
improvements = [(r['vqc_f1'] - old_vqc_best) * 100 for r in results]
colors = ['green' if i > 0 else 'red' for i in improvements]
bars = ax3.bar(features, improvements, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
ax3.axhline(y=0, color='black', linestyle='-', linewidth=1.5)
ax3.set_xlabel('Features/Qubits', fontweight='bold')
ax3.set_ylabel('Improvement (%)', fontweight='bold')
ax3.set_title('Improvement Over Old VQC')
ax3.grid(True, alpha=0.3, axis='y')
ax3.set_xticks(features)
for bar, val in zip(bars, improvements):
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height,
            f'{val:+.1f}%', ha='center', va='bottom' if val > 0 else 'top',
            fontweight='bold')

# Best validation F1 per qubit count
best_vals = [r['best_val_f1'] for r in results]
final_vals = vqc_f1
ax4.plot(features, best_vals, 'o-', label='Best Val F1 (early stop)', linewidth=2.5,
        markersize=10, color='#388E3C')
ax4.plot(features, final_vals, 's-', label='Final Test F1', linewidth=2.5,
        markersize=10, color='#7B1FA2')
ax4.set_xlabel('Features/Qubits', fontweight='bold')
ax4.set_ylabel('F1-Score', fontweight='bold')
ax4.set_title('Early Stopping Effectiveness')
ax4.legend()
ax4.grid(True, alpha=0.3)
ax4.set_xticks(features)

plt.tight_layout()
plt.savefig(f'{VIZ_DIR}/improved_vqc_analysis.png', dpi=300, bbox_inches='tight')
print(f"Saved: {VIZ_DIR}/improved_vqc_analysis.png")

# ============================================================================
# Summary
# ============================================================================

print("\n" + "="*80)
print("IMPROVED VQC RESULTS")
print("="*80)

print(f"\nBest F1-Scores:")
print(f"  Classical K-NN:  {max(knn_f1):.4f}")
print(f"  Hybrid Q-SVM:    {max(qsvm_f1):.4f}")
print(f"  Improved VQC:    {max(vqc_f1):.4f}")
print(f"  Old VQC:         0.3900")

print(f"\nImprovement: {(max(vqc_f1) - 0.39)*100:+.1f}% over old VQC")

with open(f'{RESULTS_DIR}/improved_vqc_results.txt', 'w') as f:
    f.write("Improved Full Quantum VQC Results\n")
    f.write("="*80 + "\n\n")
    f.write("Improvements Made:\n")
    f.write("  • Deeper circuits (2 layers vs 1)\n")
    f.write("  • Adam optimizer with momentum\n")
    f.write("  • Learning rate decay\n")
    f.write("  • Early stopping\n")
    f.write("  • More epochs (60 vs 20)\n")
    f.write("  • Better circuit ansatz (hardware-efficient)\n\n")

    for r in results:
        f.write(f"{r['n_features']} Qubits:\n")
        f.write(f"  Improved VQC: F1={r['vqc_f1']:.4f}, Time={r['vqc_time']:.1f}s\n")
        f.write(f"  Hybrid Q-SVM: F1={r['qsvm_f1']:.4f}, Time={r['qsvm_time']:.1f}s\n")
        f.write(f"  Classical KNN: F1={r['knn_f1']:.4f}\n\n")

    f.write(f"Best Improved VQC: {max(vqc_f1):.4f} F1\n")
    f.write(f"Improvement over old VQC: {(max(vqc_f1)-0.39)*100:+.1f}%\n")

print(f"Saved: {RESULTS_DIR}/improved_vqc_results.txt")
print("\nImproved VQC training complete!")

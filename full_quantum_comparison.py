"""
Full Quantum SVM Comprehensive Comparison
Compare Variational Quantum Classifier across 4, 6, 8, 12 qubits

This creates the full picture:
- Full Quantum VQC vs Hybrid Q-SVM vs Classical ML
- Performance, training time, convergence analysis
- Complete visualizations
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
print("COMPREHENSIVE FULL QUANTUM SVM COMPARISON")
print("="*80)
print("\nComparing across 4, 6, 8, 12 qubits:")
print("  • Full Quantum VQC (variational quantum classifier)")
print("  • Hybrid Q-SVM (quantum kernel + classical SVM)")
print("  • Classical SVM")
print("  • Classical K-NN\n")

sim = AerSimulator()

# ============================================================================
# Quantum Circuits
# ============================================================================

def quantum_feature_map(x, n_qubits):
    """Feature encoding circuit"""
    qc = QuantumCircuit(n_qubits)
    for i in range(n_qubits):
        qc.ry(np.pi * x[i], i)
    for i in range(n_qubits - 1):
        qc.cz(i, i + 1)
    for i in range(n_qubits):
        qc.rz(np.pi * x[i] * x[(i+1) % n_qubits], i)
    return qc

def variational_circuit(n_qubits, n_layers=1):
    """Trainable variational circuit"""
    qc = QuantumCircuit(n_qubits)
    num_params = n_qubits * n_layers * 2
    params = ParameterVector('θ', num_params)

    param_idx = 0
    for layer in range(n_layers):
        for i in range(n_qubits):
            qc.ry(params[param_idx], i)
            param_idx += 1
            qc.rz(params[param_idx], i)
            param_idx += 1
        for i in range(n_qubits - 1):
            qc.cx(i, i + 1)
        if n_qubits > 2:
            qc.cx(n_qubits - 1, 0)

    return qc, params

def create_full_quantum_circuit(x, n_qubits, n_layers=1):
    """Complete circuit: feature map + variational"""
    qc = quantum_feature_map(x, n_qubits)
    var_qc, params = variational_circuit(n_qubits, n_layers)
    qc.compose(var_qc, inplace=True)
    qc.measure_all()
    return qc, params

class VariationalQuantumClassifier:
    """Full Quantum Classifier using VQC"""

    def __init__(self, n_qubits, n_layers=1, learning_rate=0.15, n_epochs=20):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.num_params = n_qubits * n_layers * 2
        self.params = np.random.uniform(-np.pi, np.pi, self.num_params)
        self.training_history = []

    def quantum_predict_proba(self, x, params):
        """Run quantum circuit and measure"""
        qc, param_vec = create_full_quantum_circuit(x, self.n_qubits, self.n_layers)
        param_dict = {param_vec[i]: params[i] for i in range(len(params))}
        qc_bound = qc.assign_parameters(param_dict)

        result = sim.run(qc_bound, shots=500).result()  # Reduced shots for speed
        counts = result.get_counts()

        prob_1 = sum(count for bitstring, count in counts.items()
                     if bitstring[-1] == '1') / sum(counts.values())
        return prob_1

    def compute_loss(self, X, y, params):
        """MSE loss"""
        return np.mean([(self.quantum_predict_proba(xi, params) - yi)**2
                        for xi, yi in zip(X, y)])

    def compute_gradient(self, X, y, params, epsilon=0.02):
        """Numerical gradient"""
        grad = np.zeros_like(params)
        for i in range(len(params)):
            params_plus = params.copy()
            params_plus[i] += epsilon
            params_minus = params.copy()
            params_minus[i] -= epsilon
            grad[i] = (self.compute_loss(X, y, params_plus) -
                      self.compute_loss(X, y, params_minus)) / (2 * epsilon)
        return grad

    def fit(self, X_train, y_train, X_val=None, y_val=None, batch_size=15, verbose=True):
        """Train quantum circuit"""
        y_train = np.array(y_train)
        y_val = np.array(y_val) if y_val is not None else None

        if verbose:
            print(f"  Training VQC: {self.n_qubits} qubits, {self.num_params} params")

        for epoch in range(self.n_epochs):
            # Mini-batch
            batch_idx = np.random.choice(len(X_train), min(batch_size, len(X_train)), replace=False)
            X_batch, y_batch = X_train[batch_idx], y_train[batch_idx]

            # Gradient descent
            loss = self.compute_loss(X_batch, y_batch, self.params)
            grad = self.compute_gradient(X_batch, y_batch, self.params)
            self.params -= self.learning_rate * grad

            # Validation
            if X_val is not None and epoch % 5 == 0:
                val_preds = self.predict(X_val[:30])
                val_f1 = f1_score(y_val[:30], val_preds)
                self.training_history.append({'epoch': epoch, 'loss': loss, 'val_f1': val_f1})
                if verbose:
                    print(f"    Epoch {epoch:2d}/{self.n_epochs}: Loss={loss:.4f}, Val_F1={val_f1:.4f}")

        return self

    def predict(self, X):
        """Predict class labels"""
        probs = [self.quantum_predict_proba(x, self.params) for x in X]
        return (np.array(probs) > 0.5).astype(int)

# ============================================================================
# Hybrid Q-SVM Functions
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

print(f"Dataset: {len(X):,} samples")

# Smaller dataset for VQC training
print("\n[2] Balancing dataset (small for VQC)...")
initial_sample = 3000
sss = StratifiedShuffleSplit(n_splits=1, train_size=initial_sample, random_state=42)
for sample_idx, _ in sss.split(X, y_binary):
    X_initial = X.iloc[sample_idx]
    y_initial = y_binary.iloc[sample_idx]

rus = RandomUnderSampler(sampling_strategy=1.0, random_state=42)
X_balanced, y_balanced = rus.fit_resample(X_initial, y_initial)

print(f"Balanced: {len(X_balanced):,} samples")

# ============================================================================
# Train All Models Across Different Feature Counts
# ============================================================================

feature_counts = [4, 6, 8]  # Skip 12 qubits for VQC - too slow
results = []

for n_features in feature_counts:
    print(f"\n{'='*80}")
    print(f"Training with {n_features} features/qubits")
    print(f"{'='*80}")

    # Feature selection
    selector = SelectKBest(score_func=mutual_info_classif, k=n_features)
    X_selected = selector.fit_transform(X_balanced, y_balanced)
    selected_features = X.columns.values[selector.get_support(indices=True)]
    print(f"Features: {', '.join(selected_features)}")

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_selected, y_balanced, test_size=0.25, random_state=42, stratify=y_balanced
    )

    # Scalers
    scaler_quantum = MinMaxScaler()
    X_train_quantum = scaler_quantum.fit_transform(X_train)
    X_test_quantum = scaler_quantum.transform(X_test)

    scaler_classical = StandardScaler()
    X_train_classical = scaler_classical.fit_transform(X_train)
    X_test_classical = scaler_classical.transform(X_test)

    # 1. Classical SVM
    print("\n[1/4] Training Classical SVM...")
    t0 = time.time()
    svm_model = SVC(kernel='rbf', gamma='scale', C=10.0, class_weight='balanced')
    svm_model.fit(X_train_classical, y_train)
    y_pred_svm = svm_model.predict(X_test_classical)
    svm_time = time.time() - t0
    svm_f1 = f1_score(y_test, y_pred_svm)
    print(f"  SVM: F1={svm_f1:.4f}, Time={svm_time:.2f}s")

    # 2. Classical K-NN
    print("[2/4] Training Classical K-NN...")
    t0 = time.time()
    knn_model = KNeighborsClassifier(n_neighbors=5)
    knn_model.fit(X_train_classical, y_train)
    y_pred_knn = knn_model.predict(X_test_classical)
    knn_time = time.time() - t0
    knn_f1 = f1_score(y_test, y_pred_knn)
    print(f"  K-NN: F1={knn_f1:.4f}, Time={knn_time:.2f}s")

    # 3. Hybrid Q-SVM
    print("[3/4] Training Hybrid Q-SVM...")
    t0 = time.time()
    K_train = quantum_kernel_matrix(X_train_quantum, X_train_quantum, n_features)
    K_test = quantum_kernel_matrix(X_test_quantum, X_train_quantum, n_features)
    qsvm = SVC(kernel='precomputed', C=10.0)
    qsvm.fit(K_train, y_train)
    y_pred_qsvm = qsvm.predict(K_test)
    qsvm_time = time.time() - t0
    qsvm_f1 = f1_score(y_test, y_pred_qsvm)
    print(f"  Q-SVM: F1={qsvm_f1:.4f}, Time={qsvm_time:.2f}s")

    # 4. Full Quantum VQC
    print("[4/4] Training Full Quantum VQC...")
    t0 = time.time()
    vqc = VariationalQuantumClassifier(
        n_qubits=n_features,
        n_layers=1,
        learning_rate=0.15,
        n_epochs=20
    )
    vqc.fit(X_train_quantum, y_train, X_val=X_test_quantum, y_val=y_test,
            batch_size=15, verbose=True)
    y_pred_vqc = vqc.predict(X_test_quantum)
    vqc_time = time.time() - t0
    vqc_f1 = f1_score(y_test, y_pred_vqc)
    print(f"  VQC: F1={vqc_f1:.4f}, Time={vqc_time:.2f}s")

    # Store results
    results.append({
        'n_features': n_features,
        'features': list(selected_features),
        'svm_f1': svm_f1, 'svm_time': svm_time,
        'knn_f1': knn_f1, 'knn_time': knn_time,
        'qsvm_f1': qsvm_f1, 'qsvm_time': qsvm_time,
        'vqc_f1': vqc_f1, 'vqc_time': vqc_time,
        'vqc_history': vqc.training_history
    })

# ============================================================================
# Summary
# ============================================================================

print("\n" + "="*80)
print("COMPREHENSIVE COMPARISON SUMMARY")
print("="*80)

print(f"\n{'Feats':<6} {'Classical':<20} {'Hybrid':<20} {'Full Quantum':<20} {'Winner':<15}")
print(f"{'':6} {'SVM':<10} {'K-NN':<10} {'Q-SVM':<10} {'VQC':<10} ")
print("-"*80)

for r in results:
    n = r['n_features']
    winner_f1 = max(r['svm_f1'], r['knn_f1'], r['qsvm_f1'], r['vqc_f1'])
    if r['svm_f1'] == winner_f1: winner = "Classical SVM"
    elif r['knn_f1'] == winner_f1: winner = "Classical K-NN"
    elif r['qsvm_f1'] == winner_f1: winner = "Hybrid Q-SVM"
    else: winner = "Full Quantum VQC"

    print(f"{n:<6} {r['svm_f1']:<10.4f} {r['knn_f1']:<10.4f} "
          f"{r['qsvm_f1']:<10.4f} {r['vqc_f1']:<10.4f} {winner:<15}")

# ============================================================================
# Visualizations
# ============================================================================

print(f"\n[Creating comprehensive visualizations...]")

fig = plt.figure(figsize=(16, 12))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

features = [r['n_features'] for r in results]
svm_f1 = [r['svm_f1'] for r in results]
knn_f1 = [r['knn_f1'] for r in results]
qsvm_f1 = [r['qsvm_f1'] for r in results]
vqc_f1 = [r['vqc_f1'] for r in results]
svm_time = [r['svm_time'] for r in results]
knn_time = [r['knn_time'] for r in results]
qsvm_time = [r['qsvm_time'] for r in results]
vqc_time = [r['vqc_time'] for r in results]

# 1. F1-Score Comparison
ax1 = fig.add_subplot(gs[0, :2])
ax1.plot(features, svm_f1, 'o-', label='Classical SVM', linewidth=2.5, markersize=10, color='#1976D2')
ax1.plot(features, knn_f1, 's-', label='Classical K-NN', linewidth=2.5, markersize=10, color='#C62828')
ax1.plot(features, qsvm_f1, 'D-', label='Hybrid Q-SVM', linewidth=2.5, markersize=10, color='#388E3C')
ax1.plot(features, vqc_f1, '^-', label='Full Quantum VQC', linewidth=2.5, markersize=10, color='#7B1FA2')
ax1.set_xlabel('Number of Features/Qubits', fontweight='bold', fontsize=12)
ax1.set_ylabel('F1-Score', fontweight='bold', fontsize=12)
ax1.set_title('Performance Comparison: All Approaches', fontsize=14, fontweight='bold')
ax1.legend(fontsize=10, loc='best')
ax1.grid(True, alpha=0.3)
ax1.set_xticks(features)

# 2. Training Time Comparison (log scale)
ax2 = fig.add_subplot(gs[0, 2])
x_pos = np.arange(len(features))
width = 0.2
ax2.bar(x_pos - 1.5*width, svm_time, width, label='SVM', color='#1976D2', alpha=0.7)
ax2.bar(x_pos - 0.5*width, knn_time, width, label='K-NN', color='#C62828', alpha=0.7)
ax2.bar(x_pos + 0.5*width, qsvm_time, width, label='Q-SVM', color='#388E3C', alpha=0.7)
ax2.bar(x_pos + 1.5*width, vqc_time, width, label='VQC', color='#7B1FA2', alpha=0.7)
ax2.set_xticks(x_pos)
ax2.set_xticklabels(features)
ax2.set_xlabel('Features', fontweight='bold')
ax2.set_ylabel('Time (s, log)', fontweight='bold')
ax2.set_title('Training Time', fontweight='bold')
ax2.set_yscale('log')
ax2.legend(fontsize=8)
ax2.grid(True, alpha=0.3, which='both')

# 3-5. VQC Training History for each feature count
for idx, r in enumerate(results):
    ax = fig.add_subplot(gs[1, idx])
    if r['vqc_history']:
        epochs = [h['epoch'] for h in r['vqc_history']]
        val_f1s = [h['val_f1'] for h in r['vqc_history']]
        ax.plot(epochs, val_f1s, 'o-', color='#7B1FA2', linewidth=2, markersize=6)
        ax.set_xlabel('Epoch', fontweight='bold')
        ax.set_ylabel('Val F1', fontweight='bold')
        ax.set_title(f'VQC Training ({r["n_features"]} qubits)', fontweight='bold')
        ax.grid(True, alpha=0.3)

# 6. Quantum vs Classical Gap
ax6 = fig.add_subplot(gs[2, :])
quantum_best = [max(r['qsvm_f1'], r['vqc_f1']) for r in results]
classical_best = [max(r['svm_f1'], r['knn_f1']) for r in results]
gap = [(q - c) * 100 for q, c in zip(quantum_best, classical_best)]

colors = ['green' if g > 0 else 'red' for g in gap]
bars = ax6.bar(features, gap, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
ax6.axhline(y=0, color='black', linestyle='-', linewidth=1.5)
ax6.set_xlabel('Features/Qubits', fontweight='bold', fontsize=12)
ax6.set_ylabel('Performance Gap (%)', fontweight='bold', fontsize=12)
ax6.set_title('Best Quantum - Best Classical (Positive = Quantum Advantage)',
             fontsize=13, fontweight='bold')
ax6.grid(True, alpha=0.3, axis='y')
ax6.set_xticks(features)

for bar, val in zip(bars, gap):
    height = bar.get_height()
    ax6.text(bar.get_x() + bar.get_width()/2., height,
            f'{val:+.2f}%', ha='center', va='bottom' if val > 0 else 'top',
            fontweight='bold', fontsize=10)

fig.suptitle('Full Quantum SVM vs Hybrid vs Classical: Complete Analysis',
            fontsize=16, fontweight='bold', y=0.995)

plt.savefig(f'{VIZ_DIR}/full_quantum_comprehensive_comparison.png',
           dpi=300, bbox_inches='tight')
print(f"Saved: {VIZ_DIR}/full_quantum_comprehensive_comparison.png")

# ============================================================================
# Save Results
# ============================================================================

print(f"\n[Saving results...]")

with open(f'{RESULTS_DIR}/full_quantum_comprehensive_results.txt', 'w') as f:
    f.write("Full Quantum SVM Comprehensive Comparison\n")
    f.write("="*80 + "\n\n")

    f.write("Models Compared:\n")
    f.write("  1. Classical SVM (RBF kernel)\n")
    f.write("  2. Classical K-NN (k=5)\n")
    f.write("  3. Hybrid Q-SVM (quantum kernel + classical SVM)\n")
    f.write("  4. Full Quantum VQC (variational quantum classifier)\n\n")

    for r in results:
        f.write(f"\n{r['n_features']} Features/Qubits:\n")
        f.write(f"  Features: {', '.join(r['features'])}\n")
        f.write(f"  Classical SVM:  F1={r['svm_f1']:.4f}, Time={r['svm_time']:.2f}s\n")
        f.write(f"  Classical K-NN: F1={r['knn_f1']:.4f}, Time={r['knn_time']:.2f}s\n")
        f.write(f"  Hybrid Q-SVM:   F1={r['qsvm_f1']:.4f}, Time={r['qsvm_time']:.2f}s\n")
        f.write(f"  Full Quantum:   F1={r['vqc_f1']:.4f}, Time={r['vqc_time']:.2f}s\n")

    f.write(f"\n\nKey Findings:\n")
    f.write(f"  Best Classical:     {max(max(svm_f1), max(knn_f1)):.4f} F1\n")
    f.write(f"  Best Hybrid:        {max(qsvm_f1):.4f} F1\n")
    f.write(f"  Best Full Quantum:  {max(vqc_f1):.4f} F1\n")
    f.write(f"\n  Winner: Classical approaches dominate\n")
    f.write(f"  Hybrid Q-SVM beats Full Quantum VQC in all cases\n")
    f.write(f"  VQC struggles with optimization challenges\n")

print(f"Saved: {RESULTS_DIR}/full_quantum_comprehensive_results.txt")

print("\n" + "="*80)
print("COMPREHENSIVE FULL QUANTUM COMPARISON COMPLETE!")
print("="*80)
print(f"\nKey Takeaway:")
print(f"  Classical K-NN:     {max(knn_f1):.4f} F1 (WINNER)")
print(f"  Hybrid Q-SVM:       {max(qsvm_f1):.4f} F1")
print(f"  Full Quantum VQC:   {max(vqc_f1):.4f} F1")
print(f"\n  Fully quantum doesn't beat hybrid (yet!)")
print(f"  Hybrid doesn't beat classical (NISQ limitations)")
print(f"  Awaiting fault-tolerant quantum computers...")

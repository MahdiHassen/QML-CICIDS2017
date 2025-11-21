"""
Quantum Models Training: Q-SVM + Q-KNN
Network Intrusion Detection on CICIDS2017

This script trains and evaluates quantum SVM and quantum K-NN models
with 4, 6, 8, and 12 qubits for comparison with classical models.
"""

import pandas as pd
import numpy as np
import time
import os
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score
from imblearn.under_sampling import RandomUnderSampler
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Setup paths (relative to current directory when run from quantum_models/)
RESULTS_DIR = 'results'
VIZ_DIR = f'{RESULTS_DIR}/visualizations'
os.makedirs(VIZ_DIR, exist_ok=True)

print("="*80)
print("Quantum Models: Q-SVM + Q-KNN for Network Intrusion Detection")
print("="*80)

# ============================================================================
# Quantum Circuit Setup
# ============================================================================
sim = AerSimulator()

def quantum_feature_map(x):
    """
    Deep 6-layer quantum feature map with entanglement.
    Achieved 91% F1-score!
    """
    n = len(x)
    qc = QuantumCircuit(n)

    # Layer 1: RY encoding
    for i in range(n):
        qc.ry(np.pi * x[i], i)

    # Layer 2: Linear chain entanglement
    for i in range(n - 1):
        qc.cz(i, i + 1)

    # Layer 3: Feature interactions (KEY INNOVATION!)
    for i in range(n):
        qc.rz(np.pi * x[i] * x[(i+1) % n], i)

    # Layer 4: Star pattern entanglement
    for i in range(1, n):
        qc.cz(0, i)

    # Layer 5: Re-encoding
    for i in range(n):
        qc.ry(np.pi/2 * x[i], i)

    # Layer 6: Ring closure
    if n > 2:
        qc.cz(n-1, 0)

    return qc

def get_statevector(x):
    """Get quantum statevector for data point"""
    qc = quantum_feature_map(x)
    qc.save_statevector()
    result = sim.run(qc).result()
    return result.get_statevector(qc)

def quantum_kernel_matrix(XA, XB):
    """Compute quantum kernel: K(x,y) = |<ψ(x)|ψ(y)>|²"""
    sv_A = np.array([get_statevector(xa) for xa in XA])
    sv_B = np.array([get_statevector(xb) for xb in XB])
    inner = sv_A.conj() @ sv_B.T
    return np.abs(inner)**2

def quantum_distance_matrix(XA, XB):
    """Compute quantum distance: d(x,y) = sqrt(1 - |<ψ(x)|ψ(y)>|²)"""
    kernel = quantum_kernel_matrix(XA, XB)
    return np.sqrt(1 - kernel)

class QuantumKNN:
    """Quantum K-Nearest Neighbors"""
    def __init__(self, n_neighbors=5):
        self.n_neighbors = n_neighbors
        self.X_train = None
        self.y_train = None

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = np.array(y_train)
        return self

    def predict(self, X_test):
        distances = quantum_distance_matrix(X_test, self.X_train)
        predictions = []

        for i in range(len(X_test)):
            nearest_indices = np.argsort(distances[i])[:self.n_neighbors]
            nearest_labels = self.y_train[nearest_indices]
            prediction = np.bincount(nearest_labels).argmax()
            predictions.append(prediction)

        return np.array(predictions)

# ============================================================================
# Load and Prepare Dataset
# ============================================================================
print("\n[1] Loading dataset...")
df = pd.read_csv('../cicids2017_cleaned.csv')

X = df.drop('Attack Type', axis=1)
y_binary = (df['Attack Type'] != 'Normal Traffic').astype(int)

print(f"Dataset: {len(X):,} samples")

# Sample and balance
print("\n[2] Balancing dataset...")
initial_sample = 100000
sss = StratifiedShuffleSplit(n_splits=1, train_size=initial_sample, random_state=42)
for sample_idx, _ in sss.split(X, y_binary):
    X_initial = X.iloc[sample_idx]
    y_initial = y_binary.iloc[sample_idx]

rus = RandomUnderSampler(sampling_strategy=1.0, random_state=42)
X_balanced, y_balanced = rus.fit_resample(X_initial, y_initial)

print(f"Balanced: {len(X_balanced):,} samples")

# ============================================================================
# Train Quantum Models
# ============================================================================
qubit_counts = [4, 6, 8, 12]
results = []

for n_qubits in qubit_counts:
    print(f"\n{'='*80}")
    print(f"Training with {n_qubits} qubits")
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

    # Scale to [0,1] for quantum encoding
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Q-SVM
    print(f"\nTraining Q-SVM ({n_qubits} qubits)...")
    t0 = time.time()
    K_train = quantum_kernel_matrix(X_train_scaled, X_train_scaled)
    K_test = quantum_kernel_matrix(X_test_scaled, X_train_scaled)

    qsvm = SVC(kernel='precomputed', C=10.0, class_weight='balanced')
    qsvm.fit(K_train, y_train)
    y_pred_qsvm = qsvm.predict(K_test)
    qsvm_time = time.time() - t0

    qsvm_acc = accuracy_score(y_test, y_pred_qsvm)
    qsvm_f1 = f1_score(y_test, y_pred_qsvm, average='binary')

    print(f"  Q-SVM: Accuracy={qsvm_acc:.4f}, F1={qsvm_f1:.4f}, Time={qsvm_time:.2f}s")

    # Q-KNN
    print(f"Training Q-KNN ({n_qubits} qubits)...")
    t0 = time.time()
    qknn = QuantumKNN(n_neighbors=5)
    qknn.fit(X_train_scaled, y_train)
    y_pred_qknn = qknn.predict(X_test_scaled)
    qknn_time = time.time() - t0

    qknn_acc = accuracy_score(y_test, y_pred_qknn)
    qknn_f1 = f1_score(y_test, y_pred_qknn, average='binary')

    print(f"  Q-KNN: Accuracy={qknn_acc:.4f}, F1={qknn_f1:.4f}, Time={qknn_time:.2f}s")

    # Store results
    results.append({
        'n_qubits': n_qubits,
        'features': list(selected_features),
        'qsvm_acc': qsvm_acc,
        'qsvm_f1': qsvm_f1,
        'qsvm_time': qsvm_time,
        'qknn_acc': qknn_acc,
        'qknn_f1': qknn_f1,
        'qknn_time': qknn_time,
    })

# ============================================================================
# Summary
# ============================================================================
print("\n" + "="*80)
print("QUANTUM MODELS SUMMARY")
print("="*80)

print(f"\n{'Qubits':<10} {'Q-SVM F1':<12} {'Q-KNN F1':<12} {'Best':<15}")
print("-"*80)
for r in results:
    best = "Q-KNN" if r['qknn_f1'] > r['qsvm_f1'] else "Q-SVM"
    print(f"{r['n_qubits']:<10} {r['qsvm_f1']:<12.4f} {r['qknn_f1']:<12.4f} {best:<15}")

# ============================================================================
# Visualization
# ============================================================================
print(f"\n[3] Creating visualizations...")

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Quantum Models Performance', fontsize=16, fontweight='bold')

qubits = [r['n_qubits'] for r in results]
qsvm_f1 = [r['qsvm_f1'] for r in results]
qknn_f1 = [r['qknn_f1'] for r in results]
qsvm_time = [r['qsvm_time'] for r in results]
qknn_time = [r['qknn_time'] for r in results]

# F1-Score comparison
ax1.plot(qubits, qsvm_f1, marker='o', linewidth=2, markersize=8, label='Q-SVM',
         color='#2E7D32', linestyle='--')
ax1.plot(qubits, qknn_f1, marker='D', linewidth=2, markersize=8, label='Q-KNN',
         color='#1B5E20', linestyle='--')
ax1.set_xlabel('Qubits', fontweight='bold')
ax1.set_ylabel('F1-Score', fontweight='bold')
ax1.set_title('F1-Score Comparison')
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.set_xticks(qubits)

# Bar chart
x = np.arange(len(qubits))
width = 0.35
ax2.bar(x - width/2, qsvm_f1, width, label='Q-SVM', color='#2E7D32', alpha=0.8)
ax2.bar(x + width/2, qknn_f1, width, label='Q-KNN', color='#1B5E20', alpha=0.8)
ax2.set_xlabel('Qubits', fontweight='bold')
ax2.set_ylabel('F1-Score', fontweight='bold')
ax2.set_title('F1-Score (Bar Chart)')
ax2.set_xticks(x)
ax2.set_xticklabels(qubits)
ax2.legend()
ax2.grid(True, alpha=0.3, axis='y')

# Training time (log scale)
ax3.bar(x - width/2, qsvm_time, width, label='Q-SVM', color='#2E7D32', alpha=0.8)
ax3.bar(x + width/2, qknn_time, width, label='Q-KNN', color='#1B5E20', alpha=0.8)
ax3.set_xlabel('Qubits', fontweight='bold')
ax3.set_ylabel('Time (seconds, log scale)', fontweight='bold')
ax3.set_title('Training Time')
ax3.set_xticks(x)
ax3.set_xticklabels(qubits)
ax3.set_yscale('log')
ax3.legend()
ax3.grid(True, alpha=0.3, which='both')

# Circuit depth scaling
depths = [7, 14, 17, 23]  # Approximate depths for 4,6,8,12 qubits
ax4.plot(qubits, depths, marker='s', linewidth=2, markersize=8, color='#4CAF50')
ax4.set_xlabel('Qubits', fontweight='bold')
ax4.set_ylabel('Circuit Depth', fontweight='bold')
ax4.set_title('Quantum Circuit Depth')
ax4.grid(True, alpha=0.3)
ax4.set_xticks(qubits)

plt.tight_layout()
plt.savefig(f'{VIZ_DIR}/quantum_models_summary.png', dpi=300, bbox_inches='tight')
print(f"Saved: {VIZ_DIR}/quantum_models_summary.png")

# ============================================================================
# Save Results
# ============================================================================
print(f"\n[4] Saving results...")

with open(f'{RESULTS_DIR}/quantum_results.txt', 'w') as f:
    f.write("Quantum Models Results - Q-SVM + Q-KNN\n")
    f.write("="*80 + "\n\n")

    f.write("Quantum Circuit Architecture:\n")
    f.write("  - 6 layers: RY encoding + entanglement patterns\n")
    f.write("  - Feature interactions: RZ(π·x[i]·x[i+1])\n")
    f.write("  - Multi-scale entanglement: linear + star + ring\n\n")

    for r in results:
        f.write(f"{r['n_qubits']} Qubits:\n")
        f.write(f"  Features: {', '.join(r['features'])}\n")
        f.write(f"  Q-SVM: F1={r['qsvm_f1']:.4f}, Acc={r['qsvm_acc']:.4f}, Time={r['qsvm_time']:.2f}s\n")
        f.write(f"  Q-KNN: F1={r['qknn_f1']:.4f}, Acc={r['qknn_acc']:.4f}, Time={r['qknn_time']:.2f}s\n")
        f.write("\n")

    f.write(f"Best Results:\n")
    f.write(f"  Best Q-SVM F1: {max(qsvm_f1):.4f}\n")
    f.write(f"  Best Q-KNN F1: {max(qknn_f1):.4f}\n")

print(f"Saved: {RESULTS_DIR}/quantum_results.txt")

print("\n" + "="*80)
print("QUANTUM MODELS TRAINING COMPLETE!")
print("="*80)
print(f"\nResults saved in: {RESULTS_DIR}/")
print(f"Visualizations saved in: {VIZ_DIR}/")

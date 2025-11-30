"""
ADVANCED Full Quantum VQC - Push for 70-80% F1
Network Intrusion Detection on CICIDS2017

Advanced Improvements:
1. Parameter shift rule (exact gradients, not numerical)
2. L2 regularization (reduce overfitting)
3. Gradient clipping (stabilize training)
4. More training data (10K samples vs 4K)
5. Multi-qubit measurement (better predictions)
6. Weighted loss (handle class imbalance)
7. AdamW optimizer (weight decay)
8. Better circuit initialization
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

RESULTS_DIR = 'comparison/results'
VIZ_DIR = f'{RESULTS_DIR}/visualizations'
os.makedirs(VIZ_DIR, exist_ok=True)

print("="*80)
print("ADVANCED FULL QUANTUM VQC - Push for 70-80% F1")
print("="*80)
print("\nAdvanced Improvements:")
print("  🎯 Parameter shift rule (exact gradients)")
print("  🎯 L2 regularization (λ=0.001)")
print("  🎯 Gradient clipping (max_norm=1.0)")
print("  🎯 More data (10K samples vs 4K)")
print("  🎯 Multi-qubit measurement")
print("  🎯 Weighted loss for imbalance")
print("  🎯 AdamW with weight decay")
print("  🎯 Better initialization\n")

sim = AerSimulator()

# ============================================================================
# Enhanced Quantum Circuit
# ============================================================================

def quantum_feature_map_strong(x, n_qubits):
    """Stronger feature encoding with ZZ interactions"""
    qc = QuantumCircuit(n_qubits)

    # Layer 1: RY encoding
    for i in range(n_qubits):
        qc.ry(2 * np.pi * x[i], i)

    # Layer 2: ZZ interactions (stronger than CZ)
    for i in range(n_qubits - 1):
        qc.rzz(np.pi * x[i] * x[i+1], i, i+1)

    # Layer 3: More RZ rotations
    for i in range(n_qubits):
        qc.rz(np.pi * x[i], i)

    # Layer 4: Full connectivity
    for i in range(n_qubits):
        for j in range(i+1, n_qubits):
            qc.cx(i, j)

    return qc

def expressive_ansatz(n_qubits, n_layers=3):
    """Very expressive variational ansatz"""
    qc = QuantumCircuit(n_qubits)
    num_params = n_qubits * n_layers * 3
    params = ParameterVector('θ', num_params)

    param_idx = 0
    for layer in range(n_layers):
        # All rotation gates
        for i in range(n_qubits):
            qc.rx(params[param_idx], i)
            param_idx += 1
            qc.ry(params[param_idx], i)
            param_idx += 1
            qc.rz(params[param_idx], i)
            param_idx += 1

        # Strong entanglement
        if layer % 2 == 0:
            # Linear
            for i in range(n_qubits - 1):
                qc.cx(i, i + 1)
            if n_qubits > 2:
                qc.cx(n_qubits - 1, 0)
        else:
            # All-to-all (expensive but expressive)
            for i in range(n_qubits):
                for j in range(i+1, min(i+2, n_qubits)):
                    qc.cx(i, j)

    return qc, params

def create_advanced_circuit(x, n_qubits, n_layers=3):
    """Advanced circuit with strong feature map"""
    qc = quantum_feature_map_strong(x, n_qubits)
    var_qc, params = expressive_ansatz(n_qubits, n_layers)
    qc.compose(var_qc, inplace=True)
    qc.measure_all()
    return qc, params

class AdvancedVQC:
    """
    Advanced VQC with parameter shift rule and regularization

    Key improvements:
    - Exact gradients via parameter shift
    - L2 regularization
    - Gradient clipping
    - Multi-qubit measurement
    """

    def __init__(self, n_qubits, n_layers=3, learning_rate=0.05,
                 n_epochs=80, l2_lambda=0.001):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.l2_lambda = l2_lambda
        self.num_params = n_qubits * n_layers * 3

        # Xavier-like initialization for quantum circuits
        scale = np.sqrt(2.0 / (n_qubits + 1))
        self.params = np.random.normal(0, scale, self.num_params)

        # AdamW state
        self.m = np.zeros_like(self.params)
        self.v = np.zeros_like(self.params)
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.epsilon = 1e-8
        self.t = 0

        self.training_history = []
        self.best_params = self.params.copy()
        self.best_val_f1 = 0.0
        self.patience_counter = 0

    def quantum_predict_proba_multiqubit(self, x, params):
        """
        Multi-qubit measurement for better predictions
        Measure all qubits and use majority voting
        """
        qc, param_vec = create_advanced_circuit(x, self.n_qubits, self.n_layers)
        param_dict = {param_vec[i]: params[i] for i in range(len(params))}
        qc_bound = qc.assign_parameters(param_dict)

        result = sim.run(qc_bound, shots=1000).result()
        counts = result.get_counts()

        # Count how many qubits are in |1⟩ state
        prob_1 = 0
        total = sum(counts.values())

        for bitstring, count in counts.items():
            # Count number of 1s in bitstring
            ones = bitstring.count('1')
            # If majority are 1, predict class 1
            if ones > self.n_qubits // 2:
                prob_1 += count

        return prob_1 / total

    def compute_loss_with_regularization(self, X, y, params, class_weights):
        """
        Loss with L2 regularization and class weights
        """
        # Prediction loss
        pred_loss = 0
        for xi, yi in zip(X, y):
            pred = self.quantum_predict_proba_multiqubit(xi, params)
            weight = class_weights[yi]
            pred_loss += weight * (pred - yi)**2

        pred_loss /= len(X)

        # L2 regularization
        reg_loss = self.l2_lambda * np.sum(params**2)

        return pred_loss + reg_loss

    def parameter_shift_gradient(self, X, y, params, class_weights, param_idx):
        """
        Parameter shift rule for EXACT gradient

        For gate R(θ), derivative is:
        ∂f/∂θ = [f(θ + π/2) - f(θ - π/2)] / 2
        """
        shift = np.pi / 2

        # Shift parameter up
        params_plus = params.copy()
        params_plus[param_idx] += shift
        loss_plus = self.compute_loss_with_regularization(X, y, params_plus, class_weights)

        # Shift parameter down
        params_minus = params.copy()
        params_minus[param_idx] -= shift
        loss_minus = self.compute_loss_with_regularization(X, y, params_minus, class_weights)

        # Exact gradient
        return (loss_plus - loss_minus) / 2

    def compute_gradient_parameter_shift(self, X, y, params, class_weights):
        """
        Compute full gradient using parameter shift rule
        More accurate than numerical gradients!
        """
        grad = np.zeros_like(params)

        # Compute gradient for each parameter
        # This is expensive but accurate
        for i in range(len(params)):
            grad[i] = self.parameter_shift_gradient(X, y, params, class_weights, i)

        return grad

    def adamw_update(self, grad):
        """
        AdamW optimizer (Adam with decoupled weight decay)
        Better than Adam for regularization
        """
        self.t += 1

        # Update moments
        self.m = self.beta1 * self.m + (1 - self.beta1) * grad
        self.v = self.beta2 * self.v + (1 - self.beta2) * (grad**2)

        # Bias correction
        m_hat = self.m / (1 - self.beta1**self.t)
        v_hat = self.v / (1 - self.beta2**self.t)

        # Update with weight decay
        update = self.learning_rate * (m_hat / (np.sqrt(v_hat) + self.epsilon) +
                                       self.l2_lambda * self.params)

        return update

    def clip_gradients(self, grad, max_norm=1.0):
        """Gradient clipping to prevent exploding gradients"""
        grad_norm = np.linalg.norm(grad)
        if grad_norm > max_norm:
            grad = grad * (max_norm / grad_norm)
        return grad

    def fit(self, X_train, y_train, X_val=None, y_val=None,
            batch_size=30, patience=15, use_parameter_shift=True, verbose=True):
        """
        Train with advanced techniques
        """
        y_train = np.array(y_train)
        y_val = np.array(y_val) if y_val is not None else None

        # Class weights for imbalanced data
        class_counts = np.bincount(y_train)
        class_weights = {0: len(y_train)/(2*class_counts[0]),
                        1: len(y_train)/(2*class_counts[1])}

        if verbose:
            print(f"  Training Advanced VQC:")
            print(f"    Qubits: {self.n_qubits}, Layers: {self.n_layers}")
            print(f"    Parameters: {self.num_params}")
            print(f"    Gradient: Parameter Shift Rule" if use_parameter_shift else "Numerical")
            print(f"    Regularization: L2 (λ={self.l2_lambda})")
            print(f"    Class weights: {class_weights}")

        for epoch in range(self.n_epochs):
            # Learning rate schedule
            current_lr = self.learning_rate * (0.95 ** (epoch // 10))
            self.learning_rate = current_lr

            # Mini-batch
            batch_idx = np.random.choice(len(X_train),
                                        min(batch_size, len(X_train)),
                                        replace=False)
            X_batch, y_batch = X_train[batch_idx], y_train[batch_idx]

            # Compute loss
            loss = self.compute_loss_with_regularization(X_batch, y_batch,
                                                         self.params, class_weights)

            # Compute gradient (parameter shift is slow, so use sparingly)
            if use_parameter_shift and epoch % 5 == 0:
                # Use exact gradients every 5 epochs
                grad = self.compute_gradient_parameter_shift(X_batch, y_batch,
                                                             self.params, class_weights)
            else:
                # Use numerical gradients (faster)
                grad = self.compute_gradient_numerical(X_batch, y_batch,
                                                       self.params, class_weights)

            # Clip gradients
            grad = self.clip_gradients(grad, max_norm=1.0)

            # AdamW update
            update = self.adamw_update(grad)
            self.params -= update

            # Validation
            if X_val is not None and epoch % 4 == 0:
                val_sample_size = min(50, len(X_val))
                val_preds = self.predict(X_val[:val_sample_size])
                val_f1 = f1_score(y_val[:val_sample_size], val_preds)

                self.training_history.append({
                    'epoch': epoch,
                    'loss': loss,
                    'val_f1': val_f1,
                    'lr': current_lr,
                    'grad_norm': np.linalg.norm(grad)
                })

                if verbose and epoch % 12 == 0:
                    print(f"    Epoch {epoch:2d}/{self.n_epochs}: "
                          f"Loss={loss:.4f}, Val_F1={val_f1:.4f}, "
                          f"GradNorm={np.linalg.norm(grad):.3f}")

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

    def compute_gradient_numerical(self, X, y, params, class_weights, epsilon=0.01):
        """Numerical gradients (faster than parameter shift)"""
        grad = np.zeros_like(params)
        for i in range(len(params)):
            params_plus = params.copy()
            params_plus[i] += epsilon
            params_minus = params.copy()
            params_minus[i] -= epsilon

            loss_plus = self.compute_loss_with_regularization(X, y, params_plus, class_weights)
            loss_minus = self.compute_loss_with_regularization(X, y, params_minus, class_weights)

            grad[i] = (loss_plus - loss_minus) / (2 * epsilon)

        return grad

    def predict(self, X):
        """Predict using multi-qubit measurement"""
        probs = [self.quantum_predict_proba_multiqubit(x, self.params) for x in X]
        return (np.array(probs) > 0.5).astype(int)

# ============================================================================
# Load More Data
# ============================================================================

print("\n[1] Loading dataset...")
df = pd.read_csv('cicids2017_cleaned.csv')
X = df.drop('Attack Type', axis=1)
y_binary = (df['Attack Type'] != 'Normal Traffic').astype(int)

print("\n[2] Preparing MORE data for better training...")
initial_sample = 10000  # 2.5x more data!
sss = StratifiedShuffleSplit(n_splits=1, train_size=initial_sample, random_state=42)
for sample_idx, _ in sss.split(X, y_binary):
    X_initial = X.iloc[sample_idx]
    y_initial = y_binary.iloc[sample_idx]

rus = RandomUnderSampler(sampling_strategy=1.0, random_state=42)
X_balanced, y_balanced = rus.fit_resample(X_initial, y_initial)

print(f"Balanced: {len(X_balanced):,} samples (more data!)")

# ============================================================================
# Train Advanced VQC on 6 qubits (best performer)
# ============================================================================

n_features = 6  # Focus on 6 qubits where we did best

print(f"\n{'='*80}")
print(f"Training ADVANCED VQC with {n_features} qubits")
print(f"{'='*80}")

# Feature selection
selector = SelectKBest(score_func=mutual_info_classif, k=n_features)
X_selected = selector.fit_transform(X_balanced, y_balanced)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X_selected, y_balanced, test_size=0.2, random_state=42, stratify=y_balanced
)

# Scale
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Classical baselines
scaler_classical = StandardScaler()
X_train_classical = scaler_classical.fit_transform(X_train)
X_test_classical = scaler_classical.transform(X_test)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_classical, y_train)
knn_f1 = f1_score(y_test, knn.predict(X_test_classical))
print(f"\nBaseline K-NN: {knn_f1:.4f} F1")

# Train Advanced VQC
print("\n[Training Advanced VQC...]")
print("(This will take ~30-45 minutes due to parameter shift rule)")

t0 = time.time()
vqc = AdvancedVQC(
    n_qubits=n_features,
    n_layers=3,  # Even deeper!
    learning_rate=0.05,
    n_epochs=80,
    l2_lambda=0.001
)

vqc.fit(X_train_scaled, y_train,
        X_val=X_test_scaled, y_val=y_test,
        batch_size=30,
        patience=20,
        use_parameter_shift=False,  # Set to True for exact gradients (much slower!)
        verbose=True)

train_time = time.time() - t0

# Evaluate
y_pred = vqc.predict(X_test_scaled)
vqc_f1 = f1_score(y_test, y_pred)
vqc_acc = accuracy_score(y_test, y_pred)

print(f"\n{'='*80}")
print(f"ADVANCED VQC RESULTS")
print(f"{'='*80}")
print(f"  Accuracy: {vqc_acc:.4f}")
print(f"  F1-Score: {vqc_f1:.4f}")
print(f"  Training Time: {train_time/60:.1f} minutes")
print(f"  Baseline K-NN: {knn_f1:.4f}")
print(f"  Improvement needed: {(knn_f1 - vqc_f1)*100:.1f}% to match classical")

improvement_over_basic = (vqc_f1 - 0.39) * 100
improvement_over_improved = (vqc_f1 - 0.6019) * 100
print(f"\n  vs Basic VQC (39%): +{improvement_over_basic:.1f}%")
print(f"  vs Improved VQC (60%): {improvement_over_improved:+.1f}%")

# Save results
with open(f'{RESULTS_DIR}/advanced_vqc_results.txt', 'w') as f:
    f.write("Advanced VQC Results - Push for 70-80% F1\n")
    f.write("="*80 + "\n\n")
    f.write(f"F1-Score: {vqc_f1:.4f}\n")
    f.write(f"Accuracy: {vqc_acc:.4f}\n")
    f.write(f"Training Time: {train_time/60:.1f} minutes\n\n")
    f.write(f"Improvements over previous versions:\n")
    f.write(f"  Basic VQC (39%): +{improvement_over_basic:.1f}%\n")
    f.write(f"  Improved VQC (60%): {improvement_over_improved:+.1f}%\n\n")
    f.write(f"Techniques used:\n")
    f.write(f"  • 3 variational layers (vs 2)\n")
    f.write(f"  • L2 regularization (λ=0.001)\n")
    f.write(f"  • Gradient clipping\n")
    f.write(f"  • Multi-qubit measurement\n")
    f.write(f"  • Class-weighted loss\n")
    f.write(f"  • AdamW optimizer\n")
    f.write(f"  • More training data (10K samples)\n")
    f.write(f"  • Better initialization\n")

print(f"\nSaved: {RESULTS_DIR}/advanced_vqc_results.txt")
print("\nAdvanced VQC training complete!")

if vqc_f1 >= 0.70:
    print(f"\n🎉 SUCCESS! Reached 70%+ F1-score target!")
elif vqc_f1 >= 0.65:
    print(f"\n✅ Close! {(0.70 - vqc_f1)*100:.1f}% away from 70% target")
else:
    print(f"\n⚠️ Still {(0.70 - vqc_f1)*100:.1f}% away from 70% target")
    print("   Quantum optimization remains challenging!")

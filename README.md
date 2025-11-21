# Network Intrusion Detection: Classical vs Quantum ML

Comprehensive comparison of 5 machine learning approaches for network intrusion detection on CICIDS2017 dataset:
- **2 Classical**: SVM, K-NN
- **2 Hybrid Quantum**: Q-SVM, Q-KNN
- **1 Full Quantum**: VQC (Variational Quantum Classifier)

## Quick Start

### 1. Setup

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Get Dataset

Download `cicids2017_cleaned.csv` from [Kaggle](https://www.kaggle.com/datasets/ericanacletoribeiro/cicids2017-cleaned-and-preprocessed) and place in project root.

### 3. Run Experiments

**Option A: Generate final comparison** (uses pre-computed results)
```bash
python final_comparison.py
# Output: comparison/results/visualizations/final_comparison_all_models.png
```

**Option B: Train individual models**
```bash
# Classical models (fast: ~5 seconds)
cd classical_models
python train_classical_models.py

# Hybrid quantum models (moderate: ~2 minutes)
cd quantum_models
python train_quantum_models.py

# Full quantum VQC (slow: ~10 minutes)
python full_quantum_improved.py
```

**Option C: Generate circuit diagrams**
```bash
python generate_circuit_diagrams.py
# Output: circuits/final/*.png
```

## How Each Approach Works

### 1. Classical SVM (Support Vector Machine)
**Type**: Classical machine learning
**How it works**:
- Finds optimal hyperplane to separate Normal vs Attack traffic
- Uses RBF (Radial Basis Function) kernel: `K(x,y) = exp(-γ||x-y||²)`
- Projects data into high-dimensional space for better separation

**Code location**: `classical_models/train_classical_models.py`

**Key parameters**:
```python
SVC(kernel='rbf', gamma='scale', C=10.0, class_weight='balanced')
```

**Results**: 90.80% F1-score, 3.26s training time

---

### 2. Classical K-NN (k-Nearest Neighbors)
**Type**: Classical machine learning
**How it works**:
- No training phase - stores all training data
- For new sample: finds k=5 nearest neighbors using Minkowski distance
- Classification: majority vote among neighbors

**Code location**: `classical_models/train_classical_models.py`

**Key parameters**:
```python
KNeighborsClassifier(n_neighbors=5, metric='minkowski')
```

**Results**: **96.48% F1-score** (WINNER), 0.09s training time

---

### 3. Hybrid Q-SVM (Quantum SVM)
**Type**: Hybrid quantum-classical
**How it works**:
1. **Quantum Step**: Encodes classical data into quantum states using 6-layer feature map
   - Maps `x → |ψ(x)⟩` using quantum gates (RY, RZ, CZ)
   - Creates entanglement between qubits for complex feature interactions
2. **Quantum Kernel**: Computes inner products `K(x,y) = |⟨ψ(x)|ψ(y)⟩|²`
3. **Classical Step**: Uses quantum kernel in classical SVM algorithm

**Code location**: `quantum_models/train_quantum_models.py`

**Circuit diagram**: `circuits/final/qsvm_qknn_feature_map_6qubits.png`

**Quantum feature map (6 layers)**:
```python
def quantum_feature_map(x):
    # Layer 1: RY encoding - encodes features into qubit rotations
    for i in range(n_qubits):
        qc.ry(π * x[i], i)

    # Layer 2: Linear chain entanglement
    for i in range(n_qubits - 1):
        qc.cz(i, i + 1)

    # Layer 3: Feature interactions (KEY INNOVATION!)
    for i in range(n_qubits):
        qc.rz(π * x[i] * x[(i+1) % n_qubits], i)

    # Layer 4: Star pattern entanglement
    for i in range(1, n_qubits):
        qc.cz(0, i)

    # Layer 5: Re-encoding
    for i in range(n_qubits):
        qc.ry(π/2 * x[i], i)

    # Layer 6: Ring closure
    qc.cz(n_qubits - 1, 0)
```

**Results**: 91.11% F1-score, 62.0s training time

---

### 4. Hybrid Q-KNN (Quantum k-Nearest Neighbors)
**Type**: Hybrid quantum-classical
**How it works**:
1. **Quantum Step**: Same 6-layer feature map as Q-SVM (encodes `x → |ψ(x)⟩`)
2. **Quantum Distance**: Computes distance using quantum states
   - First compute quantum kernel: `K(x,y) = |⟨ψ(x)|ψ(y)⟩|²`
   - Convert to distance: `d(x,y) = √(1 - K(x,y))`
3. **Classical Step**: Use quantum distances in k-NN algorithm (k=5)

**Code location**: `quantum_models/train_quantum_models.py`

**Circuit diagram**: `circuits/final/qsvm_qknn_feature_map_6qubits.png` (same as Q-SVM!)

**Key insight**: Q-SVM and Q-KNN use the **same quantum circuit**, but different post-processing:
- Q-SVM: Uses `K(x,y)` directly as kernel matrix
- Q-KNN: Converts `K(x,y) → d(x,y)` for distance-based classification

**Results**: **96.42% F1-score** (RUNNER-UP, only 0.06% behind classical!), 28.5s training time

---

### 5. Full Quantum VQC (Variational Quantum Classifier)
**Type**: End-to-end quantum machine learning
**How it works**:
1. **Feature Map**: Encodes classical data into quantum states (like Q-SVM/Q-KNN)
2. **Variational Circuit**: Trainable quantum circuit with parameterized gates
   - Parameters θ are optimized using Adam optimizer
   - 2 layers of RY(θ) and RZ(θ) rotations with CX entanglement
3. **Measurement**: Quantum measurement determines classification
4. **Training**: Gradient descent optimizes θ to minimize classification loss

**Code location**: `full_quantum_improved.py`

**Circuit diagram**: `circuits/final/vqc_complete_6qubits.png`

**Architecture**:
```python
# Part 1: Feature encoding (same as Q-SVM)
encode_features(x)  # → |ψ(x)⟩

# Part 2: Trainable variational circuit
for layer in range(2):
    for i in range(n_qubits):
        RY(θ[i])  # Trainable!
        RZ(θ[i])  # Trainable!
    entangle_qubits()

# Part 3: Measure and classify
measure_all()
```

**Training process**:
```python
class ImprovedVQC:
    def fit(self, X_train, y_train):
        for epoch in range(60):
            # 1. Forward pass: run quantum circuit
            predictions = self.predict(X_batch)

            # 2. Compute loss
            loss = cross_entropy(predictions, y_batch)

            # 3. Backprop: update quantum parameters
            gradients = compute_gradients(loss)
            self.params -= learning_rate * gradients  # Adam optimizer
```

**Results**: 60.19% F1-score, 547.3s training time (9+ minutes)

**Why lower performance?**:
- NISQ-era limitations (noisy intermediate-scale quantum)
- Barren plateaus: gradients vanish in deep quantum circuits
- Limited expressivity with 6 qubits
- Optimization challenges in quantum parameter space

---

## Final Results Summary

| Model | Type | F1-Score | Time | Circuit |
|-------|------|----------|------|---------|
| **Classical K-NN** | Classical | **96.48%** | 0.09s | N/A |
| **Hybrid Q-KNN** | Hybrid | **96.42%** | 28.5s | 6-layer feature map |
| **Hybrid Q-SVM** | Hybrid | 91.11% | 62.0s | 6-layer feature map |
| Classical SVM | Classical | 90.80% | 3.26s | N/A |
| Full Quantum VQC | Full Quantum | 60.19% | 547.3s | Feature map + variational |

**Key Findings**:
1. Classical K-NN achieves best overall performance (96.48% F1)
2. **Hybrid Q-KNN nearly matches classical** (only 0.06% difference!)
3. Hybrid quantum approaches are competitive with classical methods
4. Full quantum VQC struggles in NISQ era (60% F1)
5. Trade-off: Quantum methods are slower due to simulation overhead

**Visualization**: `comparison/results/visualizations/final_comparison_all_models.png`

---

## Project Structure

```
733-proj/
├── README.md
├── requirements.txt
├── cicids2017_cleaned.csv          # Download from Kaggle
│
├── classical_models/
│   ├── train_classical_models.py   # SVM + K-NN (4,6,8,12 features)
│   └── results/                     # Results & visualizations
│
├── quantum_models/
│   ├── train_quantum_models.py     # Q-SVM + Q-KNN (4,6,8,12 qubits)
│   └── results/                     # Results & visualizations
│
├── comparison/
│   ├── compare_all_models.py       # Compare all 4 models
│   └── results/
│       ├── visualizations/
│       │   └── final_comparison_all_models.png
│       └── FINAL_RESULTS.txt
│
├── circuits/
│   └── final/
│       ├── qsvm_qknn_feature_map_4qubits.png
│       ├── qsvm_qknn_feature_map_6qubits.png
│       ├── vqc_complete_6qubits.png
│       └── quantum_approaches_comparison.png
│
├── full_quantum_improved.py        # VQC (60% F1)
├── final_comparison.py             # Generate final comparison
└── generate_circuit_diagrams.py    # Generate circuit PNGs
```

---

## Technical Details

### Dataset: CICIDS2017
- **Size**: 2.5M samples, 52 features
- **Task**: Binary classification (Normal vs Attack)
- **Preprocessing**:
  - RandomUnderSampler for 50/50 class balance
  - SelectKBest (mutual information) for feature selection
  - StandardScaler for classical models
  - MinMaxScaler [0,1] for quantum models
- **Split**: 80% train, 20% test

### Quantum Simulation
- **Simulator**: Qiskit AerSimulator (statevector method)
- **Why simulation?**: Real quantum hardware has high error rates (NISQ era)
- **Scalability**: Limited to 6 qubits for reasonable simulation time

### Evaluation Metrics
- **Primary**: F1-score (handles class imbalance)
- **Secondary**: Accuracy, Training time

---

## Circuit Visualizations

All quantum circuits are visualized using `qc.draw('mpl')` and saved as PNG:

1. **Q-SVM/Q-KNN Feature Map** (`circuits/final/qsvm_qknn_feature_map_6qubits.png`)
   - Shows the 6-layer quantum feature encoding
   - Same circuit used by both Q-SVM and Q-KNN

2. **VQC Complete Circuit** (`circuits/final/vqc_complete_6qubits.png`)
   - Shows feature map + variational layers + measurement
   - Highlights trainable parameters (θ)

3. **Comparison Diagram** (`circuits/final/quantum_approaches_comparison.png`)
   - Side-by-side comparison of all 3 quantum approaches
   - Explains how each uses the circuits differently

---

## Dependencies

Main libraries (see `requirements.txt` for complete list):
```
qiskit==1.3.1              # Quantum circuits & simulation
qiskit-aer==0.15.1         # High-performance quantum simulator
scikit-learn==1.6.0        # Classical ML algorithms
pandas==2.2.3              # Data manipulation
numpy==2.2.0               # Numerical computing
matplotlib==3.10.0         # Visualizations
imbalanced-learn==0.12.4   # Class balancing (RandomUnderSampler)
```

---

## Running Time Estimates

| Task | Time | Notes |
|------|------|-------|
| Classical models | ~5 sec | Very fast |
| Quantum models (Q-SVM + Q-KNN) | ~2 min | Moderate |
| Full quantum VQC | ~10 min | Slow (optimizer iterations) |
| Circuit diagrams | ~10 sec | Fast |
| Final comparison | ~1 sec | Uses pre-computed results |

Total time to reproduce all results: **~15 minutes**

---

## Notes

- Quantum simulations are computationally expensive on classical hardware
- Real quantum computers would be faster but have higher error rates (NISQ)
- Results demonstrate hybrid quantum methods can compete with classical ML
- VQC shows promise but needs fault-tolerant quantum computers for better performance

---

## References

### Quantum Kernel Methods & Q-SVM

1. **Havlíček, V., Córcoles, A. D., Temme, K., Harrow, A. W., Kandala, A., Chow, J. M., & Gambetta, J. M.** (2019). *Supervised learning with quantum-enhanced feature spaces*. **Nature**, 567(7747), 209-212.
   [https://doi.org/10.1038/s41586-019-0980-2](https://doi.org/10.1038/s41586-019-0980-2) | [arXiv:1804.11326](https://arxiv.org/abs/1804.11326)
   **Key contribution**: Introduced quantum kernel methods using quantum feature maps to compute kernel matrices on quantum computers, demonstrating quantum advantage for classification tasks.

2. **Schuld, M., & Killoran, N.** (2019). *Quantum machine learning in feature Hilbert spaces*. **Physical Review Letters**, 122(4), 040504.
   [https://doi.org/10.1103/PhysRevLett.122.040504](https://doi.org/10.1103/PhysRevLett.122.040504) | [arXiv:1803.07128](https://arxiv.org/abs/1803.07128)
   **Key contribution**: Theoretical framework connecting quantum circuits to kernel methods in reproducing kernel Hilbert spaces.

### Quantum k-Nearest Neighbors (Q-KNN)

3. **Afham, A., Basheer, S., & Gujrati, S.** (2020). *Quantum k-nearest neighbors algorithm*. [arXiv:2003.09187](https://arxiv.org/abs/2003.09187)
   **Key contribution**: Fidelity-based distance metric for Q-KNN: `d(x,y) = √(1 - |⟨ψ(x)|ψ(y)⟩|²)`

4. **Ruan, Y., et al.** (2017). *Quantum algorithm for k-nearest neighbors classification based on the metric of Hamming distance*. **International Journal of Theoretical Physics**, 56(11), 3496-3507.
   [https://doi.org/10.1007/s10773-017-3514-4](https://doi.org/10.1007/s10773-017-3514-4)

### Variational Quantum Classifier (VQC)

5. **Mitarai, K., Negoro, M., Kitagawa, M., & Fujii, K.** (2018). *Quantum circuit learning*. **Physical Review A**, 98(3), 032309.
   [https://doi.org/10.1103/PhysRevA.98.032309](https://doi.org/10.1103/PhysRevA.98.032309) | [arXiv:1803.00745](https://arxiv.org/abs/1803.00745)
   **Key contribution**: Proposed variational quantum circuits with trainable parameters for classification, foundational to VQC.

6. **Schuld, M., Bocharov, A., Svore, K. M., & Wiebe, N.** (2020). *Circuit-centric quantum classifiers*. **Physical Review A**, 101(3), 032308.
   [https://doi.org/10.1103/PhysRevA.101.032308](https://doi.org/10.1103/PhysRevA.101.032308) | [arXiv:1804.00633](https://arxiv.org/abs/1804.00633)
   **Key contribution**: Explored circuit architectures and training strategies for quantum classifiers.

### Barren Plateaus & Optimization Challenges

7. **McClean, J. R., Boixo, S., Smelyanskiy, V. N., Babbush, R., & Neven, H.** (2018). *Barren plateaus in quantum neural network training landscapes*. **Nature Communications**, 9(1), 4812.
   [https://doi.org/10.1038/s41467-018-07090-4](https://doi.org/10.1038/s41467-018-07090-4) | [arXiv:1803.11173](https://arxiv.org/abs/1803.11173)
   **Key contribution**: Identified that gradients vanish exponentially in randomly initialized deep quantum circuits, explaining VQC optimization difficulties.

8. **Grant, E., Wossnig, L., Ostaszewski, M., & Benedetti, M.** (2019). *An initialization strategy for addressing barren plateaus in parametrized quantum circuits*. **Quantum**, 3, 214.
   [https://doi.org/10.22331/q-2019-12-09-214](https://doi.org/10.22331/q-2019-12-09-214) | [arXiv:1903.05076](https://arxiv.org/abs/1903.05076)
   **Key contribution**: Proposed identity-block initialization to mitigate barren plateaus.

### Dataset

9. **Sharafaldin, I., Lashkari, A. H., & Ghorbani, A. A.** (2018). *Toward generating a new intrusion detection dataset and intrusion traffic characterization*. **4th International Conference on Information Systems Security and Privacy (ICISSP)**, 108-116.
   **Dataset**: [CICIDS2017 on Kaggle](https://www.kaggle.com/datasets/ericanacletoribeiro/cicids2017-cleaned-and-preprocessed)

### Quantum Computing Framework

10. **Qiskit Contributors** (2024). *Qiskit: An Open-source Framework for Quantum Computing*.
    [https://qiskit.org](https://qiskit.org) | [GitHub](https://github.com/Qiskit/qiskit)

---

## Implementation Notes

Our implementations are based on:
- **Q-SVM & Q-KNN**: Havlíček et al.'s quantum feature map approach [1], adapted with 6-layer circuit architecture
- **VQC**: Mitarai et al.'s quantum circuit learning [5], optimized with Adam and early stopping
- **Quantum distance metric**: Fidelity-based approach from Afham et al. [3]: `d(x,y) = √(1 - K(x,y))`
- **Optimization challenges**: VQC performance limited by barren plateaus (McClean et al. [7]), explaining 60% F1-score ceiling

The 6-layer quantum feature map includes:
1. Amplitude encoding (RY rotations)
2. Linear chain entanglement (CZ gates)
3. **Feature interactions** (RZ with pairwise products) - inspired by data re-uploading techniques
4. Star pattern entanglement
5. Re-encoding layer
6. Ring closure

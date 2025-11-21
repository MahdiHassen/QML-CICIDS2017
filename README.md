# Quantum Machine Learning for Network Intrusion Detection

A comparative study of classical and quantum machine learning approaches for binary classification on the CICIDS2017 network intrusion detection dataset. This work evaluates five algorithms: classical SVM and K-NN, hybrid quantum Q-SVM and Q-KNN, and a fully quantum Variational Quantum Classifier (VQC).

## Abstract

This repository implements and compares five machine learning algorithms for network intrusion detection: two classical baselines (Support Vector Machine and k-Nearest Neighbors), two hybrid quantum-classical approaches (Quantum SVM and Quantum K-NN), and one end-to-end quantum approach (Variational Quantum Classifier). Results demonstrate that hybrid quantum algorithms achieve competitive performance with classical methods (96.42% vs 96.48% F1-score), while fully quantum approaches face limitations in the current NISQ era (60.19% F1-score).

## Installation

### Prerequisites
- Python 3.8+
- 8GB RAM minimum
- CICIDS2017 dataset (download separately)

### Setup Instructions

```bash
# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Dataset Acquisition

Download the CICIDS2017 cleaned dataset from [Kaggle](https://www.kaggle.com/datasets/ericanacletoribeiro/cicids2017-cleaned-and-preprocessed) and place `cicids2017_cleaned.csv` in the project root directory.

## Reproduction Instructions

### Quick Start: Pre-computed Results

Generate final comparison visualization using pre-computed metrics:
```bash
python final_comparison.py
```
Output: `comparison/results/visualizations/final_comparison_all_models.png`

### Training Individual Models

**Classical Models** (approximate runtime: 5 seconds)
```bash
cd classical_models
python train_classical_models.py
```

**Hybrid Quantum Models** (approximate runtime: 2 minutes)
```bash
cd quantum_models
python train_quantum_models.py
```

**Full Quantum VQC** (approximate runtime: 10 minutes)
```bash
python full_quantum_improved.py
```

### Circuit Visualization

Generate quantum circuit diagrams:
```bash
python generate_circuit_diagrams.py
```
Output: `circuits/final/*.png`

## Methodology

### Classical Approaches

#### Support Vector Machine (SVM)
Implements a classical SVM with Radial Basis Function (RBF) kernel for non-linear classification. The kernel function `K(x,y) = exp(-γ||x-y||²)` projects data into a high-dimensional feature space where a linear hyperplane optimally separates attack and normal traffic patterns.

**Configuration:**
```python
SVC(kernel='rbf', gamma='scale', C=10.0, class_weight='balanced')
```

**Implementation:** `classical_models/train_classical_models.py`

#### k-Nearest Neighbors (K-NN)
Instance-based learning algorithm that classifies samples based on majority voting among k=5 nearest neighbors in feature space, using Minkowski distance metric.

**Configuration:**
```python
KNeighborsClassifier(n_neighbors=5, metric='minkowski')
```

**Implementation:** `classical_models/train_classical_models.py`

### Hybrid Quantum-Classical Approaches

#### Quantum Support Vector Machine (Q-SVM)
Hybrid algorithm that employs quantum computing for kernel estimation while retaining classical SVM optimization. The quantum component maps classical data to quantum states using a parameterized quantum circuit, computing kernels as inner products in Hilbert space.

**Quantum Kernel:**
```
K(x,y) = |⟨ψ(x)|ψ(y)⟩|²
```

where `|ψ(x)⟩ = U_φ(x)|0⟩^n` represents the quantum feature map.

**Quantum Feature Map Architecture:**

The 6-layer quantum circuit implements:

1. **Amplitude Encoding:** `RY(π·x[i])` - Encodes classical features as qubit rotation angles
2. **Linear Entanglement:** `CZ(i, i+1)` - Creates pairwise qubit correlations
3. **Feature Interactions:** `RZ(π·x[i]·x[(i+1) mod n])` - Encodes non-linear feature products
4. **Star Entanglement:** `CZ(0, i)` - Establishes global qubit correlations
5. **Re-encoding:** `RY(π/2·x[i])` - Secondary feature embedding
6. **Ring Closure:** `CZ(n-1, 0)` - Completes entanglement topology

**Circuit Diagram:** `circuits/final/qsvm_qknn_feature_map_6qubits.png`

**Implementation:** `quantum_models/train_quantum_models.py`

#### Quantum k-Nearest Neighbors (Q-KNN)
Extends the quantum kernel approach to distance-based classification. Uses the same quantum feature map as Q-SVM but converts quantum kernels to distance metrics for k-NN classification.

**Quantum Distance Metric:**
```
d(x,y) = √(1 - K(x,y)) = √(1 - |⟨ψ(x)|ψ(y)⟩|²)
```

This fidelity-based distance naturally captures similarity in quantum state space, enabling quantum-enhanced nearest neighbor search.

**Implementation:** `quantum_models/train_quantum_models.py`

### Full Quantum Approach

#### Variational Quantum Classifier (VQC)
End-to-end quantum machine learning model consisting of a feature encoding circuit followed by a trainable variational ansatz. The circuit parameters are optimized via gradient descent to minimize cross-entropy loss.

**Architecture:**

```python
# Feature encoding (identical to Q-SVM)
|ψ(x)⟩ = U_φ(x)|0⟩^n

# Variational ansatz (2 layers)
for layer in [1, 2]:
    for qubit in range(n_qubits):
        RY(θ_y[qubit, layer])
        RZ(θ_z[qubit, layer])
    apply_entanglement()

# Measurement in computational basis
measure_all()
```

**Training:** Adam optimizer with learning rate 0.1, batch size 25, 60 epochs with early stopping.

**Circuit Diagram:** `circuits/final/vqc_complete_6qubits.png`

**Implementation:** `full_quantum_improved.py`

## Experimental Results

### Performance Comparison

| Algorithm | Type | F1-Score | Training Time | Architecture |
|-----------|------|----------|---------------|--------------|
| K-NN | Classical | 96.48% | 0.09s | N/A |
| Q-KNN | Hybrid Quantum | 96.42% | 28.5s | 6-layer feature map |
| Q-SVM | Hybrid Quantum | 91.11% | 62.0s | 6-layer feature map |
| SVM | Classical | 90.80% | 3.26s | N/A |
| VQC | Full Quantum | 60.19% | 547.3s | Feature map + variational |

### Key Observations

1. **Classical Baseline:** K-NN achieves highest F1-score (96.48%) with minimal training overhead.

2. **Hybrid Quantum Competitiveness:** Q-KNN performance approaches classical K-NN (0.06% difference), demonstrating near-term quantum utility.

3. **Quantum-Classical Gap:** Hybrid approaches outperform SVM baseline, with Q-SVM achieving 91.11% F1-score.

4. **NISQ Limitations:** Full quantum VQC performance constrained to 60.19% F1-score due to barren plateau phenomena and limited qubit count.

5. **Computational Trade-off:** Quantum simulation overhead results in 300x longer training times compared to classical methods.

## Technical Specifications

### Dataset Preprocessing

**Dataset:** CICIDS2017 (2.5M samples, 52 features)
**Task:** Binary classification (Normal vs Attack traffic)

**Preprocessing Pipeline:**
1. **Class Balancing:** RandomUnderSampler (50/50 distribution, 33,776 samples)
2. **Feature Selection:** SelectKBest with mutual information criterion
3. **Normalization:**
   - Classical models: StandardScaler (zero mean, unit variance)
   - Quantum models: MinMaxScaler (range [0,1] for angle encoding)
4. **Train-Test Split:** 80/20 stratified split

### Quantum Simulation

**Simulator:** Qiskit AerSimulator (statevector method)
**Qubit Count:** 6 qubits (64-dimensional Hilbert space)
**Justification:** NISQ-era hardware limitations; real quantum devices exhibit high error rates
**Computational Complexity:** O(2^n) for n-qubit simulation on classical hardware

### Evaluation Metrics

**Primary Metric:** F1-score (harmonic mean of precision and recall)
**Justification:** Addresses class imbalance sensitivity
**Secondary Metrics:** Accuracy, training time, inference time

## Repository Structure

```
.
├── README.md                       # Documentation
├── requirements.txt                # Python dependencies
├── .gitignore                     # Git exclusions
├── cicids2017_cleaned.csv         # Dataset (user-provided)
│
├── classical_models/
│   ├── train_classical_models.py  # SVM and K-NN implementation
│   └── results/                   # Training outputs
│       ├── classical_results.txt
│       └── visualizations/
│
├── quantum_models/
│   ├── train_quantum_models.py    # Q-SVM and Q-KNN implementation
│   └── results/                   # Training outputs
│       ├── quantum_results.txt
│       └── visualizations/
│
├── circuits/
│   └── final/                     # Quantum circuit diagrams (PNG)
│       ├── qsvm_qknn_feature_map_4qubits.png
│       ├── qsvm_qknn_feature_map_6qubits.png
│       ├── vqc_complete_6qubits.png
│       └── quantum_approaches_comparison.png
│
├── comparison/
│   └── results/
│       ├── FINAL_RESULTS.txt
│       └── visualizations/
│           └── final_comparison_all_models.png
│
├── full_quantum_improved.py       # VQC implementation
├── final_comparison.py            # Aggregate results visualization
└── generate_circuit_diagrams.py   # Circuit diagram generation
```

## Dependencies

Core libraries (see `requirements.txt` for versions):

- **Quantum Computing:** qiskit, qiskit-aer
- **Machine Learning:** scikit-learn, imbalanced-learn
- **Numerical Computing:** numpy, scipy
- **Data Manipulation:** pandas
- **Visualization:** matplotlib, seaborn
- **Utilities:** pylatexenc (circuit rendering)

## References

### Quantum Kernel Methods

**[1]** Havlíček, V., Córcoles, A. D., Temme, K., Harrow, A. W., Kandala, A., Chow, J. M., & Gambetta, J. M. (2019). Supervised learning with quantum-enhanced feature spaces. *Nature*, 567(7747), 209-212. [doi:10.1038/s41586-019-0980-2](https://doi.org/10.1038/s41586-019-0980-2)

**[2]** Schuld, M., & Killoran, N. (2019). Quantum machine learning in feature Hilbert spaces. *Physical Review Letters*, 122(4), 040504. [doi:10.1103/PhysRevLett.122.040504](https://doi.org/10.1103/PhysRevLett.122.040504)

### Quantum Distance Metrics

**[3]** Afham, A., Basheer, S., & Gujrati, S. (2020). Quantum k-nearest neighbors algorithm. [arXiv:2003.09187](https://arxiv.org/abs/2003.09187)

**[4]** Ruan, Y., Xue, X., Liu, H., Tan, J., & Li, X. (2017). Quantum algorithm for k-nearest neighbors classification based on the metric of Hamming distance. *International Journal of Theoretical Physics*, 56(11), 3496-3507. [doi:10.1007/s10773-017-3514-4](https://doi.org/10.1007/s10773-017-3514-4)

### Variational Quantum Algorithms

**[5]** Mitarai, K., Negoro, M., Kitagawa, M., & Fujii, K. (2018). Quantum circuit learning. *Physical Review A*, 98(3), 032309. [doi:10.1103/PhysRevA.98.032309](https://doi.org/10.1103/PhysRevA.98.032309)

**[6]** Schuld, M., Bocharov, A., Svore, K. M., & Wiebe, N. (2020). Circuit-centric quantum classifiers. *Physical Review A*, 101(3), 032308. [doi:10.1103/PhysRevA.101.032308](https://doi.org/10.1103/PhysRevA.101.032308)

### Optimization Challenges

**[7]** McClean, J. R., Boixo, S., Smelyanskiy, V. N., Babbush, R., & Neven, H. (2018). Barren plateaus in quantum neural network training landscapes. *Nature Communications*, 9(1), 4812. [doi:10.1038/s41467-018-07090-4](https://doi.org/10.1038/s41467-018-07090-4)

**[8]** Grant, E., Wossnig, L., Ostaszewski, M., & Benedetti, M. (2019). An initialization strategy for addressing barren plateaus in parametrized quantum circuits. *Quantum*, 3, 214. [doi:10.22331/q-2019-12-09-214](https://doi.org/10.22331/q-2019-12-09-214)

### Dataset

**[9]** Sharafaldin, I., Lashkari, A. H., & Ghorbani, A. A. (2018). Toward generating a new intrusion detection dataset and intrusion traffic characterization. *4th International Conference on Information Systems Security and Privacy (ICISSP)*, 108-116.

### Framework

**[10]** Qiskit Contributors. (2024). Qiskit: An Open-source Framework for Quantum Computing. [https://qiskit.org](https://qiskit.org)

## Implementation Details

### Quantum Feature Map Design

Our 6-layer quantum feature map extends the approach of Havlíček et al. [1] with additional entanglement patterns:

- **Layers 1-2:** Standard amplitude encoding with linear entanglement
- **Layer 3:** Non-linear feature interactions via parameterized RZ gates
- **Layers 4-6:** Enhanced entanglement topology (star and ring patterns)

### Variational Quantum Classifier Training

VQC implementation follows Mitarai et al. [5] with modifications:

- **Optimizer:** Adam with adaptive learning rate
- **Regularization:** Early stopping (patience=5 epochs)
- **Batch Processing:** Mini-batch gradient descent (batch size=25)
- **Gradient Computation:** Finite difference method

### Barren Plateau Mitigation

VQC performance limited by barren plateau phenomenon [7]. Observed gradient vanishing beyond 2 variational layers, resulting in 60% F1-score ceiling. Future work may explore:

- Identity block initialization [8]
- Layerwise training protocols
- Alternative ansatz architectures

## Performance Benchmarks

Total reproduction time: Approximately 15 minutes

| Task | Duration | Hardware Requirement |
|------|----------|---------------------|
| Classical training | 5 seconds | CPU |
| Hybrid quantum training | 2 minutes | CPU (simulation) |
| VQC training | 10 minutes | CPU (simulation) |
| Circuit generation | 10 seconds | CPU |
| Results aggregation | 1 second | CPU |

## Limitations and Future Work

1. **Simulation Overhead:** Classical simulation of quantum circuits introduces computational bottleneck; deployment on actual quantum hardware would reduce training time.

2. **Qubit Scalability:** Limited to 6 qubits due to exponential memory requirements (64 complex amplitudes); fault-tolerant quantum computers could enable deeper circuits.

3. **Barren Plateaus:** VQC gradient vanishing constrains trainability; advanced initialization strategies may improve convergence.

4. **NISQ Constraints:** Current quantum hardware error rates prohibit direct deployment; error mitigation techniques required for near-term implementation.

5. **Dataset Specificity:** Results specific to CICIDS2017 network intrusion patterns; generalization to other domains requires validation.

## License

This project is available under the MIT License. See LICENSE file for details.

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{qml_cicids2017,
  title={Quantum Machine Learning for Network Intrusion Detection},
  author={},
  year={2024},
  url={https://github.com/MahdiHassen/QML-CICIDS2017}
}
```

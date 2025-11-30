# Quantum Machine Learning for Network Intrusion Detection

Comparative study of classical and quantum machine learning for binary classification on CICIDS2017.

## Quick Start

### 1. Setup Environment

```bash
# Clone repository
git clone https://github.com/MahdiHassen/QML-CICIDS2017.git
cd QML-CICIDS2017

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Download Dataset

Download `cicids2017_cleaned.csv` from [Kaggle](https://www.kaggle.com/datasets/ericanacletoribeiro/cicids2017-cleaned-and-preprocessed) and place it in the project root directory.

### 3. Run All Models

```bash
# Activate virtual environment (if not already active)
source venv/bin/activate

# Run Classical Models (SVM + KNN) - ~5 seconds
cd classical_models
python train_classical_models.py
cd ..

# Run Hybrid Quantum Models (Q-SVM + Q-KNN) - ~2-5 minutes
cd quantum_models
python train_quantum_models.py
cd ..

# Run Full Quantum VQC - ~10 minutes
python full_quantum_improved.py

# Generate Final Comparison Graph
python final_comparison.py

# Generate Quantum Circuit Diagrams
python generate_circuit_diagrams.py

# Generate Comprehensive F1 Comparison
python comprehensive_f1_comparison.py
```

## Output Files

| Script | Output Location |
|--------|-----------------|
| `train_classical_models.py` | `classical_models/results/classical_results.txt` |
| `train_quantum_models.py` | `quantum_models/results/quantum_results.txt` |
| `full_quantum_improved.py` | `comparison/results/improved_vqc_results.txt` |
| `final_comparison.py` | `comparison/results/visualizations/final_comparison_all_models.png` |
| `generate_circuit_diagrams.py` | `circuits/final/*.png` |
| `comprehensive_f1_comparison.py` | `comprehensive_f1_comparison.png` |

## Results Summary

| Algorithm | Type | F1-Score | Qubits/Features |
|-----------|------|----------|-----------------|
| Classical KNN | Classical | 98.12% | 12 features |
| Q-KNN | Hybrid Quantum | 97.38% | 8 qubits |
| Classical SVM | Classical | 93.61% | 12 features |
| Q-SVM | Hybrid Quantum | 93.17% | 12 qubits |
| SVDD | Classical | 87.40% | - |
| Q-SVDD | Quantum | 67.50% | - |
| VQC | Full Quantum | 60.19% | 4 qubits |

### SVDD Results

**Classical SVDD (Support Vector Data Description):**
- One-class classification approach for anomaly detection
- F1-Score: 87.40%

**Quantum SVDD (Q-SVDD):**
- Quantum kernel-based one-class classification
- F1-Score: 67.50%
- Uses quantum feature map for kernel computation

## Appendix

### A. Python Module Versions

```
qiskit == 1.4.5
qiskit_aer == 0.17.2
scikit-learn == 1.7.2
imbalanced-learn == 0.14.0
numpy == 2.3.5
pandas == 2.3.3
matplotlib == 3.10.7
```

To verify versions:
```python
import qiskit; print(f'qiskit == {qiskit.__version__}')
import qiskit_aer; print(f'qiskit_aer == {qiskit_aer.__version__}')
import sklearn; print(f'scikit-learn == {sklearn.__version__}')
import imblearn; print(f'imbalanced-learn == {imblearn.__version__}')
import numpy; print(f'numpy == {numpy.__version__}')
import pandas; print(f'pandas == {pandas.__version__}')
import matplotlib; print(f'matplotlib == {matplotlib.__version__}')
```

### B. Hardware Specifications

- **CPU:** Apple M2 (8 cores)
- **RAM:** 16GB
- **GPU:** Not used (quantum simulation is CPU-based)
- **Quantum Hardware:** None (all circuits simulated using Qiskit AerSimulator)

### C. Runtime Estimates

| Task | Estimated Time |
|------|----------------|
| Classical Models (SVM + KNN) | ~5 seconds |
| Hybrid Quantum Models (Q-SVM + Q-KNN) | ~2-5 minutes |
| Full Quantum VQC | ~10 minutes |
| Circuit Diagram Generation | ~10 seconds |
| **Total** | **~15 minutes** |

## Repository Structure

```
QML-CICIDS2017/
├── README.md
├── requirements.txt
├── cicids2017_cleaned.csv          # Dataset (download separately)
├── classical_models/
│   ├── train_classical_models.py   # SVM + KNN
│   └── results/
├── quantum_models/
│   ├── train_quantum_models.py     # Q-SVM + Q-KNN
│   └── results/
├── circuits/final/                 # Circuit diagrams
├── comparison/results/             # Comparison results
├── full_quantum_improved.py        # VQC
├── final_comparison.py             # Final comparison plot
├── generate_circuit_diagrams.py    # Circuit diagrams
└── comprehensive_f1_comparison.py  # F1 comparison bar chart
```

## Troubleshooting

**ModuleNotFoundError:** Run `pip install -r requirements.txt`

**Dataset not found:** Ensure `cicids2017_cleaned.csv` is in project root

**Virtual environment:** Run `source venv/bin/activate` (macOS/Linux) or `venv\Scripts\activate` (Windows)

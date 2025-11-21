"""
Classical Models Training: SVM + K-NN
Network Intrusion Detection on CICIDS2017

This script trains and evaluates classical SVM and K-NN models
with 4, 6, 8, and 12 features for comparison with quantum models.
"""

import pandas as pd
import numpy as np
import time
import os
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from imblearn.under_sampling import RandomUnderSampler
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Setup paths (relative to current directory when run from classical_models/)
RESULTS_DIR = 'results'
VIZ_DIR = f'{RESULTS_DIR}/visualizations'
os.makedirs(VIZ_DIR, exist_ok=True)

print("="*80)
print("Classical Models: SVM + K-NN for Network Intrusion Detection")
print("="*80)

# ============================================================================
# Load and Prepare Dataset
# ============================================================================
print("\n[1] Loading dataset...")
df = pd.read_csv('../cicids2017_cleaned.csv')

X = df.drop('Attack Type', axis=1)
y_binary = (df['Attack Type'] != 'Normal Traffic').astype(int)

print(f"Dataset: {len(X):,} samples, {X.shape[1]} features")
print(f"Classes: Normal={sum(y_binary==0):,}, Attack={sum(y_binary==1):,}")

# Sample and balance
print("\n[2] Balancing dataset...")
initial_sample = 100000
sss = StratifiedShuffleSplit(n_splits=1, train_size=initial_sample, random_state=42)
for sample_idx, _ in sss.split(X, y_binary):
    X_initial = X.iloc[sample_idx]
    y_initial = y_binary.iloc[sample_idx]

rus = RandomUnderSampler(sampling_strategy=1.0, random_state=42)
X_balanced, y_balanced = rus.fit_resample(X_initial, y_initial)

print(f"Balanced: {len(X_balanced):,} samples (50/50 split)")

# ============================================================================
# Train Models with Different Feature Counts
# ============================================================================
feature_counts = [4, 6, 8, 12]
results = []

for n_features in feature_counts:
    print(f"\n{'='*80}")
    print(f"Training with {n_features} features")
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

    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # SVM
    print(f"\nTraining SVM...")
    t0 = time.time()
    svm_model = SVC(kernel='rbf', gamma='scale', C=10.0, class_weight='balanced')
    svm_model.fit(X_train_scaled, y_train)
    y_pred_svm = svm_model.predict(X_test_scaled)
    svm_time = time.time() - t0

    svm_acc = accuracy_score(y_test, y_pred_svm)
    svm_f1 = f1_score(y_test, y_pred_svm, average='binary')

    print(f"  SVM: Accuracy={svm_acc:.4f}, F1={svm_f1:.4f}, Time={svm_time:.2f}s")

    # K-NN
    print(f"Training K-NN...")
    t0 = time.time()
    knn_model = KNeighborsClassifier(n_neighbors=5, metric='minkowski')
    knn_model.fit(X_train_scaled, y_train)
    y_pred_knn = knn_model.predict(X_test_scaled)
    knn_time = time.time() - t0

    knn_acc = accuracy_score(y_test, y_pred_knn)
    knn_f1 = f1_score(y_test, y_pred_knn, average='binary')

    print(f"  K-NN: Accuracy={knn_acc:.4f}, F1={knn_f1:.4f}, Time={knn_time:.2f}s")

    # Store results
    results.append({
        'n_features': n_features,
        'features': list(selected_features),
        'svm_acc': svm_acc,
        'svm_f1': svm_f1,
        'svm_time': svm_time,
        'knn_acc': knn_acc,
        'knn_f1': knn_f1,
        'knn_time': knn_time,
    })

# ============================================================================
# Summary
# ============================================================================
print("\n" + "="*80)
print("CLASSICAL MODELS SUMMARY")
print("="*80)

print(f"\n{'Features':<10} {'SVM F1':<12} {'KNN F1':<12} {'Best':<15}")
print("-"*80)
for r in results:
    best = "K-NN" if r['knn_f1'] > r['svm_f1'] else "SVM"
    print(f"{r['n_features']:<10} {r['svm_f1']:<12.4f} {r['knn_f1']:<12.4f} {best:<15}")

# ============================================================================
# Visualization
# ============================================================================
print(f"\n[3] Creating visualizations...")

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Classical Models Performance', fontsize=16, fontweight='bold')

features = [r['n_features'] for r in results]
svm_f1 = [r['svm_f1'] for r in results]
knn_f1 = [r['knn_f1'] for r in results]
svm_time = [r['svm_time'] for r in results]
knn_time = [r['knn_time'] for r in results]

# F1-Score comparison
ax1.plot(features, svm_f1, marker='s', linewidth=2, markersize=8, label='SVM', color='#1976D2')
ax1.plot(features, knn_f1, marker='^', linewidth=2, markersize=8, label='K-NN', color='#C62828')
ax1.set_xlabel('Features', fontweight='bold')
ax1.set_ylabel('F1-Score', fontweight='bold')
ax1.set_title('F1-Score Comparison')
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.set_xticks(features)

# Bar chart
x = np.arange(len(features))
width = 0.35
ax2.bar(x - width/2, svm_f1, width, label='SVM', color='#1976D2', alpha=0.8)
ax2.bar(x + width/2, knn_f1, width, label='K-NN', color='#C62828', alpha=0.8)
ax2.set_xlabel('Features', fontweight='bold')
ax2.set_ylabel('F1-Score', fontweight='bold')
ax2.set_title('F1-Score (Bar Chart)')
ax2.set_xticks(x)
ax2.set_xticklabels(features)
ax2.legend()
ax2.grid(True, alpha=0.3, axis='y')

# Training time
ax3.bar(x - width/2, svm_time, width, label='SVM', color='#1976D2', alpha=0.8)
ax3.bar(x + width/2, knn_time, width, label='K-NN', color='#C62828', alpha=0.8)
ax3.set_xlabel('Features', fontweight='bold')
ax3.set_ylabel('Time (seconds)', fontweight='bold')
ax3.set_title('Training Time')
ax3.set_xticks(x)
ax3.set_xticklabels(features)
ax3.legend()
ax3.grid(True, alpha=0.3, axis='y')

# Best model per feature count
winners = []
for r in results:
    winners.append('K-NN' if r['knn_f1'] > r['svm_f1'] else 'SVM')

knn_wins = winners.count('K-NN')
svm_wins = winners.count('SVM')

ax4.pie([knn_wins, svm_wins], labels=[f'K-NN\n({knn_wins}/4)', f'SVM\n({svm_wins}/4)'],
        autopct='%1.0f%%', colors=['#C62828', '#1976D2'], startangle=90)
ax4.set_title('Winner Distribution')

plt.tight_layout()
plt.savefig(f'{VIZ_DIR}/classical_models_summary.png', dpi=300, bbox_inches='tight')
print(f"Saved: {VIZ_DIR}/classical_models_summary.png")

# ============================================================================
# Save Results
# ============================================================================
print(f"\n[4] Saving results...")

with open(f'{RESULTS_DIR}/classical_results.txt', 'w') as f:
    f.write("Classical Models Results - SVM + K-NN\n")
    f.write("="*80 + "\n\n")

    for r in results:
        f.write(f"{r['n_features']} Features:\n")
        f.write(f"  Features: {', '.join(r['features'])}\n")
        f.write(f"  SVM:  F1={r['svm_f1']:.4f}, Acc={r['svm_acc']:.4f}, Time={r['svm_time']:.2f}s\n")
        f.write(f"  K-NN: F1={r['knn_f1']:.4f}, Acc={r['knn_acc']:.4f}, Time={r['knn_time']:.2f}s\n")
        f.write("\n")

    f.write(f"Overall:\n")
    f.write(f"  K-NN wins: {knn_wins}/4\n")
    f.write(f"  SVM wins: {svm_wins}/4\n")
    f.write(f"  Best K-NN F1: {max(knn_f1):.4f}\n")
    f.write(f"  Best SVM F1: {max(svm_f1):.4f}\n")

print(f"Saved: {RESULTS_DIR}/classical_results.txt")

print("\n" + "="*80)
print("CLASSICAL MODELS TRAINING COMPLETE!")
print("="*80)
print(f"\nResults saved in: {RESULTS_DIR}/")
print(f"Visualizations saved in: {VIZ_DIR}/")

"""
Create separate F1-score graphs for Classical KNN and Classical SVM
"""

import matplotlib.pyplot as plt
import numpy as np

# Data from training results
features = [4, 6, 8, 12]
svm_f1 = [0.8580, 0.9080, 0.9300, 0.9361]
knn_f1 = [0.9232, 0.9558, 0.9797, 0.9812]
svdd_f1 = [0.8742, 0.8329, 0.8309, 0.8046]
PATH = 'classical_models/results/visualizations/'

# ============================================================================
# Classical KNN Graph
# ============================================================================
fig1, ax1 = plt.subplots(figsize=(10, 7))

# Plot line with markers
ax1.plot(features, knn_f1, marker='o', linewidth=3, markersize=12,
         color='#C62828', label='K-NN F1-Score', markeredgecolor='white',
         markeredgewidth=2, linestyle='-', alpha=0.9)

# Add value labels on points
for x, y in zip(features, knn_f1):
    ax1.annotate(f'{y:.4f}',
                xy=(x, y),
                xytext=(0, 10),
                textcoords='offset points',
                ha='center',
                fontsize=11,
                fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='white',
                         edgecolor='#C62828', alpha=0.8))

# Styling
ax1.set_xlabel('Number of Features', fontsize=14, fontweight='bold')
ax1.set_ylabel('F1-Score', fontsize=14, fontweight='bold')
ax1.set_title('Classical K-NN: F1-Score vs Number of Features\nCICIDS2017 Dataset',
             fontsize=16, fontweight='bold', pad=20)
ax1.grid(True, alpha=0.3, linestyle='--', linewidth=1)
ax1.set_xticks(features)
ax1.set_ylim(0.8, 1.0)
ax1.set_xlim(3, 13)

# Add performance annotations
improvement = ((knn_f1[-1] - knn_f1[0]) / knn_f1[0]) * 100
ax1.text(0.98, 0.02,
        f'Performance Gain:\n{knn_f1[0]:.4f} → {knn_f1[-1]:.4f}\n(+{improvement:.2f}%)',
        transform=ax1.transAxes,
        fontsize=10,
        verticalalignment='bottom',
        horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

# Highlight best result
best_idx = knn_f1.index(max(knn_f1))
ax1.scatter(features[best_idx], knn_f1[best_idx], s=300,
           facecolors='none', edgecolors='gold', linewidth=3, zorder=5)
ax1.text(features[best_idx], knn_f1[best_idx] - 0.015, 'BEST',
        ha='center', fontsize=9, fontweight='bold', color='gold',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7))

plt.tight_layout()
plt.savefig(PATH + 'classical_knn_f1_graph.png', dpi=300, bbox_inches='tight')
print("✓ Saved: classical_knn_f1_graph.png")

# ============================================================================
# Classical SVM Graph
# ============================================================================
fig2, ax2 = plt.subplots(figsize=(10, 7))

# Plot line with markers
ax2.plot(features, svm_f1, marker='s', linewidth=3, markersize=12,
         color='#1976D2', label='SVM F1-Score', markeredgecolor='white',
         markeredgewidth=2, linestyle='-', alpha=0.9)

# Add value labels on points
for x, y in zip(features, svm_f1):
    ax2.annotate(f'{y:.4f}',
                xy=(x, y),
                xytext=(0, 10),
                textcoords='offset points',
                ha='center',
                fontsize=11,
                fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='white',
                         edgecolor='#1976D2', alpha=0.8))

# Styling
ax2.set_xlabel('Number of Features', fontsize=14, fontweight='bold')
ax2.set_ylabel('F1-Score', fontsize=14, fontweight='bold')
ax2.set_title('Classical SVM: F1-Score vs Number of Features\nCICIDS2017 Dataset',
             fontsize=16, fontweight='bold', pad=20)
ax2.grid(True, alpha=0.3, linestyle='--', linewidth=1)
ax2.set_xticks(features)
ax2.set_ylim(0.8, 1.0)
ax2.set_xlim(3, 13)

# Add performance annotations
improvement_svm = ((svm_f1[-1] - svm_f1[0]) / svm_f1[0]) * 100
ax2.text(0.98, 0.02,
        f'Performance Gain:\n{svm_f1[0]:.4f} → {svm_f1[-1]:.4f}\n(+{improvement_svm:.2f}%)',
        transform=ax2.transAxes,
        fontsize=10,
        verticalalignment='bottom',
        horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

# Highlight best result
best_idx_svm = svm_f1.index(max(svm_f1))
ax2.scatter(features[best_idx_svm], svm_f1[best_idx_svm], s=300,
           facecolors='none', edgecolors='gold', linewidth=3, zorder=5)
ax2.text(features[best_idx_svm], svm_f1[best_idx_svm] - 0.015, 'BEST',
        ha='center', fontsize=9, fontweight='bold', color='gold',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7))

plt.tight_layout()
plt.savefig(PATH + 'classical_svm_f1_graph.png', dpi=300, bbox_inches='tight')
print("✓ Saved: classical_svm_f1_graph.png")

# ============================================================================
# SVDD Graph
# ============================================================================
fig2, ax2 = plt.subplots(figsize=(10, 7))

# Plot line with markers
ax2.plot(features, svdd_f1, marker='s', linewidth=3, markersize=12,
         color="#19D219", label='SVDD F1-Score', markeredgecolor='white',
         markeredgewidth=2, linestyle='-', alpha=0.9)

# Add value labels on points
for x, y in zip(features, svdd_f1):
    ax2.annotate(f'{y:.4f}',
                xy=(x, y),
                xytext=(0, 10),
                textcoords='offset points',
                ha='center',
                fontsize=11,
                fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='white',
                         edgecolor='#19D219', alpha=0.8))

# Styling
ax2.set_xlabel('Number of Features', fontsize=14, fontweight='bold')
ax2.set_ylabel('F1-Score', fontsize=14, fontweight='bold')
ax2.set_title('Classical SVDD: F1-Score vs Number of Features\nCICIDS2017 Dataset',
             fontsize=16, fontweight='bold', pad=20)
ax2.grid(True, alpha=0.3, linestyle='--', linewidth=1)
ax2.set_xticks(features)
ax2.set_ylim(0.8, 1.0)
ax2.set_xlim(3, 13)

# Add performance annotations
improvement_svdd = ((svdd_f1[-1] - svdd_f1[0]) / svdd_f1[0]) * 100
ax2.text(0.98, 0.02,
        f'Performance Gain:\n{svdd_f1[0]:.4f} → {svdd_f1[-1]:.4f}\n({improvement_svdd:.2f}%)',
        transform=ax2.transAxes,
        fontsize=10,
        verticalalignment='bottom',
        horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

# Highlight best result
best_idx_svdd = svdd_f1.index(max(svdd_f1))
ax2.scatter(features[best_idx_svdd], svdd_f1[best_idx_svdd], s=300,
           facecolors='none', edgecolors='gold', linewidth=3, zorder=5)
ax2.text(features[best_idx_svdd], svdd_f1[best_idx_svdd] - 0.015, 'BEST',
        ha='center', fontsize=9, fontweight='bold', color='gold',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7))

plt.tight_layout()
plt.savefig(PATH + 'classical_svdd_f1_graph.png', dpi=300, bbox_inches='tight')
print("✓ Saved: classical_svdd_f1_graph.png")

# ============================================================================
# Print Summary
# ============================================================================
print("\n" + "="*70)
print(" SUMMARY: F1-Score Performance vs Features")
print("="*70)
print(f"\nClassical K-NN:")
print(f"  4 features:  F1 = {knn_f1[0]:.4f}")
print(f"  6 features:  F1 = {knn_f1[1]:.4f}")
print(f"  8 features:  F1 = {knn_f1[2]:.4f}")
print(f"  12 features: F1 = {knn_f1[3]:.4f} ← BEST")
print(f"  Improvement: +{improvement:.2f}%")

print(f"\nClassical SVM:")
print(f"  4 features:  F1 = {svm_f1[0]:.4f}")
print(f"  6 features:  F1 = {svm_f1[1]:.4f}")
print(f"  8 features:  F1 = {svm_f1[2]:.4f}")
print(f"  12 features: F1 = {svm_f1[3]:.4f} ← BEST")
print(f"  Improvement: +{improvement_svm:.2f}%")

print(f"\nClassical SVDD:")
print(f"  4 features:  F1 = {svdd_f1[0]:.4f} ← BEST")
print(f"  6 features:  F1 = {svdd_f1[1]:.4f}")
print(f"  8 features:  F1 = {svdd_f1[2]:.4f}")
print(f"  12 features: F1 = {svdd_f1[3]:.4f} ")
print(f"  Improvement: +{improvement_svdd:.2f}%")

print(f"\nKey Insights:")
print(f"  • K-NN consistently outperforms SVM across all feature counts")
print(f"  • K-NN and SVM models show diminishing returns after 8 features")
print(f"  • K-NN achieves 98.12% F1 with just 12 features")
print(f"  • SVDD achieves its best F1 with just 4 features")
print(f"  • Biggest jump for SVM: 4→6 features (+5.83%)")
print(f"  • Biggest jump for K-NN: 4→6 features (+3.53%)")

print("\n" + "="*70)
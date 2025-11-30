"""
Comprehensive F1-Score Comparison: All Models
Classical vs Quantum Approaches for Network Intrusion Detection
"""

import matplotlib.pyplot as plt
import numpy as np

# Data: Best F1 scores from all experiments
models = [
    'Classical\nKNN',
    'Q-KNN',
    'Classical\nSVM',
    'Q-SVM',
    'SVDD\n(Classical)',
    'Q-SVDD',
    'VQC\n(Full Quantum)'
]

f1_scores = [
    0.9812,  # Classical KNN (12 features)
    0.9738,  # Q-KNN (8 qubits)
    0.9361,  # Classical SVM (12 features)
    0.9317,  # Q-SVM (12 qubits)
    0.874,   # SVDD (classical)
    0.675,   # Q-SVDD (quantum)
    0.60     # VQC (4 qubits) - improved version
]

# Features/Qubits info
info = [
    '12 features',
    '8 qubits',
    '12 features',
    '12 qubits',
    'Classical',
    'Quantum',
    '4 qubits'
]

# Color scheme: Classical (blue), Quantum (green/red for poor performance)
colors = [
    '#1976D2',  # Classical KNN - blue
    '#2E7D32',  # Q-KNN - green
    '#1976D2',  # Classical SVM - blue
    '#2E7D32',  # Q-SVM - green
    '#1976D2',  # SVDD - blue
    '#C62828',  # Q-SVDD - red (poor)
    '#C62828',  # VQC - red (poor)
]

# Create figure
fig, ax = plt.subplots(figsize=(14, 8))

# Create bars
x = np.arange(len(models))
bars = ax.bar(x, f1_scores, color=colors, alpha=0.8, edgecolor='black', linewidth=2, width=0.7)

# Add value labels on bars
for i, (bar, f1, model_info) in enumerate(zip(bars, f1_scores, info)):
    height = bar.get_height()

    # F1 score on top
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
            f'{f1:.4f}',
            ha='center', va='bottom', fontweight='bold', fontsize=11)

    # Features/Qubits info inside or below bar
    y_pos = height/2 if height > 0.6 else height + 0.05
    text_color = 'white' if height > 0.6 else 'black'
    ax.text(bar.get_x() + bar.get_width()/2., y_pos,
            model_info,
            ha='center', va='center', fontsize=9,
            fontweight='bold', color=text_color)

# Styling
ax.set_ylabel('F1-Score', fontweight='bold', fontsize=14)
ax.set_xlabel('Model', fontweight='bold', fontsize=14)
ax.set_title('Comprehensive Model Comparison: Network Intrusion Detection (CICIDS2017)\nBest F1-Scores Across All Experiments',
             fontsize=16, fontweight='bold', pad=20)
ax.set_xticks(x)
ax.set_xticklabels(models, fontsize=11, fontweight='bold')
ax.set_ylim(0, 1.05)
ax.grid(True, alpha=0.3, axis='y', linestyle='--')

# Add horizontal line at 0.9 for reference
ax.axhline(y=0.9, color='gray', linestyle='--', linewidth=1.5, alpha=0.5, label='0.9 threshold')

# Add legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#1976D2', edgecolor='black', label='Classical Models'),
    Patch(facecolor='#2E7D32', edgecolor='black', label='Quantum Models (Good)'),
    Patch(facecolor='#C62828', edgecolor='black', label='Quantum Models (Poor)')
]
ax.legend(handles=legend_elements, loc='lower left', fontsize=11, framealpha=0.9)

# Calculate rankings and stats for printing (but don't display on graph)
rankings = np.argsort(f1_scores)[::-1]  # Descending order
classical_avg = np.mean([f1_scores[0], f1_scores[2], f1_scores[4]])
quantum_avg = np.mean([f1_scores[1], f1_scores[3], f1_scores[5], f1_scores[6]])

plt.tight_layout()
plt.savefig('comprehensive_f1_comparison.png', dpi=300, bbox_inches='tight')
print("✓ Saved: comprehensive_f1_comparison.png")

# Print detailed results
print("\n" + "="*80)
print("COMPREHENSIVE F1-SCORE COMPARISON")
print("="*80)
print(f"\n{'Rank':<6} {'Model':<25} {'F1-Score':<12} {'Config':<15}")
print("-"*80)
for rank, idx in enumerate(rankings, 1):
    model_name = models[idx].replace('\n', ' ')
    print(f"{rank:<6} {model_name:<25} {f1_scores[idx]:<12.4f} {info[idx]:<15}")

print("\n" + "="*80)
print("KEY FINDINGS")
print("="*80)
print(f"🥇 Best Overall:     Classical KNN (F1 = {f1_scores[0]:.4f})")
print(f"🥈 Best Quantum:     Q-KNN (F1 = {f1_scores[1]:.4f})")
print(f"📊 Classical Avg:    {classical_avg:.4f}")
print(f"⚛️  Quantum Avg:      {quantum_avg:.4f}")
print(f"📉 Gap:              {(classical_avg - quantum_avg):.4f} ({(classical_avg - quantum_avg)/classical_avg*100:.1f}%)")
print("\nQuantum Advantage Cases:")
print(f"  • Q-KNN vs Classical SVM: {(f1_scores[1] - f1_scores[2])*100:+.2f}%")
print(f"  • Q-SVM vs Classical SVM: {(f1_scores[3] - f1_scores[2])*100:+.2f}%")
print(f"  • Q-KNN vs Classical KNN: {(f1_scores[1] - f1_scores[0])*100:+.2f}%")
print("\nPoor Performers:")
print(f"  • VQC: {f1_scores[6]:.4f} (needs better training)")
print(f"  • Q-SVDD: {f1_scores[5]:.4f} (worse than classical SVDD)")
print("="*80)

plt.show()

"""
Final Comparison: All 5 Approaches
Classical (SVM, K-NN) vs Quantum (Q-SVM, Q-KNN, VQC)
"""

import matplotlib.pyplot as plt
import numpy as np
import os

os.makedirs('comparison/results/visualizations', exist_ok=True)

print("="*80)
print("Creating Final Comparison Visualization")
print("="*80)

# Results from our experiments (6 features/qubits for fair comparison)
results = {
    'Classical SVM': {'f1': 0.9080, 'time': 3.26, 'type': 'classical'},
    'Classical K-NN': {'f1': 0.9648, 'time': 0.09, 'type': 'classical'},
    'Classical SVDD': {'f1': 0.8740, 'time': 0.97, 'type': 'classical'},
    'Fully Supervised QSVDD': {'f1': 0.6562, 'time': 84.3, 'type': 'full_quantum'},
    'Hybrid Q-SVM': {'f1': 0.9111, 'time': 62.0, 'type': 'hybrid'},
    'Hybrid Q-KNN': {'f1': 0.9642, 'time': 28.5, 'type': 'hybrid'},
    'Full Quantum VQC': {'f1': 0.6019, 'time': 547.3, 'type': 'full_quantum'}
}

# Create comprehensive comparison
fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

models = list(results.keys())
f1_scores = [results[m]['f1'] for m in models]
times = [results[m]['time'] for m in models]
colors = ['#1976D2' if results[m]['type'] == 'classical'
          else '#388E3C' if results[m]['type'] == 'hybrid'
          else '#7B1FA2' for m in models]

# 1. F1-Score Comparison
ax1 = fig.add_subplot(gs[0, :2])
bars1 = ax1.bar(models, f1_scores, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
ax1.set_ylabel('F1-Score', fontweight='bold', fontsize=12)
ax1.set_title('Performance Comparison (6 Features/Qubits)', fontsize=14, fontweight='bold')
ax1.set_ylim([0, 1.0])
ax1.grid(True, alpha=0.3, axis='y')
ax1.axhline(y=0.9, color='green', linestyle='--', alpha=0.5, label='90% threshold')

# Add value labels
for bar, val in zip(bars1, f1_scores):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height + 0.02,
            f'{val:.2%}', ha='center', va='bottom', fontweight='bold', fontsize=10)

ax1.legend()
ax1.tick_params(axis='x', rotation=15)

# 2. Training Time (log scale)
ax2 = fig.add_subplot(gs[0, 2])
bars2 = ax2.bar(range(len(models)), times, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
ax2.set_yscale('log')
ax2.set_xticks(range(len(models)))
ax2.set_xticklabels([m.replace(' ', '\n') for m in models], fontsize=8)
ax2.set_ylabel('Training Time (seconds, log scale)', fontweight='bold', fontsize=10)
ax2.set_title('Computational Cost', fontsize=12, fontweight='bold')
ax2.grid(True, alpha=0.3, which='both')

# Add time labels
for i, (bar, val) in enumerate(zip(bars2, times)):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height,
            f'{val:.1f}s', ha='center', va='bottom', fontsize=8, fontweight='bold')

# 3. Performance vs Cost scatter
ax3 = fig.add_subplot(gs[1, :2])
scatter = ax3.scatter(times, f1_scores, s=500, c=colors, alpha=0.7,
                     edgecolors='black', linewidths=2)

# Add labels
for i, model in enumerate(models):
    ax3.annotate(model, (times[i], f1_scores[i]),
                fontsize=9, ha='center', va='center', fontweight='bold')

ax3.set_xlabel('Training Time (seconds, log scale)', fontweight='bold', fontsize=12)
ax3.set_ylabel('F1-Score', fontweight='bold', fontsize=12)
ax3.set_title('Performance vs Computational Cost\n(Top-Left = Best: High Performance + Low Cost)',
             fontsize=13, fontweight='bold')
ax3.set_xscale('log')
ax3.grid(True, alpha=0.3)

# Highlight zones
ax3.axhspan(0.9, 1.0, alpha=0.1, color='green', label='High Performance (>90%)')
ax3.axvspan(0.01, 10, alpha=0.1, color='blue', label='Fast (<10s)')
ax3.legend(loc='lower right', fontsize=9)

# 4. Summary table
ax4 = fig.add_subplot(gs[1, 2])
ax4.axis('off')

table_data = [['Model', 'F1', 'Time', 'Type']]
for model in models:
    f1 = f'{results[model]["f1"]:.2%}'
    time = f'{results[model]["time"]:.1f}s'
    model_type = results[model]['type'].replace('_', ' ').title()
    table_data.append([model.replace(' ', '\n'), f1, time, model_type])

table = ax4.table(cellText=table_data, cellLoc='center', loc='center',
                 colWidths=[0.35, 0.2, 0.2, 0.25])
table.auto_set_font_size(False)
table.set_fontsize(8)
table.scale(1, 3)

# Style header
for i in range(4):
    table[(0, i)].set_facecolor('#2E7D32')
    table[(0, i)].set_text_props(weight='bold', color='white')

# Highlight winner
table[(2, 0)].set_facecolor('#FFD700')  # K-NN row
table[(2, 1)].set_facecolor('#FFD700')
table[(2, 2)].set_facecolor('#FFD700')
table[(2, 3)].set_facecolor('#FFD700')

ax4.set_title('Summary Results', fontsize=12, fontweight='bold')

fig.suptitle('Final Comparison: Classical vs Quantum ML for Network Intrusion Detection',
            fontsize=16, fontweight='bold', y=0.995)

plt.savefig('comparison/results/visualizations/final_comparison_all_models.png',
           dpi=300, bbox_inches='tight', facecolor='white')
print(f"\n✅ Saved: comparison/results/visualizations/final_comparison_all_models.png")

# Save summary to text
with open('comparison/results/FINAL_RESULTS.txt', 'w') as f:
    f.write("="*80 + "\n")
    f.write("FINAL COMPARISON: Classical vs Quantum ML\n")
    f.write("Network Intrusion Detection on CICIDS2017\n")
    f.write("="*80 + "\n\n")

    f.write("Results Summary (6 Features/Qubits):\n")
    f.write("-"*80 + "\n")
    f.write(f"{'Model':<25} {'F1-Score':<15} {'Time':<15} {'Type':<20}\n")
    f.write("-"*80 + "\n")

    for model in models:
        f1 = f"{results[model]['f1']:.4f}"
        time = f"{results[model]['time']:.2f}s"
        model_type = results[model]['type'].replace('_', ' ').title()
        f.write(f"{model:<25} {f1:<15} {time:<15} {model_type:<20}\n")

    f.write("\n" + "="*80 + "\n")
    f.write("KEY FINDINGS:\n")
    f.write("="*80 + "\n\n")

    f.write("🥇 WINNER: Classical K-NN\n")
    f.write(f"   • F1-Score: 96.48%\n")
    f.write(f"   • Training Time: 0.09s\n")
    f.write(f"   • Best performance + fastest training!\n\n")

    f.write("🥈 RUNNER-UP: Hybrid Q-KNN\n")
    f.write(f"   • F1-Score: 96.42% (only 0.06% behind classical!)\n")
    f.write(f"   • Training Time: 28.5s\n")
    f.write(f"   • Shows quantum can compete with classical\n\n")

    f.write("🥉 THIRD: Hybrid Q-SVM\n")
    f.write(f"   • F1-Score: 91.11%\n")
    f.write(f"   • Training Time: 62.0s\n")
    f.write(f"   • Good quantum performance\n\n")

    f.write("📊 CLASSICAL SVM:\n")
    f.write(f"   • F1-Score: 90.80%\n")
    f.write(f"   • Training Time: 3.26s\n")
    f.write(f"   • Solid baseline\n\n")
    f.write("📊 CLASSICAL SVDD:\n")
    f.write(f"   • F1-Score: 87.40%\n")
    f.write(f"   • Training Time: 0.97s\n")
    f.write(f"   • Underperforms compared to K-NN and SVM\n\n")

    f.write("⚠️ FULLY SUPERVISED QSVDD:\n")
    f.write(f"   • F1-Score: 65.62%\n")
    f.write(f"   • Training Time: 84.3s\n")
    f.write(f"   • Struggles despite supervision\n\n")

    f.write("⚠️ FULL QUANTUM VQC:\n")
    f.write(f"   • F1-Score: 60.19%\n")
    f.write(f"   • Training Time: 547.3s (9+ minutes!)\n")
    f.write(f"   • Shows limitations of NISQ-era quantum ML\n\n")

    f.write("-"*80 + "\n")
    f.write("CONCLUSIONS:\n")
    f.write("-"*80 + "\n\n")

    f.write("1. Classical K-NN remains the best overall approach\n")
    f.write("2. Hybrid quantum approaches (Q-SVM, Q-KNN) can compete with classical\n")
    f.write("3. Q-KNN achieves 96.42% F1 - nearly matching classical K-NN!\n")
    f.write("4. Full quantum VQC struggles in NISQ era (60% F1)\n")
    f.write("5. Quantum shows promise but needs better hardware (fault-tolerant QC)\n\n")
    f.write("6. Fully supervised QSVDD struggles despite supervision (65.62% F1)\n\n")

    f.write("="*80 + "\n")
    f.write("Dataset: CICIDS2017 (Network Intrusion Detection)\n")
    f.write("Features: 6 (selected via mutual information)\n")
    f.write("Samples: 33,776 (balanced 50/50 Normal/Attack)\n")
    f.write("Metric: F1-Score (primary), Accuracy (secondary)\n")
    f.write("="*80 + "\n")

print(f"✅ Saved: comparison/results/FINAL_RESULTS.txt")

print("\n" + "="*80)
print("FINAL COMPARISON COMPLETE!")
print("="*80)
print(f"\nWinner: Classical K-NN (96.48% F1)")
print(f"Best Quantum: Q-KNN (96.42% F1) - Only 0.06% behind!")
print(f"Full Quantum VQC: 60.19% F1 (shows NISQ limitations)")

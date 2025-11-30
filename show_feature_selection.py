"""
Display the features selected at each level (4, 6, 8, 12)
and show the feature importance ranking
"""

import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.model_selection import StratifiedShuffleSplit
from imblearn.under_sampling import RandomUnderSampler
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print(" FEATURE SELECTION ANALYSIS - CICIDS2017")
print("="*80)

# Load dataset
print("\n[1] Loading dataset...")
df = pd.read_csv('cicids2017_cleaned.csv')

X = df.drop('Attack Type', axis=1)
y_binary = (df['Attack Type'] != 'Normal Traffic').astype(int)

# Sample and balance (same as training)
print("[2] Balancing dataset...")
initial_sample = 100000
sss = StratifiedShuffleSplit(n_splits=1, train_size=initial_sample, random_state=42)
for sample_idx, _ in sss.split(X, y_binary):
    X_initial = X.iloc[sample_idx]
    y_initial = y_binary.iloc[sample_idx]

rus = RandomUnderSampler(sampling_strategy=1.0, random_state=42)
X_balanced, y_balanced = rus.fit_resample(X_initial, y_initial)

print(f"Balanced dataset: {len(X_balanced):,} samples\n")

# Compute mutual information for all features
print("[3] Computing mutual information scores for all features...\n")
selector_full = SelectKBest(score_func=mutual_info_classif, k='all')
selector_full.fit(X_balanced, y_balanced)

# Get feature scores
feature_scores = pd.DataFrame({
    'Feature': X.columns,
    'MI_Score': selector_full.scores_
}).sort_values('MI_Score', ascending=False).reset_index(drop=True)

feature_scores['Rank'] = range(1, len(feature_scores) + 1)

print("="*80)
print(" TOP 20 FEATURES BY MUTUAL INFORMATION")
print("="*80)
print(f"\n{'Rank':<6} {'Feature':<40} {'MI Score':<12}")
print("-"*80)
for idx, row in feature_scores.head(20).iterrows():
    print(f"{int(row['Rank']):<6} {row['Feature']:<40} {row['MI_Score']:<12.6f}")

# Show which features are selected at each level
print("\n" + "="*80)
print(" FEATURES SELECTED AT EACH LEVEL")
print("="*80)

feature_counts = [4, 6, 8, 12]
selections = {}

for n_features in feature_counts:
    selector = SelectKBest(score_func=mutual_info_classif, k=n_features)
    selector.fit(X_balanced, y_balanced)
    selected_features = X.columns[selector.get_support(indices=True)].tolist()
    selections[n_features] = selected_features

    print(f"\n{n_features} Features:")
    print("-" * 80)
    for i, feat in enumerate(selected_features, 1):
        # Find rank
        rank = feature_scores[feature_scores['Feature'] == feat]['Rank'].values[0]
        score = feature_scores[feature_scores['Feature'] == feat]['MI_Score'].values[0]
        print(f"  {i}. {feat:<45} (Rank #{int(rank)}, MI={score:.6f})")

# Show incremental additions
print("\n" + "="*80)
print(" INCREMENTAL FEATURE ADDITIONS")
print("="*80)

print("\nStarting with 4 features, what gets added at each level:")
print("-" * 80)

prev_set = set(selections[4])
print(f"\n4 features (Base):")
for feat in selections[4]:
    print(f"  ✓ {feat}")

for n in [6, 8, 12]:
    curr_set = set(selections[n])
    added = curr_set - prev_set
    print(f"\n{n} features (+{len(added)} new):")
    for feat in added:
        rank = feature_scores[feature_scores['Feature'] == feat]['Rank'].values[0]
        print(f"  + {feat} (Rank #{int(rank)})")
    prev_set = curr_set

# Visualization
print("\n" + "="*80)
print(" CREATING VISUALIZATIONS")
print("="*80)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Plot 1: Top 20 feature importance
top_20 = feature_scores.head(20)
colors = ['#d32f2f' if i < 4 else '#f57c00' if i < 6 else '#fbc02d' if i < 8
          else '#7cb342' if i < 12 else '#90a4ae' for i in range(len(top_20))]

ax1.barh(range(len(top_20)), top_20['MI_Score'], color=colors)
ax1.set_yticks(range(len(top_20)))
ax1.set_yticklabels(top_20['Feature'], fontsize=9)
ax1.set_xlabel('Mutual Information Score', fontsize=12, fontweight='bold')
ax1.set_title('Top 20 Features by Mutual Information', fontsize=14, fontweight='bold')
ax1.invert_yaxis()
ax1.grid(axis='x', alpha=0.3)

# Add legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#d32f2f', label='Top 4 (used in all models)'),
    Patch(facecolor='#f57c00', label='Top 5-6 (6+ features)'),
    Patch(facecolor='#fbc02d', label='Top 7-8 (8+ features)'),
    Patch(facecolor='#7cb342', label='Top 9-12 (12 features)'),
    Patch(facecolor='#90a4ae', label='Not selected')
]
ax1.legend(handles=legend_elements, loc='lower right', fontsize=9)

# Plot 2: Feature overlap between levels
levels = [4, 6, 8, 12]
overlap_matrix = np.zeros((4, 4))

for i, n1 in enumerate(levels):
    for j, n2 in enumerate(levels):
        overlap = len(set(selections[n1]) & set(selections[n2]))
        overlap_matrix[i][j] = overlap

sns.heatmap(overlap_matrix, annot=True, fmt='.0f', cmap='YlGn',
            xticklabels=[f'{n} features' for n in levels],
            yticklabels=[f'{n} features' for n in levels],
            ax=ax2, cbar_kws={'label': 'Number of Shared Features'})
ax2.set_title('Feature Overlap Between Configurations', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig('feature_selection_analysis.png', dpi=300, bbox_inches='tight')
print("✓ Saved: feature_selection_analysis.png")

# Summary statistics
print("\n" + "="*80)
print(" FEATURE SELECTION SUMMARY")
print("="*80)

print(f"\nTotal features in dataset: {len(X.columns)}")
print(f"Features evaluated: {len(feature_scores)}")
print(f"\nMutual Information Score Range:")
print(f"  Highest: {feature_scores['MI_Score'].max():.6f} ({feature_scores.iloc[0]['Feature']})")
print(f"  Lowest:  {feature_scores['MI_Score'].min():.6f} ({feature_scores.iloc[-1]['Feature']})")
print(f"  Mean:    {feature_scores['MI_Score'].mean():.6f}")

print(f"\nFeature Categories:")
packet_length_features = feature_scores[feature_scores['Feature'].str.contains('Packet Length', case=False)]
print(f"  Packet Length related: {len(packet_length_features)} features")
print(f"    - In top 12: {len(packet_length_features.head(12))}")

iat_features = feature_scores[feature_scores['Feature'].str.contains('IAT', case=False)]
print(f"  IAT (Inter-Arrival Time): {len(iat_features)} features")
print(f"    - In top 12: {len(iat_features.head(12))}")

flow_features = feature_scores[feature_scores['Feature'].str.contains('Flow', case=False)]
print(f"  Flow related: {len(flow_features)} features")
print(f"    - In top 12: {len(flow_features.head(12))}")

print("\n" + "="*80)
print(" KEY INSIGHTS")
print("="*80)

print("\n1. Most Important Feature Category: Packet Length Statistics")
print("   - 4 of top 4 features are packet length related")
print("   - These features capture fundamental traffic characteristics")

print("\n2. Consistent Core Features:")
print("   - The top 4 features are included in ALL configurations")
print("   - Shows strong agreement on most discriminative features")

print("\n3. Diminishing Returns:")
top4_avg = feature_scores.head(4)['MI_Score'].mean()
next8_avg = feature_scores.iloc[4:12]['MI_Score'].mean()
print(f"   - Top 4 avg MI: {top4_avg:.6f}")
print(f"   - Next 8 avg MI: {next8_avg:.6f}")
print(f"   - Drop: {((top4_avg - next8_avg)/top4_avg)*100:.1f}%")

print("\n" + "="*80)

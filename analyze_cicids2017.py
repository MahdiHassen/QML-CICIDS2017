"""
Comprehensive Analysis of CICIDS2017 Dataset
Extracts interesting facts, statistics, and insights about the network intrusion detection dataset
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (15, 10)

print("="*80)
print(" CICIDS2017 DATASET ANALYSIS - COMPREHENSIVE INSIGHTS")
print("="*80)

# Load dataset
print("\n[1/10] Loading dataset...")
df = pd.read_csv('cicids2017_cleaned.csv')

print(f"✓ Dataset loaded successfully!")
print(f"  Total samples: {len(df):,}")
print(f"  Total features: {len(df.columns)}")

# ============================================================================
# BASIC STATISTICS
# ============================================================================
print("\n" + "="*80)
print(" [2/10] BASIC DATASET STATISTICS")
print("="*80)

print(f"\nDataset Shape: {df.shape}")
print(f"  Rows (samples): {df.shape[0]:,}")
print(f"  Columns (features): {df.shape[1]}")

print(f"\nMemory Usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

print(f"\nColumn Names:")
for i, col in enumerate(df.columns, 1):
    print(f"  {i:2d}. {col}")

# ============================================================================
# CLASS DISTRIBUTION
# ============================================================================
print("\n" + "="*80)
print(" [3/10] CLASS DISTRIBUTION (Attack vs Normal)")
print("="*80)

label_col = None
for possible_label in ['Label', 'label', 'class', 'Class', 'target']:
    if possible_label in df.columns:
        label_col = possible_label
        break

if label_col:
    class_counts = df[label_col].value_counts()
    class_percentages = df[label_col].value_counts(normalize=True) * 100

    print(f"\nClass Distribution:")
    for cls, count in class_counts.items():
        pct = class_percentages[cls]
        print(f"  {cls}: {count:,} samples ({pct:.2f}%)")

    # Calculate imbalance ratio
    if len(class_counts) == 2:
        majority = class_counts.max()
        minority = class_counts.min()
        imbalance_ratio = majority / minority
        print(f"\nClass Imbalance Ratio: {imbalance_ratio:.2f}:1")
        print(f"  (Dataset is {'highly imbalanced' if imbalance_ratio > 5 else 'relatively balanced'})")
else:
    print("No label column found")

# ============================================================================
# MISSING VALUES ANALYSIS
# ============================================================================
print("\n" + "="*80)
print(" [4/10] MISSING VALUES & DATA QUALITY")
print("="*80)

missing_counts = df.isnull().sum()
missing_percentages = (df.isnull().sum() / len(df)) * 100

if missing_counts.sum() > 0:
    print(f"\nTotal missing values: {missing_counts.sum():,}")
    print(f"\nColumns with missing values:")
    for col in missing_counts[missing_counts > 0].index:
        count = missing_counts[col]
        pct = missing_percentages[col]
        print(f"  {col}: {count:,} ({pct:.2f}%)")
else:
    print("\n✓ No missing values found! Dataset is clean.")

# Check for infinity values
inf_counts = {}
for col in df.select_dtypes(include=[np.number]).columns:
    inf_count = np.isinf(df[col]).sum()
    if inf_count > 0:
        inf_counts[col] = inf_count

if inf_counts:
    print(f"\nColumns with infinity values:")
    for col, count in inf_counts.items():
        print(f"  {col}: {count:,}")
else:
    print("✓ No infinity values found!")

# ============================================================================
# FEATURE STATISTICS
# ============================================================================
print("\n" + "="*80)
print(" [5/10] FEATURE STATISTICS")
print("="*80)

numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
if label_col and label_col in numeric_cols:
    numeric_cols.remove(label_col)

print(f"\nNumber of numeric features: {len(numeric_cols)}")

# Find features with zero variance (constant features)
zero_variance_features = []
for col in numeric_cols:
    if df[col].std() == 0:
        zero_variance_features.append(col)

if zero_variance_features:
    print(f"\nFeatures with zero variance (constant): {len(zero_variance_features)}")
    for col in zero_variance_features[:5]:  # Show first 5
        print(f"  {col}: all values = {df[col].iloc[0]}")
    if len(zero_variance_features) > 5:
        print(f"  ... and {len(zero_variance_features) - 5} more")
else:
    print("\n✓ All features have non-zero variance")

# Find features with very high correlation (potential redundancy)
print("\n[Computing feature statistics...]")
feature_ranges = df[numeric_cols].max() - df[numeric_cols].min()
feature_means = df[numeric_cols].mean()
feature_stds = df[numeric_cols].std()

print(f"\nTop 5 features with highest range (max - min):")
top_range_features = feature_ranges.nlargest(5)
for col, val in top_range_features.items():
    print(f"  {col}: {val:.2e}")

print(f"\nTop 5 features with highest standard deviation:")
top_std_features = feature_stds.nlargest(5)
for col, val in top_std_features.items():
    print(f"  {col}: {val:.2e}")

# ============================================================================
# ATTACK TYPE DIVERSITY (if available)
# ============================================================================
print("\n" + "="*80)
print(" [6/10] ATTACK TYPE DIVERSITY")
print("="*80)

# Look for attack type column
attack_type_col = None
for possible_col in ['attack_type', 'Attack Type', 'attack', 'Attack', 'Category', 'category']:
    if possible_col in df.columns:
        attack_type_col = possible_col
        break

if attack_type_col:
    attack_types = df[attack_type_col].value_counts()
    print(f"\nNumber of unique attack types: {len(attack_types)}")
    print(f"\nAttack type distribution:")
    for attack, count in attack_types.items():
        pct = (count / len(df)) * 100
        print(f"  {attack}: {count:,} ({pct:.2f}%)")
elif label_col:
    # Check if label column has more than 2 classes
    unique_labels = df[label_col].unique()
    if len(unique_labels) > 2:
        print(f"\nNumber of unique classes: {len(unique_labels)}")
        print(f"\nClass distribution:")
        for label, count in df[label_col].value_counts().items():
            pct = (count / len(df)) * 100
            print(f"  {label}: {count:,} ({pct:.2f}%)")
    else:
        print("\nNo separate attack type column found.")
        print("Dataset appears to be binary classification (Attack vs Normal)")
else:
    print("\nNo attack type or label column found")

# ============================================================================
# TEMPORAL ANALYSIS (if timestamp available)
# ============================================================================
print("\n" + "="*80)
print(" [7/10] TEMPORAL PATTERNS")
print("="*80)

timestamp_cols = [col for col in df.columns if 'time' in col.lower() or 'date' in col.lower()]
if timestamp_cols:
    print(f"\nFound timestamp columns: {timestamp_cols}")
    # Additional temporal analysis could go here
else:
    print("\nNo timestamp columns found in the dataset")
    print("(Temporal analysis not available)")

# ============================================================================
# NETWORK FLOW CHARACTERISTICS
# ============================================================================
print("\n" + "="*80)
print(" [8/10] NETWORK FLOW CHARACTERISTICS")
print("="*80)

# Common network flow features
flow_features = {
    'duration': ['duration', 'Duration', 'flow_duration'],
    'packets': ['total_fwd_packets', 'total_backward_packets', 'packets'],
    'bytes': ['total_length_fwd_packets', 'total_length_bwd_packets', 'bytes'],
    'rate': ['flow_bytes_per_sec', 'flow_packets_per_sec', 'rate']
}

found_features = {}
for category, possible_names in flow_features.items():
    for name in possible_names:
        matching = [col for col in df.columns if name.lower() in col.lower()]
        if matching:
            found_features[category] = matching
            break

if found_features:
    print("\nNetwork flow features found:")
    for category, features in found_features.items():
        print(f"\n  {category.upper()}:")
        for feature in features:
            if feature in df.columns:
                mean_val = df[feature].mean()
                median_val = df[feature].median()
                max_val = df[feature].max()
                print(f"    {feature}:")
                print(f"      Mean: {mean_val:.2e}, Median: {median_val:.2e}, Max: {max_val:.2e}")
else:
    print("\nStandard network flow features not identified")
    print("(Custom feature naming scheme may be used)")

# ============================================================================
# FEATURE CORRELATIONS
# ============================================================================
print("\n" + "="*80)
print(" [9/10] FEATURE CORRELATION ANALYSIS")
print("="*80)

if len(numeric_cols) > 1:
    print("\nComputing correlation matrix... (this may take a moment)")
    # Sample for large datasets
    sample_size = min(10000, len(df))
    df_sample = df[numeric_cols].sample(n=sample_size, random_state=42)

    correlation_matrix = df_sample.corr()

    # Find highly correlated pairs
    high_corr_pairs = []
    for i in range(len(correlation_matrix.columns)):
        for j in range(i+1, len(correlation_matrix.columns)):
            corr_val = correlation_matrix.iloc[i, j]
            if abs(corr_val) > 0.9:  # Threshold for high correlation
                high_corr_pairs.append((
                    correlation_matrix.columns[i],
                    correlation_matrix.columns[j],
                    corr_val
                ))

    if high_corr_pairs:
        print(f"\nFound {len(high_corr_pairs)} highly correlated feature pairs (|r| > 0.9):")
        for feat1, feat2, corr in sorted(high_corr_pairs, key=lambda x: abs(x[2]), reverse=True)[:10]:
            print(f"  {feat1} <-> {feat2}: r = {corr:.3f}")
        if len(high_corr_pairs) > 10:
            print(f"  ... and {len(high_corr_pairs) - 10} more pairs")
    else:
        print("\n✓ No highly correlated feature pairs found (threshold: |r| > 0.9)")

# ============================================================================
# INTERESTING FACTS SUMMARY
# ============================================================================
print("\n" + "="*80)
print(" [10/10] INTERESTING FACTS & INSIGHTS")
print("="*80)

facts = []

# Fact 1: Dataset size
facts.append(f"1. The CICIDS2017 dataset contains {len(df):,} network traffic samples with {len(df.columns)} features")

# Fact 2: Attack prevalence
if label_col:
    attack_pct = (df[label_col] != 'BENIGN').sum() / len(df) * 100 if 'BENIGN' in df[label_col].values else 50
    facts.append(f"2. Approximately {attack_pct:.1f}% of traffic is malicious/attack traffic")

# Fact 3: Data quality
if missing_counts.sum() == 0:
    facts.append("3. The dataset is remarkably clean with ZERO missing values")

# Fact 4: Feature redundancy
if len(high_corr_pairs) > 0:
    facts.append(f"4. There are {len(high_corr_pairs)} highly correlated feature pairs, suggesting potential redundancy")

# Fact 5: Dimensionality
facts.append(f"5. With {len(numeric_cols)} numeric features, this is a high-dimensional dataset requiring dimensionality reduction")

# Fact 6: Memory footprint
mem_mb = df.memory_usage(deep=True).sum() / 1024**2
facts.append(f"6. The dataset occupies {mem_mb:.1f} MB in memory, making it suitable for in-memory processing")

# Fact 7: Feature variability
zero_var_count = len(zero_variance_features)
if zero_var_count > 0:
    facts.append(f"7. {zero_var_count} features have zero variance and could be dropped without information loss")
else:
    facts.append("7. All features exhibit variability, indicating no obviously redundant constant features")

# Fact 8: Class imbalance
if label_col and len(class_counts) == 2:
    facts.append(f"8. The dataset has a {imbalance_ratio:.1f}:1 class imbalance ratio, requiring careful sampling strategies")

# Fact 9: Attack diversity
if attack_type_col and len(attack_types) > 2:
    facts.append(f"9. The dataset includes {len(attack_types)} different types of network attacks, providing diverse threat scenarios")

# Fact 10: Real-world applicability
facts.append("10. CICIDS2017 was collected from realistic network traffic, making it more representative than synthetic datasets")

print("\n📊 TOP 10 INTERESTING FACTS:")
print()
for fact in facts:
    print(f"  {fact}")

# ============================================================================
# VISUALIZATIONS
# ============================================================================
print("\n" + "="*80)
print(" GENERATING VISUALIZATIONS")
print("="*80)

fig = plt.figure(figsize=(20, 12))

# 1. Class distribution
if label_col:
    ax1 = plt.subplot(2, 3, 1)
    class_counts.plot(kind='bar', ax=ax1, color=['#2ecc71', '#e74c3c'])
    ax1.set_title('Class Distribution', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Number of Samples', fontsize=12)
    ax1.set_xlabel('Class', fontsize=12)
    ax1.tick_params(axis='x', rotation=45)
    for i, v in enumerate(class_counts.values):
        ax1.text(i, v + 100, f'{v:,}', ha='center', va='bottom', fontweight='bold')

# 2. Missing values heatmap (if any)
ax2 = plt.subplot(2, 3, 2)
if missing_counts.sum() > 0:
    missing_percentages[missing_percentages > 0].plot(kind='barh', ax=ax2, color='#e74c3c')
    ax2.set_title('Missing Values by Feature', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Percentage Missing (%)', fontsize=12)
else:
    ax2.text(0.5, 0.5, 'NO MISSING VALUES\n✓',
             ha='center', va='center', fontsize=20, fontweight='bold', color='#2ecc71')
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.axis('off')
    ax2.set_title('Data Quality', fontsize=14, fontweight='bold')

# 3. Top features by variance
ax3 = plt.subplot(2, 3, 3)
top_std_features.plot(kind='barh', ax=ax3, color='#3498db')
ax3.set_title('Top 5 Features by Std Dev', fontsize=14, fontweight='bold')
ax3.set_xlabel('Standard Deviation', fontsize=12)

# 4. Feature count summary
ax4 = plt.subplot(2, 3, 4)
feature_summary = pd.Series({
    'Numeric Features': len(numeric_cols),
    'Zero Variance': len(zero_variance_features),
    'High Correlation Pairs': len(high_corr_pairs) if 'high_corr_pairs' in locals() else 0
})
feature_summary.plot(kind='bar', ax=ax4, color=['#3498db', '#e74c3c', '#f39c12'])
ax4.set_title('Feature Analysis Summary', fontsize=14, fontweight='bold')
ax4.set_ylabel('Count', fontsize=12)
ax4.tick_params(axis='x', rotation=45)
for i, v in enumerate(feature_summary.values):
    ax4.text(i, v + 1, str(int(v)), ha='center', va='bottom', fontweight='bold')

# 5. Dataset size metrics
ax5 = plt.subplot(2, 3, 5)
size_metrics = pd.Series({
    'Total Samples': len(df) / 1000,  # in thousands
    'Features': len(df.columns),
    'Memory (MB)': mem_mb
})
colors = ['#9b59b6', '#e67e22', '#1abc9c']
size_metrics.plot(kind='bar', ax=ax5, color=colors)
ax5.set_title('Dataset Size Metrics', fontsize=14, fontweight='bold')
ax5.set_ylabel('Value', fontsize=12)
ax5.tick_params(axis='x', rotation=45)
ax5.set_yscale('log')
for i, (metric, v) in enumerate(size_metrics.items()):
    if 'Samples' in metric:
        ax5.text(i, v, f'{v:.1f}K', ha='center', va='bottom', fontweight='bold')
    else:
        ax5.text(i, v, f'{v:.1f}', ha='center', va='bottom', fontweight='bold')

# 6. Correlation heatmap (top features)
if len(numeric_cols) >= 10:
    ax6 = plt.subplot(2, 3, 6)
    # Select top 10 features by variance for correlation heatmap
    top_features = feature_stds.nlargest(10).index.tolist()
    corr_subset = df[top_features].sample(n=min(5000, len(df)), random_state=42).corr()
    sns.heatmap(corr_subset, annot=False, cmap='RdYlGn', center=0, ax=ax6,
                cbar_kws={'label': 'Correlation'}, vmin=-1, vmax=1)
    ax6.set_title('Feature Correlation Heatmap\n(Top 10 by Variance)', fontsize=14, fontweight='bold')
    ax6.tick_params(axis='both', labelsize=8)

plt.tight_layout()
plt.savefig('cicids2017_analysis_report.png', dpi=300, bbox_inches='tight')
print("\n✓ Visualization saved as 'cicids2017_analysis_report.png'")

# ============================================================================
# SAVE DETAILED REPORT
# ============================================================================
print("\n" + "="*80)
print(" SAVING DETAILED REPORT")
print("="*80)

with open('cicids2017_analysis_report.txt', 'w') as f:
    f.write("="*80 + "\n")
    f.write(" CICIDS2017 DATASET - COMPREHENSIVE ANALYSIS REPORT\n")
    f.write("="*80 + "\n\n")

    f.write("DATASET OVERVIEW\n")
    f.write("-" * 80 + "\n")
    f.write(f"Total Samples: {len(df):,}\n")
    f.write(f"Total Features: {len(df.columns)}\n")
    f.write(f"Numeric Features: {len(numeric_cols)}\n")
    f.write(f"Memory Usage: {mem_mb:.2f} MB\n\n")

    if label_col:
        f.write("CLASS DISTRIBUTION\n")
        f.write("-" * 80 + "\n")
        for cls, count in class_counts.items():
            pct = class_percentages[cls]
            f.write(f"{cls}: {count:,} samples ({pct:.2f}%)\n")
        f.write(f"\nClass Imbalance Ratio: {imbalance_ratio:.2f}:1\n\n")

    f.write("INTERESTING FACTS\n")
    f.write("-" * 80 + "\n")
    for fact in facts:
        f.write(f"{fact}\n")

    f.write("\n" + "="*80 + "\n")
    f.write("END OF REPORT\n")
    f.write("="*80 + "\n")

print("✓ Detailed report saved as 'cicids2017_analysis_report.txt'")

print("\n" + "="*80)
print(" ANALYSIS COMPLETE!")
print("="*80)
print("\nGenerated files:")
print("  1. cicids2017_analysis_report.png  (Visualization dashboard)")
print("  2. cicids2017_analysis_report.txt  (Detailed text report)")
print("\n")

import os
import time
import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM
from imblearn.under_sampling import RandomUnderSampler
import matplotlib.pyplot as plt
import kagglehub

# Paths
RESULTS_DIR = "classical_models/results"
VIZ_DIR = f"{RESULTS_DIR}/visualizations"
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(VIZ_DIR, exist_ok=True)

# Configuration
RANDOM_SEED = 42
INITIAL_SAMPLE = 100000
TEST_SVDD_SUBSET = 10000         # validation+test subset size
MAX_TRAIN_NORMAL = 5000          # cap on normals for OC-SVM
NU_GRID = [0.005, 0.01, 0.05, 0.1, 0.2, 0.3]  # boundary tightness sweep
FEATURE_COUNTS = [4, 6, 8, 12]


def set_seed(seed: int = RANDOM_SEED):
    np.random.seed(seed)

def download_cicids_cleaned() -> str:
    path = kagglehub.dataset_download("ericanacletoribeiro/cicids2017-cleaned-and-preprocessed")
    csv_path = os.path.join(path, "cicids2017_cleaned.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found at {csv_path}")
    return csv_path

def prepare_balanced_data():
    print("Loading dataset...")
    csv_path = download_cicids_cleaned()
    df = pd.read_csv(csv_path)

    X = df.drop("Attack Type", axis=1)
    y = (df["Attack Type"] != "Normal Traffic").astype(int)

    sss = StratifiedShuffleSplit(n_splits=1, train_size=INITIAL_SAMPLE, random_state=RANDOM_SEED)
    for idx, _ in sss.split(X, y):
        X_initial = X.iloc[idx]
        y_initial = y.iloc[idx]

    rus = RandomUnderSampler(sampling_strategy=1.0, random_state=RANDOM_SEED)
    X_balanced, y_balanced = rus.fit_resample(X_initial, y_initial)
    return X_balanced, y_balanced


def tune_threshold(scores, labels):
    best_tau = None
    best_f1 = -1.0
    percentiles = np.linspace(1, 99, 40)

    for p in percentiles:
        tau = np.percentile(scores, p)
        preds = (scores < tau).astype(int)  # attack if below threshold
        f1 = f1_score(labels, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_tau = tau

    return best_tau, best_f1


def run_svdd_for_features(n_features, X_balanced, y_balanced, feature_names):
    print("\n" + "-" * 80)
    print(f"Running with top-{n_features} MI features")
    print("-" * 80)

    selector = SelectKBest(score_func=mutual_info_classif, k=n_features)
    X_selected = selector.fit_transform(X_balanced, y_balanced)
    selected_features = feature_names[selector.get_support(indices=True)]
    print(f"Selected features: {', '.join(selected_features)}")

    X_train, X_test, y_train, y_test = train_test_split(
        X_selected, y_balanced, test_size=0.25, random_state=RANDOM_SEED, stratify=y_balanced
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Training data: normals only, capped for practicality
    normal_mask = y_train == 0
    X_train_normal = X_train_scaled[normal_mask]
    if len(X_train_normal) > MAX_TRAIN_NORMAL:
        X_train_normal = X_train_normal[:MAX_TRAIN_NORMAL]

    print(f"Normal samples for SVDD training: {len(X_train_normal)} (cap {MAX_TRAIN_NORMAL})")

    # Validation/test split from test subset
    subset_size = min(TEST_SVDD_SUBSET, len(X_test_scaled))
    X_test_subset = X_test_scaled[:subset_size]
    y_test_subset = y_test[:subset_size]
    val_size = subset_size // 2
    X_val, y_val = X_test_subset[:val_size], y_test_subset[:val_size]
    X_real_test, y_real_test = X_test_subset[val_size:], y_test_subset[val_size:]

    print(f"Validation subset: {X_val.shape}, Test subset: {X_real_test.shape}")

    best_model = None
    best_nu = None
    best_val_f1 = -1.0
    best_tau = 0.0

    print("\n[2] Hyperparameter sweep (nu) with validation F1 tuning (gamma fixed: scale)...")

    sweep_start = time.time()
    for nu in NU_GRID:
        model = OneClassSVM(kernel="rbf", gamma="scale", nu=nu)
        model.fit(X_train_normal)

        val_scores = model.decision_function(X_val)
        tau, f1_val = tune_threshold(val_scores, y_val)
        print(
            f"  nu={nu:>6}: val F1={f1_val:.4f}, tau={tau:.4f}"
        )

        if f1_val > best_val_f1:
            best_val_f1 = f1_val
            best_nu = nu
            best_tau = tau
            best_model = model
    train_time = time.time() - sweep_start

    if best_model is None:
        raise RuntimeError("Failed to train SVDD model.")

    print(
        f"\nBest nu={best_nu} "
        f"(val F1={best_val_f1:.4f}, tau={best_tau:.4f})"
    )

    print("\n[3] Evaluating SVDD on held-out test subset...")
    test_scores = best_model.decision_function(X_real_test)
    y_pred = (test_scores < best_tau).astype(int)
    svdd_f1 = f1_score(y_real_test, y_pred, zero_division=0)
    svdd_accuracy = accuracy_score(y_real_test, y_pred)
    cm = confusion_matrix(y_real_test, y_pred)
    report = classification_report(y_real_test, y_pred, digits=4, zero_division=0)

    print("Confusion matrix (rows=true, cols=pred):")
    print(cm)
    print("\nClassification report:")
    print(report)
    print(f"SVDD Test F1 (attack=1): {svdd_f1:.4f}")
    print(f"SVDD training/tuning time: {train_time:.2f}s")

    return {
        "n_features": n_features,
        "selected_features": list(selected_features),
        "svdd_f1": svdd_f1,
        "svdd_accuracy": svdd_accuracy,
        "svdd_train_time": train_time,
    }
def visualize_results(results):
    print(f"\n[3] Creating visualizations...")
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('SVDD Performance vs Features (MI-selected)', fontsize=16, fontweight='bold')

    features = [r['n_features'] for r in results]
    svdd_f1 = [r['svdd_f1'] for r in results]
    svdd_acc = [r['svdd_accuracy'] for r in results]
    svdd_train_time = [r['svdd_train_time'] for r in results]

    # F1-Score comparison
    ax1.plot(features, svdd_f1, marker='s', linewidth=2, markersize=8, label='SVDD (tau-tuned)', color='#1976D2')
    ax1.set_xlabel('Features', fontweight='bold')
    ax1.set_ylabel('F1-Score', fontweight='bold')
    ax1.set_title('F1-Score Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(features)

    # Bar chart: F1 and Accuracy
    x = np.arange(len(features))
    width = 0.35
    ax2.bar(x - width/2, svdd_f1, width, label='F1', color='#1976D2', alpha=0.8)
    ax2.bar(x + width/2, svdd_acc, width, label='Accuracy', color='#C62828', alpha=0.8)
    ax2.set_xlabel('Features', fontweight='bold')
    ax2.set_ylabel('Score', fontweight='bold')
    ax2.set_title('Scores (Bar Chart)')
    ax2.set_xticks(x)
    ax2.set_xticklabels(features)
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')

    # Training time
    ax3.bar(x, svdd_train_time, width, label='SVDD train+tune', color='#1976D2', alpha=0.8)
    ax3.set_xlabel('Features', fontweight='bold')
    ax3.set_ylabel('Time (seconds)', fontweight='bold')
    ax3.set_title('Training/Tuning Time')
    ax3.set_xticks(x)
    ax3.set_xticklabels(features)
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')

    # Summary table
    ax4.axis('off')
    table_data = [['Features', 'F1', 'Accuracy', 'Train (s)']]
    for r in results:
        table_data.append([
            str(r['n_features']),
            f"{r['svdd_f1']:.4f}",
            f"{r['svdd_accuracy']:.4f}",
            f"{r['svdd_train_time']:.2f}",
        ])
    table = ax4.table(cellText=table_data, cellLoc='center', loc='center', colWidths=[0.25, 0.25, 0.25, 0.25])
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 2)
    for i in range(len(table_data[0])):
        table[(0, i)].set_facecolor('#1976D2')
        table[(0, i)].set_text_props(weight='bold', color='white')

    plt.tight_layout()
    plt.savefig(f'{VIZ_DIR}/svdd_summary.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {VIZ_DIR}/svdd_summary.png")

    # Save Results
    print(f"\n[4] Saving results...")
    with open(f'{RESULTS_DIR}/svdd_results.txt', 'w') as f:
        f.write("SVDD Results (MI-selected features)\n")
        f.write("="*80 + "\n\n")

        for r in results:
            f.write(f"{r['n_features']} Features:\n")
            f.write(f"  Features: {', '.join(r['selected_features'])}\n")
            f.write(
                f"  SVDD: F1={r['svdd_f1']:.4f}, Acc={r['svdd_accuracy']:.4f}, "
                f"Train={r['svdd_train_time']:.2f}s\n"
            )
            f.write("\n")

        best_f1 = max(svdd_f1)
        best_feat = features[svdd_f1.index(best_f1)]
        f.write(f"Best F1: {best_f1:.4f} @ {best_feat} features\n")

def main():
    set_seed()
    print("=" * 80)
    print("Classical SVDD (One-Class SVM): MI feature sweep (4, 6, 8, 12)")
    print("=" * 80)

    X_balanced, y_balanced = prepare_balanced_data()
    feature_names = X_balanced.columns.values
    results = []

    for n_features in FEATURE_COUNTS:
        res = run_svdd_for_features(n_features, X_balanced, y_balanced, feature_names)
        results.append(res)

    visualize_results(results)
    print(f"Saved: {RESULTS_DIR}/svdd_results.txt")

    print("\n" + "="*80)
    print("SVDD TRAINING COMPLETE!")
    print("="*80)
    print(f"\nResults saved in: {RESULTS_DIR}/")
    print(f"Visualizations saved in: {VIZ_DIR}/")

if __name__ == "__main__":
    main()

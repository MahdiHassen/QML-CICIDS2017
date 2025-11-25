import os
import numpy as np
import pandas as pd
import time

import kagglehub
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    f1_score,
)

from scipy.optimize import minimize
from qiskit import QuantumCircuit
from qiskit.circuit.library import RealAmplitudes
from qiskit.quantum_info import Statevector
from sklearn.metrics import classification_report as sk_classification_report

# Configurations
RANDOM_SEED = 42
MAX_SAMPLES_TOTAL = 100000
TEST_QUANTUM_SUBSET = 10000
MAX_NORMAL_QSVDD = 5000
MAX_ANOM_QSVDD = 5000
N_QUBITS = 4
RESULTS_DIR = "results"
VIZ_DIR = os.path.join(RESULTS_DIR, "visualizations")
os.makedirs(VIZ_DIR, exist_ok=True)

# Fully supervised QSVDD training configurations
QSVDD_N_RESTARTS = 3
CLASS_WEIGHT_NEG = 1.0
CLASS_WEIGHT_POS = 2.0

def set_seed(seed: int = RANDOM_SEED):
    np.random.seed(seed)


def download_cicids_cleaned() -> str:
    path = kagglehub.dataset_download("ericanacletoribeiro/cicids2017-cleaned-and-preprocessed")
    csv_path = os.path.join(path, "cicids2017_cleaned.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found at {csv_path}")
    return csv_path


def load_and_preprocess(csv_path: str):
    print("Loading dataset...")
    df = pd.read_csv(csv_path)
    print("Full dataset shape:", df.shape)
    print("Attack Type value counts:")
    print(df["Attack Type"].value_counts())
    y = (df["Attack Type"] != "Normal Traffic").astype(int)

    features = df.drop(columns=["Attack Type"])
    numeric_cols = features.select_dtypes(include=["number"]).columns
    X = features[numeric_cols].copy()

    print("Number of numeric features:", len(numeric_cols))

    # Downsample for quantum experiments
    print(f"Downsampling to {MAX_SAMPLES_TOTAL} samples for practicality...")
    df_sample = df.sample(n=MAX_SAMPLES_TOTAL, random_state=RANDOM_SEED)

    y_sample = (df_sample["Attack Type"] != "Normal Traffic").astype(int)
    features_sample = df_sample.drop(columns=["Attack Type"])
    numeric_cols = features_sample.select_dtypes(include=["number"]).columns
    X_sample = features_sample[numeric_cols].copy()

    print("Sampled X shape:", X_sample.shape)
    print("Sampled y shape:", y_sample.shape)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_sample,
        y_sample,
        test_size=0.2,
        random_state=RANDOM_SEED,
        stratify=y_sample,
    )

    # Scaling + PCA
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    pca = PCA(n_components=N_QUBITS)
    X_train_reduced = pca.fit_transform(X_train_scaled)
    X_test_reduced = pca.transform(X_test_scaled)

    print("Reduced train shape:", X_train_reduced.shape)
    print("Reduced test shape:", X_test_reduced.shape)

    # Quantum train subset: use all reduced train rows
    X_train_q = X_train_reduced
    y_train_q = y_train.to_numpy()

    # Quantum val+test subset
    X_test_q = X_test_reduced[:TEST_QUANTUM_SUBSET]
    y_test_q = y_test.iloc[:TEST_QUANTUM_SUBSET].to_numpy()
    return X_train_q, y_train_q, X_test_q, y_test_q

# Quantum feature map

def build_feature_map(n_qubits: int):

    def feature_map(x):
        qc = QuantumCircuit(n_qubits)
        x = np.clip(x, -3.0, 3.0)
        x_scaled = (x / 3.0) * (np.pi / 2)
        for i, val in enumerate(x_scaled):
            qc.ry(float(val), i)
        for i in range(n_qubits - 1):
            qc.cx(i, i + 1)
        return qc
    return feature_map


def data_state(feature_map_fn, x):
    qc = feature_map_fn(x)
    return Statevector.from_instruction(qc)

# QSVDD: training, tuning, scoring

def train_qsvdd_fully_supervised(
    X_train_q,
    y_train_q,
    feature_map_fn,
    n_restarts=QSVDD_N_RESTARTS,
    class_weight_neg=CLASS_WEIGHT_NEG,
    class_weight_pos=CLASS_WEIGHT_POS,
):
    normal_idx = np.where(y_train_q == 0)[0]
    anom_idx = np.where(y_train_q == 1)[0]

    X_normal = X_train_q[normal_idx]
    X_anom = X_train_q[anom_idx]

    if len(X_normal) > MAX_NORMAL_QSVDD:
        X_normal = X_normal[:MAX_NORMAL_QSVDD]
    if len(X_anom) > MAX_ANOM_QSVDD:
        X_anom = X_anom[:MAX_ANOM_QSVDD]

    print(f"Using {X_normal.shape[0]} normal samples for supervised QSVDD.")
    print(f"Using {X_anom.shape[0]} attack samples for supervised QSVDD.")

    X_sup = np.vstack([X_normal, X_anom])
    y_sup = np.concatenate([
        np.zeros(len(X_normal), dtype=int),
        np.ones(len(X_anom), dtype=int),
    ])
    data_states_sup = [data_state(feature_map_fn, x) for x in X_sup]

    n_qubits = X_train_q.shape[1]
    center_ansatz = RealAmplitudes(
        num_qubits=n_qubits,
        reps=2,
        entanglement="full",
    )
    num_params = center_ansatz.num_parameters
    print("Supervised center ansatz parameters:", num_params)

    def center_state_from_theta(theta):
        param_binds = dict(zip(center_ansatz.parameters, theta))
        qc = center_ansatz.assign_parameters(param_binds, inplace=False)
        return Statevector.from_instruction(qc)

    def supervised_ce_cost(theta, eps=1e-8):
        center_sv = center_state_from_theta(theta)
        losses = []

        for s_vec, y in zip(data_states_sup, y_sup):
            # s is probability of being normal (y=0)
            s = np.abs(np.vdot(center_sv.data, s_vec.data))**2
            s = float(np.clip(s, eps, 1.0 - eps))

            if y == 0:
                loss_i = -class_weight_neg * np.log(s)
            else:
                loss_i = -class_weight_pos * np.log(1.0 - s)

            losses.append(loss_i)

        return float(np.mean(losses))

    # Multi-start optimization
    best_res = None
    for r in range(n_restarts):
        print(f"Supervised QSVDD optimization restart {r + 1}/{n_restarts}...")
        theta0 = np.random.uniform(0, 2 * np.pi, num_params)
        res = minimize(
            supervised_ce_cost,
            theta0,
            method="Nelder-Mead",
            options={"maxiter": 200, "disp": True},
        )
        if (best_res is None) or (res.fun < best_res.fun):
            best_res = res

    theta_opt = best_res.x
    center_sv_opt = center_state_from_theta(theta_opt)

    # Scores for normal training samples (used for threshold range)
    data_states_norm = [data_state(feature_map_fn, x) for x in X_normal]
    train_scores_norm = np.array([
        np.abs(np.vdot(center_sv_opt.data, s.data)) ** 2
        for s in data_states_norm
    ])
    return center_sv_opt, train_scores_norm


def qsvdd_score(x, feature_map_fn, center_sv):
    """Compute QSVDD score s(x) = |<phi(x)|c>|^2."""
    s = data_state(feature_map_fn, x)
    return float(np.abs(np.vdot(center_sv.data, s.data)) ** 2)


def predict_qsvdd(X_q, feature_map_fn, center_sv, tau):
    scores = np.array([
        qsvdd_score(x, feature_map_fn, center_sv) for x in X_q
    ])
    y_pred = np.where(scores >= tau, 0, 1)
    return y_pred, scores


def tune_tau(train_scores_norm, X_val_q, y_val_q, feature_map_fn, center_sv):
    taus = np.linspace(train_scores_norm.min(), train_scores_norm.max(), 100)
    best_tau = None
    best_f1 = -1.0

    for tau in taus:
        y_pred_val, _ = predict_qsvdd(X_val_q, feature_map_fn, center_sv, tau)
        f1 = f1_score(y_val_q, y_pred_val, pos_label=1)
        if f1 > best_f1:
            best_f1 = f1
            best_tau = tau
    return best_tau
def visualize_results(y_real_test_q, y_pred_qsvdd):
    report_dict = sk_classification_report(
    y_real_test_q, y_pred_qsvdd, digits=4, output_dict=True, zero_division=0
    )
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.axis("off")
    headers = ["Class", "Precision", "Recall", "F1-Score", "Support"]
    rows = []
    for label_key in ["0", "1", "macro avg", "weighted avg"]:
        if label_key not in report_dict:
            continue
        row_label = "Normal (0)" if label_key == "0" else "Attack (1)" if label_key == "1" else label_key
        row = report_dict[label_key]
        rows.append([
            row_label,
            f"{row['precision']:.4f}",
            f"{row['recall']:.4f}",
            f"{row['f1-score']:.4f}",
            int(row["support"]),
        ])

    table = ax.table(
        cellText=rows,
        colLabels=headers,
        loc="center",
        cellLoc="center",
        colWidths=[0.25, 0.15, 0.15, 0.15, 0.15],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    for j in range(len(headers)):
        table[(0, j)].set_facecolor("#1976D2")
        table[(0, j)].set_text_props(color="white", fontweight="bold")

    ax.set_title("QSVDD Classification Report (Attack=1)", fontweight="bold", fontsize=12, pad=12)
    viz_path = os.path.join(VIZ_DIR, "qsvdd_classification_report.png")
    plt.savefig(viz_path, dpi=300, bbox_inches="tight")
    print(f"Saved visualization: {viz_path}")

def main():
    set_seed(RANDOM_SEED)
    csv_path = download_cicids_cleaned()
    X_train_q, y_train_q, X_test_q, y_test_q = load_and_preprocess(csv_path)

    val_size = TEST_QUANTUM_SUBSET // 2
    X_val_q, y_val_q = X_test_q[:val_size], y_test_q[:val_size]
    X_real_test_q, y_real_test_q = X_test_q[val_size:], y_test_q[val_size:]
    n_qubits = X_train_q.shape[1]
    print("Number of qubits / PCA dims:", n_qubits)

    feature_map_fn = build_feature_map(n_qubits)
    example_circuit = feature_map_fn(X_train_q[0])
    print(example_circuit.draw())
    train_start = time.time()
    center_sv_opt, train_scores_norm = train_qsvdd_fully_supervised(
        X_train_q, y_train_q, feature_map_fn
    )
    tau = tune_tau(train_scores_norm, X_val_q, y_val_q, feature_map_fn, center_sv_opt)
    training_time_s = time.time() - train_start
    y_pred_qsvdd, test_scores = predict_qsvdd(
        X_real_test_q, feature_map_fn, center_sv_opt, tau
    )
    test_f1 = f1_score(y_real_test_q, y_pred_qsvdd, pos_label=1)
    cm = confusion_matrix(y_real_test_q, y_pred_qsvdd)
    print(cm)

    print("\nClassification report:")
    clf_report = classification_report(y_real_test_q, y_pred_qsvdd, digits=4)
    print(clf_report)
    print(f"Training time: {training_time_s:.2f} seconds")
    print(f"Test F1 (attack class = 1): {test_f1:.4f}")

    visualize_results(y_real_test_q, y_pred_qsvdd)

if __name__ == "__main__":
    main()

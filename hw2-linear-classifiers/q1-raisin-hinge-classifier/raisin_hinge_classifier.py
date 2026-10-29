import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

DATA_PATH = "001_Raisin_Dataset.xlsx"

FEATURES = [
    "Area",
    "MajorAxisLength",
    "MinorAxisLength",
    "Eccentricity",
    "ConvexArea",
    "Extent",
    "Perimeter",
]

LABEL_MAP = {"Kecimen": 0, "Besni": 1}
INV_LABEL_MAP = {0: "Kecimen", 1: "Besni"}


# ------------------------------------------------------------
# Utility functions
# ------------------------------------------------------------

def add_bias_column(X):
    return np.c_[np.ones(X.shape[0]), X]


def standardize_with_train_statistics(X_train, X_val, X_test):
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    std = np.where(std == 0, 1.0, std)

    X_train_std = (X_train - mean) / std
    X_val_std = (X_val - mean) / std
    X_test_std = (X_test - mean) / std

    return X_train_std, X_val_std, X_test_std, mean, std


def standardize_train_and_other(X_train, X_other):
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    std = np.where(std == 0, 1.0, std)

    X_train_std = (X_train - mean) / std
    X_other_std = (X_other - mean) / std

    return X_train_std, X_other_std, mean, std


# ------------------------------------------------------------
# Hinge Loss model
# ------------------------------------------------------------

def hinge_loss(w, X, y_pm):
    margins = y_pm * (X @ w)
    losses = np.maximum(0.0, 1.0 - margins)
    return losses.mean()


def hinge_subgradient(w, X, y_pm):
    margins = y_pm * (X @ w)
    active = margins < 1.0

    grad = np.zeros_like(w)

    if np.any(active):
        grad = -(X[active] * y_pm[active, None]).sum(axis=0) / X.shape[0]

    return grad


def train_hinge_classifier(X_train, y_train, learning_rate=0.05, epochs=3000):
    y_pm = np.where(y_train == 1, 1, -1)

    w = np.zeros(X_train.shape[1])
    losses = []

    for _ in range(epochs):
        grad = hinge_subgradient(w, X_train, y_pm)
        w -= learning_rate * grad
        losses.append(hinge_loss(w, X_train, y_pm))

    return w, np.array(losses)


def predict_linear(X, w):
    scores = X @ w
    predictions = (scores >= 0.0).astype(int)
    return predictions, scores


# ------------------------------------------------------------
# Metrics
# ------------------------------------------------------------

def confusion_matrix_binary(y_true, y_pred):
    cm = np.zeros((2, 2), dtype=int)

    for actual, predicted in zip(y_true, y_pred):
        cm[int(actual), int(predicted)] += 1

    return cm


def binary_metrics(y_true, y_pred):
    cm = confusion_matrix_binary(y_true, y_pred)

    tn, fp = cm[0, 0], cm[0, 1]
    fn, tp = cm[1, 0], cm[1, 1]

    accuracy = (tp + tn) / cm.sum()
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return cm, {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


# ------------------------------------------------------------
# Reporting and plotting
# ------------------------------------------------------------

def print_dataset_report(df):
    print("=" * 70)
    print("DATASET REPORT")
    print("=" * 70)

    print("Number of samples:", len(df))
    print("Number of numeric features:", len(FEATURES))
    print("Class names:", list(df["Class"].unique()))

    print("\nClass counts:")
    print(df["Class"].value_counts())

    print("\nFeature ranges:")
    print(df[FEATURES].agg(["min", "max"]).T)


def plot_loss(losses, output_path="q1_training_hinge_loss.png"):
    plt.figure(figsize=(7, 4.5))
    plt.plot(np.arange(1, len(losses) + 1), losses)
    plt.xlabel("Epoch")
    plt.ylabel("Training Hinge Loss")
    plt.title("Hinge Loss during Batch Gradient Descent")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def print_metrics(title, cm, metrics):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)

    print("Confusion matrix, rows=true and columns=predicted:")
    print(cm)

    print("\nMetrics:")
    print(f"Accuracy : {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall   : {metrics['recall']:.4f}")
    print(f"F1-score : {metrics['f1']:.4f}")
    print(f"TN: {metrics['tn']}, FP: {metrics['fp']}, FN: {metrics['fn']}, TP: {metrics['tp']}")


# ------------------------------------------------------------
# Validation hyperparameter selection
# ------------------------------------------------------------

def select_best_hyperparameters(
    X_train_b,
    y_train,
    X_val_b,
    y_val,
    learning_rates,
    epochs_options,
    selection_metric="f1",
):
    results = []
    best_result = None

    for lr in learning_rates:
        for epochs in epochs_options:
            w_candidate, losses_candidate = train_hinge_classifier(
                X_train_b,
                y_train,
                learning_rate=lr,
                epochs=epochs,
            )

            y_val_pred, _ = predict_linear(X_val_b, w_candidate)
            val_cm, val_metrics = binary_metrics(y_val, y_val_pred)

            row = {
                "learning_rate": lr,
                "epochs": epochs,
                "validation_accuracy": val_metrics["accuracy"],
                "validation_precision": val_metrics["precision"],
                "validation_recall": val_metrics["recall"],
                "validation_f1": val_metrics["f1"],
                "weights": w_candidate,
                "losses": losses_candidate,
                "validation_cm": val_cm,
            }
            results.append(row)

            if best_result is None:
                best_result = row
            else:
                current_score = row[f"validation_{selection_metric}"]
                best_score = best_result[f"validation_{selection_metric}"]

                if (
                    current_score > best_score
                    or (
                        np.isclose(current_score, best_score)
                        and (row["epochs"], row["learning_rate"])
                        < (best_result["epochs"], best_result["learning_rate"])
                    )
                ):
                    best_result = row

    results_table = pd.DataFrame(
        [
            {
                "learning_rate": r["learning_rate"],
                "epochs": r["epochs"],
                "validation_accuracy": r["validation_accuracy"],
                "validation_precision": r["validation_precision"],
                "validation_recall": r["validation_recall"],
                "validation_f1": r["validation_f1"],
            }
            for r in results
        ]
    )

    return best_result, results_table


# ------------------------------------------------------------
# Full 7-feature experiment
# ------------------------------------------------------------

def run_full_feature_experiment(df):
    X = df[FEATURES].values.astype(float)
    y = df["Class"].map(LABEL_MAP).values.astype(int)
    indices = np.arange(len(df))

    X_train, X_temp, y_train, y_temp, idx_train, idx_temp = train_test_split(
        X,
        y,
        indices,
        test_size=0.30,
        random_state=RANDOM_SEED,
        stratify=y,
    )

    X_val, X_test, y_val, y_test, idx_val, idx_test = train_test_split(
        X_temp,
        y_temp,
        idx_temp,
        test_size=0.50,
        random_state=RANDOM_SEED,
        stratify=y_temp,
    )

    X_train_std, X_val_std, X_test_std, mean, std = standardize_with_train_statistics(
        X_train,
        X_val,
        X_test,
    )

    X_train_b = add_bias_column(X_train_std)
    X_val_b = add_bias_column(X_val_std)
    X_test_b = add_bias_column(X_test_std)

    print("\n" + "=" * 70)
    print("SPLIT SIZES")
    print("=" * 70)
    print("Train:", X_train.shape[0])
    print("Validation:", X_val.shape[0])
    print("Test:", X_test.shape[0])

    learning_rates = [0.001, 0.005, 0.01, 0.05, 0.1]
    epochs_options = [500, 1000, 2000, 3000, 5000]

    best_result, validation_table = select_best_hyperparameters(
        X_train_b,
        y_train,
        X_val_b,
        y_val,
        learning_rates=learning_rates,
        epochs_options=epochs_options,
        selection_metric="f1",
    )

    print("\n" + "=" * 70)
    print("VALIDATION HYPERPARAMETER SEARCH")
    print("=" * 70)
    print(validation_table.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print("\nBest hyperparameters selected using Validation F1-score:")
    print("Learning rate:", best_result["learning_rate"])
    print("Epochs:", best_result["epochs"])
    print("Validation Accuracy:", f"{best_result['validation_accuracy']:.4f}")
    print("Validation F1-score:", f"{best_result['validation_f1']:.4f}")

    w = best_result["weights"]
    losses = best_result["losses"]

    print("\nFinal selected weight vector [bias, features...]:")
    print(w)

    print("\nWeight order:")
    print(["bias"] + FEATURES)

    y_test_pred, test_scores = predict_linear(X_test_b, w)
    test_cm, test_metrics = binary_metrics(y_test, y_test_pred)
    print_metrics("FINAL TEST EVALUATION - 7 FEATURES", test_cm, test_metrics)

    error_mask = y_test_pred != y_test
    error_indices = idx_test[error_mask]

    error_table = df.iloc[error_indices].copy()
    error_table["actual"] = [INV_LABEL_MAP[v] for v in y_test[error_mask]]
    error_table["predicted"] = [INV_LABEL_MAP[v] for v in y_test_pred[error_mask]]
    error_table["z_score"] = test_scores[error_mask]

    error_table = error_table[
        ["Area", "Perimeter", "Class", "actual", "predicted", "z_score"]
    ].head(5)

    print("\n" + "=" * 70)
    print("FIRST FIVE MISCLASSIFIED TEST SAMPLES")
    print("=" * 70)

    if len(error_table) == 0:
        print("No misclassified samples in the Test set.")
    else:
        print(error_table.to_string(index=True))

    plot_loss(losses, output_path="q1_training_hinge_loss.png")

    return {
        "w": w,
        "losses": losses,
        "test_cm": test_cm,
        "test_metrics": test_metrics,
        "error_table": error_table,
        "validation_table": validation_table,
        "best_result": best_result,
        "split_data": {
            "X_train": X_train,
            "X_val": X_val,
            "X_test": X_test,
            "y_train": y_train,
            "y_val": y_val,
            "y_test": y_test,
            "idx_test": idx_test,
        },
    }


# ------------------------------------------------------------
# 2D Area/Perimeter experiment
# ------------------------------------------------------------

def run_2d_area_perimeter_experiment(df):
    features_2d = ["Area", "Perimeter"]

    X = df[features_2d].values.astype(float)
    y = df["Class"].map(LABEL_MAP).values.astype(int)

    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=RANDOM_SEED,
        stratify=y,
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.50,
        random_state=RANDOM_SEED,
        stratify=y_temp,
    )

    X_train_std, X_val_std, X_test_std, mean, std = standardize_with_train_statistics(
        X_train,
        X_val,
        X_test,
    )

    X_train_b = add_bias_column(X_train_std)
    X_val_b = add_bias_column(X_val_std)
    X_test_b = add_bias_column(X_test_std)

    learning_rates = [0.001, 0.005, 0.01, 0.05, 0.1]
    epochs_options = [500, 1000, 2000, 3000, 5000]

    best_result_2d, validation_table_2d = select_best_hyperparameters(
        X_train_b,
        y_train,
        X_val_b,
        y_val,
        learning_rates=learning_rates,
        epochs_options=epochs_options,
        selection_metric="f1",
    )

    w = best_result_2d["weights"]

    y_test_pred, _ = predict_linear(X_test_b, w)
    test_cm, test_metrics = binary_metrics(y_test, y_test_pred)

    print("\n" + "=" * 70)
    print("2D AREA/PERIMETER EXPERIMENT")
    print("=" * 70)

    print("\n2D validation hyperparameter search:")
    print(validation_table_2d.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print("\nBest 2D hyperparameters selected using Validation F1-score:")
    print("Learning rate:", best_result_2d["learning_rate"])
    print("Epochs:", best_result_2d["epochs"])
    print("Validation Accuracy:", f"{best_result_2d['validation_accuracy']:.4f}")
    print("Validation F1-score:", f"{best_result_2d['validation_f1']:.4f}")

    print("\n2D weights [bias, Area, Perimeter]:")
    print(w)

    print_metrics("FINAL TEST EVALUATION - 2D AREA/PERIMETER", test_cm, test_metrics)

    x_grid = np.linspace(X[:, 0].min(), X[:, 0].max(), 300)

    if np.isclose(w[2], 0.0):
        raise ValueError("Cannot draw boundary because the Perimeter weight is zero.")

    y_boundary = mean[1] + std[1] * (
        -(w[0] + w[1] * ((x_grid - mean[0]) / std[0])) / w[2]
    )

    plt.figure(figsize=(7, 5))
    plt.scatter(
        X[y == 0, 0],
        X[y == 0, 1],
        s=20,
        alpha=0.65,
        label="Kecimen (0)",
    )
    plt.scatter(
        X[y == 1, 0],
        X[y == 1, 1],
        s=20,
        alpha=0.65,
        label="Besni (1)",
    )
    plt.plot(
        x_grid,
        y_boundary,
        linewidth=2.5,
        label="Decision boundary (z=0)",
    )

    plt.xlabel("Area")
    plt.ylabel("Perimeter")
    plt.title("2D Linear Decision Boundary: Area vs Perimeter")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("q1_area_perimeter_decision_boundary.png", dpi=200)
    plt.close()

    return {
        "w_2d": w,
        "test_cm_2d": test_cm,
        "test_metrics_2d": test_metrics,
        "validation_table_2d": validation_table_2d,
        "best_result_2d": best_result_2d,
    }


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

if __name__ == "__main__":
    data = pd.read_excel(DATA_PATH)

    print_dataset_report(data)

    full_results = run_full_feature_experiment(data)
    two_d_results = run_2d_area_perimeter_experiment(data)

    print("\n" + "=" * 70)
    print("OUTPUT FILES SAVED")
    print("=" * 70)
    print("1) q1_training_hinge_loss.png")
    print("2) q1_area_perimeter_decision_boundary.png")

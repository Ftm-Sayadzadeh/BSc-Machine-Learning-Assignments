import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", None)

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

DATA_PATH = "003_balance-scale.data.csv"

CANONICAL_COLUMNS = [
    "class",
    "left_weight",
    "left_distance",
    "right_weight",
    "right_distance",
]

FEATURES = [
    "left_weight",
    "left_distance",
    "right_weight",
    "right_distance",
]

CLASS_ORDER = ["L", "B", "R"]
LABEL_MAP = {"L": 0, "B": 1, "R": 2}
INV_LABEL_MAP = {0: "L", 1: "B", 2: "R"}


# ------------------------------------------------------------
# Loading and preprocessing
# ------------------------------------------------------------

def load_balance_scale_data(path):
    df = pd.read_csv(path)

    if df.shape[1] != 5:
        raise ValueError("Expected 5 columns: class and four numeric features.")

    first_col_values = set(df.iloc[:, 0].astype(str).unique())

    if first_col_values.issubset({"L", "B", "R"}):
        df = df.copy()
        df.columns = CANONICAL_COLUMNS
    else:
        df = pd.read_csv(path, header=None)
        df.columns = CANONICAL_COLUMNS

    df["class"] = df["class"].astype(str)

    for feature in FEATURES:
        df[feature] = pd.to_numeric(df[feature], errors="coerce")

    df = df.dropna(subset=["class"] + FEATURES).copy()

    df["label"] = df["class"].map(LABEL_MAP).astype(int)

    df["torque_difference"] = (
        df["left_weight"] * df["left_distance"]
        - df["right_weight"] * df["right_distance"]
    )

    return df


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


# ------------------------------------------------------------
# Logistic Regression from scratch
# ------------------------------------------------------------

def stable_sigmoid(z):
    z = np.asarray(z)
    return np.where(
        z >= 0,
        1.0 / (1.0 + np.exp(-z)),
        np.exp(z) / (1.0 + np.exp(z)),
    )


def binary_cross_entropy_loss(w, X, y, lambda_):
    z = X @ w
    data_loss = np.mean(np.logaddexp(0.0, z) - y * z)
    reg_loss = (lambda_ / (2.0 * X.shape[0])) * np.sum(w[1:] ** 2)
    return data_loss + reg_loss


def logistic_gradient(w, X, y, lambda_):
    m = X.shape[0]
    p = stable_sigmoid(X @ w)

    grad = (X.T @ (p - y)) / m

    reg_grad = np.zeros_like(w)
    reg_grad[1:] = (lambda_ / m) * w[1:]

    return grad + reg_grad


def train_binary_logistic_regression(
    X_train,
    y_train_binary,
    X_val,
    y_val_binary,
    learning_rate,
    lambda_,
    epochs,
):
    w = np.zeros(X_train.shape[1])
    train_losses = []
    val_losses = []

    for _ in range(epochs):
        grad = logistic_gradient(w, X_train, y_train_binary, lambda_)
        w -= learning_rate * grad

        train_losses.append(binary_cross_entropy_loss(w, X_train, y_train_binary, lambda_))
        val_losses.append(binary_cross_entropy_loss(w, X_val, y_val_binary, lambda_))

    return w, np.array(train_losses), np.array(val_losses)


def train_one_vs_rest(
    X_train,
    y_train,
    X_val,
    y_val,
    num_classes,
    learning_rate,
    lambda_,
    epochs,
):
    weights = []
    all_train_losses = []
    all_val_losses = []

    for class_id in range(num_classes):
        y_train_binary = (y_train == class_id).astype(int)
        y_val_binary = (y_val == class_id).astype(int)

        w, train_losses, val_losses = train_binary_logistic_regression(
            X_train,
            y_train_binary,
            X_val,
            y_val_binary,
            learning_rate,
            lambda_,
            epochs,
        )

        weights.append(w)
        all_train_losses.append(train_losses)
        all_val_losses.append(val_losses)

    W = np.vstack(weights)
    train_losses = np.vstack(all_train_losses)
    val_losses = np.vstack(all_val_losses)

    return W, train_losses, val_losses


def predict_one_vs_rest(X, W):
    scores = stable_sigmoid(X @ W.T)
    predictions = np.argmax(scores, axis=1)
    return predictions, scores


# ------------------------------------------------------------
# Metrics
# ------------------------------------------------------------

def confusion_matrix_multiclass(y_true, y_pred, num_classes):
    cm = np.zeros((num_classes, num_classes), dtype=int)

    for actual, predicted in zip(y_true, y_pred):
        cm[int(actual), int(predicted)] += 1

    return cm


def multiclass_metrics(y_true, y_pred, num_classes):
    cm = confusion_matrix_multiclass(y_true, y_pred, num_classes)

    accuracy = np.trace(cm) / cm.sum()

    per_class = []

    for class_id in range(num_classes):
        tp = cm[class_id, class_id]
        fp = cm[:, class_id].sum() - tp
        fn = cm[class_id, :].sum() - tp

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2.0 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        per_class.append(
            {
                "class_id": class_id,
                "class_name": INV_LABEL_MAP[class_id],
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
                "support": int(cm[class_id, :].sum()),
            }
        )

    macro_f1 = float(np.mean([row["f1"] for row in per_class]))

    return cm, {
        "accuracy": float(accuracy),
        "macro_f1": macro_f1,
        "per_class": per_class,
    }


# ------------------------------------------------------------
# Validation hyperparameter selection
# ------------------------------------------------------------

def select_best_hyperparameters(
    X_train_b,
    y_train,
    X_val_b,
    y_val,
    learning_rates,
    lambdas,
    epochs_options,
):
    results = []
    best_result = None
    num_classes = len(CLASS_ORDER)

    for lr in learning_rates:
        for lambda_ in lambdas:
            for epochs in epochs_options:
                W, train_losses, val_losses = train_one_vs_rest(
                    X_train_b,
                    y_train,
                    X_val_b,
                    y_val,
                    num_classes,
                    learning_rate=lr,
                    lambda_=lambda_,
                    epochs=epochs,
                )

                y_val_pred, _ = predict_one_vs_rest(X_val_b, W)
                val_cm, val_metrics = multiclass_metrics(y_val, y_val_pred, num_classes)

                row = {
                    "learning_rate": lr,
                    "lambda": lambda_,
                    "epochs": epochs,
                    "validation_accuracy": val_metrics["accuracy"],
                    "validation_macro_f1": val_metrics["macro_f1"],
                    "validation_mean_loss": float(val_losses[:, -1].mean()),
                    "W": W,
                    "train_losses": train_losses,
                    "val_losses": val_losses,
                    "validation_cm": val_cm,
                    "validation_metrics": val_metrics,
                }

                results.append(row)

                if best_result is None:
                    best_result = row
                else:
                    current_key = (
                        row["validation_macro_f1"],
                        row["validation_accuracy"],
                        -row["validation_mean_loss"],
                    )
                    best_key = (
                        best_result["validation_macro_f1"],
                        best_result["validation_accuracy"],
                        -best_result["validation_mean_loss"],
                    )

                    if current_key > best_key:
                        best_result = row

    results_table = pd.DataFrame(
        [
            {
                "learning_rate": r["learning_rate"],
                "lambda": r["lambda"],
                "epochs": r["epochs"],
                "validation_accuracy": r["validation_accuracy"],
                "validation_macro_f1": r["validation_macro_f1"],
                "validation_mean_loss": r["validation_mean_loss"],
            }
            for r in results
        ]
    )

    return best_result, results_table


# ------------------------------------------------------------
# Reporting and plotting
# ------------------------------------------------------------

def print_dataset_report(df):
    print("=" * 70)
    print("DATASET REPORT")
    print("=" * 70)

    print("Number of samples:", len(df))
    print("Number of numeric features:", len(FEATURES))
    print("Number of classes:", df["class"].nunique())
    print("Class names:", CLASS_ORDER)

    print("\nClass counts:")
    print(df["class"].value_counts().reindex(CLASS_ORDER))

    print("\nFeature ranges:")
    print(df[FEATURES].agg(["min", "max"]).T)

    print("\nTorque difference summary by class:")
    print(df.groupby("class")["torque_difference"].describe().reindex(CLASS_ORDER))


def print_multiclass_results(title, cm, metrics):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)

    print("Confusion matrix, rows=true and columns=predicted:")
    print(pd.DataFrame(cm, index=CLASS_ORDER, columns=CLASS_ORDER))

    print("\nAccuracy:", f"{metrics['accuracy']:.4f}")
    print("Macro-F1:", f"{metrics['macro_f1']:.4f}")

    print("\nPer-class metrics:")
    per_class_table = pd.DataFrame(metrics["per_class"])
    print(per_class_table.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


def plot_2d_left_right_weight(df, output_path="q3_left_weight_right_weight_scatter.png"):
    rng = np.random.default_rng(RANDOM_SEED)

    plt.figure(figsize=(7, 5))

    for class_name in CLASS_ORDER:
        subset = df[df["class"] == class_name]

        x = subset["left_weight"].values.astype(float)
        y = subset["right_weight"].values.astype(float)

        jitter_x = rng.normal(0.0, 0.04, size=len(subset))
        jitter_y = rng.normal(0.0, 0.04, size=len(subset))

        plt.scatter(
            x + jitter_x,
            y + jitter_y,
            s=28,
            alpha=0.70,
            label=class_name,
        )

    plt.xlabel("left_weight")
    plt.ylabel("right_weight")
    plt.title("Balance Scale Data: left_weight vs right_weight")
    plt.xticks([1, 2, 3, 4, 5])
    plt.yticks([1, 2, 3, 4, 5])
    plt.legend(title="Class")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_ovr_loss_curves(train_losses, val_losses, output_path="q3_ovr_loss_curves.png"):
    epochs = np.arange(1, train_losses.shape[1] + 1)

    plt.figure(figsize=(7, 4.5))

    for class_id, class_name in enumerate(CLASS_ORDER):
        plt.plot(epochs, train_losses[class_id], label=f"Train {class_name}")
        plt.plot(epochs, val_losses[class_id], linestyle="--", label=f"Val {class_name}")

    plt.xlabel("Epoch")
    plt.ylabel("Binary Cross-Entropy Loss")
    plt.title("One-vs-Rest Logistic Regression Loss Curves")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


# ------------------------------------------------------------
# Main experiment
# ------------------------------------------------------------

def run_question_3():
    df = load_balance_scale_data(DATA_PATH)
    print_dataset_report(df)

    X = df[FEATURES].values.astype(float)
    y = df["label"].values.astype(int)
    indices = np.arange(len(df))

    X_train, X_temp, y_train, y_temp, idx_train, idx_temp = train_test_split(
        X,
        y,
        indices,
        test_size=0.40,
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

    learning_rates = [0.001, 0.01, 0.05, 0.1]
    lambdas = [0.0, 0.01, 0.1]
    epochs_options = [2000, 4000, 6000]

    best_result, validation_table = select_best_hyperparameters(
        X_train_b,
        y_train,
        X_val_b,
        y_val,
        learning_rates,
        lambdas,
        epochs_options,
    )

    print("\n" + "=" * 70)
    print("VALIDATION HYPERPARAMETER SEARCH")
    print("=" * 70)
    print(validation_table.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print("\nBest hyperparameters selected using Validation Macro-F1:")
    print("Learning rate:", best_result["learning_rate"])
    print("Lambda:", best_result["lambda"])
    print("Epochs:", best_result["epochs"])
    print("Validation Accuracy:", f"{best_result['validation_accuracy']:.4f}")
    print("Validation Macro-F1:", f"{best_result['validation_macro_f1']:.4f}")
    print("Validation Mean Loss:", f"{best_result['validation_mean_loss']:.4f}")

    W = best_result["W"]
    train_losses = best_result["train_losses"]
    val_losses = best_result["val_losses"]

    print("\nLearned weight matrix W:")
    print(pd.DataFrame(W, index=CLASS_ORDER, columns=["bias"] + FEATURES))

    y_test_pred, test_scores = predict_one_vs_rest(X_test_b, W)
    test_cm, test_metrics = multiclass_metrics(y_test, y_test_pred, len(CLASS_ORDER))

    print_multiclass_results("FINAL TEST EVALUATION", test_cm, test_metrics)

    test_df = df.iloc[idx_test].copy()
    test_df["actual"] = [INV_LABEL_MAP[v] for v in y_test]
    test_df["predicted"] = [INV_LABEL_MAP[v] for v in y_test_pred]

    for class_id, class_name in enumerate(CLASS_ORDER):
        test_df[f"score_{class_name}"] = test_scores[:, class_id]

    error_table = test_df[test_df["actual"] != test_df["predicted"]].copy()

    error_columns = (
        FEATURES
        + ["torque_difference", "class", "actual", "predicted"]
        + [f"score_{class_name}" for class_name in CLASS_ORDER]
    )

    error_table_to_print = error_table[error_columns].head(5)

    print("\n" + "=" * 70)
    print("FIRST FIVE MISCLASSIFIED TEST SAMPLES")
    print("=" * 70)

    if len(error_table_to_print) == 0:
        print("No misclassified samples in the Test set.")
    else:
        print(error_table_to_print.to_string(index=True, float_format=lambda x: f"{x:.4f}"))

    print("\n" + "=" * 70)
    print("ERROR ANALYSIS USING torque_difference")
    print("=" * 70)

    if len(error_table) == 0:
        print("No test errors to analyze.")
    else:
        print("Misclassified samples grouped by actual class:")
        print(error_table["actual"].value_counts().reindex(CLASS_ORDER).fillna(0).astype(int))

        print("\nAbsolute torque_difference summary:")
        error_table["abs_torque_difference"] = error_table["torque_difference"].abs()
        print(error_table["abs_torque_difference"].describe())

        print("\nMisclassified samples with torque_difference close to zero are usually near the balance boundary.")

    plot_2d_left_right_weight(df, output_path="q3_left_weight_right_weight_scatter.png")
    plot_ovr_loss_curves(train_losses, val_losses, output_path="q3_ovr_loss_curves.png")

    validation_table.to_csv("q3_validation_hyperparameter_search.csv", index=False)
    pd.DataFrame(W, index=CLASS_ORDER, columns=["bias"] + FEATURES).to_csv("q3_weight_matrix_W.csv")
    error_table_to_print.to_csv("q3_misclassified_samples.csv", index=True)

    print("\n" + "=" * 70)
    print("OUTPUT FILES SAVED")
    print("=" * 70)
    print("1) q3_left_weight_right_weight_scatter.png")
    print("2) q3_ovr_loss_curves.png")
    print("3) q3_validation_hyperparameter_search.csv")
    print("4) q3_weight_matrix_W.csv")
    print("5) q3_misclassified_samples.csv")


if __name__ == "__main__":
    run_question_3()

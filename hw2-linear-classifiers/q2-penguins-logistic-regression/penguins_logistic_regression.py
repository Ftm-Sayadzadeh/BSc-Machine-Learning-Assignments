import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

DATA_PATH = "002_penguins.csv"

FEATURES = [
    "bill_length_mm",
    "bill_depth_mm",
    "flipper_length_mm",
    "body_mass_g",
]

TARGET_COLUMN = "species"

LABEL_MAP = {"Adelie": 0, "Gentoo": 1}
INV_LABEL_MAP = {0: "Adelie", 1: "Gentoo"}


# ------------------------------------------------------------
# Preprocessing utilities
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


def load_and_preprocess_penguins(path):
    df = pd.read_csv(path)

    df = df[df[TARGET_COLUMN].isin(["Adelie", "Gentoo"])].copy()
    df = df[[TARGET_COLUMN] + FEATURES].dropna().copy()

    df["label"] = df[TARGET_COLUMN].map(LABEL_MAP).astype(int)

    return df


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
    z = X @ w
    p = stable_sigmoid(z)

    grad = (X.T @ (p - y)) / m

    reg_grad = np.zeros_like(w)
    reg_grad[1:] = (lambda_ / m) * w[1:]

    return grad + reg_grad


def train_logistic_regression(
    X_train,
    y_train,
    X_val,
    y_val,
    learning_rate=0.01,
    lambda_=0.0,
    epochs=3000,
):
    w = np.zeros(X_train.shape[1])
    train_losses = []
    val_losses = []

    for _ in range(epochs):
        grad = logistic_gradient(w, X_train, y_train, lambda_)
        w -= learning_rate * grad

        train_losses.append(binary_cross_entropy_loss(w, X_train, y_train, lambda_))
        val_losses.append(binary_cross_entropy_loss(w, X_val, y_val, lambda_))

    return w, np.array(train_losses), np.array(val_losses)


def predict_proba(X, w):
    return stable_sigmoid(X @ w)


def predict_labels(X, w, threshold=0.5):
    probabilities = predict_proba(X, w)
    predictions = (probabilities >= threshold).astype(int)
    return predictions, probabilities


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
    print("DATASET REPORT AFTER FILTERING AND DROPPING MISSING VALUES")
    print("=" * 70)

    print("Number of samples:", len(df))
    print("Number of numeric features:", len(FEATURES))
    print("Class names:", list(df[TARGET_COLUMN].unique()))

    print("\nClass counts:")
    print(df[TARGET_COLUMN].value_counts())

    print("\nMissing values after cleaning:")
    print(df[[TARGET_COLUMN] + FEATURES].isna().sum())

    print("\nFeature ranges:")
    print(df[FEATURES].agg(["min", "max"]).T)


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


def plot_loss_curves(train_losses, val_losses, output_path="q2_logistic_loss_curves.png"):
    epochs = np.arange(1, len(train_losses) + 1)

    plt.figure(figsize=(7, 4.5))
    plt.plot(epochs, train_losses, label="Training Loss")
    plt.plot(epochs, val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Binary Cross-Entropy Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


# ------------------------------------------------------------
# Hyperparameter selection using Validation set
# ------------------------------------------------------------

def select_best_hyperparameters(
    X_train_b,
    y_train,
    X_val_b,
    y_val,
    learning_rates,
    lambdas,
    epochs,
    selection_metric="f1",
):
    results = []
    best_result = None

    for lr in learning_rates:
        for lambda_ in lambdas:
            w, train_losses, val_losses = train_logistic_regression(
                X_train_b,
                y_train,
                X_val_b,
                y_val,
                learning_rate=lr,
                lambda_=lambda_,
                epochs=epochs,
            )

            y_val_pred, y_val_prob = predict_labels(X_val_b, w)
            val_cm, val_metrics = binary_metrics(y_val, y_val_pred)

            row = {
                "learning_rate": lr,
                "lambda": lambda_,
                "epochs": epochs,
                "validation_loss": float(val_losses[-1]),
                "validation_accuracy": val_metrics["accuracy"],
                "validation_precision": val_metrics["precision"],
                "validation_recall": val_metrics["recall"],
                "validation_f1": val_metrics["f1"],
                "weights": w,
                "train_losses": train_losses,
                "val_losses": val_losses,
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
                        and row["validation_loss"] < best_result["validation_loss"]
                    )
                ):
                    best_result = row

    results_table = pd.DataFrame(
        [
            {
                "learning_rate": r["learning_rate"],
                "lambda": r["lambda"],
                "epochs": r["epochs"],
                "validation_loss": r["validation_loss"],
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
# Main experiment
# ------------------------------------------------------------

def run_question_2():
    df = load_and_preprocess_penguins(DATA_PATH)
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

    learning_rates = [0.001, 0.01, 0.1]
    lambdas = [0.0, 0.01, 0.1]
    epochs = 3000

    best_result, validation_table = select_best_hyperparameters(
        X_train_b,
        y_train,
        X_val_b,
        y_val,
        learning_rates=learning_rates,
        lambdas=lambdas,
        epochs=epochs,
        selection_metric="f1",
    )

    print("\n" + "=" * 70)
    print("VALIDATION HYPERPARAMETER SEARCH")
    print("=" * 70)
    print(validation_table.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print("\nBest hyperparameters selected using Validation F1-score:")
    print("Learning rate:", best_result["learning_rate"])
    print("Lambda:", best_result["lambda"])
    print("Epochs:", best_result["epochs"])
    print("Validation Loss:", f"{best_result['validation_loss']:.4f}")
    print("Validation Accuracy:", f"{best_result['validation_accuracy']:.4f}")
    print("Validation F1-score:", f"{best_result['validation_f1']:.4f}")

    w = best_result["weights"]
    train_losses = best_result["train_losses"]
    val_losses = best_result["val_losses"]

    print("\nFinal selected weight vector [bias, features...]:")
    print(w)

    print("\nWeight order:")
    print(["bias"] + FEATURES)

    y_test_pred, y_test_prob = predict_labels(X_test_b, w)
    test_cm, test_metrics = binary_metrics(y_test, y_test_pred)
    print_metrics("FINAL TEST EVALUATION", test_cm, test_metrics)

    test_sample_table = df.iloc[idx_test].copy()
    test_sample_table["actual"] = [INV_LABEL_MAP[v] for v in y_test]
    test_sample_table["predicted"] = [INV_LABEL_MAP[v] for v in y_test_pred]
    test_sample_table["predicted_probability_Gentoo"] = y_test_prob

    test_sample_table = test_sample_table[
        FEATURES + ["species", "actual", "predicted", "predicted_probability_Gentoo"]
    ].head(5)

    print("\n" + "=" * 70)
    print("FIVE TEST SAMPLES WITH PREDICTED PROBABILITY")
    print("=" * 70)
    print(test_sample_table.to_string(index=True))

    plot_loss_curves(
        train_losses,
        val_losses,
        output_path="q2_logistic_loss_curves.png",
    )

    validation_table.to_csv("q2_validation_hyperparameter_search.csv", index=False)

    print("\n" + "=" * 70)
    print("OUTPUT FILES SAVED")
    print("=" * 70)
    print("1) q2_logistic_loss_curves.png")
    print("2) q2_validation_hyperparameter_search.csv")


if __name__ == "__main__":
    run_question_2()

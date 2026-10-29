import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


# ==========================================
# Configuration
# ==========================================
DATA_FILE = "house-prices.csv"
OUTPUT_DIR = "outputs_corrected"
RANDOM_STATE = 42
TEST_SIZE = 0.10

FEATURE_COLUMNS = ["SqFt", "Bedrooms", "Bathrooms", "Offers", "Brick", "Neighborhood"]
TARGET_COLUMN = "Price"
NUMERICAL_FEATURES = ["SqFt", "Bedrooms", "Bathrooms", "Offers"]
CATEGORICAL_FEATURES = ["Brick", "Neighborhood"]
DISCRETE_COUNT_FEATURES = ["Bedrooms", "Bathrooms", "Offers"]


# ==========================================
# Utility functions
# ==========================================
def ensure_output_dir(path: str = OUTPUT_DIR) -> None:
    os.makedirs(path, exist_ok=True)


def save_current_plot(filename: str) -> None:
    """Save the current matplotlib figure and close it."""
    ensure_output_dir()
    path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


# ==========================================
# 1. Data loading and inspection
# ==========================================
def load_data(filename: str) -> pd.DataFrame:
    df = pd.read_csv(filename)
    expected = FEATURE_COLUMNS + [TARGET_COLUMN]
    missing_cols = [col for col in expected if col not in df.columns]
    if missing_cols:
        raise ValueError(f"These required columns are missing from the dataset: {missing_cols}")
    return df[expected].copy()


def write_initial_inspection(df: pd.DataFrame, output_file: str) -> None:
    ensure_output_dir()
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("===== INITIAL DATA INSPECTION =====\n\n")
        f.write("1) df.info():\n")
        df.info(buf=f)
        f.write("\n\n2) Null counts using df.isna().sum():\n")
        f.write(df.isna().sum().to_string())
        f.write("\n\n\n3) Descriptive statistics using df.describe():\n")
        f.write(df.describe(include="all").to_string())
        f.write("\n")


# ==========================================
# 2. Train/test-safe preprocessing
# ==========================================
def split_X_y(df: pd.DataFrame):
    # Corrected: explicitly use the assignment feature list.
    X = df[FEATURE_COLUMNS].copy()
    y = df[TARGET_COLUMN].copy()
    return X, y


def fit_impute_train(X_train: pd.DataFrame):
    """Fit numerical means and categorical modes only on the training set."""
    X = X_train.copy()
    num_means = {col: X[col].mean() for col in NUMERICAL_FEATURES}
    cat_modes = {col: X[col].mode(dropna=True)[0] for col in CATEGORICAL_FEATURES}

    for col, val in num_means.items():
        X[col] = X[col].fillna(val)
    for col, val in cat_modes.items():
        X[col] = X[col].fillna(val)
    return X, num_means, cat_modes


def transform_impute(X: pd.DataFrame, num_means: dict, cat_modes: dict):
    X = X.copy()
    for col, val in num_means.items():
        X[col] = X[col].fillna(val)
    for col, val in cat_modes.items():
        X[col] = X[col].fillna(val)
    return X


def detect_outliers_zscore(X_train: pd.DataFrame, threshold: float = 3.0):
    """Detect outliers from train only using Z-score."""
    report = {}
    for col in NUMERICAL_FEATURES:
        mean = X_train[col].mean()
        std = X_train[col].std(ddof=0)
        if std == 0 or np.isnan(std):
            mask = pd.Series(False, index=X_train.index)
        else:
            z = (X_train[col] - mean) / std
            mask = z.abs() > threshold
        report[col] = {
            "count": int(mask.sum()),
            "indices": list(X_train.index[mask]),
            "values": list(X_train.loc[mask, col]),
            "mean": float(mean),
            "std": float(std) if not np.isnan(std) else 0.0,
        }
    return report


def handle_outliers_conservatively(X_train: pd.DataFrame, X_test: pd.DataFrame, outlier_report: dict):
    """
    Conservative approach:
    - Report all outliers.
    - Do not automatically replace valid discrete/count variables such as Offers=6.
    - Winsorize only continuous numerical variables if extreme values are detected.
    In this dataset, SqFt is the main continuous predictor. Bedrooms, Bathrooms, and Offers are meaningful counts.
    """
    Xtr = X_train.copy()
    Xte = X_test.copy()
    treatment_log = []

    continuous_cols = ["SqFt"]
    for col in NUMERICAL_FEATURES:
        count = outlier_report[col]["count"]
        values = outlier_report[col]["values"]
        if count == 0:
            treatment_log.append(f"{col}: no outlier detected.")
            continue

        if col in continuous_cols:
            # Clip continuous extreme values to train 1st/99th percentiles.
            lower, upper = Xtr[col].quantile([0.01, 0.99])
            Xtr[col] = Xtr[col].clip(lower=lower, upper=upper)
            Xte[col] = Xte[col].clip(lower=lower, upper=upper)
            treatment_log.append(
                f"{col}: {count} outlier(s) detected and winsorized to train 1st/99th percentiles."
            )
        else:
            treatment_log.append(
                f"{col}: {count} outlier(s) detected with value(s) {values}; kept because this is a valid count-type feature."
            )

    return Xtr, Xte, treatment_log


def encode_categorical_train_test(X_train: pd.DataFrame, X_test: pd.DataFrame):
    X_train_enc = pd.get_dummies(X_train, columns=CATEGORICAL_FEATURES, drop_first=True)
    X_test_enc = pd.get_dummies(X_test, columns=CATEGORICAL_FEATURES, drop_first=True)
    X_test_enc = X_test_enc.reindex(columns=X_train_enc.columns, fill_value=0)
    return X_train_enc.astype(float), X_test_enc.astype(float)


def fit_minmax_train(X_train: pd.DataFrame):
    X = X_train.copy().astype(float)
    stats = {}
    for col in X.columns:
        min_val = X[col].min()
        max_val = X[col].max()
        stats[col] = {"min": float(min_val), "max": float(max_val)}
        if max_val != min_val:
            X[col] = (X[col] - min_val) / (max_val - min_val)
        else:
            X[col] = 0.0
    return X, stats


def transform_minmax(X: pd.DataFrame, stats: dict):
    X = X.copy().astype(float)
    for col, s in stats.items():
        min_val, max_val = s["min"], s["max"]
        if max_val != min_val:
            X[col] = (X[col] - min_val) / (max_val - min_val)
        else:
            X[col] = 0.0
    return X


def fit_scale_target(y_train: pd.Series):
    y_min = float(y_train.min())
    y_max = float(y_train.max())
    if y_max == y_min:
        raise ValueError("Target has zero range; Min-Max target scaling is not possible.")
    y_scaled = (y_train - y_min) / (y_max - y_min)
    return y_scaled.to_numpy(dtype=float), {"min": y_min, "max": y_max}


def transform_target(y: pd.Series, stats: dict):
    return ((y - stats["min"]) / (stats["max"] - stats["min"])).to_numpy(dtype=float)


def inverse_transform_target(y_scaled, stats: dict):
    return np.asarray(y_scaled) * (stats["max"] - stats["min"]) + stats["min"]


# ==========================================
# 3. Visualizations
# ==========================================
def create_eda_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Only for visualization: fill missing values so plots are complete."""
    df_eda = df.copy()
    for col in NUMERICAL_FEATURES:
        df_eda[col] = df_eda[col].fillna(df_eda[col].mean())
    for col in CATEGORICAL_FEATURES:
        df_eda[col] = df_eda[col].fillna(df_eda[col].mode(dropna=True)[0])
    return df_eda


def plot_correlation_heatmap(df_eda: pd.DataFrame):
    plt.figure(figsize=(8, 6))
    numeric_df = df_eda.select_dtypes(include=[np.number])
    sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Heatmap of Housing Features")
    save_current_plot("01_correlation_heatmap.png")


def plot_regression_relations(df_eda: pd.DataFrame):
    for col in NUMERICAL_FEATURES:
        plt.figure(figsize=(6, 4))
        sns.regplot(data=df_eda, x=col, y=TARGET_COLUMN, line_kws={"color": "red"})
        plt.title(f"Regression: {col} vs {TARGET_COLUMN}")
        save_current_plot(f"02_regplot_{col}.png")


def plot_categorical_impact(df_eda: pd.DataFrame):
    for col in CATEGORICAL_FEATURES:
        plt.figure(figsize=(6, 4))
        sns.boxplot(x=col, y=TARGET_COLUMN, data=df_eda)
        plt.title(f"{TARGET_COLUMN} vs {col}")
        save_current_plot(f"03_boxplot_{col}.png")


def plot_encoded_feature_relations(X_train_enc: pd.DataFrame, y_train: pd.Series):
    df_plot = X_train_enc.copy()
    df_plot[TARGET_COLUMN] = y_train.values
    for feature in X_train_enc.columns:
        plt.figure(figsize=(6, 4))
        sns.scatterplot(data=df_plot, x=feature, y=TARGET_COLUMN)
        plt.title(f"{feature} vs {TARGET_COLUMN} after encoding")
        save_current_plot(f"04_encoded_scatter_{feature}.png")


def plot_loss_curve(history):
    plt.figure(figsize=(7, 4))
    plt.plot(history)
    plt.xlabel("Epoch")
    plt.ylabel("RMSE Loss on scaled target")
    plt.title("Custom Linear Regression - RMSE Gradient Descent")
    save_current_plot("05_rmse_loss_curve.png")


# ==========================================
# 4. Custom Linear Regression with RMSE Gradient Descent
# ==========================================
class LinearRegressionGD:
    """Linear Regression optimized from scratch using RMSE gradient descent."""

    def __init__(self, learning_rate=0.05, epochs=5000, tolerance=1e-12, verbose=True):
        self.lr = learning_rate
        self.epochs = epochs
        self.tolerance = tolerance
        self.verbose = verbose
        self.w = None
        self.b = None
        self.rmse_history = []

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features, dtype=float)
        self.b = 0.0

        for epoch in range(self.epochs):
            y_pred = np.dot(X, self.w) + self.b
            error = y_pred - y
            loss = np.sqrt(np.mean(error ** 2))
            self.rmse_history.append(float(loss))

            # Gradient of RMSE = X.T @ error / (n * RMSE)
            # epsilon prevents division by zero near perfect fit.
            denom = max(loss, self.tolerance)
            dw = np.dot(X.T, error) / (n_samples * denom)
            db = np.sum(error) / (n_samples * denom)

            self.w -= self.lr * dw
            self.b -= self.lr * db

            if self.verbose and epoch % 1000 == 0:
                print(f"Epoch {epoch:<5} | scaled RMSE: {loss:.6f}")

        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        return np.dot(X, self.w) + self.b

    def score(self, X, y):
        return rmse(y, self.predict(X))


# ==========================================
# 5. Main execution
# ==========================================
def main():
    ensure_output_dir()
    df = load_data(DATA_FILE)

    inspection_path = os.path.join(OUTPUT_DIR, "report_outputs_corrected.txt")
    write_initial_inspection(df, inspection_path)

    print("=== Initial inspection ===")
    print(df.info())
    print("\nNull counts:\n", df.isna().sum())
    print("\nDescribe:\n", df.describe(include="all"))

    # Visualization after EDA-only imputation so that plots do not silently drop missing rows.
    df_eda = create_eda_dataframe(df)
    plot_correlation_heatmap(df_eda)
    plot_regression_relations(df_eda)
    plot_categorical_impact(df_eda)

    # Split before fit-based preprocessing to prevent data leakage.
    X, y = split_X_y(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    # Missing values: fit on Train, transform Train/Test.
    X_train_imp, num_means, cat_modes = fit_impute_train(X_train)
    X_test_imp = transform_impute(X_test, num_means, cat_modes)

    # Outliers: detect on Train only and treat conservatively.
    outlier_report = detect_outliers_zscore(X_train_imp, threshold=3.0)
    X_train_out, X_test_out, treatment_log = handle_outliers_conservatively(
        X_train_imp, X_test_imp, outlier_report
    )

    # Encoding: train columns define the schema.
    X_train_enc, X_test_enc = encode_categorical_train_test(X_train_out, X_test_out)
    plot_encoded_feature_relations(X_train_enc, y_train)

    # Feature normalization: fit on Train only.
    X_train_norm, x_scaler = fit_minmax_train(X_train_enc)
    X_test_norm = transform_minmax(X_test_enc, x_scaler)

    # Target scaling for stable direct RMSE optimization.
    y_train_scaled, y_scaler = fit_scale_target(y_train)
    y_test_scaled = transform_target(y_test, y_scaler)

    # Custom model using true RMSE gradient.
    print("\n=== Training custom LinearRegressionGD with RMSE gradient ===")
    model_custom = LinearRegressionGD(learning_rate=0.05, epochs=5000, verbose=True)
    model_custom.fit(X_train_norm.values, y_train_scaled)
    plot_loss_curve(model_custom.rmse_history)

    pred_train_scaled = model_custom.predict(X_train_norm.values)
    pred_test_scaled = model_custom.predict(X_test_norm.values)
    pred_train = inverse_transform_target(pred_train_scaled, y_scaler)
    pred_test = inverse_transform_target(pred_test_scaled, y_scaler)

    train_rmse_custom = rmse(y_train.values, pred_train)
    test_rmse_custom = rmse(y_test.values, pred_test)

    # Scikit-Learn comparison on the same scaled target and same normalized features.
    model_sk = LinearRegression()
    model_sk.fit(X_train_norm.values, y_train_scaled)
    sk_train = inverse_transform_target(model_sk.predict(X_train_norm.values), y_scaler)
    sk_test = inverse_transform_target(model_sk.predict(X_test_norm.values), y_scaler)
    train_rmse_sk = rmse(y_train.values, sk_train)
    test_rmse_sk = rmse(y_test.values, sk_test)

    # Save summary outputs.
    with open(inspection_path, "a", encoding="utf-8") as f:
        f.write("\n\n===== PREPROCESSING SUMMARY =====\n")
        f.write("Numerical means fitted on train:\n")
        f.write(str(num_means))
        f.write("\n\nCategorical modes fitted on train:\n")
        f.write(str(cat_modes))
        f.write("\n\nOutlier report fitted on train:\n")
        for col, info in outlier_report.items():
            f.write(f"\n{col}: count={info['count']}, values={info['values']}")
        f.write("\n\nOutlier treatment decisions:\n")
        for line in treatment_log:
            f.write(f"\n- {line}")
        f.write("\n\nEncoded columns:\n")
        f.write(str(list(X_train_enc.columns)))
        f.write("\n\n===== FINAL MODEL PERFORMANCE =====\n")
        f.write(f"Custom RMSE-GD - Train RMSE: {train_rmse_custom:.2f}\n")
        f.write(f"Custom RMSE-GD - Test RMSE:  {test_rmse_custom:.2f}\n")
        f.write(f"Scikit-Learn   - Train RMSE: {train_rmse_sk:.2f}\n")
        f.write(f"Scikit-Learn   - Test RMSE:  {test_rmse_sk:.2f}\n")

    print("\n========== FINAL RESULTS ==========")
    print(f"Custom RMSE-GD - Train RMSE: ${train_rmse_custom:.2f}")
    print(f"Custom RMSE-GD - Test RMSE:  ${test_rmse_custom:.2f}")
    print(f"Scikit-Learn   - Train RMSE: ${train_rmse_sk:.2f}")
    print(f"Scikit-Learn   - Test RMSE:  ${test_rmse_sk:.2f}")
    print(f"\nAll figures and report outputs were saved in: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
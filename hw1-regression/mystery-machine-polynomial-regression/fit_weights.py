"""Recover the weights of the mystery cubic polynomial from train.xlsx.

This is the training step that was commented out of mystery_function.py before
submission (the submitted version has the learned weights hard-coded, because
the grader did not allow pandas at prediction time).

Usage:
    python fit_weights.py
"""
import numpy as np
import pandas as pd


def build_features(x, y, z):
    """Design matrix with 20 columns: bias, pure powers, and cross terms."""
    return np.column_stack([
        np.ones(x.shape[0]),
        x, x ** 2, x ** 3,
        y, y ** 2, y ** 3,
        z, z ** 2, z ** 3,
        x * y, x ** 2 * y, x * y ** 2,
        x * z, x ** 2 * z, x * z ** 2,
        y * z, y ** 2 * z, y * z ** 2,
        x * y * z,
    ])


def main():
    df = pd.read_excel("train.xlsx")
    x = df["x"].values
    y = df["y"].values
    z = df["z"].values
    target = df["F(x, y, z)"].values

    X = build_features(x, y, z)

    # Normal equations (X^T X) w = X^T f, solved without forming an explicit inverse.
    w = np.linalg.solve(X.T @ X, X.T @ target)

    rmse = np.sqrt(np.mean((X @ w - target) ** 2))
    print(f"Training RMSE: {rmse:.6f}\n")

    print("Calculated weights (b, w_1 ... w_19), rounded to 4 decimals:")
    print(f"b    = {np.round(w[0], 4):.4f}")
    for i, coef in enumerate(w[1:], start=1):
        print(f"w_{i:<2} = {np.round(coef, 4):.4f}")


if __name__ == "__main__":
    main()

import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

SEED = 42
np.random.seed(SEED)

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

TRUE_W = 2.0
TRUE_B = 5.0

N_SAMPLES = 1000
NOISE_STD = 0.5

EPOCHS = 500
LEARNING_RATE = 0.01
MINI_BATCH_SIZE = 64


def generate_data(n_samples=1000, noise_std=0.5, seed=42):
    np.random.seed(seed)

    x = np.linspace(0, 10, n_samples)
    noise = np.random.normal(0, noise_std, size=n_samples)
    y = TRUE_W * x + TRUE_B + noise

    return x, y


def create_mini_batches(x, y, batch_size=64, shuffle=True):
    indices = np.arange(len(x))

    if shuffle:
        np.random.shuffle(indices)

    x_shuffled = x[indices]
    y_shuffled = y[indices]

    batches = []

    for start in range(0, len(x), batch_size):
        end = start + batch_size
        batches.append((x_shuffled[start:end], y_shuffled[start:end]))

    return batches


def predict(x, w, b):
    return w * x + b


def mse_loss(x, y, w, b):
    y_pred = predict(x, w, b)
    return np.mean((y - y_pred) ** 2)


def compute_gradients(x, y, w, b):
    m = len(x)
    y_pred = predict(x, w, b)
    error = y_pred - y

    dw = (2 / m) * np.sum(error * x)
    db = (2 / m) * np.sum(error)

    return dw, db


def batch_gradient_descent(x, y, learning_rate=0.01, epochs=80, init_w=0.0, init_b=0.0):
    w = init_w
    b = init_b

    losses = []
    path = []
    updates = 0

    start_time = time.perf_counter()

    for epoch in range(epochs):
        dw, db = compute_gradients(x, y, w, b)

        w -= learning_rate * dw
        b -= learning_rate * db

        updates += 1

        loss = mse_loss(x, y, w, b)
        losses.append(loss)
        path.append((w, b))

    execution_time = time.perf_counter() - start_time

    return {
        "method": "Batch GD",
        "w": w,
        "b": b,
        "losses": losses,
        "path": path,
        "updates": updates,
        "time": execution_time,
    }


def stochastic_gradient_descent(x, y, learning_rate=0.01, epochs=80, init_w=0.0, init_b=0.0):
    w = init_w
    b = init_b

    n = len(x)

    losses = []
    path = []
    updates = 0

    start_time = time.perf_counter()

    for epoch in range(epochs):
        indices = np.random.permutation(n)

        current_lr = learning_rate / (1 + 0.01 * epoch)

        for idx in indices:
            xi = x[idx:idx + 1]
            yi = y[idx:idx + 1]

            dw, db = compute_gradients(xi, yi, w, b)

            w -= current_lr * dw
            b -= current_lr * db

            updates += 1
            path.append((w, b))

        loss = mse_loss(x, y, w, b)
        losses.append(loss)

    execution_time = time.perf_counter() - start_time

    return {
        "method": "SGD",
        "w": w,
        "b": b,
        "losses": losses,
        "path": path,
        "updates": updates,
        "time": execution_time,
    }


def mini_batch_gradient_descent(
    x,
    y,
    learning_rate=0.01,
    epochs=80,
    batch_size=64,
    init_w=0.0,
    init_b=0.0
):
    w = init_w
    b = init_b

    losses = []
    path = []
    updates = 0

    start_time = time.perf_counter()

    for epoch in range(epochs):
        batches = create_mini_batches(x, y, batch_size=batch_size, shuffle=True)

        for x_batch, y_batch in batches:
            dw, db = compute_gradients(x_batch, y_batch, w, b)

            w -= learning_rate * dw
            b -= learning_rate * db

            updates += 1
            path.append((w, b))

        loss = mse_loss(x, y, w, b)
        losses.append(loss)

    execution_time = time.perf_counter() - start_time

    return {
        "method": "Mini-Batch GD",
        "w": w,
        "b": b,
        "losses": losses,
        "path": path,
        "updates": updates,
        "time": execution_time,
    }


def plot_data(x, y):
    plt.figure(figsize=(8, 5))
    plt.scatter(x, y, s=12, alpha=0.6, label="Noisy Data")
    plt.plot(x, TRUE_W * x + TRUE_B, linewidth=2, label="True Line: y = 2x + 5")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Synthetic Linear Dataset")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    path = os.path.join(OUTPUT_DIR, "data_plot.png")
    plt.savefig(path, dpi=300)
    plt.close()

    return path


def plot_loss(results):
    plt.figure(figsize=(9, 5))

    for result in results:
        plt.plot(
            np.arange(1, len(result["losses"]) + 1),
            result["losses"],
            linewidth=2,
            label=result["method"]
        )

    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("Loss Comparison")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    path = os.path.join(OUTPUT_DIR, "loss_comparison.png")
    plt.savefig(path, dpi=300)
    plt.close()

    return path


def plot_contour_paths(x, y, results):
    all_w = []
    all_b = []

    for result in results:
        path_array = np.array(result["path"])
        all_w.extend(path_array[:, 0])
        all_b.extend(path_array[:, 1])

    w_min = min(min(all_w), TRUE_W) - 1
    w_max = max(max(all_w), TRUE_W) + 1
    b_min = min(min(all_b), TRUE_B) - 2
    b_max = max(max(all_b), TRUE_B) + 2

    w_values = np.linspace(w_min, w_max, 180)
    b_values = np.linspace(b_min, b_max, 180)

    W, B = np.meshgrid(w_values, b_values)
    Z = np.zeros_like(W)

    for i in range(W.shape[0]):
        for j in range(W.shape[1]):
            Z[i, j] = mse_loss(x, y, W[i, j], B[i, j])

    plt.figure(figsize=(9, 7))
    contour = plt.contour(W, B, Z, levels=40)
    plt.clabel(contour, inline=True, fontsize=7)

    for result in results:
        path_array = np.array(result["path"])

        max_points = 400
        if len(path_array) > max_points:
            selected_indices = np.linspace(0, len(path_array) - 1, max_points).astype(int)
            path_array = path_array[selected_indices]

        plt.plot(
            path_array[:, 0],
            path_array[:, 1],
            marker="o",
            markersize=2,
            linewidth=1.5,
            label=result["method"]
        )

    plt.scatter([TRUE_W], [TRUE_B], marker="*", s=180, label="True Parameters")
    plt.xlabel("Weight (w)")
    plt.ylabel("Bias (b)")
    plt.title("Contour Plot of MSE Cost Function and Optimization Paths")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    path = os.path.join(OUTPUT_DIR, "contour_paths.png")
    plt.savefig(path, dpi=300)
    plt.close()

    return path


def plot_time_comparison(results):
    methods = [r["method"] for r in results]
    times = [r["time"] for r in results]

    plt.figure(figsize=(8, 5))
    plt.bar(methods, times)
    plt.ylabel("Execution Time (seconds)")
    plt.title("Execution Time Comparison")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    path = os.path.join(OUTPUT_DIR, "time_comparison.png")
    plt.savefig(path, dpi=300)
    plt.close()

    return path


def plot_final_fit(x, y, results):
    plt.figure(figsize=(9, 5))
    plt.scatter(x, y, s=12, alpha=0.45, label="Data")

    plt.plot(x, TRUE_W * x + TRUE_B, linewidth=3, label="True Line")

    for result in results:
        y_pred = predict(x, result["w"], result["b"])
        plt.plot(x, y_pred, linewidth=2, label=result["method"])

    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Final Fitted Lines")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    path = os.path.join(OUTPUT_DIR, "final_fit_comparison.png")
    plt.savefig(path, dpi=300)
    plt.close()

    return path


def make_results_dataframe(results):
    rows = []

    for result in results:
        rows.append({
            "Method": result["method"],
            "Final w": result["w"],
            "Final b": result["b"],
            "Final Loss": result["losses"][-1],
            "Total Updates": result["updates"],
            "Execution Time (sec)": result["time"],
        })

    df = pd.DataFrame(rows)

    csv_path = os.path.join(OUTPUT_DIR, "results.csv")
    df.to_csv(csv_path, index=False)

    return df, csv_path

def main():
    x, y = generate_data(
        n_samples=N_SAMPLES,
        noise_std=NOISE_STD,
        seed=SEED
    )

    batches = create_mini_batches(
        x,
        y,
        batch_size=MINI_BATCH_SIZE,
        shuffle=True
    )

    print("Dataset created.")
    print(f"Number of samples: {len(x)}")
    print(f"Number of mini-batches: {len(batches)}")
    print("-" * 60)

    init_w = 0.0
    init_b = 0.0

    bgd_result = batch_gradient_descent(
        x,
        y,
        learning_rate=LEARNING_RATE,
        epochs=EPOCHS,
        init_w=init_w,
        init_b=init_b
    )

    sgd_result = stochastic_gradient_descent(
        x,
        y,
        learning_rate=LEARNING_RATE,
        epochs=EPOCHS,
        init_w=init_w,
        init_b=init_b
    )

    mbgd_result = mini_batch_gradient_descent(
        x,
        y,
        learning_rate=LEARNING_RATE,
        epochs=EPOCHS,
        batch_size=MINI_BATCH_SIZE,
        init_w=init_w,
        init_b=init_b
    )

    results = [bgd_result, sgd_result, mbgd_result]

    print("Final Results:")
    for r in results:
        print(
            f"{r['method']}: "
            f"w = {r['w']:.6f}, "
            f"b = {r['b']:.6f}, "
            f"loss = {r['losses'][-1]:.6f}, "
            f"updates = {r['updates']}, "
            f"time = {r['time']:.6f} sec"
        )

    print("-" * 60)

    data_plot_path = plot_data(x, y)
    loss_plot_path = plot_loss(results)
    contour_plot_path = plot_contour_paths(x, y, results)
    time_plot_path = plot_time_comparison(results)
    fit_plot_path = plot_final_fit(x, y, results)

    results_df, csv_path = make_results_dataframe(results)

    print("Files saved:")
    print(data_plot_path)
    print(loss_plot_path)
    print(contour_plot_path)
    print(time_plot_path)
    print(fit_plot_path)
    print(csv_path)


if __name__ == "__main__":
    main()

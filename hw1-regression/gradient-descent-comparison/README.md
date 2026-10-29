# Gradient Descent Variants: Batch vs. SGD vs. Mini-Batch

A side-by-side comparison of the three gradient-descent strategies on a simple linear regression
problem, implemented from scratch with NumPy.

## Setup

- **Data:** 1,000 synthetic points from `y = 2x + 5 + noise` (Gaussian noise, std 0.5), seed 42.
- **Model / loss:** `ŷ = w·x + b` with mean squared error.
- **Training:** 500 epochs, learning rate 0.01, mini-batch size 64 (16 mini-batches per epoch).

| Variant | Gradient computed from | Updates per epoch |
|---|---|---|
| Batch GD | the whole dataset | 1 |
| SGD | one random sample | 1,000 |
| Mini-batch GD | a batch of 64 samples | 16 |

## Results

| Method | Final `w` (true 2) | Final `b` (true 5) | Final loss | Total updates | Run time |
|---|---|---|---|---|---|
| Batch GD | 2.065 | 4.587 | 0.278 | 500 | 0.03 s |
| SGD | 1.992 | 4.970 | 0.247 | 500,000 | 6.4 s |
| Mini-batch GD | 2.018 | 4.982 | **0.244** | 8,000 | 0.12 s |

(Run times depend on the machine; the table is from the saved `outputs/results.csv`.)

- The noise variance is 0.5² = 0.25, so a loss around 0.25 is the best any model can reach — SGD and mini-batch end essentially at this floor, while batch GD is still slightly above it after 500 updates.
- **Batch GD** takes a smooth, direct path but makes very few updates, so with this budget it stops farthest from the true parameters.
- **SGD** makes the most progress per epoch, but its path is noisy and it needs 500,000 updates, which makes it by far the slowest in wall-clock time here.
- **Mini-batch GD** gets the lowest loss with a fraction of SGD's updates and time: averaging 64 samples reduces the noise of the gradient, and the batch computation is vectorized.

### When to use which

- Prefer **SGD** (or mini-batches) when the dataset is very large or redundant, when the data arrives as a stream, or when the loss surface is non-convex — the noise can help escape poor regions.
- Prefer **Batch GD** for small, convex problems where a stable, deterministic convergence is wanted and one pass over the data is cheap.
- **Mini-batch GD** is the usual compromise: stable enough, fast per update, and efficient on vectorized hardware.

## Figures (`outputs/`)

| File | Shows |
|---|---|
| `data_plot.png` | The generated dataset |
| `loss_comparison.png` | MSE loss per epoch for the three methods (batch GD starts near 31 and descends smoothly but slowly; SGD starts low and stays noisy; mini-batch settles within about 30 epochs) |
| `contour_paths.png` | Contour plot of the cost surface with each method's path in `(w, b)` space |
| `time_comparison.png` | Execution time of each method |
| `final_fit_comparison.png` | Fitted lines against the data |
| `results.csv` | Final parameters, loss, number of updates, execution time |

## Run

```bash
python gradient_descent_comparison.py
```

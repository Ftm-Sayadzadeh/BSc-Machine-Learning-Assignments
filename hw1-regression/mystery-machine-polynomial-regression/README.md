# The Mystery Function — Polynomial Regression from Scratch

A hidden function `F(x, y, z)` is only known through 10,000 recorded input/output samples
(`train.xlsx`). The goal is to **recover the function** with a polynomial regression built from
scratch (no machine-learning libraries) and use it to predict `F` for new inputs.

## Approach

1. **Feature map.** The function is a cubic polynomial in three variables, so each sample is expanded into
   20 features: a constant, `x, x², x³`, `y, y², y³`, `z, z², z³`, and the cross terms
   `xy, x²y, xy², xz, x²z, xz², yz, y²z, yz², xyz`.
2. **Normal equations.** The weights minimize the sum of squared errors:
   `(XᵀX) w = Xᵀ f`, solved with `np.linalg.solve` rather than an explicit matrix inverse (more numerically stable).
3. **Hard-coded prediction.** After training, the learned weights (rounded to 4 decimals) are pasted into the
   prediction function so it runs with only `math` — the submission environment did not allow `pandas` at prediction time.
   The result is rounded **up** to two decimals, as the task required.

## Results

- Training RMSE of the 20-parameter model on `train.xlsx` is **0.0** — the data is generated exactly by a polynomial of this form.
- The recovered weights are clean numbers, which confirms the fit:

| Group | Term : weight |
|---|---|
| Constant | `b` : 0 |
| Powers of `x` | `x` : 0.50, `x²` : 1.00, `x³` : 0.25 |
| Powers of `y` | `y` : 0.60, `y²` : 0.20, `y³` : 0.40 |
| Powers of `z` | `z` : 0.25, `z²` : 0.11, `z³` : 0.23 |
| Cross terms | `xy` : 0.31, `x²y` : 0.76, `xy²` : 0.14, `xz` : 0.15, `x²z` : 0.62, `xz²` : 0.89, `yz` : 0.30, `y²z` : 0.10, `yz²` : 0.60, `xyz` : 0.21 |

- The cross terms matter: a model with only the 10 pure-power terms (no products) has an RMSE of about **248.6** on the same data.
- `mystery_function.py` reproduces the two example cases given with the assignment (`F(5, 7, 8) = 1313.3`, `F(-10, -5, 2) = -503.32`).

## Files

| File | Purpose |
|---|---|
| `fit_weights.py` | Training step: builds the 20 features from `train.xlsx` and solves for the weights (uses `pandas`/`openpyxl` to read the file) |
| `mystery_function.py` | Submitted predictor with the learned weights hard-coded; reads `x`, `y`, `z` from standard input |
| `train.xlsx` | Course-provided training data (columns `x`, `y`, `z`, `F(x, y, z)`) |

## Run

```bash
# recover the weights from the data
python fit_weights.py

# predict F(x, y, z): enter x, y and z on separate lines
python mystery_function.py
```

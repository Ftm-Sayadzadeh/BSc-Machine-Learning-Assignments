# Penguin Species — Logistic Regression from Scratch

Binary logistic regression written from scratch with NumPy to separate **Adelie** (label 0) from
**Gentoo** (label 1) penguins using four body measurements, with L2 regularization and a
validation-based hyper-parameter search.

## Data

`002_penguins.csv` — Palmer Penguins. Only the two species above and the features `bill_length_mm`,
`bill_depth_mm`, `flipper_length_mm` and `body_mass_g` are kept; rows with a missing value in any of
them are dropped. This leaves **274 penguins** (151 Adelie, 123 Gentoo).

## Approach

1. **Split** 60 % / 20 % / 20 % into train / validation / test (164 / 55 / 55), seed 42.
2. **Standardize** with training-set statistics only, then add a bias column.
3. **Model:** `p(y=1 | x) = σ(wᵀx)` with a numerically **stable sigmoid**.
4. **Loss:** binary cross-entropy plus an L2 penalty `λ/(2m) · Σ wⱼ²` that does **not** regularize the bias; trained with batch gradient descent (3,000 epochs).
5. **Search** over learning rate ∈ {0.001, 0.01, 0.1} × λ ∈ {0, 0.01, 0.1}, picking the best combination on the **validation** set by F1-score (ties broken by validation loss).

## Results

Selected hyper-parameters: **learning rate 0.1, λ = 0.**

Test set (55 penguins):

| Accuracy | Precision | Recall | F1 |
|---|---|---|---|
| 1.0000 | 1.0000 | 1.0000 | 1.0000 |

```
Confusion matrix (rows = true, columns = predicted; Adelie, Gentoo)
[[31  0]
 [ 0 24]]
```

- All nine configurations reach a validation F1 of 1.0, so the choice is decided by the validation loss: a larger learning rate converges further in 3,000 epochs (loss 0.0033 for lr 0.1 vs. 0.153 for lr 0.001), and the unregularized model has the lowest loss.
- **The two species are almost linearly separable** in these four measurements (Gentoo is clearly larger, with longer flippers and a shallower bill), so a perfect test score is expected. It also means this problem says little about the effect of regularization: there is no sign of overfitting to counter, and increasing λ only raises the validation loss slightly.
- With just 55 test samples, a score of 100 % is limited evidence of how the model would do on penguins outside this sample.
- `q2_logistic_loss_curves.png` shows training vs. validation loss per epoch. The two curves decrease together with no gap opening up, which is consistent with no overfitting. Predicted probabilities for five test penguins are printed by the script (the five printed Adelie samples all get `p(Gentoo)` below 0.02).

## Files

| File | Purpose |
|---|---|
| `penguins_logistic_regression.py` | Full pipeline: cleaning, split, training, hyper-parameter search, evaluation |
| `002_penguins.csv` | Dataset |
| `q2_validation_hyperparameter_search.csv` | Validation results for every learning rate / λ combination |
| `q2_logistic_loss_curves.png` | Training and validation loss per epoch |

## Run

```bash
python penguins_logistic_regression.py
```

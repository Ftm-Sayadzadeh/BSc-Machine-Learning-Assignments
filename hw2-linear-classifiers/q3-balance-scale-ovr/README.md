# Balance Scale — One-vs-Rest Logistic Regression

Predict which way a balance scale tips from four numbers, using a **One-vs-Rest logistic
regression written from scratch** with NumPy. The problem is small but has a realistic catch: one
class is rare, so accuracy alone is misleading and Macro-F1 is used as well.

## Data

`003_balance-scale.data.csv` — 625 scale configurations described by `left_weight`, `left_distance`,
`right_weight`, `right_distance` (each from 1 to 5). The class says where the scale tips:

| Class | Meaning | Label | Samples |
|---|---|---|---|
| `L` | tips left | 0 | 288 |
| `B` | balanced | 1 | 49 |
| `R` | tips right | 2 | 288 |

## Approach

1. **Split** 60 % / 20 % / 20 % into train / validation / test (375 / 125 / 125), seed 42.
2. **Standardize** with training-set statistics only, then add a bias column.
3. **One-vs-Rest:** one binary logistic regression per class (class *k* vs. everything else), each trained separately from zero; the weight vectors form a 3 × 5 matrix `W`. A new sample is assigned to the class with the largest sigmoid score.
4. **Hyper-parameters** (learning rate, L2 strength λ, epochs) are chosen on the validation set by **Macro-F1**; the best setting was learning rate 0.01, λ = 0, 6,000 epochs.
5. **Error analysis only:** an auxiliary column `torque_difference = left_weight·left_distance − right_weight·right_distance` is computed to explain mistakes. It is **not** used to train the model.

## Results on the test set (125 samples)

| Accuracy | Macro-F1 |
|---|---|
| 0.8880 | 0.6167 |

```
Confusion matrix (rows = true, columns = predicted; L, B, R)
    L  B   R
L  55  0   3
B   4  0   6
R   1  0  56
```

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| L | 0.9167 | 0.9483 | 0.9322 | 58 |
| B | 0.0000 | 0.0000 | 0.0000 | 10 |
| R | 0.8615 | 0.9825 | 0.9180 | 57 |

### What the numbers say

- **Accuracy hides a failure.** 88.8 % looks good, but the model **never predicts the balanced class**: its recall and F1 are 0. Macro-F1 averages the per-class scores with equal weight, so it drops to 0.62 and exposes this.
- **Why class `B` is hard.** The scale balances exactly when `left_weight × left_distance = right_weight × right_distance` — a relationship built from *products* of features. A linear model on the raw features can only weigh each number separately, so it can learn "heavier or farther on the left → `L`", but not "the two products are equal". Class `B` is also only 8 % of the data, so the learned weights for `B` stay near zero and its score is almost constant.
- **The errors are near the balance point.** `torque_difference` is exactly 0 for every `B` sample. All 14 misclassified test samples have a small absolute torque difference (mean 0.43, maximum 2): the model makes mistakes where the scale is almost balanced, not on clear cases.
- Classes `L` and `R` are easier because their score is monotone in the weights and distances, which a linear model can represent; they are the main source of the accuracy.
- **The loss curves show the same thing.** In `q3_ovr_loss_curves.png` the `L` and `R` models keep improving for all 6,000 epochs (loss ≈ 0.18), while the `B` model flattens out early at about 0.27 on the training set and 0.28 on validation — more training does not help because the raw features simply cannot express "balanced".
- A natural next step (not part of this assignment) would be to add features that capture the products, such as the two torques or the absolute torque difference, so the balanced class can be separated by a simple threshold.

## Files

| File | Purpose |
|---|---|
| `balance_scale_ovr.py` | Full pipeline: data report, split, One-vs-Rest training, validation search, evaluation, error analysis |
| `003_balance-scale.data.csv` | Dataset |
| `q3_weight_matrix_W.csv` | Learned weight matrix `W` (one row per class) |
| `q3_validation_hyperparameter_search.csv` | Validation results for every hyper-parameter combination |
| `q3_misclassified_samples.csv` | First misclassified test samples with their class scores |
| `q3_ovr_loss_curves.png` | Training and validation loss of the three binary models |
| `q3_left_weight_right_weight_scatter.png` | 2D view of the data using `left_weight` and `right_weight` |

## Run

```bash
python balance_scale_ovr.py
```

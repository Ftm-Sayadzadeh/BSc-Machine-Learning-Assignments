# Raisin Classification — Linear Classifier with Hinge Loss

A binary linear classifier trained from scratch with NumPy (no ready-made classifier) to tell two
raisin varieties apart, `Kecimen` (label 0) and `Besni` (label 1), from seven shape measurements.

## Data

`001_Raisin_Dataset.xlsx` — 900 raisins, 450 per class, with 7 numeric features: `Area`,
`MajorAxisLength`, `MinorAxisLength`, `Eccentricity`, `ConvexArea`, `Extent`, `Perimeter`.

## Approach

1. **Split** 70 % / 15 % / 15 % into train / validation / test (630 / 135 / 135 samples), seed 42.
2. **Standardize** the features with the mean and standard deviation of the **training set only**, then add a constant column for the bias.
3. **Model:** `z = wᵀx`, predict class 1 when `z ≥ 0`.
4. **Loss:** hinge loss `mean(max(0, 1 − yᵢ·wᵀxᵢ))` with labels temporarily mapped from `{0, 1}` to `{−1, +1}`, minimized by full-batch (sub)gradient descent.
5. **Hyper-parameters** (learning rate and number of epochs) are chosen on the validation set by F1-score; the best setting was a learning rate of 0.1 with 2,000 epochs.

## Results on the test set

| Features | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| All 7 features | 0.9037 | 0.9091 | 0.8955 | 0.9023 |
| Only `Area` and `Perimeter` | 0.8889 | 0.8939 | 0.8806 | 0.8872 |

Confusion matrices (rows = true, columns = predicted; order `Kecimen`, `Besni`):

```
7 features            Area + Perimeter
[[62  6]              [[61  7]
 [ 7 60]]              [ 8 59]]
```

- Precision and recall are balanced (about 0.90 each), so the model is not systematically favouring one variety.
- **Two features are almost enough.** Using only `Area` and `Perimeter` loses about 1.5 accuracy points, which suggests most of the separation is carried by size. The remaining errors lie in a region where the two varieties genuinely overlap, so a straight line cannot do much better in 2D — see `q1_area_perimeter_decision_boundary.png`.
- The first five misclassified test samples (4 `Besni` predicted as `Kecimen`, 1 the other way round) all have a score `z` close to the decision threshold (|z| ≤ 1.2), i.e. they sit near the boundary rather than being confidently wrong.
- A linear boundary is a reasonable model for this data (about 90 % accuracy) but not a perfect one.

## Files

| File | Purpose |
|---|---|
| `raisin_hinge_classifier.py` | Full pipeline: data report, split, standardization, training, validation search, test evaluation, 2D experiment |
| `001_Raisin_Dataset.xlsx` | Dataset |
| `q1_training_hinge_loss.png` | Hinge loss during training |
| `q1_area_perimeter_decision_boundary.png` | Data points and decision boundary using `Area` and `Perimeter` |

## Run

```bash
python raisin_hinge_classifier.py
```

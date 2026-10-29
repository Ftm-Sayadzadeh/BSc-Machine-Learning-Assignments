# House Price Prediction — Linear Regression from Scratch

Predict the sale price of a house from its size, rooms, number of offers, facade material and
neighborhood, using a **linear regression model trained with gradient descent that is implemented
from scratch** (loss function, gradient and parameter updates are written with NumPy).

## Data

`house-prices.csv` — 128 past sales with `SqFt`, `Bedrooms`, `Bathrooms`, `Offers`, `Brick`,
`Neighborhood` and the target `Price`. A few numeric cells are missing (2 in `SqFt`, 3 in `Bedrooms`,
1 in `Bathrooms`, 2 in `Offers`).

## Pipeline

1. **Split first, 90 % train / 10 % test** (`random_state=42`), so nothing about the test set leaks into preprocessing.
2. **Missing values** — numeric columns filled with the training mean, categorical columns with the training mode.
3. **Outliers** — detected with a z-score rule (|z| > 3) and reported per column. Continuous features (`SqFt`) would be clipped to the 1st–99th percentile of the training data, while count features such as `Offers` are valid values and are kept. In this dataset only one outlier is found (an `Offers` value), so it is reported but not changed.
4. **Encoding** — `Brick` and `Neighborhood` one-hot encoded (first level dropped).
5. **Scaling** — min-max scaling of the features and of the target, with statistics taken from the training set only.
6. **Model** — `LinearRegressionGD`: minimizes **RMSE** with gradient descent; the loss is recorded every epoch.
7. **Reference check** — scikit-learn's `LinearRegression` is fitted on the same data only to confirm the custom model.

## Results

| Model | Train RMSE | Test RMSE |
|---|---|---|
| Custom gradient descent (RMSE loss) | $10,013.58 | $9,799.80 |
| scikit-learn `LinearRegression` (reference) | $10,013.58 | $9,799.80 |

- The from-scratch model matches the closed-form least-squares solution to two decimals, so gradient descent converged to the optimum (the scaled RMSE has settled by epoch 1,000 — see `05_rmse_loss_curve.png`).
- Test RMSE is about 7.5 % of the mean price (≈ $130k), and slightly lower than the train RMSE — with only 13 test houses this difference is within noise.

## Exploratory plots (`outputs_corrected/`)

| File | Shows |
|---|---|
| `01_correlation_heatmap.png` | Correlation between numeric features and price |
| `02_regplot_*.png` | Price vs. each numeric feature with a regression line |
| `03_boxplot_*.png` | Price by `Brick` and by `Neighborhood` |
| `04_encoded_scatter_*.png` | Price vs. each encoded/scaled feature |
| `05_rmse_loss_curve.png` | Training RMSE per epoch |
| `report_outputs_corrected.txt` | Data info, null counts, outlier report and final metrics |

## Run

```bash
python house_price_regression.py
```

Figures and `report_outputs_corrected.txt` are (re)written to `outputs_corrected/`.

# BSc Machine Learning — Course Projects

Programming assignments from the **Fundamentals of Machine Learning** course
(BSc Computer Engineering, University of Isfahan, Spring 2026).

The focus of the course was on implementing the core algorithms **from scratch with NumPy**
instead of calling ready-made models, then evaluating them properly: train/validation/test
splits, scaling fitted on the training set only, no data leakage, and metrics that are
interpreted rather than just reported.

> Only the programming assignments are included here; written/theory answers are not.

## Contents

| Folder | Topic | What is implemented |
|---|---|---|
| [`hw1-regression/house-price-linear-regression`](hw1-regression/house-price-linear-regression) | Linear regression | Gradient-descent regression with RMSE loss, missing-value handling, outlier handling, one-hot encoding, min-max scaling, EDA plots |
| [`hw1-regression/mystery-machine-polynomial-regression`](hw1-regression/mystery-machine-polynomial-regression) | Polynomial regression | Recovering a hidden cubic function of 3 variables (20 parameters) with the normal equations |
| [`hw1-regression/gradient-descent-comparison`](hw1-regression/gradient-descent-comparison) | Optimization | Batch vs. SGD vs. Mini-batch gradient descent: loss curves, weight paths on the cost contour, update counts and run time |
| [`hw2-linear-classifiers/q1-raisin-hinge-classifier`](hw2-linear-classifiers/q1-raisin-hinge-classifier) | Binary linear classification | Linear classifier trained with hinge loss on the Raisin dataset, 2D decision boundary |
| [`hw2-linear-classifiers/q2-penguins-logistic-regression`](hw2-linear-classifiers/q2-penguins-logistic-regression) | Logistic regression | Binary logistic regression with L2 regularization, stable sigmoid, hyper-parameter search on a validation set |
| [`hw2-linear-classifiers/q3-balance-scale-ovr`](hw2-linear-classifiers/q3-balance-scale-ovr) | Multiclass classification | One-vs-Rest logistic regression on an imbalanced 3-class problem, Macro-F1 and error analysis |
| [`hw3-neural-network-iris`](hw3-neural-network-iris) | Neural networks | A 4-12-3 network with manual forward pass, softmax, cross-entropy, backpropagation and gradient descent on Iris |

Each project folder has its own README with the approach, results and how to run it.

## Results at a glance

| Project | Key result |
|---|---|
| House price regression | Test RMSE ≈ $9.8k (about 7.5 % of the mean price); identical to scikit-learn's closed-form solution |
| Mystery function | Training RMSE 0 — the hidden cubic function and its 20 weights are recovered exactly |
| Gradient descent comparison | Mini-batch GD reaches the lowest loss (0.244) with 8,000 updates; SGD needs 500,000 |
| Raisin (hinge loss) | Test accuracy 90.4 % with 7 features, 88.9 % with only `Area` and `Perimeter` |
| Penguins (logistic regression) | Test accuracy 100 % — Adelie and Gentoo are almost linearly separable |
| Balance Scale (One-vs-Rest) | Accuracy 88.8 % but Macro-F1 only 0.62: a linear model never predicts the rare "balanced" class |
| Iris neural network | Test accuracy 96.7 % (29 / 30) with a from-scratch network |

## Setup

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux / macOS
source .venv/bin/activate

pip install -r requirements.txt
```

Every script reads its data and writes its figures relative to the **current directory**,
so run it from inside its own project folder:

```bash
cd hw2-linear-classifiers/q1-raisin-hinge-classifier
python raisin_hinge_classifier.py
```

All experiments use a fixed random seed (42) so the results are reproducible.

## Data

The datasets are small and are included next to the code that uses them so each project runs on its own.

| Data | Used in | Origin |
|---|---|---|
| `house-prices.csv`, `train.xlsx` | HW1 | Provided by the course |
| `001_Raisin_Dataset.xlsx` | HW2 Q1 | Raisin dataset (UCI Machine Learning Repository) |
| `002_penguins.csv` | HW2 Q2 | Palmer Penguins dataset |
| `003_balance-scale.data.csv` | HW2 Q3 | Balance Scale dataset (UCI Machine Learning Repository) |
| Iris | HW3 | Loaded from scikit-learn (`sklearn.datasets.load_iris`), no file needed |

Please refer to the original sources for each dataset's license and citation.

## Author

Fatemeh Sayadzadeh — [GitHub](https://github.com/Ftm-Sayadzadeh)

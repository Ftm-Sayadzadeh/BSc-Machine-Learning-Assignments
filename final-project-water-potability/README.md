# Water Potability — Comparing Ten Classifiers Implemented from Scratch

Final course project: a complete classification pipeline that predicts whether a water sample is
safe to drink (`Potability` = 1) from nine physical and chemical measurements. All ten models are
**implemented by hand with NumPy/pandas** — no scikit-learn model classes are imported — and are
compared with cross-validation, tuned decision thresholds and several metrics, including a
safety-oriented error analysis.

The notebook keeps its executed outputs, so you can read the tables and figures on GitHub without running it.

## Data

3,276 samples, 9 features (`ph`, `Hardness`, `Solids`, `Chloramines`, `Sulfate`, `Conductivity`,
`Organic_carbon`, `Trihalomethanes`, `Turbidity`) and the target `Potability`.

- **Class balance:** 61 % non-potable (0) vs. 39 % potable (1), so always predicting "non-potable" already gives **61.0 % accuracy** — the baseline every model has to beat.
- **Missing values:** `Sulfate` 23.8 %, `ph` 15.0 %, `Trihalomethanes` 4.9 %.
- **Included as `water_potability.csv`** (522 KB, 3,276 rows) next to the notebook, so it runs as is. It is the file the notebook was run with on Kaggle (dataset `evelin13/water-potability-2`); the row count, missing-value counts and class balance above match the saved notebook outputs. Please check the Kaggle page of the dataset for its license and citation. On Kaggle you can instead attach that dataset; the notebook looks in the current folder, `../input` and `/kaggle/input`.

## Pipeline

1. **EDA** — histograms and boxplots for every feature, correlation heatmap, missing-value report. Outliers are *kept*: in water quality, extreme values can be real contamination rather than noise.
2. **Split** — stratified 80 % / 20 % train / test (seed 42), written by hand.
3. **Leak-free preprocessing** (`WaterPreprocessor`, also hand-written, fitted on training data only):
   missing-value indicator columns, a few domain features (pH distance from neutral, pH inside the WHO range 6.5–8.5, solids-to-conductivity ratio, mineral-load and disinfection-by-product indices), median imputation, and standardization for the scale-sensitive models.
4. **Models** (all from scratch): Logistic Regression, KNN, SVM with RBF kernel, Decision Tree, Gradient Boosting, AdaBoost, Random Forest, MLP — plus two bonus models, Extra Trees and Gaussian Naive Bayes.
5. **Tuning** — a small grid per model, scored by 3-fold cross-validated F1. The **decision threshold** of every model is also tuned on the validation folds (grid 0.20–0.80). The rule prefers thresholds where at most 20 % of non-potable samples are wrongly called potable ("dangerous false positives") and falls back to the best F1 if no threshold satisfies it.
6. **Evaluation** — the test set is used only for the final report; model, parameters and threshold are all chosen without it. Metrics: accuracy, precision, recall, F1, balanced accuracy, specificity, ROC-AUC, average precision, dangerous-FP rate, confusion matrices and a train-vs-test F1 gap for overfitting.

## Results on the test set (656 samples)

Sorted by cross-validated F1. "Dangerous FP rate" is the share of non-potable samples predicted potable.

| Model | CV F1 | Test accuracy | Test F1 | Test ROC-AUC | Dangerous FP rate |
|---|---|---|---|---|---|
| AdaBoost | 0.566 | 0.416 | 0.566 | 0.579 | 0.943 |
| Gradient Boosting | 0.561 | 0.390 | 0.561 | 0.555 | 1.000 |
| MLP Classifier | 0.527 | 0.590 | 0.532 | 0.626 | 0.415 |
| KNN | 0.523 | 0.488 | 0.556 | 0.598 | 0.725 |
| Random Forest | 0.507 | 0.462 | 0.537 | 0.582 | 0.755 |
| Logistic Regression | 0.486 | 0.390 | 0.559 | 0.525 | 0.993 |
| Extra Trees (bonus) | 0.485 | 0.470 | 0.539 | 0.621 | 0.738 |
| SVM (RBF) | 0.455 | 0.633 | 0.414 | **0.654** | 0.175 |
| Decision Tree | 0.411 | **0.648** | 0.438 | 0.612 | 0.163 |
| Gaussian Naive Bayes (bonus) | 0.350 | 0.613 | 0.362 | 0.577 | 0.175 |

Models selected **by cross-validation only**:

- **Best general model:** AdaBoost (CV F1 0.566; test F1 0.566, test accuracy 0.416).
- **Safety-constrained recommendation:** SVM with RBF kernel at threshold 0.60 — the best cross-validated F1 among the models that met the safety limit. On the test set it has a dangerous-FP rate of 17.5 %, accuracy 63.3 % and F1 0.414.

### How to read these numbers

- **This is a hard, noisy dataset, and the models add little over the baseline.** The best test accuracy is 64.8 % (Decision Tree) against the 61.0 % majority baseline, and the best ROC-AUC is only 0.654 (chance is 0.5). The notebook states this explicitly instead of hiding it.
- **High F1 here is misleading.** Seven models were tuned by CV to low thresholds (0.20–0.35), and none of them met the safety limit in cross-validation, so they fall back to maximizing F1 for the potable class. That makes them label most samples "potable": F1 stays between 0.53 and 0.57, but their accuracy falls *below* the 61 % baseline and 40–100 % of unsafe samples are called safe. Gradient Boosting ends up predicting every sample as potable (0 true negatives) — exactly what the trivial "always potable" rule does, with the same F1 of 0.561.
- **The accuracy-versus-safety trade-off is visible.** Models that respect the safety limit (SVM, Decision Tree, Naive Bayes) call water safe far less often and score a lower F1 (0.36–0.44) because their recall of potable water is only 28–35 %.
- **Overfitting:** the MLP has by far the largest train-vs-test F1 gap (0.45); the SVM and Decision Tree show gaps of about 0.20 and 0.16, while AdaBoost, Gradient Boosting and Logistic Regression show almost no gap — consistent with their predictions being close to the trivial rule.
- The test set was not used to pick anything: the Decision Tree has the best test accuracy but the second-lowest CV F1, and it was *not* chosen for that reason.

## Files

| File | Purpose |
|---|---|
| `water_potability_classification.ipynb` | The whole project: EDA, hand-written preprocessing and models, tuning, evaluation, final analysis |
| `water_potability.csv` | The dataset (3,276 samples, 9 features + `Potability`) |

## Run

```bash
pip install -r requirements.txt
jupyter notebook water_potability_classification.ipynb
```

Running the notebook trains all ten models (about 3–4 minutes on Kaggle) and writes figures and tables to a `water_potability_final_outputs/` folder next to the notebook.

**Reproducibility check.** A local re-run on the included dataset finished without errors and reproduced the tables above apart from small numerical differences: the from-scratch SVM moved in the third decimal (for example test accuracy 0.631 instead of 0.633 and dangerous-FP rate 0.180 instead of 0.175; the conclusions do not change), and Gradient Boosting's average precision moved in the fourth decimal. Results can therefore differ a little between environments. The tables in this README come from the outputs saved in the notebook (the original Kaggle run).

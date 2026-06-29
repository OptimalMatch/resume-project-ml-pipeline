# Milestones

Work through these in order and **commit once per milestone**. The goal is a clean git history that shows how you took a raw dataset all the way to a trained, evaluated model.

## 1. Load a real public dataset
- [ ] Pick a real public tabular dataset (e.g. UCI ML Repository, Kaggle, scikit-learn's built-in datasets, or a government open-data portal).
- [ ] Implement `load_dataset()` in `src/load_data.py` to download/read it into a pandas DataFrame.
- [ ] Print the shape and `df.head()` so you can confirm it loaded.
- **Commit:** "Load dataset into DataFrame"

## 2. Exploratory Data Analysis (EDA)
- [ ] Open `notebooks/exploration.ipynb` and load the data.
- [ ] Inspect distributions (histograms), missing values (`df.isna().sum()`), and correlations.
- [ ] Note your target column and which features look predictive.
- **Commit:** "EDA notebook with distributions and missing-value checks"

## 3. Feature engineering
- [ ] Implement transforms in `src/features.py`: impute missing values, encode categoricals, scale numerics, derive any new columns.
- [ ] Decide your feature matrix `X` and target `y`.
- **Commit:** "Feature engineering: impute, encode, scale"

## 4. Train a model
- [ ] In `src/train.py`, split the data with `train_test_split`.
- [ ] Fit a scikit-learn model (classification or regression).
- [ ] Confirm it runs end-to-end: `python -m src.train`.
- **Commit:** "Train baseline scikit-learn model"

## 5. Evaluate properly
- [ ] Report the right metrics: accuracy / F1 / ROC-AUC for classification, or RMSE / R² for regression.
- [ ] Add cross-validation (`cross_val_score`) so your numbers aren't a fluke of one split.
- **Commit:** "Evaluate with cross-validation and proper metrics"

## 6. Save the model + write up findings
- [ ] Persist the trained model with `joblib.dump`.
- [ ] Write up your findings (in the README or the notebook): what worked, what didn't, what you'd try next.
- **Commit:** "Save model and write up findings"

## Stretch
- [ ] Compare multiple models.
- [ ] Tune hyperparameters with `GridSearchCV` / `RandomizedSearchCV`.
- [ ] Wrap preprocessing + model in a scikit-learn `Pipeline`.
- [ ] Plot feature importances.

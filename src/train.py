"""Train and evaluate a supervised model, then save it.

Milestones 4-6: split the data, fit a scikit-learn model, report proper
metrics with cross-validation, and persist the trained model with joblib.

Run it once the stubs are filled in:
    python -m src.train
"""
from __future__ import annotations

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import classification_report

from src.features import build_features
from src.load_data import load_dataset

# TODO: set this to your dataset's target column name
TARGET = "target"
MODEL_PATH = "model.joblib"


def main() -> None:
    # 1. Load + engineer features
    df = load_dataset()
    X, y = build_features(df, target=TARGET)

    # 2. Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 3. Fit a model
    # TODO: try other models too (LogisticRegression, GradientBoosting, ...).
    # For a regression target, swap in a regressor + regression metrics.
    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)

    # 4. Evaluate — proper metrics, not just one accuracy number
    # TODO: for classification, also report ROC-AUC (roc_auc_score on
    # predict_proba). For regression, use RMSE and R2 instead.
    print("== Held-out test set ==")
    print(classification_report(y_test, model.predict(X_test)))

    # 5. Cross-validation so the score isn't a fluke of one split
    cv_scores = cross_val_score(model, X, y, cv=5, scoring="f1_weighted")
    print(f"5-fold CV f1_weighted: {cv_scores.mean():.3f} +/- {cv_scores.std():.3f}")

    # 6. Persist the trained model
    joblib.dump(model, MODEL_PATH)
    print(f"Saved trained model to {MODEL_PATH}")


if __name__ == "__main__":
    main()

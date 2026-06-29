"""Feature-engineering transforms.

Milestone 3: turn the raw DataFrame into a model-ready feature matrix.
Typical steps: impute missing values, encode categoricals, scale numerics,
and derive any new columns that might help the model.
"""
from __future__ import annotations

import pandas as pd


def build_features(df: pd.DataFrame, target: str) -> tuple[pd.DataFrame, pd.Series]:
    """Split a raw DataFrame into a feature matrix X and target vector y.

    TODO: implement real feature engineering. Ideas:
      - Fill missing values (df.fillna(...), or sklearn's SimpleImputer).
      - One-hot encode categoricals (pd.get_dummies, or OneHotEncoder).
      - Scale numeric columns (StandardScaler / MinMaxScaler).
      - Derive new columns (ratios, date parts, text length, ...).

    Tip: once it works, consider wrapping these steps in a
    sklearn.pipeline.Pipeline + ColumnTransformer so preprocessing
    travels with the model.
    """
    # TODO: clean / encode / scale before returning
    y = df[target]
    X = df.drop(columns=[target])

    # TODO: replace this naive encoding with something deliberate
    X = pd.get_dummies(X)

    return X, y

"""Load a public tabular dataset into a pandas DataFrame.

Milestone 1: pick a real public dataset and return it as a DataFrame.
Good sources: the UCI ML Repository, Kaggle, scikit-learn's built-in
datasets (sklearn.datasets), or any government open-data CSV.
"""
from __future__ import annotations

import pandas as pd


def load_dataset() -> pd.DataFrame:
    """Download or read a dataset and return it as a DataFrame.

    TODO: Replace this stub with a real dataset.
      Option A — read a CSV (local or URL):
          return pd.read_csv("data/your_dataset.csv")
      Option B — use a built-in scikit-learn dataset:
          from sklearn.datasets import fetch_openml
          data = fetch_openml(name="titanic", version=1, as_frame=True)
          return data.frame

    Whatever you choose, make sure the returned DataFrame has a clear
    target column you want to predict.
    """
    # TODO: load your real dataset here
    raise NotImplementedError("Implement load_dataset() — see the docstring.")


if __name__ == "__main__":
    df = load_dataset()
    print(f"Loaded dataset with shape {df.shape}")
    print(df.head())
